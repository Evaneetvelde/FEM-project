from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import meshio
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from calculs.gmsh_utils import (gmsh_init, gmsh_finalize,prepare_quadrature_and_basis, get_jacobians,end_dofs_from_nodes)
from calculs.stiffness import assemble_stiffness_and_rhs
from calculs.mass import assemble_mass
from calculs.dirichlet import theta_step

from materialsbank import get_material

# Physical ID to material name mapping (based on gmsh physical groups)
PHYSICAL_ID_MAP = {
	1: "bois",    # Physical ID 1 → Wood
	2: "beton",   # Physical ID 2 → Concrete (walls)
}


def _physical_id_to_name_map(msh: meshio.Mesh) -> dict[int, str]:
	result: dict[int, str] = {}
	for name, data in getattr(msh, "field_data", {}).items():
		phys_id = int(data[0])
		phys_dim = int(data[1])
		if phys_dim == 2:
			result[phys_id] = str(name)
	
	# If no field_data found (mesh has no named physical groups), use hardcoded mapping
	if not result:
		result = PHYSICAL_ID_MAP.copy()
	
	return result


def _element_matrices_2d(coords: np.ndarray, k_val: float, pc: float):
	x = coords[:, 0]
	y = coords[:, 1]
	det_j = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
	area = 0.5 * abs(det_j)
	b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]], dtype=float) / det_j
	c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]], dtype=float) / det_j
	ke = k_val * area * (np.outer(b, b) + np.outer(c, c))
	me = (pc * area / 12.0) * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=float)
	munit = (area / 12.0) * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=float)
	return ke, me, munit


def _assemble_system(points: np.ndarray, elems: np.ndarray, phys_ids: np.ndarray, phys_name_map: dict[int, str]):
	n_nodes = points.shape[0]
	k_glob = lil_matrix((n_nodes, n_nodes), dtype=float)
	m_glob = lil_matrix((n_nodes, n_nodes), dtype=float)
	m_unit = lil_matrix((n_nodes, n_nodes), dtype=float)
	q_node = np.zeros(n_nodes, dtype=float)
	tc_node = np.full(n_nodes, np.inf, dtype=float)

	material_usage = {}  # Track which materials are used

	for e, nodes in enumerate(elems):
		pid = int(phys_ids[e])
		mat_name = phys_name_map.get(pid, "bois")  # Default to "bois" if no mapping
		mat = get_material(mat_name)
		k_val = float(mat["k"])
		pc = float(mat["rho"]) * float(mat["c"])
		q = float(mat["Q"])
		tc = float(mat["Tc"])
		
		# Track material usage
		mat_name_lower = str(mat_name).strip().lower()
		if mat_name_lower not in material_usage:
			material_usage[mat_name_lower] = {"count": 0, "k": k_val, "rho*c": pc, "Q": q, "Tc": tc}
		material_usage[mat_name_lower]["count"] += 1
		
		ke, me, munit = _element_matrices_2d(points[nodes, :2], k_val, pc)

		for i, ni in enumerate(nodes):
			q_node[ni] = max(q_node[ni], q)
			tc_node[ni] = min(tc_node[ni], tc)
			for j, nj in enumerate(nodes):
				k_glob[ni, nj] += ke[i, j]
				m_glob[ni, nj] += me[i, j]
				m_unit[ni, nj] += munit[i, j]

	return k_glob.tocsr(), m_glob.tocsr(), m_unit.tocsr(), q_node, tc_node


def _scenario_defaults() -> dict:
	return {
		"dt": 2.0,
		"steps": 50,
		"sub_steps": 1,
		"theta": 1.0,
		"h_conv": 1.0,
		"t_amb": 293.0,
		"src_temp": 800.0,
		"src_x": 0.0,
		"src_y": 0.0,
		"src_radius": 0.5,
	}


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Simulation FEM diffusion-reaction 2D")
	parser.add_argument("--mesh", type=str, default=None, help="Nom du maillage .msh dans models/")
	parser.add_argument("--dt", type=float, default=None, help="Pas de temps")
	parser.add_argument("--steps", type=int, default=None, help="Nombre d'iterations")
	parser.add_argument("--sub-steps", dest="sub_steps", type=int, default=None, help="Sous-iterations par frame d'animation")
	parser.add_argument("--theta", type=float, default=None, help="Schema theta (1=Euler implicite)")
	parser.add_argument("--h-conv", dest="h_conv", type=float, default=None, help="Coefficient de convection")
	parser.add_argument("--t-amb", dest="t_amb", type=float, default=None, help="Temperature ambiante")
	parser.add_argument("--src-temp", dest="src_temp", type=float, default=None, help="Temperature initiale source")
	parser.add_argument("--src-x", type=float, default=None, help="X source")
	parser.add_argument("--src-y", type=float, default=None, help="Y source")
	parser.add_argument("--src-radius", type=float, default=None, help="Rayon source (2D)")
	parser.add_argument("--no-plot", dest="plot", action="store_false", help="Desactive l'affichage final")
	parser.set_defaults(plot=True)
	return parser


def run(args: argparse.Namespace) -> np.ndarray:
	defaults = _scenario_defaults()
	dt = float(args.dt if args.dt is not None else defaults["dt"])
	steps = int(args.steps if args.steps is not None else defaults["steps"])
	sub_steps = max(1, int(args.sub_steps if args.sub_steps is not None else defaults["sub_steps"]))
	theta = float(args.theta if args.theta is not None else defaults["theta"])
	h_conv = float(args.h_conv if args.h_conv is not None else defaults["h_conv"])
	t_amb = float(args.t_amb if args.t_amb is not None else defaults["t_amb"])
	src_temp = float(args.src_temp if args.src_temp is not None else defaults["src_temp"])
	src_x = float(args.src_x if args.src_x is not None else defaults["src_x"])
	src_y = float(args.src_y if args.src_y is not None else defaults["src_y"])
	src_radius = float(args.src_radius if args.src_radius is not None else defaults["src_radius"])

	mesh_file = args.mesh or "piece.msh"
	base_dir = Path(__file__).parent
	mesh_path = Path(mesh_file)
	if not mesh_path.exists():
		candidate = base_dir / mesh_file
		mesh_path = candidate if candidate.exists() else base_dir / "models" / mesh_file
	print(f"Using mesh: {mesh_path}")

	msh = meshio.read(str(mesh_path))
	pts = np.asarray(msh.points, dtype=float)[:, :2]
	elems = _get_cells(msh)
	phys = _get_physical_ids(msh, elems.shape[0])
	phys_name_map = _physical_id_to_name_map(msh)
	print(f"Maillage charge: {len(pts)} noeuds, {len(elems)} elements (2D).")

	k_mat, m_mat, m_unit, q_node, tc_node = _assemble_system(pts, elems, phys, phys_name_map)
	t = np.full(len(pts), t_amb, dtype=float)
	dist = np.hypot(pts[:, 0] - src_x, pts[:, 1] - src_y)
	t[dist <= src_radius] = src_temp

	ones = np.full(len(t), t_amb, dtype=float)
	a_lhs = m_mat.tocsr().multiply(1.0 / dt) + (k_mat + h_conv * m_unit).tocsr()

	walls_id = None
	for pid, name in phys_name_map.items():
		if any(k in name.lower() for k in ("mur", "murs", "wall", "walls")):
			walls_id = pid
			break

	fig, ax = plt.subplots(figsize=(10, 8))
	vmin_plot = t_amb
	vmax_plot = max(src_temp, float(np.max(t)))
	triang = Triangulation(pts[:, 0], pts[:, 1], elems)
	cbar = None

	def draw_frame(field: np.ndarray, time_s: float) -> None:
		nonlocal cbar
		ax.clear()
		im = ax.tripcolor(triang, np.asarray(field, dtype=float), cmap="magma", shading="gouraud", vmin=vmin_plot, vmax=vmax_plot)
		
		if walls_id is not None and phys is not None:
			wall_mask = np.asarray(phys) == walls_id
			murs_elements = elems[wall_mask]
			if len(murs_elements) > 0:
				verts = [pts[tri, :2] for tri in murs_elements]
				murs_poly = PolyCollection(verts, facecolors="#808080", edgecolors="black", linewidths=0.5, zorder=3)
				ax.add_collection(murs_poly)
		
		ax.set_aspect("equal")
		ax.set_facecolor("#1A12BD")
		ax.set_title(f"Temps: {time_s:.1f}s | Tmax: {float(np.max(field)):.1f}K")
		
		# Update or recreate colorbar
		if cbar is not None:
			cbar.remove()
		cbar = fig.colorbar(im, ax=ax, label="Temperature [K]")
		
		fig.canvas.draw_idle()
		fig.canvas.flush_events()

	def do_step() -> np.ndarray:
		nonlocal t
		h_act = (t >= tc_node).astype(float)
		src = m_unit.dot(q_node * h_act)
		conv = m_unit.dot(h_conv * ones)
		rhs = (m_mat / dt).dot(t) + src + conv
		res = spsolve(a_lhs, rhs)
		t = np.asarray(res, dtype=float)
		return np.asarray(t, dtype=float)

	plt.ion()
	plt.show(block=False)
	draw_frame(t, 0.0)
	plt.tight_layout()
	plt.pause(0.001)

	for frame in range(steps):
		for _ in range(sub_steps):
			do_step()
		t_sim = (frame + 1) * dt * sub_steps
		draw_frame(t, t_sim)
		plt.pause(0.05)  # Increased pause to ensure display updates

	plt.ioff()
	if args.plot:
		plt.show()

	return t


def main() -> None:
	parser = _build_parser()
	args = parser.parse_args()
	if args.mesh is None:
		args.mesh = "piece.msh"
	run(args)


if __name__ == "__main__":
	main()
