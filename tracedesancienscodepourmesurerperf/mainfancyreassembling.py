#main gérant la simulation FEM diffusion-reaction 2D/3D
from __future__ import annotations

import argparse
import meshio
import numpy as np
from pathlib import Path
from scipy.sparse import lil_matrix
from calculs.gmsh_utils import (
    getPhysical, gmsh_init, gmsh_finalize, open_2d_mesh,
    prepare_quadrature_and_basis, get_jacobians,
    border_dofs_from_tags
)
from calculs.stiffness import assemble_rhs_neumann, assemble_stiffness_and_rhs
from calculs.mass import assemble_mass
from calculs.plot_utils import setup_interactive_figure, plot_mesh_2d, plot_fe_solution_2d
from calculs.dirichlet import theta_step  # type: ignore
from materialsbank import get_material




def _get_cells(msh: meshio.Mesh, dim: int) -> np.ndarray:
	cell_type = "triangle" if dim == 2 else "tetra"
	if cell_type not in msh.cells_dict:
		raise ValueError(f"Le maillage ne contient pas de cellules '{cell_type}'.")
	return np.asarray(msh.cells_dict[cell_type], dtype=np.int64)


def _get_physical_ids(msh: meshio.Mesh, dim: int, n_elem: int) -> np.ndarray:
	cell_type = "triangle" if dim == 2 else "tetra"
	try:
		ids = np.asarray(msh.cell_data_dict["gmsh:physical"][cell_type], dtype=np.int64).reshape(-1)
	except Exception:
		ids = np.ones(n_elem, dtype=np.int64)
	if ids.size != n_elem:
		raise ValueError("Nombre d'IDs physiques incoherent avec le nombre d'elements.")
	return ids


def _physical_id_to_name_map(msh: meshio.Mesh, dim: int) -> dict[int, str]:
	"""Construit une table id physique -> nom de materiau depuis le maillage."""
	result: dict[int, str] = {}
	for name, data in getattr(msh, "field_data", {}).items():
		phys_id = int(data[0])
		phys_dim = int(data[1])
		if phys_dim == dim:
			result[phys_id] = str(name)
	return result


def _element_matrices_2d(coords: np.ndarray, k_val: float, pc: float):
	x = coords[:, 0]
	y = coords[:, 1]
	det_j = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
	area = 0.5 * abs(det_j)
	b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]], dtype=float) / det_j
	c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]], dtype=float) / det_j
	ke = k_val * area * (np.outer(b, b) + np.outer(c, c))
	me = (pc * area / 12.0) * np.array(
		[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=float
	)
	munit = (area / 12.0) * np.array(
		[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=float
	)
	lumped = np.full(3, area / 3.0, dtype=float)
	return ke, me, munit, lumped


def _element_matrices_3d(coords: np.ndarray, k_val: float, pc: float):
	j = (coords[1:] - coords[0]).T
	det_j = np.linalg.det(j)
	vol = abs(det_j) / 6.0
	inv_j = np.linalg.inv(j)
	g_ref = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
	g_phys = g_ref @ inv_j
	ke = k_val * vol * (g_phys @ g_phys.T)
	me = (pc * vol / 20.0) * (np.ones((4, 4)) + np.eye(4))
	munit = (vol / 20.0) * (np.ones((4, 4)) + np.eye(4))
	lumped = np.full(4, vol / 4.0, dtype=float)
	return ke, me, munit, lumped


def assemble_system(points: np.ndarray, elems: np.ndarray, phys_ids: np.ndarray, dim: int, phys_name_map: dict[int, str]):
	n_nodes = points.shape[0]
	k_glob = lil_matrix((n_nodes, n_nodes), dtype=float)
	m_glob = lil_matrix((n_nodes, n_nodes), dtype=float)
	m_unit = lil_matrix((n_nodes, n_nodes), dtype=float)
	q_node = np.zeros(n_nodes, dtype=float)
	tc_node = np.full(n_nodes, np.inf, dtype=float)
	m_lumped = np.zeros(n_nodes, dtype=float)

	for e, nodes in enumerate(elems):
		pid = int(phys_ids[e])
		mat_name = phys_name_map.get(pid, "Bois")
		mat = get_material(mat_name)
		k_val = float(mat["k"])
		pc = float(mat["rho"]) * float(mat["c"])
		q = float(mat["Q"])
		tc = float(mat["Tc"])
		coords = points[nodes, :dim]

		if dim == 2:
			ke, me, munit, lumped = _element_matrices_2d(coords, k_val, pc)
		else:
			ke, me, munit, lumped = _element_matrices_3d(coords, k_val, pc)

		for i, ni in enumerate(nodes):
			q_node[ni] = max(q_node[ni], q)
			tc_node[ni] = min(tc_node[ni], tc)
			m_lumped[ni] += lumped[i]
			for j, nj in enumerate(nodes):
				k_glob[ni, nj] += ke[i, j]
				m_glob[ni, nj] += me[i, j]
				m_unit[ni, nj] += munit[i, j]

	return k_glob.tocsr(), m_glob.tocsr(), m_unit.tocsr(), m_lumped, q_node, tc_node


def boundary_dofs(points: np.ndarray, dim: int) -> np.ndarray:
	eps = 1e-12
	masks = []
	for d in range(dim):
		mn = float(np.min(points[:, d]))
		mx = float(np.max(points[:, d]))
		masks.append(np.isclose(points[:, d], mn, atol=eps))
		masks.append(np.isclose(points[:, d], mx, atol=eps))
	bmask = np.logical_or.reduce(masks)
	return np.nonzero(bmask)[0]


def run(args: argparse.Namespace) -> np.ndarray:
	mesh_path = "models" + args.mesh
    with open(mesh_path, "r") as f:
	defaults = scenario_defaults(args.dim)
	dt = float(args.dt if args.dt is not None else defaults["dt"])
	steps = int(args.steps if args.steps is not None else defaults["steps"])
	theta = float(args.theta if args.theta is not None else defaults["theta"])
	h_conv = float(args.h_conv if args.h_conv is not None else defaults["h_conv"])
	t_amb = float(args.t_amb if args.t_amb is not None else defaults["t_amb"])
	src_temp = float(args.src_temp if args.src_temp is not None else defaults["src_temp"])
	src_x = float(args.src_x if args.src_x is not None else defaults["src_x"])
	src_y = float(args.src_y if args.src_y is not None else defaults["src_y"])
	src_z = float(args.src_z if args.src_z is not None else defaults["src_z"])
	src_radius = float(args.src_radius if args.src_radius is not None else defaults["src_radius"])
	src_z_max = float(args.src_z_max if args.src_z_max is not None else defaults["src_z_max"])

	msh = meshio.read(str(mesh_path))
	pts = np.asarray(msh.points, dtype=float)[:, :3]

	elems = _get_cells(msh, args.dim)
	phys = _get_physical_ids(msh, args.dim, elems.shape[0])
	phys_name_map = _physical_id_to_name_map(msh, args.dim)
	print(f"Maillage charge: {len(pts)} noeuds, {len(elems)} elements (dim={args.dim}).")

	k_mat, m_mat, m_unit, _, q_node, tc_node = assemble_system(pts, elems, phys, args.dim, phys_name_map)

	t = np.full(len(pts), t_amb, dtype=float)
	if args.dim == 2:
		dist = np.hypot(pts[:, 0] - src_x, pts[:, 1] - src_y)
		t[dist <= src_radius] = src_temp
	else:
		t[pts[:, 2] < src_z_max] = src_temp

	empty_dofs = np.array([], dtype=int)
	empty_vals = np.array([], dtype=float)
	ones = np.full(len(t), t_amb, dtype=float)
	k_eff = k_mat + h_conv * m_unit

	for step in range(steps):
		h = (t >= tc_node).astype(float)
		src = m_unit.dot(q_node * h)
		conv = m_unit.dot(h_conv * ones)
		f_n = src + conv
		t = theta_step(
			m_mat,
			k_eff,
			f_n,
			f_n,
			t,
			dt=dt,
			theta=theta,
			dirichlet_dofs=empty_dofs,
			dir_vals_np1=empty_vals,
		)
		if (step + 1) % max(1, steps // 10) == 0:
			print(f"Step {step + 1}/{steps} | Tmax={np.max(t):.2f} K")

	if args.plot:
		import matplotlib.pyplot as plt

		if args.dim == 2:
			plt.figure(figsize=(8, 6))
			plt.tripcolor(pts[:, 0], pts[:, 1], elems, t, shading="gouraud", cmap="magma", vmin=t_amb, vmax=max(src_temp, np.max(t)))
			plt.colorbar(label="Temperature [K]")
			plt.axis("equal")
			plt.title("Resultat final 2D")
			plt.tight_layout()
			plt.show()
		else:
			from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

			fig = plt.figure(figsize=(9, 7))
			ax = fig.add_subplot(111, projection="3d")
			m = t > (t_amb + 5.0)
			sc = ax.scatter(pts[m, 0], pts[m, 1], pts[m, 2], c=t[m], cmap="hot", s=8, vmin=t_amb, vmax=max(src_temp, np.max(t)))
			plt.colorbar(sc, label="Temperature [K]")
			ax.set_title("Resultat final 3D")
			plt.tight_layout()
			plt.show()

	return t


class Scenario:
	"""Scenario de simulation: geometrie, materiaux, conditions initiales."""
	def __init__(self, name: str, dim: int, mesh_file: str, defaults: dict):
		self.name = name
		self.dim = dim
		self.mesh_file = mesh_file
		self.defaults = defaults
		self.features = {}


def load_scenario_from_file(filepath: Path) -> Scenario | None:
	"""Charge un scenario depuis un fichier texte.
	
	Format futur (JSON ou YAML):
	- geometrie: pieces, murs, etages
	- materiaux: type, proprietes
	- conditions initiales: sources
	
	TODO: implementer selon format choisi.
	"""
	if not filepath.exists():
		return None
	# Placeholder: a completer avec parsing JSON/YAML
	return None


def build_parser()-> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Simulation FEM diffusion-reaction 2D/3D")
	parser.add_argument("--scenario", type=str, default=None, help="Fichier scenario (JSON/YAML) decrivant pieces, murs, materiaux")
	parser.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Dimension du calcul")
	parser.add_argument("--mesh", type=str, default=None, help="Nom du maillage .msh dans models/")
	parser.add_argument("--dt", type=float, default=None, help="Pas de temps")
	parser.add_argument("--steps", type=int, default=None, help="Nombre d'iterations")
	parser.add_argument("--theta", type=float, default=None, help="Schema theta (1=Euler implicite)")
	parser.add_argument("--h-conv", dest="h_conv", type=float, default=None, help="Coefficient de convection")
	parser.add_argument("--t-amb", dest="t_amb", type=float, default=None, help="Temperature ambiante")
	parser.add_argument("--src-temp", dest="src_temp", type=float, default=None, help="Temperature initiale source")
	parser.add_argument("--src-x", type=float, default=None, help="X source")
	parser.add_argument("--src-y", type=float, default=None, help="Y source")
	parser.add_argument("--src-z", type=float, default=None, help="Z source (3D)")
	parser.add_argument("--src-radius", type=float, default=None, help="Rayon source (2D)")
	parser.add_argument("--src-z-max", type=float, default=None, help="Hauteur max de la source 3D")
	parser.add_argument("--no-plot", dest="plot", action="store_false", help="Desactive l'affichage final")
	parser.set_defaults(plot=True)
	return parser


def main():
	parser = build_parser()
	args = parser.parse_args()
	
	if args.scenario:
		scenario = load_scenario_from_file(Path(args.scenario))
		if scenario:
			args.dim = scenario.dim
			if not args.mesh:
				args.mesh = scenario.mesh_file
			print(f"Scenario charge: {scenario.name}")
		else:
			print(f"Avertissement: scenario {args.scenario} non trouve ou invalide")
	
	if args.mesh is None:
		args.mesh = "piece.msh" if args.dim == 2 else "immeuble.msh"
	run(args)


if __name__ == "__main__":
	main()
