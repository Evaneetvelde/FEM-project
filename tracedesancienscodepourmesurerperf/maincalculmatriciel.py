#main gérant la simulation FEM diffusion-reaction 2D/3D
from __future__ import annotations

import csv
import meshio
import numpy as np
import argparse
import time

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Patch
from scipy.sparse import csr_matrix
from pathlib import Path

from calculs.mass import assemble_mass
from calculs.stiffness import assemble_stiffness_and_rhs
from calculs.dirichlet import theta_step

from materialsbank import get_material, get_material_color

PHYSICAL_ID_MAP = {
	1: "bois",
	2: "beton",
}


def _physical_id_to_name_map(msh: meshio.Mesh) -> dict[int, str]:
	result: dict[int, str] = {}
	for name, data in getattr(msh, "field_data", {}).items():
		phys_id = int(data[0])
		phys_dim = int(data[1])
		if phys_dim == 2:
			result[phys_id] = str(name)
	if not result:
		result = PHYSICAL_ID_MAP.copy()
	return result


def _load_mesh_data(mesh_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
	"""Load mesh points, elements, and physical IDs using meshio.

	Returns:
		(points_2d, triangles, physical_ids, phys_id_to_name_map)
	"""
	msh = meshio.read(str(mesh_path))

	pts = np.asarray(msh.points, dtype=float)[:, :2]
	elems = msh.cells_dict.get("triangle", np.array([], dtype=int))
	elems = np.asarray(elems, dtype=int)
	phys = msh.cell_data_dict.get("gmsh:physical", {}).get("triangle", np.ones(len(elems), dtype=int))
	phys = np.asarray(phys, dtype=int)
	phys_name_map = _physical_id_to_name_map(msh)

	return pts, elems, phys, phys_name_map


def _build_p1_triangle_quadrature(points: np.ndarray, elems: np.ndarray):
	"""Build quadrature data compatible with calculs.mass/stiffness for P1 triangles."""
	quad_ref = np.array(
		[
			[1.0 / 6.0, 1.0 / 6.0],
			[2.0 / 3.0, 1.0 / 6.0],
			[1.0 / 6.0, 2.0 / 3.0],
		],
		dtype=float,
	)
	w = np.full(3, 1.0 / 6.0, dtype=float)
	n_ref = np.array(
		[
			[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
			[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
			[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
		],
		dtype=float,
	)
	grad_ref = np.array(
		[
			[-1.0, -1.0, 0.0],
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
		],
		dtype=float,
	)

	ne = len(elems)
	ngp = len(w)
	jac = np.zeros((ne, ngp, 3, 3), dtype=float)
	det = np.zeros((ne, ngp), dtype=float)
	coords = np.zeros((ne, ngp, 3), dtype=float)

	for e, nodes in enumerate(elems):
		p0, p1, p2 = points[nodes]
		j11 = p1[0] - p0[0]
		j12 = p2[0] - p0[0]
		j21 = p1[1] - p0[1]
		j22 = p2[1] - p0[1]
		det_j = j11 * j22 - j12 * j21
		jac_e = np.array(
			[
				[j11, j12, 0.0],
				[j21, j22, 0.0],
				[0.0, 0.0, 1.0],
			],
			dtype=float,
		)
		for g, (xi, eta) in enumerate(quad_ref):
			jac[e, g] = jac_e
			det[e, g] = abs(det_j)
			coords[e, g, :2] = (1.0 - xi - eta) * p0 + xi * p1 + eta * p2

	return w, n_ref, np.repeat(grad_ref[None, :, :], ngp, axis=0), jac, det, coords


def _assemble_system(points: np.ndarray, elems: np.ndarray, phys_ids: np.ndarray, phys_name_map: dict[int, str]):
	n_nodes = points.shape[0]
	q_node = np.zeros(n_nodes, dtype=float)
	tc_node = np.full(n_nodes, np.inf, dtype=float)
	tag_to_dof = np.arange(n_nodes, dtype=int)
	elem_tags = np.arange(len(elems), dtype=int)
	conn = np.asarray(elems, dtype=int)
	conn_flat = conn.reshape(-1)
	w, n_ref, grad_ref, jac, det, coords = _build_p1_triangle_quadrature(points, elems)

	m_unit = assemble_mass(elem_tags, conn_flat, det, w, n_ref, tag_to_dof).tocsr()
	k_glob = csr_matrix((n_nodes, n_nodes), dtype=float)
	m_glob = csr_matrix((n_nodes, n_nodes), dtype=float)
	processed_materials: set[str] = set()

	for e, nodes in enumerate(conn):
		pid = int(phys_ids[e])
		mat_name = phys_name_map.get(pid, "bois")
		mat = get_material(mat_name)

		for ni in nodes:
			q_node[ni] = max(q_node[ni], float(mat["Q"]))
			tc_node[ni] = min(tc_node[ni], float(mat["Tc"]))

		mat_key = str(mat_name).strip().lower()
		if mat_key in processed_materials:
			continue

		mat_mask = np.array(
			[str(phys_name_map.get(int(pid_local), "bois")).strip().lower() == mat_key for pid_local in phys_ids],
			dtype=bool,
		)
		group_tags = elem_tags[mat_mask]
		group_conn = conn[mat_mask]
		group_conn_flat = group_conn.reshape(-1)
		group_jac = jac[mat_mask]
		group_det = det[mat_mask]
		group_coords = coords[mat_mask]
		pc = float(mat["rho"]) * float(mat["c"])
		k_val = float(mat["k"])

		group_mass = assemble_mass(group_tags, group_conn_flat, group_det, w, n_ref, tag_to_dof).tocsr()
		group_stiffness, _ = assemble_stiffness_and_rhs(group_tags,group_conn_flat,group_jac,group_det,group_coords,w,n_ref,grad_ref,lambda _x, k=k_val: k,lambda _x: 0.0,tag_to_dof,)

		m_glob = m_glob + pc * group_mass
		k_glob = k_glob + group_stiffness.tocsr()
		processed_materials.add(mat_key)

	return k_glob, m_glob, m_unit, q_node, tc_node


def _add_material_overlays(ax, points: np.ndarray, elems: np.ndarray, phys_ids: np.ndarray, phys_name_map: dict[int, str]):
	"""Draw subtle material-colored overlays and return legend handles."""
	legend_handles: list[Patch] = []
	seen_materials: set[str] = set()

	for pid in np.unique(phys_ids):
		mat_name = str(phys_name_map.get(int(pid), "bois")).strip().lower()
		mask = phys_ids == pid
		if not np.any(mask):
			continue

		color = get_material_color(mat_name)
		verts = [points[tri, :2] for tri in elems[mask]]
		collection = PolyCollection(
			verts,
			facecolors=color,
			edgecolors=color,
			linewidths=0.35,
			alpha=0.14,
			zorder=3,
		)
		ax.add_collection(collection)

		if mat_name not in seen_materials:
			mat = get_material(mat_name)
			legend_handles.append(
				Patch(facecolor=color, edgecolor=color, alpha=0.35, label=str(mat["name"]))
			)
			seen_materials.add(mat_name)

	return legend_handles


def _add_opaque_walls(ax, points: np.ndarray, elems: np.ndarray, phys_ids: np.ndarray, phys_name_map: dict[int, str]):
	"""Draw wall elements as opaque overlays above the temperature field."""
	wall_mask = np.array(
		[str(phys_name_map.get(int(pid), "bois")).strip().lower() == "beton" for pid in phys_ids],
		dtype=bool,
	)
	if not np.any(wall_mask):
		return None

	verts = [points[tri, :2] for tri in elems[wall_mask]]
	return ax.add_collection(
		PolyCollection(
			verts,
			facecolors=get_material_color("beton"),
			edgecolors="black",
			linewidths=0.5,
			alpha=1.0,
			zorder=4,
		)
	)


def _scenario_defaults() -> dict:
	return {
		"dt": 10.0,
		"steps": 2000,
		"sub_steps": 1,
		"theta": 1.0,
		"h_conv": 1.0,
		"t_amb": 293.0,
		"src_temp": 800.0,
		"src_x": 0.0,
		"src_y": 0.0,
		"src_radius": 0.05,
	}


def _resolve_save_targets(save_arg: str) -> tuple[Path, Path, Path, Path]:
	"""Return output_dir, animation_path, setup_png_path, timings_csv_path."""
	save_path = Path(save_arg)
	if save_path.suffix:
		output_dir = save_path.parent / save_path.stem
		animation_path = output_dir / save_path.name
	else:
		output_dir = save_path
		animation_path = output_dir / "animation.mp4"
	setup_png_path = output_dir / "setup_initial.png"
	timings_csv_path = output_dir / "timings.csv"
	return output_dir, animation_path, setup_png_path, timings_csv_path


def _write_timings_csv(csv_path: Path, timing_rows: list[dict[str, object]]) -> None:
	fieldnames = ["phase", "seconds", "frame", "details"]
	with csv_path.open("w", newline="", encoding="utf-8") as fh:
		writer = csv.DictWriter(fh, fieldnames=fieldnames)
		writer.writeheader()
		for row in timing_rows:
			writer.writerow(row)

def run(args: argparse.Namespace):
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
	timing_rows: list[dict[str, object]] = []

	def record_timing(phase: str, seconds: float, frame: int | str = "", details: str = "") -> None:
		timing_rows.append(
			{
				"phase": phase,
				"seconds": f"{seconds:.6f}",
				"frame": frame,
				"details": details,
			}
		)

	mesh_file = args.mesh or "piece.msh"
	base_dir = Path(__file__).parent
	mesh_path = Path(mesh_file)
	if not mesh_path.exists():
		candidate = base_dir / mesh_file
		mesh_path = candidate if candidate.exists() else base_dir / "models" / mesh_file
	print(f"Using mesh: {mesh_path}")

	t0 = time.perf_counter()
	pts, elems, phys, phys_name_map = _load_mesh_data(mesh_path)
	record_timing("mesh_load", time.perf_counter() - t0, details=str(mesh_path))
	print(f"Maillage charge: {len(pts)} noeuds, {len(elems)} elements (2D).")

	t0 = time.perf_counter()
	k_mat, m_mat, m_unit, q_node, tc_node = _assemble_system(pts, elems, phys, phys_name_map)
	record_timing("system_assembly", time.perf_counter() - t0, details=f"nodes={len(pts)};elements={len(elems)}")

	t0 = time.perf_counter()
	t = np.full(len(pts), t_amb, dtype=float)
	dist = np.hypot(pts[:, 0] - src_x, pts[:, 1] - src_y)
	t[dist <= src_radius] = src_temp
	record_timing("initial_conditions", time.perf_counter() - t0, details=f"src_x={src_x};src_y={src_y};src_radius={src_radius};src_temp={src_temp}")

	ones = np.full(len(t), t_amb, dtype=float)
	empty_dofs = np.array([], dtype=int)
	empty_vals = np.array([], dtype=float)
	k_eff = k_mat + h_conv * m_unit

	def _setup_axes(local_t: np.ndarray, title: str):
		fig, ax = plt.subplots(figsize=(10, 8))
		im = ax.tripcolor(pts[:, 0], pts[:, 1], elems, local_t, cmap="magma", shading="gouraud", vmin=t_amb, vmax=1500)
		legend_handles = _add_material_overlays(ax, pts, elems, phys, phys_name_map)
		_add_opaque_walls(ax, pts, elems, phys, phys_name_map)
		plt.colorbar(im, ax=ax, label="Temperature [K]")
		ax.set_facecolor("#1A12BD")
		ax.set_aspect("equal")
		ax.set_title(title)
		if legend_handles:
			ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, title="Materiaux")
		return fig, ax, im

	output_dir = animation_path = setup_png_path = timings_csv_path = None
	if getattr(args, "save", None):
		t0 = time.perf_counter()
		output_dir, animation_path, setup_png_path, timings_csv_path = _resolve_save_targets(args.save)
		output_dir.mkdir(parents=True, exist_ok=True)
		record_timing("output_directory_prepare", time.perf_counter() - t0, details=str(output_dir))

	if args.plot:
		t0 = time.perf_counter()
		fig_init, _ax_init, _im_init = _setup_axes(t, "Setup initial")
		record_timing("initial_setup_figure", time.perf_counter() - t0)
		plt.tight_layout()
		if setup_png_path is not None:
			t1 = time.perf_counter()
			fig_init.savefig(setup_png_path, dpi=200, bbox_inches="tight")
			record_timing("initial_setup_png_save", time.perf_counter() - t1, details=str(setup_png_path))
		plt.show()
		plt.close(fig_init)
	elif setup_png_path is not None:
		t0 = time.perf_counter()
		fig_init, _ax_init, _im_init = _setup_axes(t, "Setup initial")
		record_timing("initial_setup_figure", time.perf_counter() - t0)
		plt.tight_layout()
		t1 = time.perf_counter()
		fig_init.savefig(setup_png_path, dpi=200, bbox_inches="tight")
		record_timing("initial_setup_png_save", time.perf_counter() - t1, details=str(setup_png_path))
		plt.close(fig_init)

	t0 = time.perf_counter()
	fig, ax, im = _setup_axes(t, "Temps: 0.0s | Tmax: {:.1f}K".format(float(np.max(t))))
	record_timing("animation_figure", time.perf_counter() - t0)
	state = {"t": t, "time": 0.0}

	def advance_state(current_t: np.ndarray, current_time: float) -> tuple[np.ndarray, float, float]:
		t = current_t
		sim_time = float(current_time)
		t0 = time.perf_counter()
		for _ in range(sub_steps):
			h_act = (t >= tc_node).astype(float)
			src = m_unit.dot(q_node * h_act)
			conv = h_conv * m_unit.dot(ones)
			rhs = src + conv
			t = np.asarray(
				theta_step(
					m_mat,
					k_eff,
					rhs,
					rhs,
					t,
					dt=dt,
					theta=theta,
					dirichlet_dofs=empty_dofs,
					dir_vals_np1=empty_vals,
				),
				dtype=float,
			)
			sim_time += dt
		return t, sim_time, time.perf_counter() - t0

	if not args.plot:
		headless_frames = [t.copy()]
		headless_times = [0.0]
		t0 = time.perf_counter()
		for frame_idx in range(steps):
			t, sim_time, elapsed = advance_state(t, headless_times[-1])
			record_timing("frame_calculation", elapsed, frame=frame_idx, details=f"sim_time={sim_time:.5f}")
			headless_frames.append(t.copy())
			headless_times.append(sim_time)
		state["t"] = t
		state["time"] = headless_times[-1]
		record_timing("headless_calculation_total", time.perf_counter() - t0, details=f"frames={steps}")

		def update_anim(frame_idx: int):
			frame_t = headless_frames[frame_idx]
			frame_time = headless_times[frame_idx]
			im.set_array(frame_t)
			ax.set_title(f"Temps: {frame_time:.1f}s | Tmax: {np.max(frame_t):.1f}K")
			return [im]

		ani = FuncAnimation(fig, update_anim, frames=len(headless_frames), interval=100, blit=False, repeat=False)
	else:
		def update_anim(frame_idx: int, state: dict[str, np.ndarray]):
			t_local, sim_time, _elapsed = advance_state(state["t"], float(state["time"]))
			state["t"] = t_local
			state["time"] = sim_time
			im.set_array(t_local)
			ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(t_local):.1f}K")
			return [im]

		ani = FuncAnimation(fig, lambda frame_idx: update_anim(frame_idx, state), frames=steps, interval=100, blit=False, repeat=False)
	plt.tight_layout()

	if getattr(args, "save", None):
		print(f"Sauvegarde de l'animation dans {animation_path} ...")
		try:
			t0 = time.perf_counter()
			writer = FFMpegWriter(fps=15, metadata={"artist": "Simulation"}, bitrate=2000)
			ani.save(str(animation_path), writer=writer)
			record_timing("animation_save", time.perf_counter() - t0, details=str(animation_path))
			print("Enregistrement termine.")
		except Exception as e:
			print(f"Echec de la sauvegarde: {e}")
		finally:
			if timings_csv_path is not None:
				_write_timings_csv(timings_csv_path, timing_rows)

	if args.plot:
		plt.show()

def build_parser() -> argparse.ArgumentParser:
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
	parser.add_argument("--save", dest="save", type=str, default=None, help="Nom de fichier MP4 pour sauvegarder l'animation (requires ffmpeg)")
	parser.set_defaults(plot=True)
	return parser

def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	if args.mesh is None:
		args.mesh = "piece.msh"
	run(args)


if __name__ == "__main__":
	main()
