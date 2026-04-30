from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import meshio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.sparse import csr_matrix

from calculs.dirichlet import theta_step
from calculs.mass import assemble_mass
from calculs.stiffness import assemble_stiffness_and_rhs
from materialsbank import get_material, get_material_color, get_material_overlay_alpha

PHYSICAL_ID_MAP_2D = {
	1: "bois",
	2: "beton",
}

PHYSICAL_ID_MAP_3D = {
	1: "bois",
	2: "beton",
}

ROLE_KEYWORDS = {
	"wall": {"mur", "murs", "wall", "walls"},
	"window": {"fenetre", "fenetres", "window", "windows", "glass", "verre"},
	"floor": {"sol", "floor", "slab", "dalle"},
	"door": {"porte", "portes", "door", "doors"},
	"column": {"colonne", "colonnes", "column", "columns", "pillar", "pillars"},
}

THERMAL_CMAP = LinearSegmentedColormap.from_list(
	"thermal_white_red",
	[
		(1.0, 1.0, 1.0, 0.55),
		(1.0, 0.92, 0.92, 0.72),
		(1.0, 0.45, 0.25, 0.9),
		(1.0, 0.0, 0.0, 1.0),
	],
	N=256,
)


def _physical_id_to_name_map(msh: meshio.Mesh, dim: int) -> dict[int, str]:
	return (PHYSICAL_ID_MAP_2D if dim == 2 else PHYSICAL_ID_MAP_3D).copy()


def _format_region_label(raw_name: str) -> str:
	label = str(raw_name).strip().replace("_", " ")
	return " ".join(part.capitalize() for part in label.split()) or "Region"


def _infer_region_role(raw_name: str) -> str:
	name = str(raw_name).strip().lower()
	tokens = {token for token in name.replace("-", "_").split("_") if token}
	for role, keywords in ROLE_KEYWORDS.items():
		if any(keyword in name for keyword in keywords) or tokens.intersection(keywords):
			return role
	return "region"


def _build_visual_regions(phys_ids: np.ndarray, phys_name_map: dict[int, str]) -> list[dict[str, object]]:
	regions: list[dict[str, object]] = []
	for pid in sorted({int(pid) for pid in phys_ids}):
		raw_name = str(phys_name_map.get(pid, f"region_{pid}")).strip()
		material_key = raw_name.lower()
		material = get_material(material_key)
		role = _infer_region_role(raw_name)
		label = _format_region_label(raw_name)
		if label.lower() == str(material["name"]).lower():
			legend_label = str(material["name"])
		else:
			legend_label = f"{label} ({material['name']})"
		regions.append(
			{
				"pid": pid,
				"raw_name": raw_name,
				"label": label,
				"legend_label": legend_label,
				"material_key": material_key,
				"material": material,
				"role": role,
				"color": get_material_color(material_key),
				"overlay_alpha": get_material_overlay_alpha(material_key, 2),
				"overlay_alpha_3d": get_material_overlay_alpha(material_key, 3),
				"solid_fill": role in {"wall", "window", "door", "column"},
				"edge_width": 0.65 if role == "wall" else 0.4,
			}
		)
	return regions


def _load_mesh_data(mesh_path: Path, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
	msh = meshio.read(str(mesh_path))
	cell_type = "triangle" if dim == 2 else "tetra"
	pts = np.asarray(msh.points, dtype=float)[:, :dim]
	elems = np.asarray(msh.cells_dict.get(cell_type, np.array([], dtype=int)), dtype=int)
	phys = np.asarray(
		msh.cell_data_dict.get("gmsh:physical", {}).get(cell_type, np.ones(len(elems), dtype=int)),
		dtype=int,
	)

	# Fusion des noeuds doublons (repare les maillages non-conformes/entites separees)
	unique_pts, inverse = np.unique(np.round(pts, 7), axis=0, return_inverse=True)
	pts = unique_pts
	elems = inverse[elems]

	phys_name_map = _physical_id_to_name_map(msh, dim)
	return pts, elems, phys, phys_name_map


def _build_p1_triangle_quadrature(points: np.ndarray, elems: np.ndarray):
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


def _build_p1_tetra_quadrature(points: np.ndarray, elems: np.ndarray):
	a = 0.5854101966249685
	b = 0.1381966011250105
	bary = np.array(
		[
			[a, b, b, b],
			[b, a, b, b],
			[b, b, a, b],
			[b, b, b, a],
		],
		dtype=float,
	)
	w = np.full(4, 1.0 / 24.0, dtype=float)
	n_ref = bary.copy()
	grad_ref = np.array(
		[
			[-1.0, -1.0, -1.0],
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0],
		],
		dtype=float,
	)

	ne = len(elems)
	ngp = len(w)
	jac = np.zeros((ne, ngp, 3, 3), dtype=float)
	det = np.zeros((ne, ngp), dtype=float)
	coords = np.zeros((ne, ngp, 3), dtype=float)

	for e, nodes in enumerate(elems):
		p0, p1, p2, p3 = points[nodes]
		jac_e = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
		det_j = float(np.linalg.det(jac_e))
		for g, shape_vals in enumerate(bary):
			jac[e, g] = jac_e
			det[e, g] = abs(det_j)
			coords[e, g] = shape_vals[0] * p0 + shape_vals[1] * p1 + shape_vals[2] * p2 + shape_vals[3] * p3

	return w, n_ref, np.repeat(grad_ref[None, :, :], ngp, axis=0), jac, det, coords


def _build_p1_quadrature(points: np.ndarray, elems: np.ndarray, dim: int):
	return _build_p1_triangle_quadrature(points, elems) if dim == 2 else _build_p1_tetra_quadrature(points, elems)


def _assemble_system(points: np.ndarray, elems: np.ndarray, phys_ids: np.ndarray, phys_name_map: dict[int, str], dim: int):
	n_nodes = points.shape[0]
	q_node = np.zeros(n_nodes, dtype=float)
	tc_node = np.full(n_nodes, np.inf, dtype=float)
	tag_to_dof = np.arange(n_nodes, dtype=int)
	elem_tags = np.arange(len(elems), dtype=int)
	conn = np.asarray(elems, dtype=int)
	conn_flat = conn.reshape(-1)
	w, n_ref, grad_ref, jac, det, coords = _build_p1_quadrature(points, elems, dim)

	m_unit = assemble_mass(elem_tags, conn_flat, det, w, n_ref, tag_to_dof).tocsr()
	k_glob = csr_matrix((n_nodes, n_nodes), dtype=float)
	m_glob = csr_matrix((n_nodes, n_nodes), dtype=float)
	processed_materials: set[str] = set()

	for e, nodes in enumerate(conn):
		pid = int(phys_ids[e])
		mat_name = str(phys_name_map.get(pid, "bois")).strip().lower()
		mat = get_material(mat_name)

		for ni in nodes:
			q_node[ni] = max(q_node[ni], float(mat["Q"]))
			tc_node[ni] = min(tc_node[ni], float(mat["Tc"]))

		if mat_name in processed_materials:
			continue

		mat_mask = np.array(
			[str(phys_name_map.get(int(pid_local), "bois")).strip().lower() == mat_name for pid_local in phys_ids],
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
		group_stiffness, _ = assemble_stiffness_and_rhs(
			group_tags,
			group_conn_flat,
			group_jac,
			group_det,
			group_coords,
			w,
			n_ref,
			grad_ref,
			lambda _x, k=k_val: k,
			lambda _x: 0.0,
			tag_to_dof,
		)

		m_glob = m_glob + pc * group_mass
		k_glob = k_glob + group_stiffness.tocsr()
		processed_materials.add(mat_name)

	return k_glob, m_glob, m_unit, q_node, tc_node


def _extract_boundary_faces(elems: np.ndarray, phys_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	face_map: dict[tuple[int, int, int], tuple[int, int]] = {}
	for elem, pid in zip(elems, phys_ids, strict=False):
		local_faces = (
			(elem[0], elem[1], elem[2]),
			(elem[0], elem[1], elem[3]),
			(elem[0], elem[2], elem[3]),
			(elem[1], elem[2], elem[3]),
		)
		for face in local_faces:
			key = tuple(sorted(int(v) for v in face))
			if key in face_map:
				count, existing_pid = face_map[key]
				face_map[key] = (count + 1, existing_pid)
			else:
				face_map[key] = (1, int(pid))

	boundary_faces: list[tuple[int, int, int]] = []
	boundary_phys: list[int] = []
	for key, (count, pid) in face_map.items():
		if count == 1:
			boundary_faces.append(key)
			boundary_phys.append(pid)
	return np.asarray(boundary_faces, dtype=int), np.asarray(boundary_phys, dtype=int)


def _build_region_legend_handles(regions: list[dict[str, object]]) -> list[Patch]:
	legend_handles: list[Patch] = []
	for region in regions:
		legend_handles.append(
			Patch(
				facecolor=str(region["color"]),
				edgecolor="black" if bool(region["solid_fill"]) else str(region["color"]),
				alpha=0.45 if bool(region["solid_fill"]) else 0.30,
				label=str(region["legend_label"]),
			)
		)
	return legend_handles


def _add_region_overlays_2d(ax, points: np.ndarray, elems: np.ndarray, phys_ids: np.ndarray, regions: list[dict[str, object]]):
	for region in regions:
		mask = phys_ids == int(region["pid"])
		if not np.any(mask):
			continue
		verts = [points[tri, :2] for tri in elems[mask]]
		edgecolor = "black" if bool(region["solid_fill"]) else str(region["color"])
		alpha = 1.0 if bool(region["solid_fill"]) else float(region["overlay_alpha"])
		linewidth = float(region["edge_width"]) if bool(region["solid_fill"]) else 0.35
		zorder = 4 if bool(region["solid_fill"]) else 3
		ax.add_collection(
			PolyCollection(
				verts,
				facecolors=str(region["color"]),
				edgecolors=edgecolor,
				linewidths=linewidth,
				alpha=alpha,
				zorder=zorder,
			)
		)


def _add_region_overlays_3d(
	ax,
	points: np.ndarray,
	boundary_faces: np.ndarray,
	boundary_phys: np.ndarray,
	regions: list[dict[str, object]],
	include_solid_fill: bool,
):
	collections = []
	for region in regions:
		mask = boundary_phys == int(region["pid"])
		if not np.any(mask):
			continue
		face_vertices = [points[face] for face in boundary_faces[mask]]
		if include_solid_fill and bool(region["solid_fill"]):
			poly = Poly3DCollection(
				face_vertices,
				facecolors=str(region["color"]),
				edgecolors="black",
				linewidths=float(region["edge_width"]) * 0.4,
				alpha=1.0,
				zorder=3,
			)
		else:
			poly = Poly3DCollection(
				face_vertices,
				facecolors=str(region["color"]),
				edgecolors=str(region["color"]),
				linewidths=0.15,
				alpha=float(region["overlay_alpha_3d"]),
				zorder=1,
			)
		ax.add_collection3d(poly)
		collections.append(poly)
	return collections


def _add_region_edges_3d(ax, points: np.ndarray, boundary_faces: np.ndarray, boundary_phys: np.ndarray, regions: list[dict[str, object]]):
	collections = []
	for region in regions:
		if not bool(region["solid_fill"]):
			continue
		mask = boundary_phys == int(region["pid"])
		if not np.any(mask):
			continue
		region_edges = _build_boundary_edges(boundary_faces[mask])
		segments = [[points[i], points[j]] for i, j in region_edges]
		collection = Line3DCollection(
			segments,
			colors="black",
			linewidths=max(0.55, float(region["edge_width"])),
			alpha=0.95,
		)
		ax.add_collection3d(collection)
		collections.append(collection)
	return collections


def _build_boundary_edges(boundary_faces: np.ndarray) -> np.ndarray:
	edge_set: set[tuple[int, int]] = set()
	for face in boundary_faces:
		i, j, k = [int(v) for v in face]
		edge_set.add(tuple(sorted((i, j))))
		edge_set.add(tuple(sorted((i, k))))
		edge_set.add(tuple(sorted((j, k))))
	return np.asarray(sorted(edge_set), dtype=int)


def _plot_3d_mesh_preview(points: np.ndarray, boundary_faces: np.ndarray, title: str):
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection="3d")
	edges = _build_boundary_edges(boundary_faces)
	segments = [[points[i], points[j]] for i, j in edges]
	ax.add_collection3d(Line3DCollection(segments, colors="black", linewidths=0.25, alpha=0.65))
	ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="#444444", alpha=0.35, depthshade=False)
	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	_set_equal_3d_axes(ax, points)
	return fig, ax


def _plot_3d_filled_preview(points: np.ndarray, boundary_faces: np.ndarray, boundary_phys: np.ndarray, regions: list[dict[str, object]], title: str):
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection="3d")
	_add_region_overlays_3d(ax, points, boundary_faces, boundary_phys, regions, include_solid_fill=False)
	_add_region_overlays_3d(ax, points, boundary_faces, boundary_phys, regions, include_solid_fill=True)
	_add_region_edges_3d(ax, points, boundary_faces, boundary_phys, regions)
	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	_set_equal_3d_axes(ax, points)
	legend_handles = _build_region_legend_handles(regions)
	if legend_handles:
		ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, title="Objets / Materiaux")
	return fig, ax


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
	mins = np.min(points, axis=0)
	maxs = np.max(points, axis=0)
	center = 0.5 * (mins + maxs)
	radius = 0.5 * float(np.max(maxs - mins))
	if radius <= 0.0:
		radius = 1.0
	ax.set_xlim(center[0] - radius, center[0] + radius)
	ax.set_ylim(center[1] - radius, center[1] + radius)
	ax.set_zlim(center[2] - radius, center[2] + radius)


def _scenario_defaults(dim: int) -> dict[str, float | int]:
	if dim == 3:
		return {
			"dt": 10000.0,
			"steps": 400,
			"sub_steps": 1,
			"theta": 1.0,
			"h_conv": 1.0,
			"t_amb": 293.0,
			"src_temp": 800.0,
			"src_x": 1.0,
			"src_y": 1.0,
			"src_z": 0.5,
			"src_radius": 1.0,
		}
	return {
		"dt": 100.0,
		"steps": 2000,
		"sub_steps": 1,
		"theta": 1.0,
		"h_conv": 1.0,
		"t_amb": 293.0,
		"src_temp": 800.0,
		"src_x": 0.0,
		"src_y": 0.0,
		"src_z": 0.0,
		"src_radius": 0.05,
	}


def _resolve_save_targets(save_arg: str) -> tuple[Path, Path, Path, Path]:
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
	if not args.plot:
		plt.switch_backend("Agg")

	dim = int(args.dim)
	defaults = _scenario_defaults(dim)
	dt = float(args.dt if args.dt is not None else defaults["dt"])
	steps = int(args.steps if args.steps is not None else defaults["steps"])
	sub_steps = max(1, int(args.sub_steps if args.sub_steps is not None else defaults["sub_steps"]))
	theta = float(args.theta if args.theta is not None else defaults["theta"])
	h_conv = float(args.h_conv if args.h_conv is not None else defaults["h_conv"])
	t_amb = float(args.t_amb if args.t_amb is not None else defaults["t_amb"])
	src_temp = float(args.src_temp if args.src_temp is not None else defaults["src_temp"])
	src_x = float(args.src_x if args.src_x is not None else defaults["src_x"])
	src_y = float(args.src_y if args.src_y is not None else defaults["src_y"])
	src_z = float(args.src_z if args.src_z is not None else defaults["src_z"])
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

	mesh_file = args.mesh or ("piece.msh" if dim == 2 else "immeuble.msh")
	base_dir = Path(__file__).parent
	mesh_path = Path(mesh_file)
	if not mesh_path.exists():
		candidate = base_dir / mesh_file
		mesh_path = candidate if candidate.exists() else base_dir / "models" / mesh_file
	print(f"Using mesh: {mesh_path}")

	t0 = time.perf_counter()
	pts, elems, phys, phys_name_map = _load_mesh_data(mesh_path, dim)
	record_timing("mesh_load", time.perf_counter() - t0, details=f"{mesh_path};dim={dim}")
	print(f"Maillage charge: {len(pts)} noeuds, {len(elems)} elements ({dim}D).")

	t0 = time.perf_counter()
	k_mat, m_mat, m_unit, q_node, tc_node = _assemble_system(pts, elems, phys, phys_name_map, dim)
	record_timing("system_assembly", time.perf_counter() - t0, details=f"nodes={len(pts)};elements={len(elems)};dim={dim}")

	t0 = time.perf_counter()
	t = np.full(len(pts), t_amb, dtype=float)
	if dim == 2:
		dist = np.hypot(pts[:, 0] - src_x, pts[:, 1] - src_y)
	else:
		dist = np.sqrt((pts[:, 0] - src_x) ** 2 + (pts[:, 1] - src_y) ** 2 + (pts[:, 2] - src_z) ** 2)
	t[dist <= src_radius] = src_temp
	record_timing(
		"initial_conditions",
		time.perf_counter() - t0,
		details=f"src_x={src_x};src_y={src_y};src_z={src_z};src_radius={src_radius};src_temp={src_temp}",
	)

	ones = np.full(len(t), t_amb, dtype=float)
	empty_dofs = np.array([], dtype=int)
	empty_vals = np.array([], dtype=float)
	k_eff = k_mat + h_conv * m_unit
	boundary_faces = boundary_phys = None
	visual_regions = _build_visual_regions(phys, phys_name_map)
	if dim == 3:
		boundary_faces, boundary_phys = _extract_boundary_faces(elems, phys)

	def _setup_axes(local_t: np.ndarray, title: str):
		if dim == 2:
			fig, ax = plt.subplots(figsize=(10, 8))
			im = ax.tripcolor(pts[:, 0], pts[:, 1], elems, local_t, cmap=THERMAL_CMAP, shading="gouraud", vmin=t_amb, vmax=1500, alpha=0.86)
			_add_region_overlays_2d(ax, pts, elems, phys, visual_regions)
			plt.colorbar(im, ax=ax, label="Temperature [K]")
			ax.set_facecolor("#1A12BD")
			ax.set_aspect("equal")
			ax.set_title(title)
			legend_handles = _build_region_legend_handles(visual_regions)
			if legend_handles:
				ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, title="Objets / Materiaux")
			return fig, ax, {"field": im}

		fig = plt.figure(figsize=(16, 8))
		ax_mesh = fig.add_subplot(121, projection="3d")
		ax_full = fig.add_subplot(122, projection="3d")

		edges = _build_boundary_edges(boundary_faces)
		segments = [[pts[i], pts[j]] for i, j in edges]
		mesh_lines = Line3DCollection(segments, colors="black", linewidths=0.2, alpha=0.45)
		ax_mesh.add_collection3d(mesh_lines)
		mesh_scatter = ax_mesh.scatter(
			pts[:, 0],
			pts[:, 1],
			pts[:, 2],
			c=local_t,
			cmap=THERMAL_CMAP,
			s=5,
			vmin=t_amb,
			vmax=max(1500.0, float(np.max(local_t))),
			alpha=0.82,
			depthshade=False,
		)

		_add_region_overlays_3d(ax_full, pts, boundary_faces, boundary_phys, visual_regions, include_solid_fill=False)
		face_temps = np.mean(local_t[boundary_faces], axis=1)
		norm = Normalize(vmin=t_amb, vmax=max(1500.0, float(np.max(local_t))))
		face_colors = THERMAL_CMAP(norm(face_temps))
		full_surface = Poly3DCollection(
			[pts[face] for face in boundary_faces],
			facecolors=face_colors,
			edgecolors="none",
			linewidths=0.0,
			alpha=0.56,
		)
		ax_full.add_collection3d(full_surface)
		region_fills = _add_region_overlays_3d(ax_full, pts, boundary_faces, boundary_phys, visual_regions, include_solid_fill=True)
		region_edges = _add_region_edges_3d(ax_full, pts, boundary_faces, boundary_phys, visual_regions)

		fig.colorbar(
			mesh_scatter,
			ax=[ax_mesh, ax_full],
			label="Temperature [K]",
			location="bottom",
			fraction=0.045,
			pad=0.08,
			shrink=0.92,
		)

		ax_mesh.set_title(f"{title} | Vue maillage")
		ax_full.set_title(f"{title} | Vue pleine")
		for local_ax in (ax_mesh, ax_full):
			local_ax.set_xlabel("x")
			local_ax.set_ylabel("y")
			local_ax.set_zlabel("z")
			_set_equal_3d_axes(local_ax, pts)
		legend_handles = _build_region_legend_handles(visual_regions)
		if legend_handles:
			ax_full.legend(handles=legend_handles, loc="upper right", framealpha=0.9, title="Objets / Materiaux")
		return fig, ax_full, {"mesh_scatter": mesh_scatter, "full_surface": full_surface, "mesh_ax": ax_mesh, "full_ax": ax_full, "region_fills": region_fills, "region_edges": region_edges}

	output_dir = animation_path = setup_png_path = timings_csv_path = None
	if getattr(args, "save", None):
		t0 = time.perf_counter()
		output_dir, animation_path, setup_png_path, timings_csv_path = _resolve_save_targets(args.save)
		output_dir.mkdir(parents=True, exist_ok=True)
		record_timing("output_directory_prepare", time.perf_counter() - t0, details=str(output_dir))

	if args.plot:
		if dim == 3:
			fig_mesh, _ax_mesh = _plot_3d_mesh_preview(pts, boundary_faces, "Maillage 3D")
			#plt.tight_layout()
			plt.show()
			plt.close(fig_mesh)
			fig_full, _ax_full = _plot_3d_filled_preview(pts, boundary_faces, boundary_phys, visual_regions, "Batiment 3D plein")
			#plt.tight_layout()
			plt.show()
			plt.close(fig_full)
		t0 = time.perf_counter()
		fig_init, _ax_init, _visuals_init = _setup_axes(t, "Setup initial")
		record_timing("initial_setup_figure", time.perf_counter() - t0)
		#plt.tight_layout()
		if setup_png_path is not None:
			t1 = time.perf_counter()
			fig_init.savefig(setup_png_path, dpi=200, bbox_inches="tight")
			record_timing("initial_setup_png_save", time.perf_counter() - t1, details=str(setup_png_path))
		plt.show()
		plt.close(fig_init)
	elif setup_png_path is not None:
		t0 = time.perf_counter()
		fig_init, _ax_init, _visuals_init = _setup_axes(t, "Setup initial")
		record_timing("initial_setup_figure", time.perf_counter() - t0)
		#plt.tight_layout()
		t1 = time.perf_counter()
		fig_init.savefig(setup_png_path, dpi=200, bbox_inches="tight")
		record_timing("initial_setup_png_save", time.perf_counter() - t1, details=str(setup_png_path))
		plt.close(fig_init)

	t0 = time.perf_counter()
	fig, ax, visuals = _setup_axes(t, "Temps: 0.0s | Tmax: {:.1f}K".format(float(np.max(t))))
	record_timing("animation_figure", time.perf_counter() - t0)
	state = {"t": t, "time": 0.0}
	ani = None

	def _update_visual(local_t: np.ndarray, sim_time: float):
		if dim == 2:
			field = visuals["field"]
			field.set_array(local_t)
			ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(local_t):.1f}K")
			return [field]
		else:
			mesh_scatter = visuals["mesh_scatter"]
			full_surface = visuals["full_surface"]
			mesh_ax = visuals["mesh_ax"]
			full_ax = visuals["full_ax"]
			mesh_scatter.set_array(local_t)
			mesh_scatter.set_clim(vmin=t_amb, vmax=max(1500.0, float(np.max(local_t))))
			face_temps = np.mean(local_t[boundary_faces], axis=1)
			norm = Normalize(vmin=t_amb, vmax=max(1500.0, float(np.max(local_t))))
			full_surface.set_facecolor(THERMAL_CMAP(norm(face_temps)))
			mesh_ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(local_t):.1f}K | Vue maillage")
			full_ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(local_t):.1f}K | Vue pleine")
			return [mesh_scatter, full_surface]

	def advance_state(current_t: np.ndarray, current_time: float) -> tuple[np.ndarray, float, float]:
		t_local = current_t
		sim_time = float(current_time)
		t0 = time.perf_counter()
		for _ in range(sub_steps):
			h_act = (t_local >= tc_node).astype(float)
			src = m_unit.dot(q_node * h_act)
			conv = h_conv * m_unit.dot(ones)
			rhs = src + conv
			t_local = np.asarray(
				theta_step(
					m_mat,
					k_eff,
					rhs,
					rhs,
					t_local,
					dt=dt,
					theta=theta,
					dirichlet_dofs=empty_dofs,
					dir_vals_np1=empty_vals,
				),
				dtype=float,
			)
			sim_time += dt
		return t_local, sim_time, time.perf_counter() - t0

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
		record_timing("headless_calculation_total", time.perf_counter() - t0, details=f"frames={steps};dim={dim}")

		def update_anim(frame_idx: int):
			return _update_visual(headless_frames[frame_idx], headless_times[frame_idx])

		if getattr(args, "save", None):
			ani = FuncAnimation(fig, update_anim, frames=len(headless_frames), interval=100, blit=False, repeat=False)
		else:
			plt.close(fig)
	else:
		def update_anim(frame_idx: int, local_state: dict[str, np.ndarray]):
			t_local, sim_time, _elapsed = advance_state(local_state["t"], float(local_state["time"]))
			local_state["t"] = t_local
			local_state["time"] = sim_time
			return _update_visual(t_local, sim_time)

		ani = FuncAnimation(fig, lambda frame_idx: update_anim(frame_idx, state), frames=steps, interval=100, blit=False, repeat=False)

	if ani is not None:
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
	parser = argparse.ArgumentParser(description="Simulation FEM diffusion-reaction 2D/3D")
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
	parser.add_argument("--src-z", type=float, default=None, help="Z source (3D)")
	parser.add_argument("--src-radius", type=float, default=None, help="Rayon source initial")
	parser.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Dimension du calcul")
	parser.add_argument("--2d", dest="dim", action="store_const", const=2, help="Force le mode 2D")
	parser.add_argument("--3d", dest="dim", action="store_const", const=3, help="Force le mode 3D")
	parser.add_argument("--no-plot", dest="plot", action="store_false", help="Desactive l'affichage final")
	parser.add_argument("--save", dest="save", type=str, default=None, help="Nom de fichier MP4 pour sauvegarder l'animation")
	parser.set_defaults(plot=True)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	if args.mesh is None:
		args.mesh = "piece.msh" if int(args.dim) == 2 else "immeuble.msh"
	run(args)


if __name__ == "__main__":
	main()
