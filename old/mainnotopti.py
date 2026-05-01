from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import meshio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.sparse import csr_matrix, diags

from calculs.dirichlet import theta_step
from calculs.mass import assemble_mass
from calculs.stiffness import assemble_stiffness_and_rhs
from materialsbank import get_burn_material_name, get_material, get_material_color, get_material_overlay_alpha

PHYSICAL_ID_MAP_2D = {
	1: "bois",
	2: "beton",
	3: "verre",
	4: "isolation",
	5: "air",
	6: "metal",
	7: "meche",
	8: "explosif",
}

PHYSICAL_ID_MAP_3D = {
	1: "bois",
	2: "beton",
	3: "verre",
	4: "isolation",
	5: "air",
	6: "metal",
	7: "meche",
	8: "explosif",
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

THERMAL_CMAP_3D = LinearSegmentedColormap.from_list(
	"thermal_3d_white_purple_red",
	[
		(1.0, 1.0, 1.0, 1.0),
		(0.45, 0.0, 0.75, 1.0),
		(1.0, 0.0, 0.0, 1.0),
	],
	N=256,
)
THERMAL_3D_VMIN = 200.0
THERMAL_3D_VMAX = 10000.0


@dataclass
class BoundaryConditionField:
	weights: np.ndarray
	h: np.ndarray
	t_ext: np.ndarray
	dofs: np.ndarray
	loss_matrix: csr_matrix
	rhs: np.ndarray


@dataclass
class VolumeLossField:
	linear_coeff: float
	radiation_coeff: float
	t_ext: float
	loss_matrix: csr_matrix
	rhs: np.ndarray


@dataclass
class ElementwiseSystem:
	k_mat: csr_matrix
	m_mat: csr_matrix
	m_unit: csr_matrix
	unit_load_local: np.ndarray
	q_node: np.ndarray
	tc_node: np.ndarray
	elem_material_names: np.ndarray
	w: np.ndarray
	n_ref: np.ndarray
	grad_ref: np.ndarray
	jac: np.ndarray
	det: np.ndarray
	coords: np.ndarray


@dataclass
class VerticalHeatTransferField:
	enabled: bool
	targets: list[np.ndarray]
	dz: list[np.ndarray]
	element_volumes: np.ndarray


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


def _element_tc_values(phys_ids: np.ndarray, phys_name_map: dict[int, str]) -> np.ndarray:
	return np.asarray([float(get_material(str(phys_name_map.get(int(pid), "bois")).strip().lower())["Tc"]) for pid in phys_ids], dtype=float)


def _initial_element_materials(phys_ids: np.ndarray, phys_name_map: dict[int, str]) -> np.ndarray:
	return np.asarray([str(phys_name_map.get(int(pid), "bois")).strip().lower() for pid in phys_ids], dtype=object)


def _node_reaction_fields(elems: np.ndarray, elem_material_names: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
	q_node = np.zeros(n_nodes, dtype=float)
	tc_node = np.full(n_nodes, np.inf, dtype=float)
	for nodes, mat_name in zip(elems, elem_material_names, strict=False):
		mat = get_material(str(mat_name))
		for ni in nodes:
			q_node[int(ni)] = max(q_node[int(ni)], float(mat["Q"]))
			tc_node[int(ni)] = min(tc_node[int(ni)], float(mat["Tc"]))
	return q_node, tc_node


def _local_unit_matrices_from_quadrature(
	elems: np.ndarray,
	w: np.ndarray,
	n_ref: np.ndarray,
	grad_ref: np.ndarray,
	jac: np.ndarray,
	det: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	ne = len(elems)
	nloc = elems.shape[1]
	ngp = len(w)
	unit_mass = np.zeros((ne, nloc, nloc), dtype=float)
	unit_stiffness = np.zeros((ne, nloc, nloc), dtype=float)
	unit_load = np.zeros((ne, nloc), dtype=float)

	for e in range(ne):
		for g in range(ngp):
			wg_det = float(w[g] * det[e, g])
			inv_jac = np.linalg.inv(jac[e, g])
			grads = np.array([inv_jac @ grad_ref[g, a] for a in range(nloc)], dtype=float)
			for a in range(nloc):
				unit_load[e, a] += wg_det * float(n_ref[g, a])
				for b in range(nloc):
					unit_mass[e, a, b] += wg_det * float(n_ref[g, a] * n_ref[g, b])
					unit_stiffness[e, a, b] += wg_det * float(np.dot(grads[a], grads[b]))

	return unit_mass, unit_stiffness, unit_load


def _local_unit_matrices(points: np.ndarray, elems: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	w, n_ref, grad_ref, jac, det, _coords = _build_p1_quadrature(points, elems, dim)
	return _local_unit_matrices_from_quadrature(elems, w, n_ref, grad_ref, jac, det)


def _assemble_vector_from_local(elems: np.ndarray, local_values: np.ndarray, n_nodes: int) -> np.ndarray:
	values = np.zeros(n_nodes, dtype=float)
	np.add.at(values, elems.reshape(-1), local_values.reshape(-1))
	return values


def _assemble_material_matrices(
	elems: np.ndarray,
	elem_material_names: np.ndarray,
	w: np.ndarray,
	n_ref: np.ndarray,
	grad_ref: np.ndarray,
	jac: np.ndarray,
	det: np.ndarray,
	coords: np.ndarray,
	n_nodes: int,
	use_burn_delta: bool = False,
) -> tuple[csr_matrix, csr_matrix]:
	tag_to_dof = np.arange(n_nodes, dtype=int)
	k_mat = csr_matrix((n_nodes, n_nodes), dtype=float)
	m_mat = csr_matrix((n_nodes, n_nodes), dtype=float)

	for mat_name in sorted({str(name) for name in elem_material_names}):
		mask = np.asarray(elem_material_names, dtype=object) == mat_name
		if not np.any(mask):
			continue
		mat = get_material(mat_name)
		if use_burn_delta:
			burn_mat = get_material(get_burn_material_name(mat_name))
			k_coeff = float(burn_mat["k"]) - float(mat["k"])
			m_coeff = float(burn_mat["rho"]) * float(burn_mat["c"]) - float(mat["rho"]) * float(mat["c"])
		else:
			k_coeff = float(mat["k"])
			m_coeff = float(mat["rho"]) * float(mat["c"])

		elem_tags = np.flatnonzero(mask)
		conn = elems[mask].reshape(-1)
		if m_coeff != 0.0:
			m_mat = m_mat + (m_coeff * assemble_mass(elem_tags, conn, det[mask], w, n_ref, tag_to_dof).tocsr())
		if k_coeff != 0.0:
			k_group, _rhs = assemble_stiffness_and_rhs(
				elem_tags,
				conn,
				jac[mask],
				det[mask],
				coords[mask],
				w,
				n_ref,
				grad_ref,
				lambda _x, coeff=k_coeff: coeff,
				lambda _x: 0.0,
				tag_to_dof,
			)
			k_mat = k_mat + k_group.tocsr()

	return k_mat, m_mat


def _default_vertical_air_radius(points: np.ndarray, elems: np.ndarray) -> float:
	centroids = np.mean(points[elems], axis=1)
	spans = np.ptp(centroids[:, :2], axis=0)
	area = max(float(spans[0] * spans[1]), 1e-12)
	return 1.5 * float(np.sqrt(area / max(len(elems), 1)))


def _build_vertical_heat_transfer_field(
	points: np.ndarray,
	elems: np.ndarray,
	unit_load_local: np.ndarray,
	dim: int,
	enabled: bool,
	attenuation_per_m: float,
	horizontal_radius: float,
) -> VerticalHeatTransferField:
	element_volumes = np.sum(unit_load_local, axis=1)
	if dim != 3 or not enabled:
		return VerticalHeatTransferField(False, [], [], element_volumes)

	centroids = np.mean(points[elems], axis=1)
	radius = float(horizontal_radius) if horizontal_radius > 0.0 else _default_vertical_air_radius(points, elems)
	attenuation = max(0.0, float(attenuation_per_m))
	targets: list[np.ndarray] = []
	dz_by_source: list[np.ndarray] = []

	for source_idx, source_center in enumerate(centroids):
		dz = centroids[:, 2] - source_center[2]
		horizontal_dist = np.linalg.norm(centroids[:, :2] - source_center[:2], axis=1)
		mask = (dz > 0.0) & (horizontal_dist <= radius)
		local_factors = np.maximum(0.0, 1.0 - attenuation * dz[mask])
		valid = local_factors > 0.0
		targets.append(np.flatnonzero(mask)[valid])
		dz_by_source.append(dz[mask][valid])

	return VerticalHeatTransferField(True, targets, dz_by_source, element_volumes)


def _assemble_elementwise_system(
	points: np.ndarray,
	elems: np.ndarray,
	phys_ids: np.ndarray,
	phys_name_map: dict[int, str],
	dim: int,
) -> ElementwiseSystem:
	n_nodes = points.shape[0]
	elem_material_names = _initial_element_materials(phys_ids, phys_name_map)
	w, n_ref, grad_ref, jac, det, coords = _build_p1_quadrature(points, elems, dim)
	_unit_mass, _unit_stiffness, unit_load = _local_unit_matrices_from_quadrature(elems, w, n_ref, grad_ref, jac, det)

	k_mat, m_mat = _assemble_material_matrices(elems, elem_material_names, w, n_ref, grad_ref, jac, det, coords, n_nodes)
	m_unit = assemble_mass(np.arange(len(elems)), elems.reshape(-1), det, w, n_ref, np.arange(n_nodes, dtype=int)).tocsr()
	q_node, tc_node = _node_reaction_fields(elems, elem_material_names, n_nodes)

	return ElementwiseSystem(
		k_mat=k_mat,
		m_mat=m_mat,
		m_unit=m_unit,
		unit_load_local=unit_load,
		q_node=q_node,
		tc_node=tc_node,
		elem_material_names=elem_material_names,
		w=w,
		n_ref=n_ref,
		grad_ref=grad_ref,
		jac=jac,
		det=det,
		coords=coords,
	)


def _apply_burn_deltas(system: ElementwiseSystem, elems: np.ndarray, burned_indices: np.ndarray) -> None:
	if len(burned_indices) == 0:
		return
	n_nodes = system.k_mat.shape[0]
	delta_k, delta_m = _assemble_material_matrices(
		elems[burned_indices],
		system.elem_material_names[burned_indices],
		system.w,
		system.n_ref,
		system.grad_ref,
		system.jac[burned_indices],
		system.det[burned_indices],
		system.coords[burned_indices],
		n_nodes,
		use_burn_delta=True,
	)
	system.k_mat = system.k_mat + delta_k
	system.m_mat = system.m_mat + delta_m
	for elem_idx in burned_indices:
		system.elem_material_names[int(elem_idx)] = get_burn_material_name(str(system.elem_material_names[int(elem_idx)]))
	system.q_node, system.tc_node = _node_reaction_fields(elems, system.elem_material_names, n_nodes)


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


def _extract_boundary_edges(elems: np.ndarray, phys_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	edge_map: dict[tuple[int, int], tuple[int, int]] = {}
	for elem, pid in zip(elems, phys_ids, strict=False):
		local_edges = (
			(elem[0], elem[1]),
			(elem[1], elem[2]),
			(elem[2], elem[0]),
		)
		for edge in local_edges:
			key = tuple(sorted(int(v) for v in edge))
			if key in edge_map:
				count, existing_pid = edge_map[key]
				edge_map[key] = (count + 1, existing_pid)
			else:
				edge_map[key] = (1, int(pid))

	boundary_edges: list[tuple[int, int]] = []
	boundary_phys: list[int] = []
	for key, (count, pid) in edge_map.items():
		if count == 1:
			boundary_edges.append(key)
			boundary_phys.append(pid)
	return np.asarray(boundary_edges, dtype=int), np.asarray(boundary_phys, dtype=int)


def _boundary_weights_2d(points: np.ndarray, boundary_edges: np.ndarray) -> np.ndarray:
	weights = np.zeros(len(points), dtype=float)
	for i, j in boundary_edges:
		length = float(np.linalg.norm(points[int(i), :2] - points[int(j), :2]))
		weights[int(i)] += 0.5 * length
		weights[int(j)] += 0.5 * length
	return weights


def _boundary_weights_3d(points: np.ndarray, boundary_faces: np.ndarray) -> np.ndarray:
	weights = np.zeros(len(points), dtype=float)
	for i, j, k in boundary_faces:
		p0 = points[int(i)]
		p1 = points[int(j)]
		p2 = points[int(k)]
		area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
		weights[int(i)] += area / 3.0
		weights[int(j)] += area / 3.0
		weights[int(k)] += area / 3.0
	return weights


def _build_boundary_condition_field(
	points: np.ndarray,
	boundary_entities: np.ndarray,
	dim: int,
	h_conv: float,
	t_ext: float,
) -> BoundaryConditionField:
	weights = _boundary_weights_2d(points, boundary_entities) if dim == 2 else _boundary_weights_3d(points, boundary_entities)
	h = np.zeros(len(points), dtype=float)
	external_temperature = np.full(len(points), float(t_ext), dtype=float)
	dofs = np.flatnonzero(weights > 0.0)
	h[dofs] = float(h_conv)

	diag_values = h * weights
	loss_matrix = diags(diag_values, offsets=0, shape=(len(points), len(points)), format="csr")
	rhs = diag_values * external_temperature
	return BoundaryConditionField(
		weights=weights,
		h=h,
		t_ext=external_temperature,
		dofs=dofs,
		loss_matrix=loss_matrix,
		rhs=rhs,
	)


def _build_volume_loss_field(
	m_unit: csr_matrix,
	n_nodes: int,
	general_loss: float,
	vent_loss: float,
	radiation_loss: float,
	t_ext: float,
) -> VolumeLossField:
	linear_coeff = max(0.0, float(general_loss)) + max(0.0, float(vent_loss))
	ambient = np.full(n_nodes, float(t_ext), dtype=float)
	loss_matrix = (linear_coeff * m_unit).tocsr()
	rhs = linear_coeff * m_unit.dot(ambient)
	return VolumeLossField(
		linear_coeff=linear_coeff,
		radiation_coeff=max(0.0, float(radiation_loss)),
		t_ext=float(t_ext),
		loss_matrix=loss_matrix,
		rhs=np.asarray(rhs, dtype=float),
	)


def _radiation_loss_rhs(m_unit: csr_matrix, local_t: np.ndarray, volume_loss: VolumeLossField) -> np.ndarray:
	if volume_loss.radiation_coeff <= 0.0:
		return np.zeros_like(local_t)
	radiation_t = np.clip(local_t, 0.0, 5000.0)
	external_t = max(0.0, min(5000.0, volume_loss.t_ext))
	power_density = volume_loss.radiation_coeff * (radiation_t**4 - external_t**4)
	return np.asarray(m_unit.dot(power_density), dtype=float)


def _heat_release_rate(material: dict[str, object], elem_temp: float, burn_age: float) -> float:
	_ = elem_temp
	peak_hrr = max(0.0, float(material.get("hrr", 0.0)))
	duration = max(0.0, float(material.get("hrr_duration", 0.0)))
	if peak_hrr <= 0.0 or duration <= 0.0 or burn_age >= duration:
		return 0.0

	ramp_end = 0.10 * duration
	decay_start = 0.75 * duration
	if burn_age < ramp_end:
		return peak_hrr * (burn_age / max(ramp_end, 1e-12))
	if burn_age < decay_start:
		return peak_hrr
	return peak_hrr * max(0.0, (duration - burn_age) / max(duration - decay_start, 1e-12))


def _hrr_source_rhs(
	system: ElementwiseSystem,
	elems: np.ndarray,
	burned_elements: np.ndarray,
	local_t: np.ndarray,
	burn_times: np.ndarray,
	sim_time: float,
	vertical_transfer: VerticalHeatTransferField | None = None,
	vertical_attenuation: float = 0.25,
) -> np.ndarray:
	burned_indices = np.flatnonzero(burned_elements)
	if len(burned_indices) == 0:
		return np.zeros(system.m_mat.shape[0], dtype=float)

	elem_temperatures = np.mean(local_t[elems[burned_indices]], axis=1)
	local_loads = np.zeros((len(burned_indices), elems.shape[1]), dtype=float)
	vertical_load_by_element: dict[int, np.ndarray] = {}
	for local_idx, elem_idx in enumerate(burned_indices):
		material = get_material(str(system.elem_material_names[int(elem_idx)]))
		burn_age = max(0.0, sim_time - float(burn_times[int(elem_idx)]))
		hrr = _heat_release_rate(material, float(elem_temperatures[local_idx]), burn_age)
		local_loads[local_idx] = hrr * system.unit_load_local[int(elem_idx)]

		if vertical_transfer is not None and vertical_transfer.enabled and hrr > 0.0:
			source_power = hrr * float(vertical_transfer.element_volumes[int(elem_idx)])
			for target_idx, dz in zip(vertical_transfer.targets[int(elem_idx)], vertical_transfer.dz[int(elem_idx)], strict=False):
				factor = max(0.0, 1.0 - max(0.0, float(vertical_attenuation)) * float(dz))
				if factor <= 0.0:
					continue
				target_volume = max(float(vertical_transfer.element_volumes[int(target_idx)]), 1e-12)
				target_load = source_power * factor * system.unit_load_local[int(target_idx)] / target_volume
				if int(target_idx) in vertical_load_by_element:
					vertical_load_by_element[int(target_idx)] += target_load
				else:
					vertical_load_by_element[int(target_idx)] = target_load.copy()

	rhs = _assemble_vector_from_local(elems[burned_indices], local_loads, system.m_mat.shape[0])
	if vertical_load_by_element:
		target_indices = np.asarray(list(vertical_load_by_element.keys()), dtype=int)
		target_loads = np.asarray([vertical_load_by_element[int(idx)] for idx in target_indices], dtype=float)
		rhs += _assemble_vector_from_local(elems[target_indices], target_loads, system.m_mat.shape[0])
	return rhs


def _update_burned_elements(
	burned_elements: np.ndarray,
	elems: np.ndarray,
	local_t: np.ndarray,
	elem_tc: np.ndarray,
) -> np.ndarray:
	elem_temperatures = np.mean(local_t[elems], axis=1)
	newly_burned = (~burned_elements) & (elem_temperatures >= elem_tc)
	burned_elements[newly_burned] = True
	return np.flatnonzero(newly_burned)


def _burned_triangle_vertices(points: np.ndarray, elems: np.ndarray, burned_elements: np.ndarray) -> list[np.ndarray]:
	return [points[tri, :2] for tri in elems[burned_elements]]


def _burned_tetra_face_vertices(points: np.ndarray, elems: np.ndarray, burned_elements: np.ndarray) -> list[np.ndarray]:
	face_vertices: list[np.ndarray] = []
	for tet in elems[burned_elements]:
		i, j, k, l = [int(v) for v in tet]
		face_vertices.extend(
			[
				points[[i, j, k]],
				points[[i, j, l]],
				points[[i, k, l]],
				points[[j, k, l]],
			]
		)
	return face_vertices


def _burned_boundary_face_mask(boundary_faces: np.ndarray, elems: np.ndarray, burned_elements: np.ndarray) -> np.ndarray:
	burned_faces: set[tuple[int, int, int]] = set()
	for tet in elems[burned_elements]:
		i, j, k, l = [int(v) for v in tet]
		burned_faces.add(tuple(sorted((i, j, k))))
		burned_faces.add(tuple(sorted((i, j, l))))
		burned_faces.add(tuple(sorted((i, k, l))))
		burned_faces.add(tuple(sorted((j, k, l))))
	return np.asarray([tuple(sorted(int(v) for v in face)) in burned_faces for face in boundary_faces], dtype=bool)


def _burned_boundary_face_vertices(points: np.ndarray, boundary_faces: np.ndarray, elems: np.ndarray, burned_elements: np.ndarray) -> list[np.ndarray]:
	mask = _burned_boundary_face_mask(boundary_faces, elems, burned_elements)
	return [points[face] for face in boundary_faces[mask]]


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


def _add_source_marker_2d(ax, src_x: float, src_y: float, src_radius: float):
	marker = ax.scatter(
		[src_x],
		[src_y],
		marker="X",
		s=115,
		c="#ff2d00",
		edgecolors="black",
		linewidths=0.9,
		zorder=8,
		label="Source",
	)
	if src_radius > 0.0:
		ax.add_patch(
			Circle(
				(src_x, src_y),
				src_radius,
				fill=False,
				edgecolor="#ff2d00",
				linewidth=1.4,
				alpha=0.85,
				zorder=7,
			)
		)
	return marker


def _add_source_marker_3d(ax, src_x: float, src_y: float, src_z: float):
	return ax.scatter(
		[src_x],
		[src_y],
		[src_z],
		marker="X",
		s=95,
		c="#ff2d00",
		edgecolors="black",
		linewidths=0.8,
		depthshade=False,
		zorder=8,
		label="Source",
	)


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
			"dt": 100.0,
			"steps": 400,
			"sub_steps": 1,
			"theta": 1.0,
			"h_conv": 10.0,
			"general_loss": 0.2,
			"vent_loss": 1.0,
			"radiation_loss": 5.0e-8,
			"vertical_air_transfer": 1,
			"vertical_air_attenuation": 0.25,
			"vertical_air_radius": 0.0,
			"vertical_air_random_delta": 0.2,
			"t_amb": 293.0,
			"src_temp": 800.0,
			"src_x": 2.0,
			"src_y": 2.0,
			"src_z": 0.5,
			"src_radius": 1.0,
		}
	return {
		"dt": 50.0,
		"steps": 2000,
		"sub_steps": 1,
		"theta": 1.0,
		"h_conv": 10.0,
		"general_loss": 0.2,
		"vent_loss": 1.0,
		"radiation_loss": 5.0e-8,
		"vertical_air_transfer": 0,
		"vertical_air_attenuation": 0.25,
		"vertical_air_radius": 0.0,
		"vertical_air_random_delta": 0.0,
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
	general_loss = float(args.general_loss if args.general_loss is not None else defaults["general_loss"])
	vent_loss = float(args.vent_loss if args.vent_loss is not None else defaults["vent_loss"])
	radiation_loss = float(args.radiation_loss if args.radiation_loss is not None else defaults["radiation_loss"])
	vertical_air_transfer = bool(args.vertical_air_transfer if args.vertical_air_transfer is not None else defaults["vertical_air_transfer"])
	vertical_air_attenuation = float(args.vertical_air_attenuation if args.vertical_air_attenuation is not None else defaults["vertical_air_attenuation"])
	vertical_air_radius = float(args.vertical_air_radius if args.vertical_air_radius is not None else defaults["vertical_air_radius"])
	vertical_air_random_delta = max(0.0, float(args.vertical_air_random_delta if args.vertical_air_random_delta is not None else defaults["vertical_air_random_delta"]))
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
	base_dir = PROJECT_ROOT
	mesh_path = Path(mesh_file)
	if not mesh_path.exists():
		candidate = base_dir / mesh_file
		mesh_path = candidate if candidate.exists() else base_dir / "models" / mesh_file
	print(f"Using mesh: {mesh_path}")

	t0 = time.perf_counter()
	pts, elems, phys, phys_name_map = _load_mesh_data(mesh_path, dim)
	record_timing("mesh_load", time.perf_counter() - t0, details=f"{mesh_path};dim={dim}")
	print(f"Maillage charge: {len(pts)} noeuds, {len(elems)} elements ({dim}D).")

	elem_tc = _element_tc_values(phys, phys_name_map)
	burned_elements = np.zeros(len(elems), dtype=bool)
	burn_times = np.full(len(elems), np.inf, dtype=float)
	t0 = time.perf_counter()
	system = _assemble_elementwise_system(pts, elems, phys, phys_name_map, dim)
	record_timing("system_assembly", time.perf_counter() - t0, details=f"nodes={len(pts)};elements={len(elems)};dim={dim}")
	t0 = time.perf_counter()
	vertical_transfer = _build_vertical_heat_transfer_field(
		pts,
		elems,
		system.unit_load_local,
		dim,
		vertical_air_transfer,
		vertical_air_attenuation,
		vertical_air_radius,
	)
	record_timing(
		"vertical_air_transfer_prepare",
		time.perf_counter() - t0,
		details=f"enabled={vertical_transfer.enabled};attenuation={vertical_air_attenuation};radius={vertical_air_radius}",
	)

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
	initial_burned_indices = _update_burned_elements(burned_elements, elems, t, elem_tc)
	if len(initial_burned_indices):
		burn_times[initial_burned_indices] = 0.0
		_apply_burn_deltas(system, elems, initial_burned_indices)
		record_timing("element_burn_initial", 0.0, details=f"changed_elements={len(initial_burned_indices)}")

	empty_dofs = np.array([], dtype=int)
	empty_vals = np.array([], dtype=float)
	boundary_faces = boundary_phys = boundary_edges = boundary_edge_phys = None
	visual_regions = _build_visual_regions(phys, phys_name_map)
	if dim == 3:
		boundary_faces, boundary_phys = _extract_boundary_faces(elems, phys)
		bc_field = _build_boundary_condition_field(pts, boundary_faces, dim, h_conv, t_amb)
	else:
		boundary_edges, boundary_edge_phys = _extract_boundary_edges(elems, phys)
		bc_field = _build_boundary_condition_field(pts, boundary_edges, dim, h_conv, t_amb)
	volume_loss = _build_volume_loss_field(system.m_unit, len(pts), general_loss, vent_loss, radiation_loss, t_amb)
	k_eff = system.k_mat + bc_field.loss_matrix + volume_loss.loss_matrix

	def apply_material_changes(burned_indices: np.ndarray) -> None:
		nonlocal volume_loss, k_eff
		if len(burned_indices) == 0:
			return
		t_update = time.perf_counter()
		_apply_burn_deltas(system, elems, burned_indices)
		volume_loss = _build_volume_loss_field(system.m_unit, len(pts), general_loss, vent_loss, radiation_loss, t_amb)
		k_eff = system.k_mat + bc_field.loss_matrix + volume_loss.loss_matrix
		record_timing(
			"material_burn_delta_update",
			time.perf_counter() - t_update,
			details=f"changed_elements={len(burned_indices)}",
		)

	def _setup_axes(local_t: np.ndarray, title: str):
		if dim == 2:
			fig, ax = plt.subplots(figsize=(10, 8))
			im = ax.tripcolor(pts[:, 0], pts[:, 1], elems, local_t, cmap=THERMAL_CMAP, shading="gouraud", vmin=t_amb, vmax=1500, alpha=0.86)
			_add_region_overlays_2d(ax, pts, elems, phys, visual_regions)
			burned_collection = PolyCollection(
				_burned_triangle_vertices(pts, elems, burned_elements),
				facecolors="black",
				edgecolors="black",
				linewidths=0.45,
				alpha=0.28,
				zorder=9,
			)
			ax.add_collection(burned_collection)
			source_marker = _add_source_marker_2d(ax, src_x, src_y, src_radius)
			plt.colorbar(im, ax=ax, label="Temperature [K]")
			ax.set_facecolor("#1A12BD")
			ax.set_aspect("equal")
			ax.set_title(title)
			legend_handles = _build_region_legend_handles(visual_regions)
			legend_handles.append(source_marker)
			legend_handles.append(Patch(facecolor="black", edgecolor="black", alpha=0.28, label="Brule"))
			if legend_handles:
				ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, title="Objets / Materiaux")
			return fig, ax, {"field": im, "burned_collection": burned_collection, "source_marker": source_marker}

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
			cmap=THERMAL_CMAP_3D,
			s=5,
			vmin=THERMAL_3D_VMIN,
			vmax=THERMAL_3D_VMAX,
			alpha=0.82,
			depthshade=False,
		)

		_add_region_overlays_3d(ax_full, pts, boundary_faces, boundary_phys, visual_regions, include_solid_fill=False)
		face_temps = np.mean(local_t[boundary_faces], axis=1)
		norm = Normalize(vmin=THERMAL_3D_VMIN, vmax=THERMAL_3D_VMAX)
		face_colors = THERMAL_CMAP_3D(norm(face_temps))
		burned_boundary_mask = _burned_boundary_face_mask(boundary_faces, elems, burned_elements)
		face_colors[burned_boundary_mask] = (0.0, 0.0, 0.0, 1.0)
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
		burned_tetra_surface = Poly3DCollection(
			_burned_boundary_face_vertices(pts, boundary_faces, elems, burned_elements),
			facecolors=(0.0, 0.0, 0.0, 1.0),
			edgecolors=(0.0, 0.0, 0.0, 1.0),
			linewidths=0.12,
			alpha=0.9,
			zorder=9,
		)
		ax_full.add_collection3d(burned_tetra_surface)
		source_marker_mesh = _add_source_marker_3d(ax_mesh, src_x, src_y, src_z)
		source_marker_full = _add_source_marker_3d(ax_full, src_x, src_y, src_z)

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
		legend_handles.append(source_marker_full)
		legend_handles.append(Patch(facecolor="black", edgecolor="black", alpha=0.9, label="Brule"))
		if legend_handles:
			ax_full.legend(handles=legend_handles, loc="upper right", framealpha=0.9, title="Objets / Materiaux")
		return fig, ax_full, {"mesh_scatter": mesh_scatter, "full_surface": full_surface, "mesh_ax": ax_mesh, "full_ax": ax_full, "region_fills": region_fills, "region_edges": region_edges, "burned_tetra_surface": burned_tetra_surface, "source_marker_mesh": source_marker_mesh, "source_marker_full": source_marker_full}

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
	rng = np.random.default_rng()

	def _update_visual(local_t: np.ndarray, sim_time: float):
		if dim == 2:
			field = visuals["field"]
			burned_collection = visuals["burned_collection"]
			field.set_array(local_t)
			burned_collection.set_verts(_burned_triangle_vertices(pts, elems, burned_elements))
			ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(local_t):.1f}K")
			return [field, burned_collection]
		else:
			mesh_scatter = visuals["mesh_scatter"]
			full_surface = visuals["full_surface"]
			burned_tetra_surface = visuals["burned_tetra_surface"]
			mesh_ax = visuals["mesh_ax"]
			full_ax = visuals["full_ax"]
			mesh_scatter.set_array(local_t)
			mesh_scatter.set_clim(vmin=THERMAL_3D_VMIN, vmax=THERMAL_3D_VMAX)
			face_temps = np.mean(local_t[boundary_faces], axis=1)
			norm = Normalize(vmin=THERMAL_3D_VMIN, vmax=THERMAL_3D_VMAX)
			face_colors = THERMAL_CMAP_3D(norm(face_temps))
			burned_boundary_mask = _burned_boundary_face_mask(boundary_faces, elems, burned_elements)
			face_colors[burned_boundary_mask] = (0.0, 0.0, 0.0, 1.0)
			full_surface.set_facecolor(face_colors)
			burned_tetra_surface.set_verts(_burned_boundary_face_vertices(pts, boundary_faces, elems, burned_elements))
			burned_tetra_surface.set_facecolor((0.0, 0.0, 0.0, 1.0))
			burned_tetra_surface.set_edgecolor((0.0, 0.0, 0.0, 1.0))
			burned_tetra_surface.set_alpha(0.9)
			mesh_ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(local_t):.1f}K | Vue maillage")
			full_ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(local_t):.1f}K | Vue pleine")
			return [mesh_scatter, full_surface, burned_tetra_surface]

	def advance_state(current_t: np.ndarray, current_time: float) -> tuple[np.ndarray, float, float]:
		t_local = current_t
		sim_time = float(current_time)
		t0 = time.perf_counter()
		for _ in range(sub_steps):
			vertical_attenuation_step = vertical_air_attenuation
			if vertical_transfer.enabled and vertical_air_random_delta > 0.0:
				vertical_attenuation_step = float(
					vertical_air_attenuation
					* rng.uniform(max(0.0, 1.0 - vertical_air_random_delta), 1.0 + vertical_air_random_delta)
				)
			src = _hrr_source_rhs(system, elems, burned_elements, t_local, burn_times, sim_time, vertical_transfer, vertical_attenuation_step)
			radiation_loss_rhs = _radiation_loss_rhs(system.m_unit, t_local, volume_loss)
			rhs = src + bc_field.rhs + volume_loss.rhs - radiation_loss_rhs
			t_local = np.asarray(
				theta_step(
					system.m_mat,
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
			burned_indices = _update_burned_elements(burned_elements, elems, t_local, elem_tc)
			if len(burned_indices):
				burn_times[burned_indices] = sim_time + dt
				record_timing("element_burn_update", 0.0, details=f"changed_elements={len(burned_indices)}")
				apply_material_changes(burned_indices)
			sim_time += dt
		return t_local, sim_time, time.perf_counter() - t0

	if not args.plot:
		headless_frames = [t.copy()]
		headless_times = [0.0]
		headless_burned = [burned_elements.copy()]
		t0 = time.perf_counter()
		for frame_idx in range(steps):
			t, sim_time, elapsed = advance_state(t, headless_times[-1])
			record_timing("frame_calculation", elapsed, frame=frame_idx, details=f"sim_time={sim_time:.5f}")
			headless_frames.append(t.copy())
			headless_times.append(sim_time)
			headless_burned.append(burned_elements.copy())
		state["t"] = t
		state["time"] = headless_times[-1]
		record_timing("headless_calculation_total", time.perf_counter() - t0, details=f"frames={steps};dim={dim}")

		def update_anim(frame_idx: int):
			burned_elements[:] = headless_burned[frame_idx]
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
	parser.add_argument("--general-loss", dest="general_loss", type=float, default=None, help="Perte volumique lineaire generale [W/m3/K]")
	parser.add_argument("--vent-loss", dest="vent_loss", type=float, default=None, help="Perte volumique de ventilation [W/m3/K]")
	parser.add_argument("--radiation-loss", dest="radiation_loss", type=float, default=None, help="Coefficient radiatif volumique [W/m3/K4]")
	parser.add_argument("--vertical-air-transfer", dest="vertical_air_transfer", type=int, choices=[0, 1], default=None, help="Active le transfert vertical simplifie de HRR en 3D")
	parser.add_argument("--vertical-air-attenuation", dest="vertical_air_attenuation", type=float, default=None, help="Attenuation verticale du transfert air [1/m]")
	parser.add_argument("--vertical-air-radius", dest="vertical_air_radius", type=float, default=None, help="Rayon horizontal du transfert vertical; 0=auto")
	parser.add_argument("--vertical-air-random-delta", dest="vertical_air_random_delta", type=float, default=None, help="Variation aleatoire de l'attenuation verticale a chaque sous-pas; 0=desactive")
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
