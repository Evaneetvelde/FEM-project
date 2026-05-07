#simulation diffusion-reacction 2D/3D
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import meshio
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.sparse import csr_matrix, diags

from calculs.dirichlet import precompute_dirichlet_dofs, theta_step_fast
# Les kernels Numba sont utilises dans preassemble_* et assemble_*_from_preassembled.
# On garde rows/cols/unit_data pour reassembler vite quand les coefficients changent.
from calculs.mass import assemble_mass_from_preassembled, preassemble_mass_unit
from calculs.stiffness import assemble_stiffness_from_preassembled, preassemble_stiffness_unit
from materialsbank import get_burn_material_name, get_material, get_material_color, get_material_overlay_alpha
from structural_fun import prepare_structural_fun, update_structural_fun

PHYSICAL_ID_MAP = { # lien id mesh -> materiau
	1: "bois", 
	2: "beton",
	3: "verre",
	4: "isolation",
	5: "air",
	6: "metal",
	7: "meche",
	8: "explosif",
	9: "viande",
	10: "vegetation",
	11: "ptfe",
	12: "acier",
}

ROLE_KEYWORDS = { # lien materiau -> role dans la visualisation
	"wall": {"mur", "murs", "wall", "walls", "beton"},
	"window": {"fenetre", "fenetres", "window", "windows", "glass", "verre"},
	"floor": {"sol", "floor", "slab", "dalle"},
	"door": {"porte", "portes", "door", "doors"},
	"column": {"colonne", "colonnes", "column", "columns", "pillar", "pillars", "acier", "steel"},
	"vegetation": {"vegetation", "vegetal", "plante", "plantes", "tree", "trees"},
}

THERMAL_CMAP = LinearSegmentedColormap.from_list( # lien temperature -> couleur 2D
	"thermal_white_red",
	[
		(1.0, 1.0, 1.0, 0.55),
		(1.0, 0.92, 0.92, 0.72),
		(1.0, 0.45, 0.25, 0.9),
		(1.0, 0.0, 0.0, 1.0),
	],
	N=256,
)

THERMAL_CMAP_3D = LinearSegmentedColormap.from_list( # lien temperature -> couleur 3D
	"thermal_3d_white_purple_red",
	[
		(1.0, 1.0, 1.0, 1.0),
		(0.45, 0.0, 0.75, 1.0),
		(1.0, 0.0, 0.0, 1.0),
	],
	N=256,
)
THERMAL_3D_VMIN = 200.0 # pour cap les valeurs lors du débuggages des anomalies
THERMAL_3D_VMAX = 1500.0


@dataclass
class BoundaryConditionField: # class CL
	weights: np.ndarray
	h: np.ndarray
	t_ext: np.ndarray
	dofs: np.ndarray
	loss_matrix: csr_matrix
	rhs: np.ndarray


@dataclass
class VolumeLossField: # class simplification perte de volume
	linear_coeff: float
	radiation_coeff: float
	t_ext: float
	loss_matrix: csr_matrix
	rhs: np.ndarray


@dataclass
class ElementwiseSystem: # class systeme element par element
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
	mass_rows: np.ndarray
	mass_cols: np.ndarray
	mass_unit_data: np.ndarray
	mass_n_nodes: int
	mass_nloc: int
	stiffness_rows: np.ndarray
	stiffness_cols: np.ndarray
	stiffness_unit_data: np.ndarray
	stiffness_n_nodes: int
	stiffness_nloc: int


@dataclass
class VerticalHeatTransferField: # class simplification transfert de chaleur vertical
	enabled: bool
	targets: list[np.ndarray]
	dz: list[np.ndarray]
	element_volumes: np.ndarray


@dataclass
class HorizontalAirTransferField: # class simplification mouvements d'air horizontaux
	enabled: bool
	targets: list[np.ndarray]
	distances: list[np.ndarray]
	element_volumes: np.ndarray
	power_fraction: float


def _physical_id_to_name_map(msh: meshio.Mesh, dim: int) -> dict[int, str]: #visuel 
	"""
	helper mapping id mesh -> materiau

	param: msh: meshio.Mesh
	param: dim: dimension du maillage (2 ou 3)
	return: dict[int, str] mapping id mesh -> nom du materiau
	"""
	return PHYSICAL_ID_MAP.copy()


def _format_region_label(raw_name: str) -> str: # visuel
	"""
	helper formatage label region

	param: raw_name: nom brut de la region
	return: str label formaté
	"""
	label = str(raw_name).strip().replace("_", " ")
	return " ".join(part.capitalize() for part in label.split()) or "Region"


def _infer_region_role(raw_name: str) -> str: # visuel
	"""
	helper inférence role region 
	
	param: raw_name: nom brut de la region
	return: str role inféré (wall, window, floor, door, column ou region)
	"""
	name = str(raw_name).strip().lower()
	tokens = {token for token in name.replace("-", "_").split("_") if token}
	for role, keywords in ROLE_KEYWORDS.items():
		if any(keyword in name for keyword in keywords) or tokens.intersection(keywords):
			return role
	return "region"


def _build_visual_regions(phys_ids: np.ndarray, phys_name_map: dict[int, str]) -> list[dict[str, object]]: # visuel
	"""
	helper construction regions pour visualisation

	param: phys_ids: tableau des ids physiques par element
	param: phys_name_map: mapping id mesh -> nom du materiau
	return: list[dict[str, object]] liste des regions 
	"""
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


def load_mesh_data(mesh_path: Path, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str], int]: # chargement du maillage
	"""
	Charge le maillage, en mergeant les noeuds doublons dans la 3D (répare la diffusion sol-> mur)

	param: mesh_path: chemin vers le fichier de maillage
	param: dim: dimension souhaitée (2 ou 3), auto-détectée si les éléments du type demandé sont absents
	return: tuple contenant les points, les éléments, les ids physiques, le mapping id->nom physique et la dimension finale utilisée
	"""
	if not mesh_path.exists() or mesh_path.stat().st_size < 100:
		raise FileNotFoundError(f"Le maillage {mesh_path} est introuvable, vide ou corrompu. Verifiez qu'il a bien ete genere en 3D.")
	msh = meshio.read(str(mesh_path))
	cell_type = "triangle" if dim == 2 else "tetra"
	elems = np.asarray(msh.cells_dict.get(cell_type, np.array([], dtype=int)), dtype=int)
	
	# Auto-detect dimension if requested cell type has no elements
	if len(elems) == 0:
		if dim == 2 and "tetra" in msh.cells_dict:
			dim = 3
			cell_type = "tetra"
			elems = np.asarray(msh.cells_dict["tetra"], dtype=int)
		elif dim == 3 and "triangle" in msh.cells_dict:
			dim = 2
			cell_type = "triangle"
			elems = np.asarray(msh.cells_dict["triangle"], dtype=int)
	
	pts = np.asarray(msh.points, dtype=float)[:, :dim]
	phys = np.asarray(
		msh.cell_data_dict.get("gmsh:physical", {}).get(cell_type, np.ones(len(elems), dtype=int)),
		dtype=int,
	)

	# Fusion des noeuds doublons (repare les maillages non-conformes/entites separees)
	unique_pts, inverse = np.unique(np.round(pts, 7), axis=0, return_inverse=True)
	pts = unique_pts
	elems = inverse[elems]

	phys_name_map = _physical_id_to_name_map(msh, dim)
	return pts, elems, phys, phys_name_map, dim


@njit(cache=True)
def Nbuild_quadra_2D(points: np.ndarray, elems: np.ndarray, n_ref: np.ndarray): #process numba
	ne = len(elems)
	ngp = n_ref.shape[0]
	jac = np.zeros((ne, ngp, 3, 3), dtype=np.float64)
	det = np.zeros((ne, ngp), dtype=np.float64)
	coords = np.zeros((ne, ngp, 3), dtype=np.float64)

	for e in range(ne):                       # comme petite matrice, moins couteux que de faire les sous matrice pour le jacobien
		i0 = elems[e, 0]
		i1 = elems[e, 1]
		i2 = elems[e, 2]
		p0x = points[i0, 0]
		p0y = points[i0, 1]
		p1x = points[i1, 0]
		p1y = points[i1, 1]
		p2x = points[i2, 0]
		p2y = points[i2, 1]
		j11 = p1x - p0x
		j12 = p2x - p0x
		j21 = p1y - p0y
		j22 = p2y - p0y
		det_j = abs(j11 * j22 - j12 * j21)

		for g in range(ngp):
			jac[e, g, 0, 0] = j11
			jac[e, g, 0, 1] = j12
			jac[e, g, 1, 0] = j21
			jac[e, g, 1, 1] = j22
			jac[e, g, 2, 2] = 1.0
			det[e, g] = det_j
			coords[e, g, 0] = n_ref[g, 0] * p0x + n_ref[g, 1] * p1x + n_ref[g, 2] * p2x
			coords[e, g, 1] = n_ref[g, 0] * p0y + n_ref[g, 1] * p1y + n_ref[g, 2] * p2y

	return jac, det, coords


@njit(cache=True)
def _det3(jac_e: np.ndarray) -> float: #process numba
	return (																	#comme petite matrice, moins couteux
		jac_e[0, 0] * (jac_e[1, 1] * jac_e[2, 2] - jac_e[1, 2] * jac_e[2, 1])
		- jac_e[0, 1] * (jac_e[1, 0] * jac_e[2, 2] - jac_e[1, 2] * jac_e[2, 0])
		+ jac_e[0, 2] * (jac_e[1, 0] * jac_e[2, 1] - jac_e[1, 1] * jac_e[2, 0])
	)


@njit(cache=True) 
def Nbuild_quadra_3D(points: np.ndarray, elems: np.ndarray, bary: np.ndarray): #process numba
	ne = len(elems)
	ngp = bary.shape[0]
	jac = np.zeros((ne, ngp, 3, 3), dtype=np.float64)
	det = np.zeros((ne, ngp), dtype=np.float64)
	coords = np.zeros((ne, ngp, 3), dtype=np.float64)

	for e in range(ne):                                                        # même justification que précedement
		i0 = elems[e, 0]
		i1 = elems[e, 1]
		i2 = elems[e, 2]
		i3 = elems[e, 3]
		jac_e = np.empty((3, 3), dtype=np.float64)
		for d in range(3):
			p0 = points[i0, d]
			jac_e[d, 0] = points[i1, d] - p0
			jac_e[d, 1] = points[i2, d] - p0
			jac_e[d, 2] = points[i3, d] - p0
		det_j = abs(_det3(jac_e))

		for g in range(ngp):
			for r in range(3):
				for c in range(3):
					jac[e, g, r, c] = jac_e[r, c]
			det[e, g] = det_j
			for d in range(3):
				coords[e, g, d] = (
					bary[g, 0] * points[i0, d]
					+ bary[g, 1] * points[i1, d]
					+ bary[g, 2] * points[i2, d]
					+ bary[g, 3] * points[i3, d]
				)

	return jac, det, coords


def build_triangle_quadra(points: np.ndarray, elems: np.ndarray): # calcul 
	"""
	Construit la quadrature des éléments 2D de type triangle P1

	param: points: tableau des coordonnées des points du maillage
	param: elems: tableau des éléments du maillage  
	"""
	w = np.full(3, 1.0 / 6.0, dtype=float)
	n_ref = np.array( # quadrature de degré 2 pour le triangle, 3 points de quadrature
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

	ngp = len(w)
	jac, det, coords = Nbuild_quadra_2D( np.asarray(points, dtype=np.float64),np.asarray(elems, dtype=np.int64),n_ref,)

	return w, n_ref, np.repeat(grad_ref[None, :, :], ngp, axis=0), jac, det, coords


def build_tetra_quadra(points: np.ndarray, elems: np.ndarray): #calcul
	"""
	Construit la quadrature des éléments 3D de type tétraèdre P1

	param: points: tableau des coordonnées des points du maillage
	param: elems: tableau des éléments du maillage
	"""
	a = 0.5854101966249685 
	b = 0.1381966011250105
	bary = np.array( # quadrature de degré 2 pour le tétraèdre, 4 points de quadrature
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

	ngp = len(w)
	jac, det, coords = Nbuild_quadra_3D(
		np.asarray(points, dtype=np.float64),
		np.asarray(elems, dtype=np.int64),
		bary,
	)

	return w, n_ref, np.repeat(grad_ref[None, :, :], ngp, axis=0), jac, det, coords


def _build_p1_quadrature(points: np.ndarray, elems: np.ndarray, dim: int): # calcul
	"""
	wrapper des constructions de quadrature

	param: dim: dimension du maillage
	"""
	return build_triangle_quadra(points, elems) if dim == 2 else build_tetra_quadra(points, elems)


def _element_tc_values(phys_ids: np.ndarray, phys_name_map: dict[int, str]) -> np.ndarray: #données
	"""
	helper extraction temperature de combustion par element

	param: phys_ids: tableau des ids physiques par element
	param: phys_name_map: mapping id mesh -> nom du materiau
	return: tableau des temperature de combustion par element
	"""
	return np.asarray([float(get_material(str(phys_name_map.get(int(pid), "bois")).strip().lower())["Tc"]) for pid in phys_ids], dtype=float)


def _initial_element_materials(phys_ids: np.ndarray, phys_name_map: dict[int, str]) -> np.ndarray: # données
	"""
	helper extraction nom du materiau par element

	param: phys_ids: tableau des ids physiques par element
	param: phys_name_map: mapping id mesh -> nom du materiau
	return: tableau des noms de materiau par element
	"""
	return np.asarray([str(phys_name_map.get(int(pid), "bois")).strip().lower() for pid in phys_ids], dtype=object)


def _node_reaction_fields(elems: np.ndarray, elem_material_names: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]: # calcul
	"""
	helper construction champs de reaction aux noeuds (q et tc)

	param: elems: tableau des éléments du maillage
	param: elem_material_names: tableau des noms de materiau par element
	param: n_nodes: nombre de noeuds dans le maillage
	return: tuple de tableaux (q_node, tc_node) contenant q et tc 
	"""
	q_node = np.zeros(n_nodes, dtype=float)
	tc_node = np.full(n_nodes, np.inf, dtype=float)
	for nodes, mat_name in zip(elems, elem_material_names, strict=False):
		mat = get_material(str(mat_name))
		for ni in nodes:
			q_node[int(ni)] = max(q_node[int(ni)], float(mat["Q"]))
			tc_node[int(ni)] = min(tc_node[int(ni)], float(mat["Tc"]))
	return q_node, tc_node

def _assemble_vector_from_local(elems: np.ndarray, local_values: np.ndarray, n_nodes: int) -> np.ndarray:
	"""
	helper assembly d'un vecteur à partir de valeurs locales

	param: elems: tableau des éléments du maillage
	param: local_values: tableau des valeurs locales
	param: n_nodes: nombre de noeuds dans le maillage
	return: tableau du vecteur assemblé
	"""
	values = np.zeros(n_nodes, dtype=float)
	np.add.at(values, elems.reshape(-1), local_values.reshape(-1))
	return values


def _material_coefficients( elem_material_names: np.ndarray,use_burn_delta: bool = False,) -> tuple[np.ndarray, np.ndarray]:# données
	"""
	helpeur extraction des coefficients thermiques (k et rho*c) par element, avec option de delta de combustion

	param: elem_material_names: tableau des noms de materiau par element
	param: use_burn_delta: si True, retourne les deltas de k et rho*c entre le materiau brûlé et non-brûlé
	return: tuple de tableaux (k_coeffs, m_coeffs) contenant k et rho*c
	"""

	k_coeffs = np.zeros(len(elem_material_names), dtype=float)
	m_coeffs = np.zeros(len(elem_material_names), dtype=float)
	for idx, mat_name in enumerate(elem_material_names):
		mat = get_material(mat_name)
		if use_burn_delta:
			burn_mat = get_material(get_burn_material_name(mat_name))
			k_coeffs[idx] = float(burn_mat["k"]) - float(mat["k"])
			m_coeffs[idx] = float(burn_mat["rho"]) * float(burn_mat["c"]) - float(mat["rho"]) * float(mat["c"])
		else:
			k_coeffs[idx] = float(mat["k"])
			m_coeffs[idx] = float(mat["rho"]) * float(mat["c"])
	return k_coeffs, m_coeffs

def _default_vertical_air_radius(points: np.ndarray, elems: np.ndarray) -> float: # calcul
	"""
	Simule un rayon d'action de la monté de la chaleur au étage dans la simplification 3D

	param: points: tableau des coordonnées des points du maillage
	param: elems: tableau des éléments du maillage
	return: rayon d'action 
	"""
	centroids = np.mean(points[elems], axis=1)
	spans = np.ptp(centroids[:, :2], axis=0)
	area = max(float(spans[0] * spans[1]), 1e-12)
	return 1.5 * float(np.sqrt(area / max(len(elems), 1)))


def _default_horizontal_air_radius(points: np.ndarray, elems: np.ndarray, dim: int) -> float: # calcul
	"""
	Simule le rayon d'action horizontal des mouvements d'air en 2D/3D.
	"""
	centroids = np.mean(points[elems], axis=1)
	horizontal = centroids[:, :2] if dim == 3 else centroids[:, :dim]
	spans = np.ptp(horizontal, axis=0)
	area = max(float(np.prod(np.maximum(spans, 1e-12))), 1e-12)
	return 2.0 * float(np.sqrt(area / max(len(elems), 1)))


def _build_vertical_heat_transfer_field(points: np.ndarray, elems: np.ndarray, unit_load_local: np.ndarray, dim: int, enabled: bool, attenuation_per_m: float, horizontal_radius: float) -> VerticalHeatTransferField: # calcul
	"""
	Construit le champ de transfert de chaleur vertical

	param: points: tableau des coordonnées des points du maillage
	param: elems: tableau des éléments du maillage
	param: unit_load_local: tableau des charges unitaires locales par élément
	param: dim: dimension du maillage
	param: enabled: bool indiquant si le transfert de chaleur vertical est activé
	param: attenuation_per_m: coefficient d'atténuation de la chaleur par mètre de distance verticale
	param: horizontal_radius: rayon d'influence horizontal pour le transfert de chaleur vertical, ou 0 pour auto-détection
	return: VerticalHeatTransferField contenant les informations nécessaires pour appliquer le transfert de chaleur vertical
	"""
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


def _build_horizontal_air_transfer_field(points: np.ndarray, elems: np.ndarray, unit_load_local: np.ndarray, dim: int, enabled: bool, attenuation_per_m: float, radius: float, power_fraction: float) -> HorizontalAirTransferField: # calcul
	"""
	Construit le champ de transfert horizontal de chaleur favorise par les mouvements d'air.
	"""
	element_volumes = np.sum(unit_load_local, axis=1)
	if not enabled:
		return HorizontalAirTransferField(False, [], [], element_volumes, 0.0)

	centroids = np.mean(points[elems], axis=1)
	horizontal = centroids[:, :2] if dim == 3 else centroids[:, :dim]
	effective_radius = float(radius) if radius > 0.0 else _default_horizontal_air_radius(points, elems, dim)
	attenuation = max(0.0, float(attenuation_per_m))
	targets: list[np.ndarray] = []
	distances_by_source: list[np.ndarray] = []

	for source_idx, source_center in enumerate(horizontal):
		distances = np.linalg.norm(horizontal - source_center, axis=1)
		mask = (distances > 0.0) & (distances <= effective_radius)
		local_factors = np.maximum(0.0, 1.0 - attenuation * distances[mask])
		valid = local_factors > 0.0
		targets.append(np.flatnonzero(mask)[valid])
		distances_by_source.append(distances[mask][valid])

	return HorizontalAirTransferField(True, targets, distances_by_source, element_volumes, max(0.0, float(power_fraction)))


def _assemble_elementwise_system(points: np.ndarray, elems: np.ndarray, phys_ids: np.ndarray, phys_name_map: dict[int, str], dim: int ) -> ElementwiseSystem: # calcul
	"""
	Helper assamblant le système élément par élément

	param: points: tableau des coordonnées des points du maillage
	param: elems: tableau des éléments du maillage
	param: phys_ids: tableau des ids physiques par élément
	param: phys_name_map: mapping id mesh -> nom du materiau
	param: dim: dimension du maillage
	return: ElementwiseSystem contenant les matrices, champs et informations nécessaires pour la simulation
	"""
	n_nodes = points.shape[0]
	elem_material_names = _initial_element_materials(phys_ids, phys_name_map)
	w, n_ref, grad_ref, jac, det, coords = _build_p1_quadrature(points, elems, dim)
	unit_load = np.einsum("eg,g,ga->ea", det, w, n_ref, optimize=True)
	tag_to_dof = np.arange(n_nodes, dtype=int)
	mass_rows, mass_cols, mass_unit_data, mass_n_nodes, mass_nloc = preassemble_mass_unit(elems.reshape(-1), det, w, n_ref, tag_to_dof)
	stiffness_rows, stiffness_cols, stiffness_unit_data, stiffness_n_nodes, stiffness_nloc = preassemble_stiffness_unit(elems.reshape(-1), jac, det, w, grad_ref, tag_to_dof)
	k_coeffs, m_coeffs = _material_coefficients(elem_material_names)
	k_mat = assemble_stiffness_from_preassembled(stiffness_rows, stiffness_cols, stiffness_unit_data, stiffness_n_nodes, stiffness_nloc, k_coeffs)
	m_mat = assemble_mass_from_preassembled(mass_rows, mass_cols, mass_unit_data, mass_n_nodes, mass_nloc, m_coeffs)
	m_unit = assemble_mass_from_preassembled(mass_rows, mass_cols, mass_unit_data, mass_n_nodes, mass_nloc)
	q_node, tc_node = _node_reaction_fields(elems, elem_material_names, n_nodes)

	return ElementwiseSystem(k_mat=k_mat, m_mat=m_mat, m_unit=m_unit, unit_load_local=unit_load, q_node=q_node, tc_node=tc_node, elem_material_names=elem_material_names, w=w, n_ref=n_ref, grad_ref=grad_ref, jac=jac, det=det, coords=coords, mass_rows=mass_rows, mass_cols=mass_cols, mass_unit_data=mass_unit_data, mass_n_nodes=mass_n_nodes, mass_nloc=mass_nloc, stiffness_rows=stiffness_rows, stiffness_cols=stiffness_cols, stiffness_unit_data=stiffness_unit_data,stiffness_n_nodes=stiffness_n_nodes, stiffness_nloc=stiffness_nloc )

def _apply_burn_deltas(system: ElementwiseSystem, elems: np.ndarray, burned_indices: np.ndarray) -> None: #calcul
	"""
	Applique la modification au différentes matrice suite à la combustion d'éléments

	param: system: ElementwiseSystem contenant les matrices et champs à modifier
	param: elems: tableau des éléments du maillage
	param: burned_indices: indices des éléments qui ont brûlé
	"""
	if len(burned_indices) == 0:
		return
	n_nodes = system.k_mat.shape[0]
	local_delta_k, local_delta_m = _material_coefficients(system.elem_material_names[burned_indices], use_burn_delta=True)
	delta_k_coeffs = np.zeros(len(elems), dtype=float)
	delta_m_coeffs = np.zeros(len(elems), dtype=float)
	delta_k_coeffs[burned_indices] = local_delta_k
	delta_m_coeffs[burned_indices] = local_delta_m
	delta_k = assemble_stiffness_from_preassembled( system.stiffness_rows, system.stiffness_cols, system.stiffness_unit_data, system.stiffness_n_nodes, system.stiffness_nloc, delta_k_coeffs)
	delta_m = assemble_mass_from_preassembled( system.mass_rows, system.mass_cols, system.mass_unit_data, system.mass_n_nodes, system.mass_nloc, delta_m_coeffs )
	delta_k.eliminate_zeros()
	delta_m.eliminate_zeros()
	system.k_mat = system.k_mat + delta_k
	system.m_mat = system.m_mat + delta_m
	for elem_idx in burned_indices:
		system.elem_material_names[int(elem_idx)] = get_burn_material_name(str(system.elem_material_names[int(elem_idx)]))
	system.q_node, system.tc_node = _node_reaction_fields(elems, system.elem_material_names, n_nodes)

def _extract_boundary_faces(elems: np.ndarray, phys_ids: np.ndarray, include_material_interfaces: bool = False) -> tuple[np.ndarray, np.ndarray]: # calcul
	"""
	helper extraction faces bords

	param: elems: tableau des éléments du maillage
	param: phys_ids: tableau des ids physiques par élément
	return: tuple de tableaux (boundary_faces, boundary_phys) contenant faces de bord et ids
	"""
	face_map: dict[tuple[int, int, int], list[int]] = {}
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
				face_map[key].append(int(pid))
			else:
				face_map[key] = [int(pid)]

	boundary_faces: list[tuple[int, int, int]] = []
	boundary_phys: list[int] = []
	for key, pids in face_map.items():
		if len(pids) == 1:
			boundary_faces.append(key)
			boundary_phys.append(pids[0])
		elif include_material_interfaces and len(set(pids)) > 1:
			visible_pid = next((pid for pid in pids if pid != 5), pids[0])
			boundary_faces.append(key)
			boundary_phys.append(-visible_pid)
	return np.asarray(boundary_faces, dtype=int), np.asarray(boundary_phys, dtype=int)

def _extract_boundary_edges(elems: np.ndarray, phys_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]: # calcul
	"""
	helper extraction arêtes bords

	param: elems: tableau des éléments du maillage
	param: phys_ids: tableau des ids physiques par élément
	return: tuple de tableaux (boundary_edges, boundary_phys) contenant arêtes de bord et ids
	"""
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

def _boundary_weights_2d(points: np.ndarray, boundary_edges: np.ndarray) -> np.ndarray: # calcul
	"""
	helper calcul poids contributions frontière 2D

	param: points: tableau des coordonnées des points du maillage
	param: boundary_edges: tableau des arêtes de bord
	return: tableau des poids de contribution aux conditions de bord pour chaque noeud
	"""
	weights = np.zeros(len(points), dtype=float)
	for i, j in boundary_edges:
		length = float(np.linalg.norm(points[int(i), :2] - points[int(j), :2]))
		weights[int(i)] += 0.5 * length
		weights[int(j)] += 0.5 * length
	return weights


def _boundary_weights_3d(points: np.ndarray, boundary_faces: np.ndarray) -> np.ndarray: # calcul
	"""
	helper calcul poids contributions frontière 3D

	param: points: tableau des coordonnées des points du maillage
	param: boundary_faces: tableau des faces de bord
	return: tableau des poids de contribution aux conditions de bord pour chaque noeud
	"""
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


def _build_boundary_condition_field(points: np.ndarray, boundary_entities: np.ndarray, dim: int, h_conv: float, t_ext: float ) -> BoundaryConditionField: # calcul
	"""
	Construit le champ de conditions de bord convectives

	param: points: tableau des coordonnées des points du maillage
	param: boundary_entities: tableau des entités de bord
	param: dim: dimension du maillage
	param: h_conv: coefficient de convection pour les conditions de bord convectives
	param: t_ext: température extérieure pour les conditions de bord convectives
	return: BoundaryConditionField 
	"""
	weights = _boundary_weights_2d(points, boundary_entities) if dim == 2 else _boundary_weights_3d(points, boundary_entities)
	h = np.zeros(len(points), dtype=float)
	external_temperature = np.full(len(points), float(t_ext), dtype=float)
	dofs = np.flatnonzero(weights > 0.0)
	h[dofs] = float(h_conv)

	diag_values = h * weights
	loss_matrix = diags(diag_values, offsets=0, shape=(len(points), len(points)), format="csr")
	rhs = diag_values * external_temperature
	return BoundaryConditionField( weights=weights, h=h, t_ext=external_temperature, dofs=dofs, loss_matrix=loss_matrix, rhs=rhs)


def _build_volume_loss_field(m_unit: csr_matrix, n_nodes: int, general_loss: float, vent_loss: float, radiation_loss: float, t_ext: float) -> VolumeLossField:
	"""
	Helper construisant le champ de perte de chaleur volumique

	param: m_unit: matrice de masse unitaire pour le maillage
	param: n_nodes: nombre de noeuds dans le maillage
	param: general_loss: coefficient de perte de chaleur générale (ex: diffusion vers l'extérieur)
	param: vent_loss: coefficient de perte de chaleur par ventilation (ex: diffusion sol->mur
	param: radiation_loss: coefficient de perte de chaleur par rayonnement
	param: t_ext: température extérieure pour les pertes de chaleur convectives et radiatives
	return: VolumeLossField
	"""
	linear_coeff = max(0.0, float(general_loss)) + max(0.0, float(vent_loss))
	ambient = np.full(n_nodes, float(t_ext), dtype=float)
	loss_matrix = (linear_coeff * m_unit).tocsr()
	rhs = linear_coeff * m_unit.dot(ambient)
	return VolumeLossField(linear_coeff=linear_coeff, radiation_coeff=max(0.0, float(radiation_loss)), t_ext=float(t_ext), loss_matrix=loss_matrix, rhs=np.asarray(rhs, dtype=float))


def _radiation_loss_rhs(m_unit: csr_matrix, local_t: np.ndarray, volume_loss: VolumeLossField) -> np.ndarray: # calcul
	"""
	helper calcul du terme source de perte de chaleur par rayonnement

	param: m_unit: matrice de masse unitaire pour le maillage
	param: local_t: tableau des températures locales par élément
	param: volume_loss: VolumeLossField contenant les informations nécessaires pour calculer la perte de chaleur
	return: tableau du terme source de perte de chaleur par rayonnement à appliquer au système
	"""
	if volume_loss.radiation_coeff <= 0.0:
		return np.zeros_like(local_t)
	radiation_t = np.clip(local_t, 0.0, 5000.0)
	external_t = max(0.0, min(5000.0, volume_loss.t_ext))
	power_density = volume_loss.radiation_coeff * (radiation_t**4 - external_t**4)
	return np.asarray(m_unit.dot(power_density), dtype=float)


def _heat_release_rate(material: dict[str, object], elem_temp: float, burn_age: float) -> float: # calcul
	"""
	Calcul la cheleur émise par un élément brulé 

	param: material: dictionnaire contenant les propriétés du matériau de l'élément
	param: elem_temp: température de l'élément
	param: burn_age: temps écoulé depuis que l'élément a commencé à brûler
	return: taux de libération de chaleur de l'élément
	"""
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


def _hrr_source_rhs(system: ElementwiseSystem, elems: np.ndarray, burned_elements: np.ndarray, local_t: np.ndarray, burn_times: np.ndarray,csim_time: float,vertical_transfer: VerticalHeatTransferField | None = None,vertical_attenuation: float = 0.25,horizontal_transfer: HorizontalAirTransferField | None = None,horizontal_attenuation: float = 0.25) -> np.ndarray: # calcul
	"""
	Calcul le terme source de perte de chaleur par libération de chaleur

	param: system: ElementwiseSystem contenant les matrices et champs à utiliser pour le calcul
	param: elems: tableau des éléments du maillage
	param: burned_elements: tableau booléen indiquant quels éléments sont brûlés
	param: local_t: tableau des températures locales par élément
	param: burn_times: tableau des temps de combustion par élément
	param: csim_time: temps actuel de la simulation
	param: vertical_transfer: VerticalHeatTransferField contenant les informations pour le transfert de chaleur vertical, ou None si désactivé
	param: vertical_attenuation: coefficient d'atténuation de la chaleur pour le transfert de chaleur vertical
	return: tableau du terme source de libération de chaleur
	"""
	burned_indices = np.flatnonzero(burned_elements)
	if len(burned_indices) == 0:
		return np.zeros(system.m_mat.shape[0], dtype=float)

	elem_temperatures = np.mean(local_t[elems[burned_indices]], axis=1)
	local_loads = np.zeros((len(burned_indices), elems.shape[1]), dtype=float)
	vertical_load_by_element: dict[int, np.ndarray] = {}
	horizontal_load_by_element: dict[int, np.ndarray] = {}
	for local_idx, elem_idx in enumerate(burned_indices):
		material = get_material(str(system.elem_material_names[int(elem_idx)]))
		burn_age = max(0.0, csim_time - float(burn_times[int(elem_idx)]))
		hrr = _heat_release_rate(material, float(elem_temperatures[local_idx]), burn_age)
		local_loads[local_idx] = hrr * system.unit_load_local[int(elem_idx)]

		if vertical_transfer is not None and vertical_transfer.enabled and hrr > 0.0:
			source_power = hrr * float(vertical_transfer.element_volumes[int(elem_idx)])
			target_indices = vertical_transfer.targets[int(elem_idx)]
			target_dz = vertical_transfer.dz[int(elem_idx)]
			factors = np.asarray(
				[max(0.0, 1.0 - max(0.0, float(vertical_attenuation)) * float(dz)) for dz in target_dz],
				dtype=float,
			)
			total_factor = float(np.sum(factors))
			if total_factor > 0.0:
				# Evite de dupliquer la puissance quand plusieurs cellules au-dessus sont ciblees.
				vertical_power = source_power * min(1.0, total_factor)
				for target_idx, factor in zip(target_indices, factors, strict=False):
					if factor <= 0.0:
						continue
					target_volume = max(float(vertical_transfer.element_volumes[int(target_idx)]), 1e-12)
					target_power = vertical_power * factor / total_factor
					target_load = target_power * system.unit_load_local[int(target_idx)] / target_volume
					if int(target_idx) in vertical_load_by_element:
						vertical_load_by_element[int(target_idx)] += target_load
					else:
						vertical_load_by_element[int(target_idx)] = target_load.copy()

		if horizontal_transfer is not None and horizontal_transfer.enabled and hrr > 0.0 and horizontal_transfer.power_fraction > 0.0:
			source_power = hrr * float(horizontal_transfer.element_volumes[int(elem_idx)])
			target_indices = horizontal_transfer.targets[int(elem_idx)]
			target_distances = horizontal_transfer.distances[int(elem_idx)]
			if len(target_indices) == 0:
				continue
			unburned_mask = ~burned_elements[target_indices]
			target_indices = target_indices[unburned_mask]
			target_distances = target_distances[unburned_mask]
			factors = np.asarray(
				[max(0.0, 1.0 - max(0.0, float(horizontal_attenuation)) * float(distance)) for distance in target_distances],
				dtype=float,
			)
			total_factor = float(np.sum(factors))
			if total_factor <= 0.0:
				continue
			# Les mouvements d'air redistribuent une part plafonnee de la puissance sans la dupliquer.
			horizontal_power = source_power * min(horizontal_transfer.power_fraction, total_factor)
			for target_idx, factor in zip(target_indices, factors, strict=False):
				if factor <= 0.0:
					continue
				target_volume = max(float(horizontal_transfer.element_volumes[int(target_idx)]), 1e-12)
				target_power = horizontal_power * factor / total_factor
				target_load = target_power * system.unit_load_local[int(target_idx)] / target_volume
				if int(target_idx) in horizontal_load_by_element:
					horizontal_load_by_element[int(target_idx)] += target_load
				else:
					horizontal_load_by_element[int(target_idx)] = target_load.copy()

	rhs = _assemble_vector_from_local(elems[burned_indices], local_loads, system.m_mat.shape[0])
	if vertical_load_by_element:
		target_indices = np.asarray(list(vertical_load_by_element.keys()), dtype=int)
		target_loads = np.asarray([vertical_load_by_element[int(idx)] for idx in target_indices], dtype=float)
		rhs += _assemble_vector_from_local(elems[target_indices], target_loads, system.m_mat.shape[0])
	if horizontal_load_by_element:
		target_indices = np.asarray(list(horizontal_load_by_element.keys()), dtype=int)
		target_loads = np.asarray([horizontal_load_by_element[int(idx)] for idx in target_indices], dtype=float)
		rhs += _assemble_vector_from_local(elems[target_indices], target_loads, system.m_mat.shape[0])
	return rhs


def _update_burned_elements(burned_elements: np.ndarray, elems: np.ndarray, local_t: np.ndarray, elem_tc: np.ndarray, active_elements: np.ndarray | None = None ) -> np.ndarray: # calcul
	"""
	Update les éléments brulés

	param: burned_elements: tableau booléen indiquant quels éléments sont brûlés
	param: elems: tableau des éléments du maillage
	param: local_t: tableau des températures locales par élément
	param: elem_tc: tableau des températures de combustion par élément
	param: active_elements: tableau booléen indiquant quels éléments sont actuellement actifs, ou None pour ignorer cette condition
	return: tableau des indices des éléments qui viennent de brûler 
	"""
	candidates = ~burned_elements
	if active_elements is not None:
		candidates &= active_elements
	if not np.any(candidates):
		return np.array([], dtype=int)

	candidate_indices = np.flatnonzero(candidates)
	elem_temperatures = np.mean(local_t[elems[candidate_indices]], axis=1)
	newly_burned = candidate_indices[elem_temperatures >= elem_tc[candidate_indices]]
	burned_elements[newly_burned] = True
	return newly_burned


def _update_element_activity(active_elements: np.ndarray, idle_steps: np.ndarray, burned_elements: np.ndarray, elems: np.ndarray, local_t: np.ndarray, elem_tc: np.ndarray, freeze_steps: int, temp_margin: float) -> tuple[int, int]: # opti
	"""
	Update les éléments actifs en fonction de leur température et de leur temps d'inactivité

	param: active_elements: tableau booléen indiquant quels éléments sont actuellement actifs
	param: idle_steps: tableau des nombres de pas d'inactivité par élément
	param: burned_elements: tableau booléen indiquant quels éléments sont brûlés
	param: elems: tableau des éléments du maillage
	param: local_t: tableau des températures locales par élément
	param: elem_tc: tableau des températures de combustion par élément
	param: freeze_steps: nombre de pas d'inactivité avant de geler un élément
	param: temp_margin: marge de température pour considérer un élément comme froid
	return: tuple (n_checked, n_frozen) indiquant le nombre d'éléments vérifiés et le nombre d'éléments qui viennent d'être gelés
	"""
	if freeze_steps <= 0:
		active_elements[:] = True
		idle_steps[:] = 0
		return len(active_elements), 0

	checkable = active_elements & (~burned_elements)
	if not np.any(checkable):
		return 0, int(np.count_nonzero(~active_elements))

	check_indices = np.flatnonzero(checkable)
	elem_temperatures = np.mean(local_t[elems[check_indices]], axis=1)
	cold_stable = elem_temperatures < (elem_tc[check_indices] - float(temp_margin))
	idle_steps[check_indices[cold_stable]] += 1
	idle_steps[check_indices[~cold_stable]] = 0

	to_freeze = check_indices[cold_stable & (idle_steps[check_indices] >= freeze_steps)]
	active_elements[to_freeze] = False
	return len(check_indices), int(len(to_freeze))

def _reactivate_near_hot_nodes(active_elements: np.ndarray, idle_steps: np.ndarray, burned_elements: np.ndarray, elems: np.ndarray, local_t: np.ndarray, elem_tc: np.ndarray,temp_margin: float) -> int: #opti
	"""
	Update les éléments inactifs en les réactivant s'ils sont proches d'éléments chauds

	param: active_elements: tableau booléen indiquant quels éléments sont actuellement actifs
	param: idle_steps: tableau des nombres de pas d'inactivité par élément
	param: burned_elements: tableau booléen indiquant quels éléments sont brûlés
	param: elems: tableau des éléments du maillage
	param: local_t: tableau des températures locales par élément
	param: elem_tc: tableau des températures de combustion par élément
	param: temp_margin: marge de température pour considérer un élément comme chaud
	return: nombre d'éléments qui viennent d'être réactivés
	"""
	inactive = (~active_elements) & (~burned_elements) # Seuls les éléments inactifs et non brûlés peuvent être réactivés
	if not np.any(inactive):
		return 0

	inactive_indices = np.flatnonzero(inactive)
	max_node_t = np.max(local_t[elems[inactive_indices]], axis=1)
	to_reactivate = inactive_indices[max_node_t >= (elem_tc[inactive_indices] - float(temp_margin))]
	active_elements[to_reactivate] = True
	idle_steps[to_reactivate] = 0
	return int(len(to_reactivate))

def _build_node_neighbors(elems: np.ndarray, n_nodes: int) -> list[np.ndarray]: #opti
	"""
	Helper construisant la liste des voisins de chaque noeud à partir des éléments du maillage

	param: elems: tableau des éléments du maillage
	param: n_nodes: nombre de noeuds dans le maillage
	return: liste de tableaux contenant les indices des noeuds voisins pour chaque noeud
	"""
	neighbors: list[set[int]] = [set() for _ in range(n_nodes)]
	for element in elems:
		nodes = [int(v) for v in element]
		for node in nodes:
			neighbors[node].update(other for other in nodes if other != node)
	return [np.asarray(sorted(local_neighbors), dtype=int) for local_neighbors in neighbors]

def _thaw_frozen_nodes(frozen_nodes: np.ndarray, node_idle_steps: np.ndarray, node_neighbors: list[np.ndarray], local_t: np.ndarray, node_tc: np.ndarray, thaw_delta: float, thaw_tc_margin: float) -> int: #opti
	"""
	helper réactivant les noeuds gelés 

	param: frozen_nodes: tableau booléen indiquant quels noeuds sont gelés
	param: node_idle_steps: tableau des nombres de pas d'inactivité par noeud
	param: node_neighbors: liste de tableaux contenant les indices des noeuds voisins pour chaque noeud
	param: local_t: tableau des températures locales par noeud
	param: node_tc: tableau des températures de combustion par noeud
	param: thaw_delta: seuil de différence de température avec les voisins pour réactiver un noeud gelé
	param: thaw_tc_margin: marge de température pour considérer un voisin comme chaud pour réactiver un noeud gelé
	return: nombre de noeuds qui viennent d'être réactivés
	"""
	frozen_indices = np.flatnonzero(frozen_nodes)
	if len(frozen_indices) == 0:
		return 0

	to_thaw: list[int] = []
	for node_idx in frozen_indices:
		neighbors = node_neighbors[int(node_idx)]
		if len(neighbors) == 0:
			continue
		neighbor_delta = float(np.max(np.abs(local_t[neighbors] - local_t[int(node_idx)])))
		neighbor_hot = bool(np.any(local_t[neighbors] >= node_tc[neighbors] - float(thaw_tc_margin)))
		if neighbor_delta >= thaw_delta or neighbor_hot:
			to_thaw.append(int(node_idx))

	if not to_thaw:
		return 0
	thaw_indices = np.asarray(to_thaw, dtype=int)
	frozen_nodes[thaw_indices] = False
	node_idle_steps[thaw_indices] = 0
	return int(len(thaw_indices))

def _update_frozen_nodes(frozen_nodes: np.ndarray, node_idle_steps: np.ndarray, old_t: np.ndarray, new_t: np.ndarray, node_tc: np.ndarray, freeze_steps: int, freeze_delta: float, freeze_tc_margin: float, max_frozen_fraction: float ) -> tuple[int, int]: #opti
	"""
	helper mettant à jour les noeuds gelés en fonction de leur température, de leur temps d'inactivité et de la température de combustion

	param: frozen_nodes: tableau booléen indiquant quels noeuds sont gelés
	param: node_idle_steps: tableau des nombres de pas d'inactivité par noeud
	param: old_t: tableau des températures locales par noeud à l'étape précédente
	param: new_t: tableau des températures locales par noeud à l'étape actuelle
	param: node_tc: tableau des températures de combustion par noeud
	param: freeze_steps: nombre de pas d'inactivité avant de geler un noeud
	param: freeze_delta: seuil de différence de température avec l'étape précédente pour considérer un noeud comme stable
	param: freeze_tc_margin: marge de température pour considérer un noeud comme froid pour le gel
	param: max_frozen_fraction: fraction maximale de noeuds qui peuvent être gelés en même temps
	return: tuple (n_checked, n_frozen) indiquant le nombre de noeuds vérifiés et le nombre de noeuds qui viennent d'être gelés
	"""
	if freeze_steps <= 0:
		frozen_count = int(np.count_nonzero(frozen_nodes))
		frozen_nodes[:] = False
		node_idle_steps[:] = 0
		return 0, -frozen_count

	candidates = ~frozen_nodes
	if not np.any(candidates):
		return 0, 0

	candidate_indices = np.flatnonzero(candidates)
	stable = np.abs(new_t[candidate_indices] - old_t[candidate_indices]) <= float(freeze_delta)
	cool = new_t[candidate_indices] < (node_tc[candidate_indices] - float(freeze_tc_margin))
	steady = stable & cool
	node_idle_steps[candidate_indices[steady]] += 1
	node_idle_steps[candidate_indices[~steady]] = 0

	to_freeze = candidate_indices[steady & (node_idle_steps[candidate_indices] >= freeze_steps)]
	max_frozen = int(max(0.0, min(1.0, float(max_frozen_fraction))) * len(frozen_nodes))
	available = max(0, max_frozen - int(np.count_nonzero(frozen_nodes)))
	if available <= 0:
		return len(candidate_indices), 0
	if len(to_freeze) > available:
		to_freeze = to_freeze[:available]

	frozen_nodes[to_freeze] = True
	node_idle_steps[to_freeze] = 0
	return len(candidate_indices), int(len(to_freeze))

def _burned_triangle_vertices(points: np.ndarray, elems: np.ndarray, burned_elements: np.ndarray) -> list[np.ndarray]: # visuel
	"""
	Helper construisant la liste des coordonnées des triangles brûlés

	param: points: tableau des coordonnées des points du maillage
	param: elems: tableau des éléments du maillage
	param: burned_elements: tableau booléen indiquant quels éléments sont brûlés
	return: liste de tableaux contenant les coordonnées des sommets de chaque triangle brûlé
	"""
	return [points[tri, :2] for tri in elems[burned_elements]]

def _burned_boundary_face_mask(boundary_faces: np.ndarray, elems: np.ndarray, burned_elements: np.ndarray) -> np.ndarray: # visuel
	"""
	Helper construisant un masque indiquant quelles faces de bord sont brûlées
	"""
	burned_faces: set[tuple[int, int, int]] = set()
	for tet in elems[burned_elements]:
		i, j, k, l = [int(v) for v in tet]
		burned_faces.add(tuple(sorted((i, j, k))))
		burned_faces.add(tuple(sorted((i, j, l))))
		burned_faces.add(tuple(sorted((i, k, l))))
		burned_faces.add(tuple(sorted((j, k, l))))
	return np.asarray([tuple(sorted(int(v) for v in face)) in burned_faces for face in boundary_faces], dtype=bool)

def _burned_boundary_face_vertices(points: np.ndarray, boundary_faces: np.ndarray, elems: np.ndarray, burned_elements: np.ndarray) -> list[np.ndarray]: # visuel
	"""
	Helper construisant la liste des coordonnées des faces frontières brûlées
	"""
	mask = _burned_boundary_face_mask(boundary_faces, elems, burned_elements)
	return [points[face] for face in boundary_faces[mask]]

def _build_region_legend_handles(regions: list[dict[str, object]]) -> list[Patch]: # visuel
	"""
	Helper les legendes des régions
	"""
	legend_handles: list[Patch] = []
	for region in regions:
		legend_handles.append(Patch( facecolor=str(region["color"]), edgecolor="black" if bool(region["solid_fill"]) else str(region["color"]), alpha=0.45 if bool(region["solid_fill"]) else 0.30, label=str(region["legend_label"])))
	return legend_handles

def _add_region_overlays_2d(ax, points: np.ndarray, elems: np.ndarray, phys_ids: np.ndarray, regions: list[dict[str, object]]): # visuel
	"""
	Helper ajoutant les superpositions de régions sur une visualisation 2D
	"""
	for region in regions:
		mask = phys_ids == int(region["pid"])
		if not np.any(mask):
			continue
		verts = [points[tri, :2] for tri in elems[mask]]
		edgecolor = "black" if bool(region["solid_fill"]) else str(region["color"])
		alpha = 1.0 if bool(region["solid_fill"]) else float(region["overlay_alpha"])
		linewidth = float(region["edge_width"]) if bool(region["solid_fill"]) else 0.35
		zorder = 4 if bool(region["solid_fill"]) else 3
		ax.add_collection(PolyCollection(verts, facecolors=str(region["color"]), edgecolors=edgecolor, linewidths=linewidth, alpha=alpha, zorder=zorder))

def _add_region_overlays_3d(ax, points: np.ndarray, boundary_faces: np.ndarray, boundary_phys: np.ndarray, regions: list[dict[str, object]],include_solid_fill: bool): # visuel
	"""
	Helper ajoutant les superpositions de régions sur une visualisation 3D
	"""
	collections = []
	for region in regions:
		mask = np.abs(boundary_phys) == int(region["pid"])
		if not np.any(mask):
			continue
		face_vertices = [points[face] for face in boundary_faces[mask]]
		has_interface_faces = bool(np.any(boundary_phys[mask] < 0))
		facecolor = "#050505" if has_interface_faces else str(region["color"])
		edgecolor = "black" if has_interface_faces else str(region["color"])
		alpha = 1.0 if has_interface_faces else float(region["overlay_alpha_3d"])
		linewidth = 0.55 if has_interface_faces else 0.15
		if include_solid_fill and bool(region["solid_fill"]):
			poly = Poly3DCollection(face_vertices, facecolors=facecolor, edgecolors="black", linewidths=max(linewidth, float(region["edge_width"]) * 0.4), alpha=alpha, zorder=5 if has_interface_faces else 3)
		else:
			poly = Poly3DCollection(face_vertices, facecolors=facecolor, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha,zorder=5 if has_interface_faces else 1,)
		ax.add_collection3d(poly)
		collections.append(poly)
	return collections

def _add_region_edges_3d(ax, points: np.ndarray, boundary_faces: np.ndarray, boundary_phys: np.ndarray, regions: list[dict[str, object]]): #visuel 
	"""
	helper ajoutant les arêtes des régions sur une visualisation 3D
	"""
	collections = []
	for region in regions:
		mask = np.abs(boundary_phys) == int(region["pid"])
		if not np.any(mask):
			continue
		has_interface_edges = bool(np.any(boundary_phys[mask] < 0))
		if not has_interface_edges and not bool(region["solid_fill"]):
			continue
		region_edges = _build_boundary_edges(boundary_faces[mask])
		segments = [[points[i], points[j]] for i, j in region_edges]
		collection = Line3DCollection(segments, colors="black", linewidths=1.1 if has_interface_edges else max(0.55, float(region["edge_width"])), alpha=1.0 if has_interface_edges else 0.95)
		ax.add_collection3d(collection)
		collections.append(collection)
	return collections

def _build_boundary_edges(boundary_faces: np.ndarray) -> np.ndarray: # visuel
	"""
	Helper construisant la liste des arêtes de bord à partir des faces de bord
	"""
	edge_set: set[tuple[int, int]] = set()
	for face in boundary_faces:
		i, j, k = [int(v) for v in face]
		edge_set.add(tuple(sorted((i, j))))
		edge_set.add(tuple(sorted((i, k))))
		edge_set.add(tuple(sorted((j, k))))
	return np.asarray(sorted(edge_set), dtype=int)

def _add_source_marker_2d(ax, src_x: float, src_y: float, src_radius: float, src_temp: float): # visuel
	"""
	Helper ajoutant un marqueur pour la source de chaleur sur une visualisation 2D
	"""
	marker = ax.scatter([src_x], [src_y], marker="X", s=115, c="#ff2d00", edgecolors="black", linewidths=0.9, zorder=8, label="Source: "+str(src_temp)+" K")
	radius_patch = None
	if src_radius > 0.0:
		radius_patch = Circle( (src_x, src_y), src_radius, fill=False, edgecolor="#ff2d00", linewidth=1.4, alpha=0.85, zorder=7)
		ax.add_patch(radius_patch)
	return marker, radius_patch

def _add_source_marker_3d(ax, src_x: float, src_y: float, src_z: float, src_temp: float, points: np.ndarray | None = None): #visuel
	"""
	Helper ajoutant un marqueur pour la source de chaleur sur une visualisation 3D
	"""
	marker = ax.scatter([src_x], [src_y], [src_z], marker="X", s=220, c="#ff2d00", edgecolors="white", linewidths=1.8, depthshade=False, zorder=20, label="Source: "+str(src_temp)+" K")
	if points is not None and len(points):
		mins = np.min(points, axis=0)
		maxs = np.max(points, axis=0)
		segments = [
			[(mins[0], src_y, src_z), (maxs[0], src_y, src_z)],
			[(src_x, mins[1], src_z), (src_x, maxs[1], src_z)],
			[(src_x, src_y, mins[2]), (src_x, src_y, maxs[2])],
		]
		beacon = Line3DCollection(segments, colors="#ff2d00", linewidths=2.4, alpha=1.0, zorder=19)
		ax.add_collection3d(beacon)
	else:
		beacon = None
	return marker, beacon


def _plot_3d_mesh_preview(points: np.ndarray, boundary_faces: np.ndarray, title: str): # visuel
	"""
	Affiche le maillage 3D de l'environnement
	"""
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


def _plot_3d_filled_preview(points: np.ndarray, boundary_faces: np.ndarray, boundary_phys: np.ndarray, regions: list[dict[str, object]], title: str): # visuel
	"""
	Affiche le maillage 3D de l'environnement avec les régions remplies
	"""
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


def _set_equal_3d_axes(ax, points: np.ndarray) -> None: # visuel
	"""
	Gère l'affichage des axes
	"""
	mins = np.min(points, axis=0)
	maxs = np.max(points, axis=0)
	center = 0.5 * (mins + maxs)
	radius = 0.5 * float(np.max(maxs - mins))
	if radius <= 0.0:
		radius = 1.0
	ax.set_xlim(center[0] - radius, center[0] + radius)
	ax.set_ylim(center[1] - radius, center[1] + radius)
	ax.set_zlim(center[2] - radius, center[2] + radius)


SCENARIO_KEYS = {
	"dt",
	"steps",
	"sub_steps",
	"theta",
	"h_conv",
	"general_loss",
	"vent_loss",
	"radiation_loss",
	"vertical_air_transfer",
	"vertical_air_attenuation",
	"vertical_air_radius",
	"vertical_air_random_delta",
	"horizontal_air_transfer",
	"horizontal_air_attenuation",
	"horizontal_air_radius",
	"horizontal_air_power_fraction",
	"horizontal_air_random_delta",
	"structural_fun",
	"structural_fun_radius",
	"structural_fun_load_factor",
	"t_amb",
	"src_temp",
	"src_x",
	"src_y",
	"src_z",
	"src_radius",
	"source_time",
}


def _default_scenario_path(dim: int) -> Path:
	return PROJECT_ROOT / "models" / f"default_{int(dim)}d.txt"


def _scenario_defaults(dim: int, allowed_keys: set[str] | None = None) -> tuple[dict[str, float | int], Path | None]: # main
	"""
	Charge les parametres depuis models/default_2d.txt ou models/default_3d.txt.
	"""
	defaults: dict[str, float | int] = {}
	allowed = allowed_keys if allowed_keys is not None else set(SCENARIO_KEYS)
	default_path = _default_scenario_path(dim)
	if not default_path.exists():
		raise FileNotFoundError(f"Scenario par defaut introuvable: {default_path}")
	defaults.update(_read_scenario_file(default_path, allowed))
	return defaults, default_path


def _parse_scenario_value(raw_value: str) -> float | int:
	value = raw_value.split("#", 1)[0].strip()
	if not value:
		raise ValueError("valeur de scenario vide")
	lower = value.lower()
	if lower in {"true", "yes", "on"}:
		return 1
	if lower in {"false", "no", "off"}:
		return 0
	try:
		if any(marker in value for marker in (".", "e", "E")):
			return float(value)
		return int(value)
	except ValueError:
		return float(value)


def _read_scenario_file(scenario_path: Path, allowed_keys: set[str]) -> dict[str, float | int]:
	values: dict[str, float | int] = {}
	for lineno, raw_line in enumerate(scenario_path.read_text(encoding="utf-8").splitlines(), start=1):
		line = raw_line.strip()
		if not line or line.startswith("#"):
			continue
		separator = "=" if "=" in line else ":" if ":" in line else None
		if separator is None:
			raise ValueError(f"{scenario_path}:{lineno}: utilisez key=value")
		key, raw_value = line.split(separator, 1)
		key = key.strip().replace("-", "_")
		if key not in allowed_keys:
			raise ValueError(f"{scenario_path}:{lineno}: parametre inconnu '{key}'")
		values[key] = _parse_scenario_value(raw_value)
	return values


def _scenario_txt_candidates(mesh_path: Path, dim: int) -> list[Path]:
	candidates = [mesh_path.with_suffix(".txt")]
	for scenario_path in (
		PROJECT_ROOT / "models" / Path(mesh_path.name).with_suffix(".txt"),
		PROJECT_ROOT / "models" / "perf" / Path(mesh_path.name).with_suffix(".txt"),
		_default_scenario_path(dim),
	):
		if scenario_path not in candidates:
			candidates.append(scenario_path)
	return candidates


def _load_scenario_defaults(dim: int, mesh_path: Path) -> tuple[dict[str, float | int], Path | None]: # main
	"""
	loed les parametres de simulation depuis un fichier txt associé au maillage, ou depuis le scenario par defaut si aucun fichier n'est trouvé
	"""
	allowed_keys = set(SCENARIO_KEYS)
	defaults, default_path = _scenario_defaults(dim, allowed_keys)
	for scenario_path in _scenario_txt_candidates(mesh_path, dim):
		if scenario_path.exists():
			if scenario_path == default_path:
				return defaults, default_path
			file_values = _read_scenario_file(scenario_path, allowed_keys)
			defaults.update(file_values)
			return defaults, scenario_path
	return defaults, None


def _resolve_mesh_path(mesh_file: str) -> Path: # main
	"""
	Gère les chemins d'accès au txt du maillage
	"""
	mesh_path = Path(mesh_file)
	if mesh_path.is_absolute() and mesh_path.exists():
		return mesh_path

	candidates = [
		PROJECT_ROOT / mesh_path,
		PROJECT_ROOT / "models" / mesh_path,
		PROJECT_ROOT / "models" / mesh_path.name,
		PROJECT_ROOT / "models" / "perf" / mesh_path.name,
		mesh_path,
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	return PROJECT_ROOT / "models" / mesh_path


def _resolve_save_targets(save_arg: str) -> tuple[Path, Path, Path, Path]: # main
	"""
	Gère les chemins de sauvegarde
	"""
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


def _write_timings_csv(csv_path: Path, timing_rows: list[dict[str, object]]) -> None: # main
	"""
	Sauve les données de performance temporel
	"""
	fieldnames = ["phase", "seconds", "frame", "details"]
	with csv_path.open("w", newline="", encoding="utf-8") as fh:
		writer = csv.DictWriter(fh, fieldnames=fieldnames)
		writer.writeheader()
		for row in timing_rows:
			writer.writerow(row)


def run(args: argparse.Namespace): # main
	"""
	Gère le lancement de la simulation
	"""
	if not args.plot:
		plt.switch_backend("Agg")

	dim = int(args.dim)
	mesh_file = args.mesh or ("piece.msh" if dim == 2 else "immeuble.msh")
	mesh_path = _resolve_mesh_path(mesh_file)
	defaults, scenario_path = _load_scenario_defaults(dim, mesh_path)
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
	horizontal_air_transfer = bool(args.horizontal_air_transfer if args.horizontal_air_transfer is not None else defaults["horizontal_air_transfer"])
	horizontal_air_attenuation = float(args.horizontal_air_attenuation if args.horizontal_air_attenuation is not None else defaults["horizontal_air_attenuation"])
	horizontal_air_radius = float(args.horizontal_air_radius if args.horizontal_air_radius is not None else defaults["horizontal_air_radius"])
	horizontal_air_power_fraction = max(0.0, float(args.horizontal_air_power_fraction if args.horizontal_air_power_fraction is not None else defaults["horizontal_air_power_fraction"]))
	horizontal_air_random_delta = max(0.0, float(args.horizontal_air_random_delta if args.horizontal_air_random_delta is not None else defaults["horizontal_air_random_delta"]))
	structural_fun_enabled = bool(args.structural_fun if args.structural_fun is not None else defaults["structural_fun"])
	structural_fun_radius = max(0.0, float(args.structural_fun_radius if args.structural_fun_radius is not None else defaults["structural_fun_radius"]))
	structural_fun_load_factor = max(0.0, float(args.structural_fun_load_factor if args.structural_fun_load_factor is not None else defaults["structural_fun_load_factor"]))
	t_amb = float(args.t_amb if args.t_amb is not None else defaults["t_amb"])
	src_temp = float(args.src_temp if args.src_temp is not None else defaults["src_temp"])
	src_x = float(args.src_x if args.src_x is not None else defaults["src_x"])
	src_y = float(args.src_y if args.src_y is not None else defaults["src_y"])
	src_z = float(args.src_z if args.src_z is not None else defaults["src_z"])
	src_radius = float(args.src_radius if args.src_radius is not None else defaults["src_radius"])
	source_time = max(0.0, float(args.source_time if args.source_time is not None else defaults.get("source_time", 0.0)))
	element_freeze_steps = max(0, int(args.element_freeze_steps))
	element_freeze_margin = max(0.0, float(args.element_freeze_margin))
	node_freeze_steps = max(0, int(args.node_freeze_steps))
	node_freeze_delta = max(0.0, float(args.node_freeze_delta))
	node_freeze_margin = max(0.0, float(args.node_freeze_margin))
	node_thaw_delta = max(0.0, float(args.node_thaw_delta))
	node_thaw_margin = max(0.0, float(args.node_thaw_margin))
	max_frozen_node_fraction = max(0.0, min(0.98, float(args.max_frozen_node_fraction)))
	show_burned_elements = not bool(args.hide_burned_elements)
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

	print(f"Using mesh: {mesh_path}")
	print(f"Using scenario: {scenario_path if scenario_path is not None else 'built-in fallback'}")

	t0 = time.perf_counter()
	pts, elems, phys, phys_name_map, dim = load_mesh_data(mesh_path, dim)
	record_timing("mesh_load", time.perf_counter() - t0, details=f"{mesh_path};dim={dim}")
	print(f"Maillage charge: {len(pts)} noeuds, {len(elems)} elements ({dim}D).")

	elem_tc = _element_tc_values(phys, phys_name_map)
	burned_elements = np.zeros(len(elems), dtype=bool)
	burn_times = np.full(len(elems), np.inf, dtype=float)
	active_elements = np.ones(len(elems), dtype=bool)
	element_idle_steps = np.zeros(len(elems), dtype=int)
	frozen_nodes = np.zeros(len(pts), dtype=bool)
	node_idle_steps = np.zeros(len(pts), dtype=int)
	node_neighbors = _build_node_neighbors(elems, len(pts))
	t0 = time.perf_counter()
	system = _assemble_elementwise_system(pts, elems, phys, phys_name_map, dim)
	record_timing("system_assembly", time.perf_counter() - t0, details=f"nodes={len(pts)};elements={len(elems)};dim={dim}")
	t0 = time.perf_counter()
	structural_state = prepare_structural_fun(
		pts,
		elems,
		np.sum(system.unit_load_local, axis=1),
		bool(structural_fun_enabled and dim == 3),
		structural_fun_radius,
		structural_fun_load_factor,
	)
	record_timing(
		"structural_fun_prepare",
		time.perf_counter() - t0,
		details=f"enabled={structural_state.enabled};radius={structural_fun_radius};load_factor={structural_fun_load_factor}",
	)
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
	horizontal_transfer = _build_horizontal_air_transfer_field(
		pts,
		elems,
		system.unit_load_local,
		dim,
		horizontal_air_transfer,
		horizontal_air_attenuation,
		horizontal_air_radius,
		horizontal_air_power_fraction,
	)
	record_timing(
		"horizontal_air_transfer_prepare",
		time.perf_counter() - t0,
		details=f"enabled={horizontal_transfer.enabled};attenuation={horizontal_air_attenuation};radius={horizontal_air_radius};power_fraction={horizontal_air_power_fraction}",
	)

	t0 = time.perf_counter()
	t = np.full(len(pts), t_amb, dtype=float)
	if dim == 2:
		dist = np.hypot(pts[:, 0] - src_x, pts[:, 1] - src_y)
	else:
		dist = np.sqrt((pts[:, 0] - src_x) ** 2 + (pts[:, 1] - src_y) ** 2 + (pts[:, 2] - src_z) ** 2)
	source_nodes = dist <= src_radius
	t[source_nodes] = src_temp
	record_timing(
		"initial_conditions",
		time.perf_counter() - t0,
		details=f"src_x={src_x};src_y={src_y};src_z={src_z};src_radius={src_radius};src_temp={src_temp};source_time={source_time}",
	)
	initial_burned_indices = _update_burned_elements(burned_elements, elems, t, elem_tc, active_elements)
	if len(initial_burned_indices):
		burn_times[initial_burned_indices] = 0.0
		_apply_burn_deltas(system, elems, initial_burned_indices)
		record_timing("element_burn_initial", 0.0, details=f"changed_elements={len(initial_burned_indices)}")
		active_elements[initial_burned_indices] = False
		element_idle_steps[initial_burned_indices] = 0
	if structural_state.enabled:
		initial_broken_indices = update_structural_fun(structural_state, system.elem_material_names, t, elems, burned_elements)
		if len(initial_broken_indices):
			record_timing("structural_fun_break_initial", 0.0, details=f"broken_elements={len(initial_broken_indices)}")

	empty_dofs = np.array([], dtype=int)
	empty_vals = np.array([], dtype=float)
	free_dofs = precompute_dirichlet_dofs(len(pts), empty_dofs)
	boundary_faces = boundary_phys = visual_faces = visual_phys = boundary_edges = boundary_edge_phys = None
	visual_regions = _build_visual_regions(phys, phys_name_map)
	if dim == 3:
		boundary_faces, boundary_phys = _extract_boundary_faces(elems, phys)
		visual_faces, visual_phys = _extract_boundary_faces(elems, phys, include_material_interfaces=True)
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

	def source_visible(sim_time: float) -> bool:
		return sim_time <= source_time if source_time > 0.0 else sim_time <= 0.0

	def _setup_axes(local_t: np.ndarray, title: str):
		if dim == 2:
			fig, ax = plt.subplots(figsize=(10, 8))
			im = ax.tripcolor(pts[:, 0], pts[:, 1], elems, local_t, cmap=THERMAL_CMAP, shading="gouraud", vmin=t_amb, vmax=1500, alpha=0.86)
			_add_region_overlays_2d(ax, pts, elems, phys, visual_regions)
			burned_collection = PolyCollection(
				_burned_triangle_vertices(pts, elems, burned_elements) if show_burned_elements else [],
				facecolors="black",
				edgecolors="black",
				linewidths=0.45,
				alpha=0.28 if show_burned_elements else 0.0,
				zorder=9,
			)
			ax.add_collection(burned_collection)
			source_marker, source_radius_patch = _add_source_marker_2d(ax, src_x, src_y, src_radius, src_temp)
			plt.colorbar(im, ax=ax, label="Temperature [K]")
			ax.set_facecolor("#1A12BD")
			ax.set_aspect("equal")
			ax.set_title(title)
			legend_handles = _build_region_legend_handles(visual_regions)
			legend_handles.append(source_marker)
			if show_burned_elements:
				legend_handles.append(Patch(facecolor="black", edgecolor="black", alpha=0.28, label="Brule"))
			if legend_handles:
				ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, title="Objets / Materiaux")
			return fig, ax, {"field": im, "burned_collection": burned_collection, "source_marker": source_marker, "source_radius_patch": source_radius_patch}

		fig = plt.figure(figsize=(16, 8))
		ax_mesh = fig.add_subplot(121, projection="3d")
		ax_full = fig.add_subplot(122, projection="3d")

		edges = _build_boundary_edges(visual_faces)
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

		_add_region_overlays_3d(ax_full, pts, visual_faces, visual_phys, visual_regions, include_solid_fill=False)
		face_temps = np.mean(local_t[visual_faces], axis=1)
		norm = Normalize(vmin=THERMAL_3D_VMIN, vmax=THERMAL_3D_VMAX)
		face_colors = THERMAL_CMAP_3D(norm(face_temps))
		if show_burned_elements:
			burned_boundary_mask = _burned_boundary_face_mask(visual_faces, elems, burned_elements)
			face_colors[burned_boundary_mask] = (0.0, 0.0, 0.0, 1.0)
		full_surface = Poly3DCollection(
			[pts[face] for face in visual_faces],
			facecolors=face_colors,
			edgecolors="none",
			linewidths=0.0,
			alpha=0.56,
		)
		ax_full.add_collection3d(full_surface)
		region_fills = _add_region_overlays_3d(ax_full, pts, visual_faces, visual_phys, visual_regions, include_solid_fill=True)
		region_edges = _add_region_edges_3d(ax_full, pts, visual_faces, visual_phys, visual_regions)
		burned_tetra_surface = Poly3DCollection(
			_burned_boundary_face_vertices(pts, visual_faces, elems, burned_elements) if show_burned_elements else [],
			facecolors=(0.0, 0.0, 0.0, 1.0),
			edgecolors=(0.0, 0.0, 0.0, 1.0),
			linewidths=0.12,
			alpha=0.9 if show_burned_elements else 0.0,
			zorder=9,
		)
		ax_full.add_collection3d(burned_tetra_surface)
		source_marker_mesh, source_beacon_mesh = _add_source_marker_3d(ax_mesh, src_x, src_y, src_z, src_temp, pts)
		source_marker_full, source_beacon_full = _add_source_marker_3d(ax_full, src_x, src_y, src_z, src_temp, pts)
		broken_points = structural_state.centroids[structural_state.broken] if structural_state.enabled else np.empty((0, 3))
		broken_marker_mesh = ax_mesh.scatter(
			broken_points[:, 0] if len(broken_points) else [],
			broken_points[:, 1] if len(broken_points) else [],
			broken_points[:, 2] if len(broken_points) else [],
			marker="x",
			s=95,
			c="#ffd400",
			linewidths=2.2,
			depthshade=False,
			zorder=10,
			label="Rupture structurelle",
		)
		broken_marker_full = ax_full.scatter(
			broken_points[:, 0] if len(broken_points) else [],
			broken_points[:, 1] if len(broken_points) else [],
			broken_points[:, 2] if len(broken_points) else [],
			marker="x",
			s=120,
			c="#ffd400",
			linewidths=2.4,
			depthshade=False,
			zorder=10,
			label="Rupture structurelle",
		)

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
		if show_burned_elements:
			legend_handles.append(Patch(facecolor="black", edgecolor="black", alpha=0.9, label="Brule"))
		if structural_state.enabled:
			legend_handles.append(broken_marker_full)
		if legend_handles:
			ax_full.legend(handles=legend_handles, loc="upper right", framealpha=0.9, title="Objets / Materiaux")
		return fig, ax_full, {"mesh_scatter": mesh_scatter, "full_surface": full_surface, "mesh_ax": ax_mesh, "full_ax": ax_full, "region_fills": region_fills, "region_edges": region_edges, "burned_tetra_surface": burned_tetra_surface, "source_marker_mesh": source_marker_mesh, "source_marker_full": source_marker_full, "source_beacon_mesh": source_beacon_mesh, "source_beacon_full": source_beacon_full, "broken_marker_mesh": broken_marker_mesh, "broken_marker_full": broken_marker_full}

	def _setup_3d_initial_overview(local_t: np.ndarray):
		fig = plt.figure(figsize=(20, 7))
		ax_mesh = fig.add_subplot(131, projection="3d")
		ax_full = fig.add_subplot(132, projection="3d")
		ax_setup = fig.add_subplot(133, projection="3d")

		edges = _build_boundary_edges(visual_faces)
		segments = [[pts[i], pts[j]] for i, j in edges]
		ax_mesh.add_collection3d(Line3DCollection(segments, colors="black", linewidths=0.2, alpha=0.55))
		ax_mesh.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="#444444", alpha=0.28, depthshade=False)

		_add_region_overlays_3d(ax_full, pts, visual_faces, visual_phys, visual_regions, include_solid_fill=False)
		_add_region_overlays_3d(ax_full, pts, visual_faces, visual_phys, visual_regions, include_solid_fill=True)
		_add_region_edges_3d(ax_full, pts, visual_faces, visual_phys, visual_regions)

		face_temps = np.mean(local_t[visual_faces], axis=1)
		norm = Normalize(vmin=THERMAL_3D_VMIN, vmax=THERMAL_3D_VMAX)
		setup_surface = Poly3DCollection(
			[pts[face] for face in visual_faces],
			facecolors=THERMAL_CMAP_3D(norm(face_temps)),
			edgecolors="none",
			linewidths=0.0,
			alpha=0.60,
		)
		ax_setup.add_collection3d(setup_surface)
		_add_region_overlays_3d(ax_setup, pts, visual_faces, visual_phys, visual_regions, include_solid_fill=True)
		_add_region_edges_3d(ax_setup, pts, visual_faces, visual_phys, visual_regions)
		_add_source_marker_3d(ax_setup, src_x, src_y, src_z, src_temp, pts)

		ax_mesh.set_title("Maillage 3D")
		ax_full.set_title("Batiment 3D plein")
		ax_setup.set_title("Setup initial")
		for local_ax in (ax_mesh, ax_full, ax_setup):
			local_ax.set_xlabel("x")
			local_ax.set_ylabel("y")
			local_ax.set_zlabel("z")
			_set_equal_3d_axes(local_ax, pts)
		return fig

	output_dir = animation_path = setup_png_path = timings_csv_path = None
	if getattr(args, "save", None):
		t0 = time.perf_counter()
		output_dir, animation_path, setup_png_path, timings_csv_path = _resolve_save_targets(args.save)
		output_dir.mkdir(parents=True, exist_ok=True)
		record_timing("output_directory_prepare", time.perf_counter() - t0, details=str(output_dir))

	if args.plot:
		if dim == 3:
			t0 = time.perf_counter()
			fig_init = _setup_3d_initial_overview(t)
			record_timing("initial_setup_figure", time.perf_counter() - t0, details="3d_overview_triptych")
		else:
			t0 = time.perf_counter()
			fig_init, _ax_init, _visuals_init = _setup_axes(t, "Setup initial")
			record_timing("initial_setup_figure", time.perf_counter() - t0)
		if setup_png_path is not None:
			t1 = time.perf_counter()
			fig_init.savefig(setup_png_path, dpi=200, bbox_inches="tight")
			record_timing("initial_setup_png_save", time.perf_counter() - t1, details=str(setup_png_path))
		#plt.tight_layout()
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

	should_render = bool(args.plot or getattr(args, "save", None))
	fig = ax = visuals = None
	if should_render:
		t0 = time.perf_counter()
		fig, ax, visuals = _setup_axes(t, "Temps: 0.0s | Tmax: {:.1f}K".format(float(np.max(t))))
		record_timing("animation_figure", time.perf_counter() - t0)
	state = {"t": t, "time": 0.0}
	ani = None
	rng = np.random.default_rng()
	cached_frozen_dofs = empty_dofs
	cached_free_dofs = free_dofs

	def _update_visual(local_t: np.ndarray, sim_time: float):
		if visuals is None or ax is None:
			return []
		if dim == 2:
			field = visuals["field"]
			burned_collection = visuals["burned_collection"]
			field.set_array(local_t)
			burned_collection.set_verts(_burned_triangle_vertices(pts, elems, burned_elements) if show_burned_elements else [])
			visible = source_visible(sim_time)
			visuals["source_marker"].set_visible(visible)
			if visuals.get("source_radius_patch") is not None:
				visuals["source_radius_patch"].set_visible(visible)
			ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(local_t):.1f}K")
			return [field, burned_collection]
		else:
			mesh_scatter = visuals["mesh_scatter"]
			full_surface = visuals["full_surface"]
			burned_tetra_surface = visuals["burned_tetra_surface"]
			broken_marker_mesh = visuals["broken_marker_mesh"]
			broken_marker_full = visuals["broken_marker_full"]
			mesh_ax = visuals["mesh_ax"]
			full_ax = visuals["full_ax"]
			visible = source_visible(sim_time)
			visuals["source_marker_mesh"].set_visible(visible)
			visuals["source_marker_full"].set_visible(visible)
			if visuals.get("source_beacon_mesh") is not None:
				visuals["source_beacon_mesh"].set_visible(visible)
			if visuals.get("source_beacon_full") is not None:
				visuals["source_beacon_full"].set_visible(visible)
			mesh_scatter.set_array(local_t)
			mesh_scatter.set_clim(vmin=THERMAL_3D_VMIN, vmax=THERMAL_3D_VMAX)
			face_temps = np.mean(local_t[visual_faces], axis=1)
			norm = Normalize(vmin=THERMAL_3D_VMIN, vmax=THERMAL_3D_VMAX)
			face_colors = THERMAL_CMAP_3D(norm(face_temps))
			if show_burned_elements:
				burned_boundary_mask = _burned_boundary_face_mask(visual_faces, elems, burned_elements)
				face_colors[burned_boundary_mask] = (0.0, 0.0, 0.0, 1.0)
			full_surface.set_facecolor(face_colors)
			burned_tetra_surface.set_verts(_burned_boundary_face_vertices(pts, visual_faces, elems, burned_elements) if show_burned_elements else [])
			burned_tetra_surface.set_facecolor((0.0, 0.0, 0.0, 1.0))
			burned_tetra_surface.set_edgecolor((0.0, 0.0, 0.0, 1.0))
			burned_tetra_surface.set_alpha(0.9 if show_burned_elements else 0.0)
			broken_points = structural_state.centroids[structural_state.broken] if structural_state.enabled else np.empty((0, 3))
			xs = broken_points[:, 0] if len(broken_points) else []
			ys = broken_points[:, 1] if len(broken_points) else []
			zs = broken_points[:, 2] if len(broken_points) else []
			broken_marker_mesh._offsets3d = (xs, ys, zs)
			broken_marker_full._offsets3d = (xs, ys, zs)
			mesh_ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(local_t):.1f}K | Vue maillage")
			full_ax.set_title(f"Temps: {sim_time:.1f}s | Tmax: {np.max(local_t):.1f}K | Vue pleine")
			return [mesh_scatter, full_surface, burned_tetra_surface, broken_marker_mesh, broken_marker_full]

	def advance_state(current_t: np.ndarray, current_time: float) -> tuple[np.ndarray, float, float]:
		nonlocal cached_frozen_dofs, cached_free_dofs
		t_local = current_t
		sim_time = float(current_time)
		t0 = time.perf_counter()
		for _ in range(sub_steps):
			old_t = t_local.copy()
			thawed_nodes = _thaw_frozen_nodes(
				frozen_nodes,
				node_idle_steps,
				node_neighbors,
				t_local,
				system.tc_node,
				node_thaw_delta,
				node_thaw_margin,
			)
			vertical_attenuation_step = vertical_air_attenuation
			if vertical_transfer.enabled and vertical_air_random_delta > 0.0:
				vertical_attenuation_step = float(
					vertical_air_attenuation
					* rng.uniform(max(0.0, 1.0 - vertical_air_random_delta), 1.0 + vertical_air_random_delta)
				)
			horizontal_attenuation_step = horizontal_air_attenuation
			if horizontal_transfer.enabled and horizontal_air_random_delta > 0.0:
				horizontal_attenuation_step = float(
					horizontal_air_attenuation
					* rng.uniform(max(0.0, 1.0 - horizontal_air_random_delta), 1.0 + horizontal_air_random_delta)
				)
			src = _hrr_source_rhs(
				system,
				elems,
				burned_elements,
				t_local,
				burn_times,
				sim_time,
				vertical_transfer,
				vertical_attenuation_step,
				horizontal_transfer,
				horizontal_attenuation_step,
			)
			radiation_loss_rhs = _radiation_loss_rhs(system.m_unit, t_local, volume_loss)
			rhs = src + bc_field.rhs + volume_loss.rhs - radiation_loss_rhs
			source_active = source_time > 0.0 and sim_time < source_time
			if source_active:
				step_frozen_nodes = frozen_nodes.copy()
				step_frozen_nodes[source_nodes] = True
				frozen_dofs = np.flatnonzero(step_frozen_nodes)
			else:
				frozen_dofs = np.flatnonzero(frozen_nodes)
			frozen_vals = t_local[frozen_dofs]
			if source_active and len(frozen_dofs):
				frozen_vals = frozen_vals.copy()
				frozen_vals[source_nodes[frozen_dofs]] = src_temp
			if len(frozen_dofs) == 0:
				step_free_dofs = free_dofs
				cached_frozen_dofs = empty_dofs
				cached_free_dofs = free_dofs
			elif np.array_equal(frozen_dofs, cached_frozen_dofs):
				step_free_dofs = cached_free_dofs
			else:
				cached_frozen_dofs = frozen_dofs.copy()
				cached_free_dofs = precompute_dirichlet_dofs(len(pts), cached_frozen_dofs)
				step_free_dofs = cached_free_dofs
			t_local = np.asarray(
				theta_step_fast(
					system.m_mat,
					k_eff,
					rhs,
					rhs,
					t_local,
					dt=dt,
					theta=theta,
					dirichlet_dofs=frozen_dofs if len(frozen_dofs) else empty_dofs,
					dir_vals_np1=frozen_vals if len(frozen_dofs) else empty_vals,
					free_dofs=step_free_dofs,
				),
				dtype=float,
			)
			if source_active:
				t_local[source_nodes] = src_temp
			checked_nodes, newly_frozen_nodes = _update_frozen_nodes(
				frozen_nodes,
				node_idle_steps,
				old_t,
				t_local,
				system.tc_node,
				node_freeze_steps,
				node_freeze_delta,
				node_freeze_margin,
				max_frozen_node_fraction,
			)
			if newly_frozen_nodes or thawed_nodes:
				record_timing(
					"node_freeze_update",
					0.0,
					details=f"checked={checked_nodes};frozen={newly_frozen_nodes};thawed={thawed_nodes};frozen_total={int(np.count_nonzero(frozen_nodes))}",
				)
			reactivated_count = _reactivate_near_hot_nodes(
				active_elements,
				element_idle_steps,
				burned_elements,
				elems,
				t_local,
				elem_tc,
				element_freeze_margin,
			)
			burned_indices = _update_burned_elements(burned_elements, elems, t_local, elem_tc, active_elements)
			if len(burned_indices):
				burn_times[burned_indices] = sim_time + dt
				record_timing("element_burn_update", 0.0, details=f"changed_elements={len(burned_indices)}")
				apply_material_changes(burned_indices)
				active_elements[burned_indices] = False
				element_idle_steps[burned_indices] = 0
			if structural_state.enabled:
				broken_indices = update_structural_fun(structural_state, system.elem_material_names, t_local, elems, burned_elements)
				if len(broken_indices):
					record_timing("structural_fun_break_update", 0.0, frame="", details=f"broken_elements={len(broken_indices)};total={int(np.count_nonzero(structural_state.broken))}")
			checked_count, frozen_count = _update_element_activity(
				active_elements,
				element_idle_steps,
				burned_elements,
				elems,
				t_local,
				elem_tc,
				element_freeze_steps,
				element_freeze_margin,
			)
			if frozen_count or reactivated_count:
				record_timing(
					"element_activity_update",
					0.0,
					details=f"checked={checked_count};frozen={frozen_count};reactivated={reactivated_count};active={int(np.count_nonzero(active_elements))}",
				)
			sim_time += dt
		return t_local, sim_time, time.perf_counter() - t0

	if not args.plot:
		t0 = time.perf_counter()
		max_wall_seconds = float(getattr(args, "max_wall_seconds", 0.0) or 0.0)
		actual_frames = 0
		if getattr(args, "save", None):
			headless_frames = [t.copy()]
			headless_times = [0.0]
			headless_burned = [burned_elements.copy()]
			for frame_idx in range(steps):
				t, sim_time, elapsed = advance_state(t, headless_times[-1])
				record_timing("frame_calculation", elapsed, frame=frame_idx, details=f"sim_time={sim_time:.5f}")
				headless_frames.append(t.copy())
				headless_times.append(sim_time)
				headless_burned.append(burned_elements.copy())
				actual_frames = frame_idx + 1
				if max_wall_seconds > 0.0 and time.perf_counter() - t0 >= max_wall_seconds:
					break
			state["t"] = t
			state["time"] = headless_times[-1]

			def update_anim(frame_idx: int):
				burned_elements[:] = headless_burned[frame_idx]
				return _update_visual(headless_frames[frame_idx], headless_times[frame_idx])

			ani = FuncAnimation(fig, update_anim, frames=len(headless_frames), interval=100, blit=False, repeat=False)
		else:
			sim_time = 0.0
			for frame_idx in range(steps):
				t, sim_time, elapsed = advance_state(t, sim_time)
				record_timing("frame_calculation", elapsed, frame=frame_idx, details=f"sim_time={sim_time:.5f}")
				actual_frames = frame_idx + 1
				if max_wall_seconds > 0.0 and time.perf_counter() - t0 >= max_wall_seconds:
					break
			state["t"] = t
			state["time"] = sim_time
		record_timing("headless_calculation_total", time.perf_counter() - t0, details=f"frames={actual_frames};requested_frames={steps};dim={dim}")
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

	if getattr(args, "timings_csv", None):
		timings_path = Path(args.timings_csv)
		timings_path.parent.mkdir(parents=True, exist_ok=True)
		_write_timings_csv(timings_path, timing_rows)

	if args.plot:
		plt.show()


def build_parser() -> argparse.ArgumentParser: # terminal
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
	parser.add_argument("--horizontal-air-transfer", dest="horizontal_air_transfer", type=int, choices=[0, 1], default=None, help="Active l'avance horizontale des flammes par mouvements d'air en 2D/3D")
	parser.add_argument("--horizontal-air-attenuation", dest="horizontal_air_attenuation", type=float, default=None, help="Attenuation horizontale du transfert air [1/m]")
	parser.add_argument("--horizontal-air-radius", dest="horizontal_air_radius", type=float, default=None, help="Rayon horizontal de l'effet de zone; 0=auto")
	parser.add_argument("--horizontal-air-power-fraction", dest="horizontal_air_power_fraction", type=float, default=None, help="Fraction max de puissance redistribuee horizontalement")
	parser.add_argument("--horizontal-air-random-delta", dest="horizontal_air_random_delta", type=float, default=None, help="Variation aleatoire de l'attenuation horizontale a chaque sous-pas; 0=desactive")
	parser.add_argument("--structural-fun", dest="structural_fun", type=int, nargs="?", const=1, choices=[0, 1], default=None, help="Active l'add-on fun de rupture structurelle simplifiee en 3D")
	parser.add_argument("--structural-fun-radius", dest="structural_fun_radius", type=float, default=None, help="Rayon horizontal de reprise des charges; 0=auto")
	parser.add_argument("--structural-fun-load-factor", dest="structural_fun_load_factor", type=float, default=None, help="Facteur multiplicatif sur les charges de poids")
	parser.add_argument("--t-amb", dest="t_amb", type=float, default=None, help="Temperature ambiante")
	parser.add_argument("--src-temp", dest="src_temp", type=float, default=None, help="Temperature initiale source")
	parser.add_argument("--src-x", type=float, default=None, help="X source")
	parser.add_argument("--src-y", type=float, default=None, help="Y source")
	parser.add_argument("--src-z", type=float, default=None, help="Z source (3D)")
	parser.add_argument("--src-radius", type=float, default=None, help="Rayon source initial")
	parser.add_argument("--source-time", "--temps-source", "--temps-sources", dest="source_time", type=float, default=None, help="Temps de simulation pendant lequel la source reste active; 0=source initiale seulement")
	parser.add_argument("--element-freeze-steps", dest="element_freeze_steps", type=int, default=25, help="Nombre de steps froids avant de ne plus retester un element; 0=desactive")
	parser.add_argument("--element-freeze-margin", dest="element_freeze_margin", type=float, default=25.0, help="Marge sous Tc pour considerer un element froid/stable [K]")
	parser.add_argument("--node-freeze-steps", dest="node_freeze_steps", type=int, default=20, help="Nombre de steps quasi stationnaires avant Dirichlet temporaire; 0=desactive")
	parser.add_argument("--node-freeze-delta", dest="node_freeze_delta", type=float, default=0.05, help="Variation max par step pour geler un noeud [K]")
	parser.add_argument("--node-freeze-margin", dest="node_freeze_margin", type=float, default=50.0, help="Marge sous Tc requise pour geler un noeud [K]")
	parser.add_argument("--node-thaw-delta", dest="node_thaw_delta", type=float, default=2.0, help="Ecart max avec un voisin avant degel [K]")
	parser.add_argument("--node-thaw-margin", dest="node_thaw_margin", type=float, default=35.0, help="Marge sous Tc d'un voisin avant degel [K]")
	parser.add_argument("--max-frozen-node-fraction", dest="max_frozen_node_fraction", type=float, default=0.90, help="Fraction maximale de noeuds geles")
	parser.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Dimension du calcul")
	parser.add_argument("-2d", "--2d", dest="dim", action="store_const", const=2, help="Force le mode 2D")
	parser.add_argument("-3d", "--3d", dest="dim", action="store_const", const=3, help="Force le mode 3D")
	parser.add_argument("--no-plot", dest="plot", action="store_false", help="Desactive l'affichage final")
	parser.add_argument("--hide-burned-elements", dest="hide_burned_elements", action="store_true", help="Masque l'affichage noir des elements brules sans desactiver leur calcul")
	parser.add_argument("--save", dest="save", type=str, default=None, help="Nom de fichier MP4 pour sauvegarder l'animation")
	parser.add_argument("--timings-csv", dest="timings_csv", type=str, default=None, help="Chemin CSV pour sauvegarder les timings sans exporter d'animation")
	parser.add_argument("--max-wall-seconds", dest="max_wall_seconds", type=float, default=None, help="Arrete proprement le calcul headless apres cette duree murale.")
	parser.set_defaults(plot=True)
	return parser


def main() -> None: # main lui-même
	parser = build_parser()
	args = parser.parse_args()
	if args.mesh is None:
		args.mesh = "piece.msh" if int(args.dim) == 2 else "immeuble.msh"
	run(args)


if __name__ == "__main__": #BOOM
	main()
