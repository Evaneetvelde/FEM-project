from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from materialsbank import get_material


@dataclass
class StructuralFunState:
	enabled: bool
	centroids: np.ndarray
	volumes: np.ndarray
	support_targets: list[np.ndarray]
	support_factors: list[np.ndarray]
	broken: np.ndarray
	load_factor: float
	min_strength_factor: float


def _auto_support_radius(centroids: np.ndarray, n_elems: int) -> float:
	spans = np.ptp(centroids[:, :2], axis=0)
	area = max(float(spans[0] * spans[1]), 1e-12)
	return 1.75 * float(np.sqrt(area / max(n_elems, 1)))


def prepare_structural_fun(points: np.ndarray, elems: np.ndarray, volumes: np.ndarray, enabled: bool, radius: float, load_factor: float, min_strength_factor: float = 0.08) -> StructuralFunState:
	centroids = np.mean(points[elems], axis=1)
	n_elems = len(elems)
	if points.shape[1] < 3 or not enabled:
		return StructuralFunState(False, centroids, np.asarray(volumes, dtype=float), [], [], np.zeros(n_elems, dtype=bool), float(load_factor), float(min_strength_factor))

	effective_radius = float(radius) if radius > 0.0 else _auto_support_radius(centroids, n_elems)
	support_targets: list[np.ndarray] = []
	support_factors: list[np.ndarray] = []
	for source_idx, source_center in enumerate(centroids):
		dz = source_center[2] - centroids[:, 2]
		horizontal_dist = np.linalg.norm(centroids[:, :2] - source_center[:2], axis=1)
		mask = (dz > 0.0) & (horizontal_dist <= effective_radius)
		target_indices = np.flatnonzero(mask)
		if len(target_indices) == 0:
			support_targets.append(np.array([], dtype=int))
			support_factors.append(np.array([], dtype=float))
			continue
		factors = np.maximum(0.0, 1.0 - horizontal_dist[target_indices] / max(effective_radius, 1e-12))
		total = float(np.sum(factors))
		if total <= 0.0:
			factors = np.full(len(target_indices), 1.0 / len(target_indices), dtype=float)
		else:
			factors = factors / total
		support_targets.append(target_indices)
		support_factors.append(factors.astype(float))

	return StructuralFunState(True, centroids, np.asarray(volumes, dtype=float), support_targets, support_factors, np.zeros(n_elems, dtype=bool), max(0.0, float(load_factor)), max(0.0, float(min_strength_factor)))


def update_structural_fun(state: StructuralFunState, elem_material_names: np.ndarray, local_t: np.ndarray, elems: np.ndarray, burned_elements: np.ndarray) -> np.ndarray:
	if not state.enabled:
		return np.array([], dtype=int)

	g = 9.81
	n_elems = len(elems)
	elem_temps = np.mean(local_t[elems], axis=1)
	weights = np.zeros(n_elems, dtype=float)
	capacity = np.full(n_elems, np.inf, dtype=float)
	structural_mask = np.zeros(n_elems, dtype=bool)

	for elem_idx, material_name in enumerate(elem_material_names):
		material = get_material(str(material_name))
		rho = max(0.0, float(material.get("rho", 0.0)))
		weights[elem_idx] = rho * float(state.volumes[elem_idx]) * g
		if not bool(material.get("structural", False)):
			continue
		structural_mask[elem_idx] = True
		strength = max(0.0, float(material.get("compressive_strength", 0.0)))
		temp_coeff = max(0.0, float(material.get("strength_temp_coeff", 0.0)))
		temp_damage = max(0.0, elem_temps[elem_idx] - 293.0)
		strength_factor = max(state.min_strength_factor, 1.0 - temp_coeff * temp_damage)
		area = max(float(state.volumes[elem_idx]), 1e-12) ** (2.0 / 3.0)
		capacity[elem_idx] = strength * strength_factor * area

	supported_load = weights.copy()
	for source_idx, target_indices in enumerate(state.support_targets):
		if len(target_indices) == 0:
			continue
		supported_load[target_indices] += weights[source_idx] * state.support_factors[source_idx]

	candidates = burned_elements & structural_mask & (~state.broken)
	newly_broken = np.flatnonzero(candidates & (supported_load * state.load_factor > capacity))
	state.broken[newly_broken] = True
	return newly_broken
