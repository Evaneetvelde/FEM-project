from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import meshio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIRNAME = "plots"


@dataclass(frozen=True)
class VersionSpec:
	order: int
	id: str
	label: str
	script: Path
	description: str


@dataclass(frozen=True)
class MeshSpec:
	id: str
	label: str
	dim: int
	path: Path
	category: str


VERSIONS = [
	VersionSpec(
		1,
		"main_sans_rien",
		"Main sans optimisation",
		PROJECT_ROOT / "old" / "mainnotopti.py",
		"Reference lente: assemblage classique et fonctions profs.",
	),
	VersionSpec(
		2,
		"fancy_reassembling",
		"Fancy reassembling",
		PROJECT_ROOT / "old" / "mainfancyreassembling.py",
		"Ajoute gel/degel et retest selectif des elements/noeuds.",
	),
	VersionSpec(
		3,
		"calcul_matriciel",
		"Calcul matriciel",
		PROJECT_ROOT / "old" / "maincalculmatriciel.py",
		"Ajoute Numba, preassemblage, matrices unitaires et assemblage matriciel.",
	),
	VersionSpec(
		4,
		"main_opti",
		"Main opti",
		PROJECT_ROOT / "main.py",
		"Fusion finale: calcul matriciel + gel/degel + mode headless optimise.",
	),
]

BENCHMARK_MESHES = [
	MeshSpec("2d_peu_noeuds", "2D peu de noeuds", 2, PROJECT_ROOT / "models" / "perf" / "plot_2d_small.msh", "peu"),
	MeshSpec("2d_moyen_noeuds", "2D moyen", 2, PROJECT_ROOT / "models" / "perf" / "perf_2d_middle.msh", "moyen"),
	MeshSpec("2d_beaucoup_noeuds", "2D beaucoup de noeuds", 2, PROJECT_ROOT / "models" / "perf" / "perf_2d_big.msh", "beaucoup"),
	MeshSpec("2d_complexe", "2D complexe", 2, PROJECT_ROOT / "models" / "perf" / "plot_2d_complex.msh", "complexe"),
	MeshSpec("3d_peu_noeuds", "3D peu de noeuds", 3, PROJECT_ROOT / "models" / "perf" / "plot_3d_small.msh", "peu"),
	MeshSpec("3d_moyen_noeuds", "3D moyen", 3, PROJECT_ROOT / "models" / "perf" / "plot_3d_middle.msh", "moyen"),
	MeshSpec("3d_beaucoup_noeuds", "3D beaucoup de noeuds", 3, PROJECT_ROOT / "models" / "perf" / "plot_3d_big.msh", "beaucoup"),
	MeshSpec("3d_complexe", "3D complexe", 3, PROJECT_ROOT / "models" / "perf" / "plot_3d_complex.msh", "complexe"),
]


SUMMARY_PHASES = {
	"mesh_load": "mesh_load",
	"system_assembly": "system_assembly",
	"vertical_air_transfer_prepare": "vertical_air_transfer_prepare",
	"initial_conditions": "initial_conditions",
	"frame_calculation": "frames_sum",
	"headless_calculation_total": "headless_total",
	"material_burn_delta_update": "burn_delta_updates",
	"node_freeze_update": "node_freeze_updates",
	"element_activity_update": "element_activity_updates",
}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as fh:
		writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _read_timing_rows(path: Path) -> list[dict[str, str]]:
	if not path.exists():
		return []
	with path.open("r", newline="", encoding="utf-8") as fh:
		return list(csv.DictReader(fh))


def _safe_float(value: object, default: float = 0.0) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def _mesh_counts(mesh: MeshSpec) -> tuple[int, int]:
	msh = meshio.read(str(mesh.path))
	cell_type = "triangle" if mesh.dim == 2 else "tetra"
	n_nodes = int(len(msh.points))
	n_elements = int(len(msh.cells_dict.get(cell_type, [])))
	return n_nodes, n_elements


def _select_meshes(args: argparse.Namespace) -> list[MeshSpec]:
	if args.mesh:
		mesh_path = Path(args.mesh)
		if not mesh_path.exists():
			candidate = PROJECT_ROOT / args.mesh
			mesh_path = candidate if candidate.exists() else PROJECT_ROOT / "models" / args.mesh
		if not mesh_path.exists():
			candidate = PROJECT_ROOT / "models" / "perf" / Path(args.mesh).name
			mesh_path = candidate if candidate.exists() else mesh_path
		return [MeshSpec("custom", f"custom {mesh_path.name}", int(args.dim), mesh_path, "custom")]

	if args.suite == "2d":
		return [mesh for mesh in BENCHMARK_MESHES if mesh.dim == 2]
	if args.suite == "3d":
		return [mesh for mesh in BENCHMARK_MESHES if mesh.dim == 3]
	if args.suite == "small":
		return [mesh for mesh in BENCHMARK_MESHES if mesh.category == "peu"]
	return BENCHMARK_MESHES


def _validate_meshes(meshes: list[MeshSpec]) -> None:
	missing = [mesh.path for mesh in meshes if not mesh.path.exists()]
	if missing:
		formatted = "\n".join(f"  - {path}" for path in missing)
		raise FileNotFoundError(f"Maillage(s) benchmark introuvable(s):\n{formatted}")


def _build_common_args(args: argparse.Namespace, mesh: MeshSpec) -> list[str]:
	steps = max(int(args.steps), 1_000_000) if args.run_seconds is not None else int(args.steps)
	cmd = [
		"--steps",
		str(steps),
		"--sub-steps",
		str(args.sub_steps),
		"--no-plot",
	]
	if mesh.dim == 3:
		cmd.append("--3d")
	else:
		cmd.append("--2d")
	cmd.extend(["--mesh", str(mesh.path)])
	if args.dt is not None:
		cmd.extend(["--dt", str(args.dt)])
	if args.theta is not None:
		cmd.extend(["--theta", str(args.theta)])
	if args.run_seconds is not None:
		cmd.extend(["--max-wall-seconds", str(args.run_seconds)])
	return cmd


def _run_one(version: VersionSpec, mesh: MeshSpec, repeat_idx: int, args: argparse.Namespace, run_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
	raw_csv = run_dir / "raw" / mesh.id / f"{version.order:02d}_{version.id}_r{repeat_idx}.csv"
	stdout_path = run_dir / "logs" / mesh.id / f"{version.order:02d}_{version.id}_r{repeat_idx}.stdout.txt"
	stderr_path = run_dir / "logs" / mesh.id / f"{version.order:02d}_{version.id}_r{repeat_idx}.stderr.txt"

	cmd = [sys.executable, str(version.script), *_build_common_args(args, mesh), "--timings-csv", str(raw_csv)]
	start = time.perf_counter()
	proc = subprocess.run(
		cmd,
		cwd=PROJECT_ROOT,
		text=True,
		capture_output=True,
		check=False,
	)
	wall_seconds = time.perf_counter() - start

	stdout_path.parent.mkdir(parents=True, exist_ok=True)
	stdout_path.write_text(proc.stdout, encoding="utf-8")
	stderr_path.write_text(proc.stderr, encoding="utf-8")

	timing_rows: list[dict[str, object]] = []
	for row_idx, row in enumerate(_read_timing_rows(raw_csv)):
		timing_rows.append(
			{
				"run_id": run_dir.name,
				"mesh_id": mesh.id,
				"mesh_label": mesh.label,
				"mesh_dim": mesh.dim,
				"mesh_category": mesh.category,
				"repeat": repeat_idx,
				"version_order": version.order,
				"version_id": version.id,
				"version_label": version.label,
				"phase": row.get("phase", ""),
				"seconds": row.get("seconds", ""),
				"frame": row.get("frame", ""),
				"details": row.get("details", ""),
				"source_row": row_idx,
			}
		)

	status = {
		"run_id": run_dir.name,
		"mesh_id": mesh.id,
		"mesh_label": mesh.label,
		"mesh_dim": mesh.dim,
		"mesh_category": mesh.category,
		"mesh_path": str(mesh.path.relative_to(PROJECT_ROOT)) if mesh.path.is_relative_to(PROJECT_ROOT) else str(mesh.path),
		"repeat": repeat_idx,
		"version_order": version.order,
		"version_id": version.id,
		"version_label": version.label,
		"description": version.description,
		"script": str(version.script.relative_to(PROJECT_ROOT)),
		"returncode": proc.returncode,
		"wall_seconds": f"{wall_seconds:.6f}",
		"timings_csv": str(raw_csv.relative_to(run_dir)),
		"stdout": str(stdout_path.relative_to(run_dir)),
		"stderr": str(stderr_path.relative_to(run_dir)),
		"command": " ".join(cmd),
	}
	return timing_rows, status


def _summarize(raw_rows: list[dict[str, object]], statuses: list[dict[str, object]]) -> list[dict[str, object]]:
	status_by_key = {(str(s["mesh_id"]), int(s["version_order"]), int(s["repeat"])): s for s in statuses}
	grouped: dict[tuple[str, int, str, int], list[dict[str, object]]] = {}
	for row in raw_rows:
		key = (str(row["mesh_id"]), int(row["version_order"]), str(row["version_id"]), int(row["repeat"]))
		grouped.setdefault(key, []).append(row)

	summary_rows: list[dict[str, object]] = []
	for (mesh_id, version_order, version_id, repeat_idx), rows in sorted(grouped.items()):
		version_label = str(rows[0]["version_label"])
		status = status_by_key.get((mesh_id, version_order, repeat_idx), {})
		phase_sums = {alias: 0.0 for alias in SUMMARY_PHASES.values()}
		frame_count = 0
		frame_max = 0.0
		frame_min = 0.0
		frame_values: list[float] = []

		for row in rows:
			phase = str(row["phase"])
			seconds = _safe_float(row["seconds"])
			if phase in SUMMARY_PHASES:
				phase_sums[SUMMARY_PHASES[phase]] += seconds
			if phase == "frame_calculation":
				frame_values.append(seconds)

		if frame_values:
			frame_count = len(frame_values)
			frame_min = min(frame_values)
			frame_max = max(frame_values)
		measured_compute_total = (
			phase_sums["mesh_load"]
			+ phase_sums["system_assembly"]
			+ phase_sums["vertical_air_transfer_prepare"]
			+ phase_sums["initial_conditions"]
			+ phase_sums["headless_total"]
		)

		summary_rows.append(
			{
				"version_order": version_order,
				"version_id": version_id,
				"version_label": version_label,
				"mesh_id": mesh_id,
				"mesh_label": rows[0]["mesh_label"],
				"mesh_dim": rows[0]["mesh_dim"],
				"mesh_category": rows[0]["mesh_category"],
				"repeat": repeat_idx,
				"returncode": status.get("returncode", ""),
				"wall_seconds": status.get("wall_seconds", ""),
				"measured_compute_total": f"{measured_compute_total:.6f}",
				"mesh_load": f"{phase_sums['mesh_load']:.6f}",
				"system_assembly": f"{phase_sums['system_assembly']:.6f}",
				"vertical_air_transfer_prepare": f"{phase_sums['vertical_air_transfer_prepare']:.6f}",
				"initial_conditions": f"{phase_sums['initial_conditions']:.6f}",
				"frames_sum": f"{phase_sums['frames_sum']:.6f}",
				"frames_avg": f"{(sum(frame_values) / frame_count if frame_count else 0.0):.6f}",
				"frames_min": f"{frame_min:.6f}",
				"frames_max": f"{frame_max:.6f}",
				"frame_count": frame_count,
				"headless_total": f"{phase_sums['headless_total']:.6f}",
				"burn_delta_updates": f"{phase_sums['burn_delta_updates']:.6f}",
				"node_freeze_updates": f"{phase_sums['node_freeze_updates']:.6f}",
				"element_activity_updates": f"{phase_sums['element_activity_updates']:.6f}",
			}
		)
	return summary_rows


def _best_by_mesh_version(summary_rows: list[dict[str, object]], metric: str) -> dict[tuple[str, str], dict[str, object]]:
	best: dict[tuple[str, str], dict[str, object]] = {}
	for row in summary_rows:
		mesh_id = str(row["mesh_id"])
		version_id = str(row["version_id"])
		key = (mesh_id, version_id)
		value = _safe_float(row.get(metric))
		if key not in best or value < _safe_float(best[key].get(metric), float("inf")):
			best[key] = row
	return best


def _comparison_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
	best_rows = sorted(_best_by_mesh_version(summary_rows, "measured_compute_total").values(), key=lambda row: (str(row["mesh_id"]), int(row["version_order"])))
	if not best_rows:
		return []

	rows: list[dict[str, object]] = []
	mesh_ids = sorted({str(row["mesh_id"]) for row in best_rows})
	for mesh_id in mesh_ids:
		mesh_rows = [row for row in best_rows if str(row["mesh_id"]) == mesh_id]
		baseline = mesh_rows[0]
		baseline_total = _safe_float(baseline["measured_compute_total"])
		previous_total = baseline_total
		for row in mesh_rows:
			total = _safe_float(row["measured_compute_total"])
			speedup = baseline_total / total if total > 0.0 else 0.0
			gain_vs_baseline = 100.0 * (1.0 - total / baseline_total) if baseline_total > 0.0 else 0.0
			gain_vs_previous = 100.0 * (1.0 - total / previous_total) if previous_total > 0.0 else 0.0
			rows.append(
				{
					"mesh_id": row["mesh_id"],
					"mesh_label": row["mesh_label"],
					"mesh_dim": row["mesh_dim"],
					"mesh_category": row["mesh_category"],
					"version_order": row["version_order"],
					"version_id": row["version_id"],
					"version_label": row["version_label"],
					"best_repeat": row["repeat"],
					"measured_compute_total": row["measured_compute_total"],
					"headless_total": row["headless_total"],
					"wall_seconds": row["wall_seconds"],
					"mesh_load": row["mesh_load"],
					"system_assembly": row["system_assembly"],
					"initial_conditions": row["initial_conditions"],
					"frames_sum": row["frames_sum"],
					"frames_avg": row["frames_avg"],
					"speedup_vs_main_sans_rien": f"{speedup:.6f}",
					"gain_pct_vs_main_sans_rien": f"{gain_vs_baseline:.2f}",
					"gain_pct_vs_previous_version": f"{gain_vs_previous:.2f}",
				}
			)
			previous_total = total
	return rows


def _phase_comparison_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
	phase_sums: dict[tuple[str, int, str, str, int], float] = {}
	labels: dict[tuple[str, int, str], tuple[str, str, object, object]] = {}
	for row in raw_rows:
		mesh_id = str(row["mesh_id"])
		version_order = int(row["version_order"])
		version_id = str(row["version_id"])
		repeat_idx = int(row["repeat"])
		phase = str(row["phase"])
		labels[(mesh_id, version_order, version_id)] = (str(row["version_label"]), str(row["mesh_label"]), row["mesh_dim"], row["mesh_category"])
		phase_sums[(mesh_id, version_order, version_id, phase, repeat_idx)] = phase_sums.get((mesh_id, version_order, version_id, phase, repeat_idx), 0.0) + _safe_float(row["seconds"])

	best_phase: dict[tuple[str, int, str, str], float] = {}
	for (mesh_id, version_order, version_id, phase, _repeat_idx), seconds in phase_sums.items():
		key = (mesh_id, version_order, version_id, phase)
		if key not in best_phase or seconds < best_phase[key]:
			best_phase[key] = seconds

	baseline_by_phase = {
		(mesh_id, phase): seconds
		for (mesh_id, version_order, _version_id, phase), seconds in best_phase.items()
		if version_order == 1
	}
	rows: list[dict[str, object]] = []
	for (mesh_id, version_order, version_id, phase), seconds in sorted(best_phase.items()):
		baseline = baseline_by_phase.get((mesh_id, phase), 0.0)
		speedup = baseline / seconds if seconds > 0.0 else 0.0
		version_label, mesh_label, mesh_dim, mesh_category = labels.get((mesh_id, version_order, version_id), ("", "", "", ""))
		rows.append(
			{
				"mesh_id": mesh_id,
				"mesh_label": mesh_label,
				"mesh_dim": mesh_dim,
				"mesh_category": mesh_category,
				"version_order": version_order,
				"version_id": version_id,
				"version_label": version_label,
				"phase": phase,
				"best_seconds": f"{seconds:.6f}",
				"speedup_vs_main_sans_rien_same_phase": f"{speedup:.6f}",
				"gain_pct_vs_main_sans_rien_same_phase": f"{(100.0 * (1.0 - seconds / baseline) if baseline > 0.0 else 0.0):.2f}",
			}
		)
	return rows


def _global_summary_rows(summary_rows: list[dict[str, object]], group_by_dim: bool = False) -> list[dict[str, object]]:
	grouped: dict[tuple[object, int, str], list[dict[str, object]]] = {}
	for row in summary_rows:
		group_key = row["mesh_dim"] if group_by_dim else "all"
		key = (group_key, int(row["version_order"]), str(row["version_id"]))
		grouped.setdefault(key, []).append(row)

	aggregates: list[dict[str, object]] = []
	for (group_key, version_order, version_id), rows in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])):
		totals = [_safe_float(row["measured_compute_total"]) for row in rows]
		walls = [_safe_float(row["wall_seconds"]) for row in rows]
		frames = [int(row.get("frame_count", 0) or 0) for row in rows]
		total_compute = sum(totals)
		total_wall = sum(walls)
		aggregates.append(
			{
				"group": f"{group_key}D" if group_by_dim else "all",
				"version_order": version_order,
				"version_id": version_id,
				"version_label": str(rows[0]["version_label"]),
				"mesh_runs": len(rows),
				"total_measured_compute": f"{total_compute:.6f}",
				"avg_measured_compute": f"{(total_compute / len(rows) if rows else 0.0):.6f}",
				"total_wall_seconds": f"{total_wall:.6f}",
				"avg_wall_seconds": f"{(total_wall / len(rows) if rows else 0.0):.6f}",
				"total_frames": sum(frames),
				"avg_frame_seconds": f"{(sum(_safe_float(row['frames_sum']) for row in rows) / max(sum(frames), 1)):.6f}",
			}
		)

	baseline_by_group = {
		str(row["group"]): _safe_float(row["total_measured_compute"])
		for row in aggregates
		if int(row["version_order"]) == 1
	}
	previous_by_group: dict[str, float] = {}
	for row in aggregates:
		group = str(row["group"])
		total = _safe_float(row["total_measured_compute"])
		baseline = baseline_by_group.get(group, 0.0)
		previous = previous_by_group.get(group, baseline)
		row["speedup_vs_main_sans_rien"] = f"{(baseline / total if total > 0.0 else 0.0):.6f}"
		row["gain_pct_vs_main_sans_rien"] = f"{(100.0 * (1.0 - total / baseline) if baseline > 0.0 else 0.0):.2f}"
		row["gain_pct_vs_previous_version"] = f"{(100.0 * (1.0 - total / previous) if previous > 0.0 else 0.0):.2f}"
		previous_by_group[group] = total
	return aggregates


def _index_rows(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, object]]:
	return {str(row[key]): row for row in rows}


def _write_plots(run_dir: Path, mesh_rows: list[dict[str, object]], comparison_rows: list[dict[str, object]], global_rows: list[dict[str, object]] | None = None) -> list[Path]:
	if not comparison_rows:
		return []

	try:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		import numpy as np
	except ImportError as exc:
		print(f"[perf] Graphiques ignores: matplotlib/numpy indisponible ({exc})")
		return []

	plots_dir = run_dir / PLOTS_DIRNAME
	plots_dir.mkdir(parents=True, exist_ok=True)
	mesh_by_id = _index_rows(mesh_rows, "mesh_id")
	version_ids = [version.id for version in VERSIONS]
	version_labels = {version.id: version.label for version in VERSIONS}
	version_colors = {
		"main_sans_rien": "#5B8DEF",
		"fancy_reassembling": "#F6A04D",
		"calcul_matriciel": "#58B368",
		"main_opti": "#D45B7A",
	}
	outputs: list[Path] = []

	def best(metric: str, mesh_id: str, version_id: str) -> float:
		values = [
			_safe_float(row.get(metric))
			for row in comparison_rows
			if str(row["mesh_id"]) == mesh_id and str(row["version_id"]) == version_id
		]
		return values[0] if values else 0.0

	def save_current(name: str) -> None:
		path = plots_dir / name
		plt.tight_layout()
		plt.savefig(path, dpi=180)
		plt.close()
		outputs.append(path)

	for dim in (2, 3):
		dim_meshes = [row for row in mesh_rows if int(row["mesh_dim"]) == dim]
		if not dim_meshes:
			continue
		mesh_ids = [str(row["mesh_id"]) for row in dim_meshes]
		labels = [str(row["mesh_label"]) for row in dim_meshes]
		x = np.arange(len(mesh_ids))
		width = 0.18

		plt.figure(figsize=(12, 6))
		for idx, version_id in enumerate(version_ids):
			values = [best("measured_compute_total", mesh_id, version_id) for mesh_id in mesh_ids]
			offset = (idx - (len(version_ids) - 1) / 2.0) * width
			plt.bar(x + offset, values, width, label=version_labels[version_id], color=version_colors.get(version_id))
		plt.xticks(x, labels, rotation=18, ha="right")
		plt.ylabel("Temps calcule mesure (s)")
		plt.title(f"Temps total par version - {dim}D")
		plt.grid(axis="y", alpha=0.25)
		plt.legend()
		save_current(f"temps_total_{dim}d.png")

		plt.figure(figsize=(12, 6))
		for idx, version_id in enumerate(version_ids):
			values = [best("speedup_vs_main_sans_rien", mesh_id, version_id) for mesh_id in mesh_ids]
			offset = (idx - (len(version_ids) - 1) / 2.0) * width
			plt.bar(x + offset, values, width, label=version_labels[version_id], color=version_colors.get(version_id))
		plt.axhline(1.0, color="#222222", linewidth=1.0)
		plt.xticks(x, labels, rotation=18, ha="right")
		plt.ylabel("Acceleration vs main sans optimisation (x)")
		plt.title(f"Speedup par version - {dim}D")
		plt.grid(axis="y", alpha=0.25)
		plt.legend()
		save_current(f"speedup_{dim}d.png")

		plt.figure(figsize=(12, 6))
		elements = [int(mesh_by_id[mesh_id]["elements"]) for mesh_id in mesh_ids]
		for version_id in version_ids:
			values = [best("measured_compute_total", mesh_id, version_id) for mesh_id in mesh_ids]
			plt.plot(elements, values, marker="o", linewidth=2, label=version_labels[version_id], color=version_colors.get(version_id))
		plt.xlabel("Nombre d'elements")
		plt.ylabel("Temps calcule mesure (s)")
		plt.title(f"Scalabilite selon le nombre d'elements - {dim}D")
		plt.grid(True, alpha=0.25)
		plt.legend()
		save_current(f"scalabilite_{dim}d.png")

	phase_names = ["mesh_load", "system_assembly", "initial_conditions", "frames_sum"]
	phase_labels = ["Maillage", "Assemblage", "Initialisation", "Frames"]
	for dim in (2, 3):
		rows = [row for row in comparison_rows if int(row["mesh_dim"]) == dim and str(row["version_id"]) == "main_opti"]
		if not rows:
			continue
		rows = sorted(rows, key=lambda row: int(mesh_by_id[str(row["mesh_id"])]["elements"]))
		x = np.arange(len(rows))
		bottom = np.zeros(len(rows))
		plt.figure(figsize=(12, 6))
		for phase, label in zip(phase_names, phase_labels, strict=True):
			values = np.asarray([_safe_float(row.get(phase)) for row in rows], dtype=float)
			plt.bar(x, values, bottom=bottom, label=label)
			bottom += values
		plt.xticks(x, [str(row["mesh_label"]) for row in rows], rotation=18, ha="right")
		plt.ylabel("Temps (s)")
		plt.title(f"Repartition des phases - Main opti {dim}D")
		plt.grid(axis="y", alpha=0.25)
		plt.legend()
		save_current(f"phases_main_opti_{dim}d.png")

	if global_rows:
		all_rows = [row for row in global_rows if str(row["group"]) == "all"]
		if all_rows:
			all_rows = sorted(all_rows, key=lambda row: int(row["version_order"]))
			labels = [str(row["version_label"]) for row in all_rows]
			colors = [version_colors.get(str(row["version_id"])) for row in all_rows]
			x = np.arange(len(all_rows))

			plt.figure(figsize=(10, 6))
			plt.bar(x, [_safe_float(row["total_measured_compute"]) for row in all_rows], color=colors)
			plt.xticks(x, labels, rotation=15, ha="right")
			plt.ylabel("Temps calcule cumule (s)")
			plt.title("Temps global cumule sur tous les maillages")
			plt.grid(axis="y", alpha=0.25)
			save_current("global_temps_total.png")

			plt.figure(figsize=(10, 6))
			plt.bar(x, [_safe_float(row["speedup_vs_main_sans_rien"]) for row in all_rows], color=colors)
			plt.axhline(1.0, color="#222222", linewidth=1.0)
			plt.xticks(x, labels, rotation=15, ha="right")
			plt.ylabel("Acceleration globale (x)")
			plt.title("Amelioration globale vs main sans optimisation")
			plt.grid(axis="y", alpha=0.25)
			save_current("global_speedup.png")

		dim_rows = [row for row in global_rows if str(row["group"]) != "all"]
		if dim_rows:
			groups = sorted({str(row["group"]) for row in dim_rows})
			x = np.arange(len(groups))
			width = 0.18
			plt.figure(figsize=(10, 6))
			for idx, version_id in enumerate(version_ids):
				values = [
					_safe_float(next((row["speedup_vs_main_sans_rien"] for row in dim_rows if str(row["group"]) == group and str(row["version_id"]) == version_id), 0.0))
					for group in groups
				]
				offset = (idx - (len(version_ids) - 1) / 2.0) * width
				plt.bar(x + offset, values, width, label=version_labels[version_id], color=version_colors.get(version_id))
			plt.axhline(1.0, color="#222222", linewidth=1.0)
			plt.xticks(x, groups)
			plt.ylabel("Acceleration globale par dimension (x)")
			plt.title("Amelioration globale 2D / 3D")
			plt.grid(axis="y", alpha=0.25)
			plt.legend()
			save_current("global_speedup_par_dimension.png")

	return outputs


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Benchmark des versions FEM et export CSV comparatif.")
	parser.add_argument("--suite", choices=["all", "2d", "3d", "small"], default="all", help="Suite de maillages benchmark a lancer.")
	parser.add_argument("--mesh", type=str, default=None, help="Maillage unique a utiliser au lieu de la suite.")
	parser.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Dimension du benchmark.")
	parser.add_argument("--steps", type=int, default=20, help="Nombre de steps par run, ou plafond si --run-seconds est utilise.")
	parser.add_argument("--sub-steps", dest="sub_steps", type=int, default=1, help="Sous-steps par step.")
	parser.add_argument("--run-seconds", type=float, default=None, help="Duree murale cible de chaque run; 180 correspond a 3 minutes.")
	parser.add_argument("--dt", type=float, default=None, help="Pas de temps optionnel.")
	parser.add_argument("--theta", type=float, default=None, help="Theta optionnel.")
	parser.add_argument("--repeats", type=int, default=1, help="Nombre de repetitions par version.")
	parser.add_argument("--out-dir", type=str, default=None, help="Dossier de sortie. Par defaut perf/results/<timestamp>.")
	parser.add_argument("--only", nargs="*", default=None, choices=[v.id for v in VERSIONS], help="Limiter a certaines versions.")
	parser.add_argument("--no-plots", dest="plots", action="store_false", help="Ne pas generer les graphiques PNG.")
	parser.set_defaults(plots=True)
	return parser


def main() -> None:
	args = build_parser().parse_args()
	run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "perf" / "results" / run_name
	run_dir.mkdir(parents=True, exist_ok=True)

	selected = [v for v in VERSIONS if args.only is None or v.id in set(args.only)]
	meshes = _select_meshes(args)
	_validate_meshes(meshes)
	raw_rows: list[dict[str, object]] = []
	status_rows: list[dict[str, object]] = []
	mesh_rows: list[dict[str, object]] = []

	for mesh in meshes:
		n_nodes, n_elements = _mesh_counts(mesh)
		mesh_rows.append(
			{
				"mesh_id": mesh.id,
				"mesh_label": mesh.label,
				"mesh_dim": mesh.dim,
				"mesh_category": mesh.category,
				"mesh_path": str(mesh.path.relative_to(PROJECT_ROOT)) if mesh.path.is_relative_to(PROJECT_ROOT) else str(mesh.path),
				"nodes": n_nodes,
				"elements": n_elements,
			}
		)

	for repeat_idx in range(1, max(1, args.repeats) + 1):
		for mesh in meshes:
			for version in selected:
				print(f"[perf] {mesh.label} - repeat {repeat_idx}/{args.repeats} - {version.label}")
				timing_rows, status = _run_one(version, mesh, repeat_idx, args, run_dir)
				raw_rows.extend(timing_rows)
				status_rows.append(status)
				if int(status["returncode"]) != 0:
					print(f"[perf] ECHEC {mesh.id}/{version.id}, voir {status['stderr']}")

	summary_rows = _summarize(raw_rows, status_rows)
	comparison = _comparison_rows(summary_rows)
	phase_comparison = _phase_comparison_rows(raw_rows)
	global_summary = _global_summary_rows(summary_rows, group_by_dim=False) + _global_summary_rows(summary_rows, group_by_dim=True)

	_write_csv(
		run_dir / "raw_timings_combined.csv",
		raw_rows,
		["run_id", "mesh_id", "mesh_label", "mesh_dim", "mesh_category", "repeat", "version_order", "version_id", "version_label", "phase", "seconds", "frame", "details", "source_row"],
	)
	_write_csv(
		run_dir / "mesh_suite.csv",
		mesh_rows,
		["mesh_id", "mesh_label", "mesh_dim", "mesh_category", "mesh_path", "nodes", "elements"],
	)
	_write_csv(
		run_dir / "run_status.csv",
		status_rows,
		["run_id", "mesh_id", "mesh_label", "mesh_dim", "mesh_category", "mesh_path", "repeat", "version_order", "version_id", "version_label", "description", "script", "returncode", "wall_seconds", "timings_csv", "stdout", "stderr", "command"],
	)
	_write_csv(
		run_dir / "summary_by_run.csv",
		summary_rows,
		[
			"version_order",
			"version_id",
			"version_label",
			"mesh_id",
			"mesh_label",
			"mesh_dim",
			"mesh_category",
			"repeat",
			"returncode",
			"wall_seconds",
			"measured_compute_total",
			"mesh_load",
			"system_assembly",
			"vertical_air_transfer_prepare",
			"initial_conditions",
			"frames_sum",
			"frames_avg",
			"frames_min",
			"frames_max",
			"frame_count",
			"headless_total",
			"burn_delta_updates",
			"node_freeze_updates",
			"element_activity_updates",
		],
	)
	_write_csv(
		run_dir / "comparison_summary.csv",
		comparison,
		[
			"version_order",
			"mesh_id",
			"mesh_label",
			"mesh_dim",
			"mesh_category",
			"version_id",
			"version_label",
			"best_repeat",
			"measured_compute_total",
			"headless_total",
			"wall_seconds",
			"mesh_load",
			"system_assembly",
			"initial_conditions",
			"frames_sum",
			"frames_avg",
			"speedup_vs_main_sans_rien",
			"gain_pct_vs_main_sans_rien",
			"gain_pct_vs_previous_version",
		],
	)
	_write_csv(
		run_dir / "comparison_by_phase.csv",
		phase_comparison,
		["mesh_id", "mesh_label", "mesh_dim", "mesh_category", "version_order", "version_id", "version_label", "phase", "best_seconds", "speedup_vs_main_sans_rien_same_phase", "gain_pct_vs_main_sans_rien_same_phase"],
	)
	_write_csv(
		run_dir / "global_summary.csv",
		global_summary,
		[
			"group",
			"version_order",
			"version_id",
			"version_label",
			"mesh_runs",
			"total_measured_compute",
			"avg_measured_compute",
			"total_wall_seconds",
			"avg_wall_seconds",
			"total_frames",
			"avg_frame_seconds",
			"speedup_vs_main_sans_rien",
			"gain_pct_vs_main_sans_rien",
			"gain_pct_vs_previous_version",
		],
	)
	plot_paths = _write_plots(run_dir, mesh_rows, comparison, global_summary) if args.plots else []

	print(f"[perf] CSV ecrits dans {run_dir}")
	if plot_paths:
		print(f"[perf] Graphiques ecrits dans {run_dir / PLOTS_DIRNAME}")


if __name__ == "__main__":
	main()
