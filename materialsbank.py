from __future__ import annotations


MATERIALS = {
	"bois": {
		"name": "Bois",
		"k": 0.20,
		"rho": 600.0,
		"c": 2000.0,
		"Q": 1.8e7,
		"Tc": 500.0,
		"hrr": 8.0e6,
		"hrr_duration": 1800.0,
		"color": "#A66A3F",
		"overlay_alpha_2d": 0.14,
		"overlay_alpha_3d": 0.08,
	},
	"beton": {
		"name": "Murs/Structure",
		"k": 1.40,
		"rho": 2200.0,
		"c": 1000.0,
		"Q": 1.2e7,
		"Tc": 520.0,
		"hrr": 0.0,
		"hrr_duration": 0.0,
		"color": "#8D939A",
		"overlay_alpha_2d": 0.14,
		"overlay_alpha_3d": 0.08,
	},
	"verre": {
		"name": "Verre",
		"k": 0.80,
		"rho": 2500.0,
		"c": 840.0,
		"Q": 0.0,
		"Tc": 2000.0,
		"hrr": 0.0,
		"hrr_duration": 0.0,
		"color": "#5FACC5",
		"overlay_alpha_2d": 0.10,
		"overlay_alpha_3d": 0.06,
	},
	"isolation": {
		"name": "Isolation",
		"k": 0.04,
		"rho": 30.0,
		"c": 1400.0,
		"Q": 0.0,
		"Tc": 2000.0,
		"hrr": 3.0e6,
		"hrr_duration": 900.0,
		"color": "#E8DFA8",
		"overlay_alpha_2d": 0.14,
		"overlay_alpha_3d": 0.08,
	},
	"metal": {
		"name": "Metal",
		"k": 45.0,
		"rho": 7850.0,
		"c": 500.0,
		"Q": 0.0,
		"Tc": 10000.0,
		"hrr": 0.0,
		"hrr_duration": 0.0,
		"color": "#B8BEC6",
		"overlay_alpha_2d": 0.16,
		"overlay_alpha_3d": 0.10,
	},
	"meche": {
		"name": "Meche",
		"k": 0.18,
		"rho": 900.0,
		"c": 1400.0,
		"Q": 3.0e6,
		"Tc": 520.0,
		"hrr": 1.2e7,
		"hrr_duration": 20.0,
		"color": "#4A3325",
		"overlay_alpha_2d": 0.18,
		"overlay_alpha_3d": 0.11,
	},
	"explosif": {
		"name": "Explosif",
		"k": 0.30,
		"rho": 1700.0,
		"c": 1000.0,
		"Q": 3.0e6,
		"Tc": 560.0,
		"hrr": 2.5e8,
		"hrr_duration": 0.6,
		"color": "#3B3530",
		"overlay_alpha_2d": 0.20,
		"overlay_alpha_3d": 0.12,
	},
	"air": {
		"name": "Air",
		"k": 0.5,#0.026,
		"rho": 1.2,
		"c": 1000.0,
		"Q": 0.0,
		"Tc": 10000.0,
		"hrr": 0.0,
		"hrr_duration": 0.0,
		"color": "#D7EEF8",
		"overlay_alpha_2d": 0.06,
		"overlay_alpha_3d": 0.04,
	},
}


def _make_burn_material(base_key: str, color: str):
	base = MATERIALS[base_key]
	return {
		**base,
		"name": f"{base['name']} brule",
		"k": float(base["k"]) * 0.65,
		"rho": float(base["rho"]) * 0.75,
		"c": float(base["c"]),
		"Q": 0.0,
		"Tc": float("inf"),
		"hrr": float(base["hrr"]),
		"hrr_duration": float(base["hrr_duration"]),
		"color": color,
		"overlay_alpha_2d": min(0.28, float(base["overlay_alpha_2d"]) + 0.08),
		"overlay_alpha_3d": min(0.18, float(base["overlay_alpha_3d"]) + 0.05),
	}


MATERIALS.update(
	{
		"bois_burn": _make_burn_material("bois", "#2B1B14"),
		"beton_burn": _make_burn_material("beton", "#4D5053"),
		"verre_burn": _make_burn_material("verre", "#2A5966"),
		"isolation_burn": _make_burn_material("isolation", "#6F6949"),
		"metal_burn": _make_burn_material("metal", "#555A60"),
		"meche_burn": _make_burn_material("meche", "#17110D"),
		"explosif_burn": _make_burn_material("explosif", "#0E0D0C"),
		"air_burn": _make_burn_material("air", "#8EA6AF"),
	}
)


def get_burn_material_name(name: str) -> str:
	"""Retourne la variante brulee du materiau si elle existe."""
	key = str(name).strip().lower()
	burn_key = key if key.endswith("_burn") else f"{key}_burn"
	return burn_key if burn_key in MATERIALS else key


def get_material(name: str):
	"""Retourne le dictionnaire du materiau associe au nom donne."""
	return MATERIALS.get(str(name).strip().lower(), MATERIALS["bois"])


def get_material_color(name: str) -> str:
	"""Retourne la couleur hex associee au materiau demande."""
	return str(get_material(name).get("color", "#A66A3F"))


def get_material_overlay_alpha(name: str, dim: int) -> float:
	"""Retourne une opacite de surcouche adaptee au materiau et a la dimension."""
	material = get_material(name)
	key = "overlay_alpha_2d" if int(dim) == 2 else "overlay_alpha_3d"
	return float(material.get(key, 0.12))
