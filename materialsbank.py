from __future__ import annotations


MATERIALS = {
	"bois": {
		"name": "Bois",
		"k": 0.20,
		"rho": 600.0,
		"c": 2000.0,
		"Q": 1.8e7,
		"Tc": 500.0,
		"hrr": 2.0e6,
		"hrr_duration": 1800.0,
		"color": "#A66A3F",
		"overlay_alpha_2d": 0.14,
		"overlay_alpha_3d": 0.08,
		"structural": True,
		"compressive_strength": 35.0e6,
		"tensile_strength": 4.0e6,
		"strength_temp_coeff": 0.0012,
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
		"structural": True,
		"compressive_strength": 25.0e6,
		"tensile_strength": 2.5e6,
		"strength_temp_coeff": 0.0018,
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
		"structural": True,
		"compressive_strength": 700.0e6,
		"tensile_strength": 45.0e6,
		"strength_temp_coeff": 0.0025,
	},
	"isolation": {
		"name": "Isolation",
		"k": 0.04,
		"rho": 30.0,
		"c": 1400.0,
		"Q": 0.0,
		"Tc": 2000.0,
		"hrr": 8.0e5,
		"hrr_duration": 900.0,
		"color": "#E8DFA8",
		"overlay_alpha_2d": 0.14,
		"overlay_alpha_3d": 0.08,
		"structural": False,
		"compressive_strength": 0.15e6,
		"tensile_strength": 0.03e6,
		"strength_temp_coeff": 0.0030,
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
		"structural": True,
		"compressive_strength": 250.0e6,
		"tensile_strength": 250.0e6,
		"strength_temp_coeff": 0.0010,
	},
	"ptfe": {
		"name": "acier",
		"k": 0.25,
		"rho": 2200.0,
		"c": 1000.0,
		"Q": 0.0,
		"Tc": 600.0,
		"hrr": 0.0,
		"hrr_duration": 0.0,
		"color": "#F4F5EE",
		"overlay_alpha_2d": 0.90,
		"overlay_alpha_3d": 0.86,
		"structural": False,
		"compressive_strength": 20.0e6,
		"tensile_strength": 25.0e6,
		"strength_temp_coeff": 0.0030,
	},
	"meche": {
		"name": "Meche",
		"k": 0.18,
		"rho": 900.0,
		"c": 1400.0,
		"Q": 3.0e6,
		"Tc": 520.0,
		"hrr": 3.0e6,
		"hrr_duration": 20.0,
		"color": "#4A3325",
		"overlay_alpha_2d": 0.18,
		"overlay_alpha_3d": 0.11,
		"structural": False,
		"compressive_strength": 0.5e6,
		"tensile_strength": 0.1e6,
		"strength_temp_coeff": 0.0030,
	},
	"explosif": {
		"name": "Explosif",
		"k": 0.30,
		"rho": 1700.0,
		"c": 1000.0,
		"Q": 3.0e6,
		"Tc": 560.0,
		"hrr": 2.5e7,
		"hrr_duration": 0.6,
		"color": "#3B3530",
		"overlay_alpha_2d": 0.20,
		"overlay_alpha_3d": 0.12,
		"structural": False,
		"compressive_strength": 0.2e6,
		"tensile_strength": 0.05e6,
		"strength_temp_coeff": 0.0040,
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
		"structural": False,
		"compressive_strength": 0.0,
		"tensile_strength": 0.0,
		"strength_temp_coeff": 0.0,
	},
	"viande": {
		"name": "Viande",
		"k": 0.45,
		"rho": 1050.0,
		"c": 3500.0,
		"Q": 2.5e6,
		"Tc": 570.0,
		"hrr": 6.0e5,
		"hrr_duration": 900.0,
		"color": "#B85C5A",
		"overlay_alpha_2d": 0.16,
		"overlay_alpha_3d": 0.10,
		"structural": False,
		"compressive_strength": 0.08e6,
		"tensile_strength": 0.02e6,
		"strength_temp_coeff": 0.0040,
	},
	"vegetation": {
		"name": "Vegetation",
		"k": 0.12,
		"rho": 350.0,
		"c": 1900.0,
		"Q": 1.0e7,
		"Tc": 480.0,
		"hrr": 1.2e6,
		"hrr_duration": 1200.0,
		"color": "#4F8A45",
		"overlay_alpha_2d": 0.16,
		"overlay_alpha_3d": 0.10,
		"structural": False,
		"compressive_strength": 0.5e6,
		"tensile_strength": 0.08e6,
		"strength_temp_coeff": 0.0035,
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
		"overlay_alpha_2d": min(0.95, float(base["overlay_alpha_2d"]) + 0.08),
		"overlay_alpha_3d": min(0.90, float(base["overlay_alpha_3d"]) + 0.05),
		"compressive_strength": float(base.get("compressive_strength", 0.0)) * 0.35,
		"tensile_strength": float(base.get("tensile_strength", 0.0)) * 0.25,
	}


MATERIALS.update(
	{
		"bois_burn": _make_burn_material("bois", "#2B1B14"),
		"beton_burn": _make_burn_material("beton", "#4D5053"),
		"verre_burn": _make_burn_material("verre", "#2A5966"),
		"isolation_burn": _make_burn_material("isolation", "#6F6949"),
		"metal_burn": _make_burn_material("metal", "#555A60"),
		"ptfe_burn": _make_burn_material("ptfe", "#D0D2C9"),
		"meche_burn": _make_burn_material("meche", "#17110D"),
		"explosif_burn": _make_burn_material("explosif", "#0E0D0C"),
		"air_burn": _make_burn_material("air", "#8EA6AF"),
		"viande_burn": _make_burn_material("viande", "#4A2523"),
		"vegetation_burn": _make_burn_material("vegetation", "#1F2B18"),
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
