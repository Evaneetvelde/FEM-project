from __future__ import annotations


MATERIALS = {
	"bois": {
		"name": "Bois",
		"k": 0.30,
		"rho": 500.0,
		"c": 1500.0,
		"Q": 1.0e6,
		"Tc": 480.0,
		"color": "#A66A3F",
		"overlay_alpha_2d": 0.14,
		"overlay_alpha_3d": 0.08,
	},
	"beton": {
		"name": "Beton",
		"k": 1.40,
		"rho": 2400.0,
		"c": 880.0,
		"Q": 0.0,
		"Tc": 2000.0,
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
		"color": "#E8DFA8",
		"overlay_alpha_2d": 0.14,
		"overlay_alpha_3d": 0.08,
	},
	"air": {
		"name": "Air",
		"k": 0.03,
		"rho": 1.2,
		"c": 1000.0,
		"Q": 0.0,
		"Tc": 2000.0,
		"color": "#D7EEF8",
		"overlay_alpha_2d": 0.06,
		"overlay_alpha_3d": 0.04,
	},
}
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
