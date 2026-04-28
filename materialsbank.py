# banque de matériaux centralisée
from __future__ import annotations

# "exemple":{"name":"Exemple", "k": conductivité float kg.m/s^3.K, "rho": masse volumique float kg/m^3, "c": chaleur spécifique float J/kg.K, "Q": chaleur latente float J/kg, "Tc": température de combustion float K}
MATERIALS = {
	"bois": {"name": "Bois", "k": 0.30, "rho": 500.0, "c": 1500.0, "Q": 1.0e6, "Tc": 480.0},
	"beton": {"name": "Beton", "k": 1.40, "rho": 2400.0, "c": 880.0, "Q": 0.0, "Tc": 2000.0},
	"verre": {"name": "Verre", "k": 0.80, "rho": 2500.0, "c": 840.0, "Q": 0.0, "Tc": 2000.0},
	"isolation": {"name": "Isolation", "k": 0.04, "rho": 30.0, "c": 1400.0, "Q": 0.0, "Tc": 2000.0},
	"air": {"name": "Air", "k": 0.03, "rho": 1.2, "c": 1000.0, "Q": 0.0, "Tc": 2000.0},
}

def get_material(name: str):
	"""Retourne le dictionnaire du materiau associe au nom donne.
	:param name: Nom du materiau ou alias
	:return: Dictionnaire des proprietes du materiau
	"""
	key = str(name).strip().lower()
	return MATERIALS.get(key, MATERIALS["bois"])