# Projet FEM - Diffusion/Reaction thermique 2D/3D

## But du projet

Ce projet simule la diffusion thermique avec source de reaction (type combustion) en 2D et 3D par elements finis.

Le point d'entree est `main.py`, qui:
- charge un maillage dans `models/`,
- recupere les proprietes materiaux dans `materialsbank.py`,
- utilise le code de base enseignant dans `calculs/`,
- lance le calcul transitoire en schema theta.

## Structure du projet

```text
FEM-project/
	main.py                 # Point d'entree unique 2D/3D
	materialsbank.py        # Banque de materiaux centralisee
	models/                 # Fichiers de geometrie/maillage
		piece.geo
		piece.msh
		immeuble.geo
		immeuble.msh
	calculs/                   # Code de base fourni (enseignant)
		dirichlet.py
		errors.py
		gmsh_utils.py
		mass.py
		stiffness.py
		plot_utils.py
		main_diffusion_1d.py
		main_diffusion_2d.py
	trace.../               # contient tout les différents mains, pour comparer les optis entre elles

```

## Commandes principales

### 1) Calcul 2D (maillage `piece.msh`)

```bash
python main.py --dim 2
```

### 2) Calcul 3D (maillage `immeuble.msh`)

```bash
python main.py --dim 3
```

### 3) Lancer sans affichage (utile pour test rapide)

```bash
python main.py --dim 2 --steps 20 --no-plot
python main.py --dim 3 --steps 20 --no-plot
```

### 4) Exemples de parametres

```bash
python main.py --dim 2 --dt 2.0 --steps 400 --theta 1.0 --h-conv 30
python main.py --dim 3 --dt 5.0 --steps 300 --src-x 1 --src-y 1 --src-z 0 --src-box 1.0
```
