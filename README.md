# Projet FEM - Diffusion Thermique 2D par Éléments Finis

## Vue d'ensemble

Ce projet simule la diffusion thermique avec source de reaction (type combustion) en 2D et 3D par elements finis.

Le point d'entree est `main.py`, qui:
- charge un maillage dans `models/`,
- recupere les proprietes materiaux dans `materialsbank.py`,
- utilise le code de base enseignant dans `calculs/`,
- lance le calcul transitoire en schema theta.

## Structure du projet

```
FEM-project/
├── main.py                      # Point d'entrée principal (refactorisé)
├── diffusion_2D_fem.py          # Script simple didactique
├── materialsbank.py             # Banque de matériaux
├── README.md
├── requirements.txt
├── models/                      # Géométries et maillages
│   ├── piece.geo
│   ├── piece.msh               # Maillage 2D simple (1611 nœuds, 3084 triangles)
│   ├── immeuble.geo
│   └── immeuble.msh            # Maillage 2D complexe (building)
├── calculs/                     # Code base fourni (enseignant)
│   ├── dirichlet.py
│   ├── errors.py
│   ├── gmsh_utils.py
│   ├── mass.py
│   ├── stiffness.py
│   ├── plot_utils.py
│   └── main_diffusion_1d.py
├── tracedesancienscodepourmesurerperf/  # Anciennes versions pour benchmark
└── old/                         # base Martin
```

## Installation et dépendances

Installer les dépendances :
```bash
pip install -r requirements.txt
```

Dépendances principales :
- `numpy` : calcul matriciel
- `scipy` : solveurs creux (sparse solvers)
- `matplotlib` : visualisation et animation
- `meshio` : lecture/écriture de maillages
- `ffmpeg` (système) : optionnel, pour exporter les animations en MP4

Vérifier l'installation de ffmpeg :
```bash
ffmpeg -version
```

## Utilisation rapide

### Option 1 : Script simple (`diffusion_2D_fem.py`)

Pour un lancement simple et rapide :
```bash
python FEM-project/diffusion_2D_fem.py
```

**Affiche** :
- Animation FuncAnimation en temps réel avec Tmax et temps simulé mis à jour.
- Conditions : maillage `piece.msh`, 1000 frames, 10 sous-itérations par frame.

### Option 2 : Script principal avec options (`main.py`)

Lancement par défaut :
```bash
python FEM-project/main.py
```

Avec options (exemples) :
```bash
# Changer le maillage
python FEM-project/main.py --mesh immeuble.msh

# Réduire le nombre de frames (pour aller plus vite)
python FEM-project/main.py --steps 100 --sub-steps 2

# Augmenter le pas de temps (dt)
python FEM-project/main.py --dt 5.0

# Personnaliser les paramètres physiques
python FEM-project/main.py --src-temp 1000 --src-radius 0.1

# Sauvegarder l'animation en MP4 (nécessite ffmpeg)
python FEM-project/main.py --save simulation.mp4

# Sauvegarder sans affichage interactif
python FEM-project/main.py --save simulation.mp4 --no-plot

# Combinaison : accélérer ET sauvegarder
python FEM-project/main.py --steps 100 --sub-steps 5 --save simulation_fast.mp4
```

## Options CLI de `main.py`

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--mesh` | str | `piece.msh` | Nom du maillage dans `models/` |
| `--steps` | int | 50 | Nombre de frames d'animation |
| `--sub-steps` | int | 1 | Sous-itérations par frame (augmente précision/temps) |
| `--dt` | float | 2.0 | Pas de temps [s] |
| `--h-conv` | float | 1.0 | Coefficient de convection [W/m²K] |
| `--t-amb` | float | 293.0 | Température ambiante [K] |
| `--src-temp` | float | 800.0 | Température initiale de la source [K] |
| `--src-x`, `--src-y` | float | 0.0 | Position X,Y de la source [m] |
| `--src-radius` | float | 0.05 | Rayon de la source chauffante [m] |
| `--save` | str | None | Nom du fichier MP4 pour sauvegarder (ex: `sim.mp4`) |
| `--no-plot` | flag | False | Désactiver l'affichage interactif (utilise avec `--save`) |

## Paramètres physiques

Les matériaux sont définis dans `materialsbank.py` :

| Matériau | k [W/mK] | ρc [J/m³K] | Q [W/m³] | Tc [K] |
|----------|----------|-----------|----------|--------|
| Bois | 5.0 | 480 000 | 1.0e6 | 480 |
| Béton | 1.0 | 500 000 | 1.2e6 | 450 |
| Air | 2.0 | 1 200 | 0.0 | 2000 |

Où :
- `k` : conductivité thermique
- `ρc` : capacité thermique volumique
- `Q` : terme source de chaleur (combustion)
- `Tc` : seuil de combustion (H>0 si T≥Tc)

## Schéma numérique

Le projet utilise **Euler implicite** (θ=1) pour la discrétisation temporelle :

$$M \frac{T^{n+1} - T^n}{\Delta t} + K T^{n+1} = Q^n$$

Où :
- M : matrice de masse (inertie thermique)
- K : matrice de raideur (conduction + convection)
- Q : vecteur source (combustion)
- Δt : pas de temps

## Exemple de workflow

### 1. Simulation rapide et sauvegarde MP4

```bash
python FEM-project/main.py --steps 50 --sub-steps 2 --save output.mp4
```

Génère `output.mp4` en ~30 secondes (dépend de ffmpeg).

### 2. Étude de sensibilité sur le rayon source

```bash
for r in 0.05 0.1 0.2; do
  python FEM-project/main.py --src-radius $r --save sim_r${r}.mp4 --no-plot
done
```

### 3. Comparaison visuelle : simple vs main

```bash
# Terminal 1 : simple
python FEM-project/diffusion_2D_fem.py

# Terminal 2 : main.py avec les mêmes paramètres
python FEM-project/main.py --steps 100 --sub-steps 10 --dt 4.0
```

## Architecture et améliorations

### Améliorations apportées à `main.py`

1. **FuncAnimation robuste** :
   - Passage d'état explicite via dictionnaire pour éviter les problèmes de portée.
   - Lambda pour wrapper le callback.
   - `blit=False` pour garantir le rafraîchissement complet à chaque frame.

2. **Sauvegarde MP4** :
   - Utilise `FFMpegWriter` de matplotlib.
   - FPS configurable (15 fps par défaut).
   - Gestion d'erreur si ffmpeg n'est pas installé.

3. **Code modularisé** :
   - `_load_mesh_data()` : chargement robuste du maillage via meshio.
   - `_assemble_system()` : assemblage FEM.
   - `_element_matrices_2d()` : matrices élémentaires (rigidité, masse).
   - Gestion des Physical IDs via fonction dédiée.

4. **CLI flexible** :
   - Arguments pour tous les paramètres physiques et numériques.
   - Defaults sensés pour lancement rapide.

### Améliorations apportées à `diffusion_2D_fem.py`

1. **Affichage temps/Tmax correct** :
   - Variable `_sim_time` cumulée à chaque frame.
   - Format : "Temps : X.X s | Tmax : Y.Y K".

2. **FuncAnimation simple** :
   - Utilise `global T, _sim_time` pour gérer l'état.
   - Blit=True pour performance (compatible avec ce script simple).

3. **Didactique** :
   - Code court (~115 lignes avec commentaires).
   - Facile à comprendre et modifier pour l'enseignement.

## Dépannage

### Animation figée (n'affiche que la première frame)

**Cause** : `blit=True` avec artists complexes ou portée de variable cassée.

**Solution** : 
- Vérifier que `--no-plot` n'est pas actif.
- Utiliser `main.py` qui a la gestion d'état correcte.

### Erreur "ffmpeg not found"

**Cause** : ffmpeg n'est pas installé ou pas dans le PATH.

**Solutions** :
1. Installer ffmpeg :
   - Windows : `choco install ffmpeg` ou télécharger depuis ffmpeg.org
   - Linux : `sudo apt install ffmpeg`
   - macOS : `brew install ffmpeg`

2. Ou utiliser sans sauvegarde MP4 :
   ```bash
   python FEM-project/main.py  # affiche seulement
   ```

### Lenteur de la simulation

**Causes** : `sub_steps` grand, `dt` petit, maillage fin.

**Solutions** :
- Réduire `--steps` pour moins de frames.
- Augmenter `--dt` pour des pas plus grands.
- Réduire `--sub-steps` (minimum 1).

## Résultats attendus

Pour le maillage `piece.msh` avec paramètres par défaut :

- **Temps de calcul** : ~1-2 min (50 frames × 1 sub_step).
- **Évolution Tmax** : 293 K (ambiant) → ~970 K (après 50 frames × 2s).
- **Visualisation** : gradient de température visible, combustion en zone source, diffusion vers les bords.
- **Sauvegarde MP4** : ~3-5 MB pour 50 frames.

## Notes techniques

- Schéma numérique : Euler implicite (stable inconditionnellement).
- Maillage : triangles P1 (linéaires), générés par Gmsh.
- Solveur : SparseLU (scipy.sparse.linalg.spsolve).
- Animation : FuncAnimation de matplotlib avec rendu Gouraud.
- Backend matplotlib : auto (utilise le backend par défaut du système).

## Auteur et contact

Projet FEM - Université (Q6).

Pour questions ou améliorations : vérifier les issues ou discuter en équipe.
