# Projet FEM - Diffusion Thermique 2D

## Vue d'ensemble

Ce projet simule une diffusion thermique transitoire en 2D par éléments finis, avec :

- conduction dans des matériaux heterogenes,
- convection vers une ambiance a température imposée,
- source de combustion activee au-dessus d'une temperature seuil,
- animation de la température dans le maillage,
- export du setup initial, de l'animation et de timings CSV.

Le point d'entrée principal est `main.py`.

## Structure du projet

```text
FEM-project/
|-- main.py
|-- materialsbank.py
|-- README.md
|-- requirements.txt
|-- models/
|   |-- piece.geo
|   |-- piece.msh
|   |-- immeuble.geo
|   `-- immeuble.msh
|-- calculs/
|   |-- dirichlet.py
|   |-- errors.py
|   |-- gmsh_utils.py
|   |-- mass.py
|   |-- plot_utils.py
|   `-- stiffness.py
|-- old/
`-- tracedesancienscodepourmesurerperf/
```

## Installation

```bash
pip install -r requirements.txt    
ffmpeg -version                    #pour la sauvegarde d'animation
```

## Utilisation rapide

Lancement interactif :

```bash
python main.py
```

Test rapide :

```bash
python main.py --steps 20 --sub-steps 1
```

Export sans affichage interactif :

```bash
python main.py --steps 20 --save run1.mp4 --no-plot
```

Avec `--save`, le script cree un dossier de sortie contenant :

- un PNG du setup initial,
- l'animation exportee,
- un `timings.csv`.

Si tu fais :

```bash
python main.py --save run1.mp4 --no-plot
```

alors le dossier cree sera :

```text
run1/
|-- run1.mp4
|-- setup_initial.png
`-- timings.csv
```

## Options CLI

| Option | Type | Defaut | Description |
|---|---|---:|---|
| `--mesh` | str | `piece.msh` | Maillage dans `models/` |
| `--dt` | float | `10.0` | Pas de temps |
| `--steps` | int | `2000` | Nombre de frames / pas affiches |
| `--sub-steps` | int | `1` | Sous-iterations de calcul par frame |
| `--theta` | float | `1.0` | Schema theta, `1.0` = Euler implicite |
| `--h-conv` | float | `1.0` | Coefficient de convection |
| `--t-amb` | float | `293.0` | Temperature ambiante [K] |
| `--src-temp` | float | `800.0` | Temperature initiale de la source [K] |
| `--src-x` | float | `0.0` | Position X de la source |
| `--src-y` | float | `0.0` | Position Y de la source |
| `--src-radius` | float | `0.05` | Rayon de la source initiale |
| `--save` | str | `None` | Nom du MP4 ou dossier de sortie |
| `--no-plot` | flag | `False` | Desactive l'affichage interactif |

## Materiaux

Les materiaux sont definis dans `materialsbank.py`.

| Materiau | k [W/m.K] | rho [kg/m3] | c [J/kg.K] | rho*c [J/m3.K] | Q | Tc [K] | Couleur |
|---|---:|---:|---:|---:|---:|---:|---|
| Bois | 0.30 | 500.0 | 1500.0 | 750000 | 1.0e6 | 480.0 | brun |
| Beton | 1.40 | 2400.0 | 880.0 | 2112000 | 0.0 | 2000.0 | gris |
| Verre | 0.80 | 2500.0 | 840.0 | 2100000 | 0.0 | 2000.0 | bleu clair |
| Isolation | 0.04 | 30.0 | 1400.0 | 42000 | 0.0 | 2000.0 | beige |
| Air | 0.03 | 1.2 | 1000.0 | 1200 | 0.0 | 2000.0 | bleu tres pale |

Signification :

- `k` : conductivite thermique,
- `rho*c` : inertie thermique volumique,
- `Q` : source de combustion locale,
- `Tc` : seuil d'activation de la combustion.

## Theorie de diffusion utilisee

### Equation continue

Le modele thermique exploite dans `main.py` correspond a une equation de diffusion-reaction-convection de type :

```math
\rho c \frac{\partial T}{\partial t}
- \nabla \cdot (k \nabla T)
+ h (T - T_{amb})
= q(x)\,H(T - T_c)
```

avec :

- `T(x,t)` : temperature,
- `rho c` : capacite thermique volumique,
- `k` : conductivite thermique,
- `-div(k grad T)` : diffusion / conduction thermique,
- `h (T - T_amb)` : echange convectif avec l'air ambiant,
- `q(x)` : intensite de la source de combustion,
- `H(T - Tc)` : activation de type Heaviside, egale a `1` quand `T >= Tc`, sinon `0`.

### Sens physique des termes

- `grad T` mesure la pente locale de temperature.
- `k grad T` represente le flux thermique de conduction.
- `div(k grad T)` mesure combien ce flux entre ou sort localement.
- `h (T - T_amb)` refroidit le solide quand il est plus chaud que l'ambiance, et le rechauffe s'il est plus froid.
- `q(x) H(T - Tc)` injecte de l'energie seulement si le seuil local de combustion est depasse.

## Equation FEM utilisee dans `main.py`

### Forme matricielle

Le code resout a chaque pas :

```math
M \frac{T^{n+1} - T^n}{\Delta t}
+
(K + h M_u) T^{n+1}
=
S(T^n) + h M_u T_{amb}
```

Dans le code, cela correspond a :

```python
k_eff = k_mat + h_conv * m_unit
src = m_unit.dot(q_node * h_act)
conv = h_conv * m_unit.dot(ones)
rhs = src + conv
t = theta_step(m_mat, k_eff, rhs, rhs, t, dt=dt, theta=theta, ...)
```

### Signification des objets du code

- `t` : vecteur de temperatures nodales.
- `dt` : pas de temps.
- `m_mat` : matrice de masse thermique globale.
- `k_mat` : matrice de conduction globale.
- `m_unit` : matrice de masse unitaire, utilisee pour projeter les termes volumiques.
- `h_conv` : coefficient de convection.
- `ones = np.full(len(t), t_amb)` : vecteur contenant la temperature ambiante.
- `q_node` : source locale de combustion par noeud.
- `tc_node` : seuil local de combustion par noeud.
- `h_act = (t >= tc_node).astype(float)` : active la combustion localement.
- `src = m_unit.dot(q_node * h_act)` : second membre de combustion.
- `conv = h_conv * m_unit.dot(ones)` : terme de convection vers l'ambiance.
- `k_eff = k_mat + h_conv * m_unit` : operateur conduction + convection.

### Forme implicite effectivement resolue

Comme `theta = 1.0` par defaut, `theta_step(...)` applique Euler implicite :

```math
\left(\frac{M}{\Delta t} + K + h M_u \right) T^{n+1}
=
\frac{M}{\Delta t} T^n + S(T^n) + h M_u T_{amb}
```

## Assemblage FEM

Les matrices globales sont construites dans `_assemble_system()` en s'appuyant sur `calculs/` :

- `assemble_mass(...)` pour la masse,
- `assemble_stiffness_and_rhs(...)` pour la raideur de conduction,
- `theta_step(...)` pour l'avance temporelle.

Pipeline principal :

1. lecture du maillage via `meshio`,
2. recuperation des triangles et des IDs physiques,
3. construction d'une quadrature P1 triangle,
4. assemblage de `m_unit`,
5. assemblage de `m_mat` par groupe de materiau via `rho * c`,
6. assemblage de `k_mat` par groupe de materiau via `k`,
7. calcul des vecteurs `q_node` et `tc_node`.

## Affichage et animation

Le `main.py` :

- affiche un setup initial,
- colore legerement les zones de materiaux,
- redessine les murs en beton de facon opaque par-dessus,
- anime ensuite le champ de temperature.

Le temps affiche dans le titre correspond au temps simule cumule.

## Export et timings

Quand `--save` est utilise, le script exporte :

- `setup_initial.png`,
- l'animation MP4,
- `timings.csv`.

Le CSV contient notamment :

- `mesh_load`,
- `system_assembly`,
- `initial_conditions`,
- `initial_setup_figure`,
- `initial_setup_png_save`,
- `animation_figure`,
- `animation_save`.

## Resultats 
à observer nous même

