# Projet FEM - Diffusion Thermique 2D/3D

## Vue d'ensemble

Ce projet simule une diffusion thermique transitoire en 2D et 3D par éléments finis, avec :

- conduction dans des matériaux heterogenes,
- convection vers une ambiance a température imposée,
- pertes volumiques generales, ventilation et rayonnement,
- source de combustion activee au-dessus d'une temperature seuil,
- animation de la température dans le maillage,
- export du setup initial, de l'animation et de timings CSV.

Le point d'entrée principal est `main.py`.

Une version experimentale orientee optimisation est disponible dans `main_elementwise.py`. Elle preassemble les contributions element par element et applique seulement les deltas normal -> brule quand un triangle/tetraedre bascule.

La gestion 3D est deja prise en charge par `main.py` et les maillages du dossier `models/`. Les modules du dossier `calculs/` n'ont pas ete modifies, car ils supportent deja la 3D.

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

Forcer la version 3D :

```bash
python main.py --3d
```

En 3D, le maillage utilise par defaut `models/immeuble.msh` ; en 2D, `models/piece.msh` reste le choix par defaut.

Tester la version element par element :

```bash
python main_elementwise.py
python main_elementwise.py --3d
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
| `--2d` | flag | `False` | Force l'execution en 2D |
| `--3d` | flag | `False` | Force l'execution en 3D |
| `--dt` | float | `10.0` | Pas de temps |
| `--steps` | int | `2000` | Nombre de frames / pas affiches |
| `--sub-steps` | int | `1` | Sous-iterations de calcul par frame |
| `--theta` | float | `1.0` | Schema theta, `1.0` = Euler implicite |
| `--h-conv` | float | `10.0` | Coefficient de convection de frontiere [W/m2/K] |
| `--general-loss` | float | `0.2` | Perte volumique lineaire generale [W/m3/K] |
| `--vent-loss` | float | `1.0` | Perte volumique de ventilation [W/m3/K] |
| `--radiation-loss` | float | `5.0e-8` | Coefficient radiatif volumique [W/m3/K4] |
| `--vertical-air-transfer` | int | `1` en 3D | Active le transfert vertical simplifie de HRR |
| `--vertical-air-attenuation` | float | `0.25` | Perte du transfert vertical par metre [1/m] |
| `--vertical-air-radius` | float | `0.0` | Rayon horizontal du transfert vertical, `0=auto` |
| `--vertical-air-random-delta` | float | `0.0` | Variation aleatoire de l'attenuation verticale a chaque sous-pas |
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

Les valeurs HRR actuelles sont des ordres de grandeur effectifs pour le modele volumique :

- `bois` : `hrr = 8.0e6 W/m3`, `hrr_duration = 1800 s`,
- `isolation` : `hrr = 3.0e6 W/m3`, `hrr_duration = 900 s`,
- `beton`, `verre`, `air` : `hrr = 0`, non combustibles dans ce modele.

Signification :

- `k` : conductivite thermique,
- `rho*c` : inertie thermique volumique,
- `Q` : source de combustion locale,
- `Tc` : seuil d'activation de la combustion.
- `hrr` : heat release rate volumique maximal [W/m3].
- `hrr_duration` : duree effective de degagement HRR [s].

Chaque materiau possede aussi une variante `*_burn` dans la banque de donnees. Pendant la simulation, chaque element garde un etat `pas brule / brule` : en 2D chaque triangle qui depasse le `Tc` de son materiau devient noir ; en 3D chaque tetraedre qui depasse le `Tc` devient noir. Quand un element brule, son materiau courant passe en variante `*_burn`, les matrices thermiques prennent les caracteristiques cramees, et la chaleur degagee vient de la loi HRR du materiau.

## Theorie de diffusion utilisee

### Equation continue

Le modele thermique exploite dans `main.py` correspond a une equation de diffusion-reaction-convection de type :

```math
\rho c \frac{\partial T}{\partial t}
- \nabla \cdot (k \nabla T)
+ a_v (T - T_{amb})
+ a_r (T^4 - T_{amb}^4)
= q(x)\,H(T - T_c)
```

avec une condition de bord de type Robin :

```math
-k \nabla T \cdot n = h_b (T - T_{amb})
```

avec :

- `T(x,t)` : temperature,
- `rho c` : capacite thermique volumique,
- `k` : conductivite thermique,
- `-div(k grad T)` : diffusion / conduction thermique,
- `h_b (T - T_amb)` : echange convectif sur la frontiere,
- `a_v (T - T_amb)` : pertes volumiques lineaires generales et ventilation,
- `a_r (T^4 - T_amb^4)` : perte radiative volumique,
- `q(x)` : intensite de la source de combustion,
- `H(T - Tc)` : activation de type Heaviside, egale a `1` quand `T >= Tc`, sinon `0`.

### Sens physique des termes

- `grad T` mesure la pente locale de temperature.
- `k grad T` represente le flux thermique de conduction.
- `div(k grad T)` mesure combien ce flux entre ou sort localement.
- `h_b (T - T_amb)` refroidit le solide par sa frontiere.
- `a_v (T - T_amb)` simule les pertes reparties dans tout le volume.
- `a_r (T^4 - T_amb^4)` devient dominant a haute temperature.
- `HRR(x,t)` injecte l'energie degagee par les elements deja crames.
- En 3D, le HRR peut aussi etre transporte vers les elements au-dessus avec un facteur simplifie `max(0, 1 - 0.25 dz)`.
- `--vertical-air-random-delta d` multiplie l'attenuation verticale a chaque sous-pas par un facteur aleatoire dans `[1-d, 1+d]`.

## Equation FEM utilisee dans `main.py`

### Forme matricielle

Le code resout a chaque pas :

```math
M \frac{T^{n+1} - T^n}{\Delta t}
+
(K + B + A_v M_u) T^{n+1}
=
H_{RR}(T^n) + B T_{amb} + A_v M_u T_{amb} - R(T^n)
```

Dans le code, cela correspond a :

```python
k_eff = k_mat + bc_field.loss_matrix + volume_loss.loss_matrix
src = _hrr_source_rhs(system, elems, burned_elements, t_local, burn_times, sim_time)
radiation_loss_rhs = _radiation_loss_rhs(m_unit, t_local, volume_loss)
rhs = src + bc_field.rhs + volume_loss.rhs - radiation_loss_rhs
t = theta_step(m_mat, k_eff, rhs, rhs, t, dt=dt, theta=theta, ...)
```

### Signification des objets du code

- `t` : vecteur de temperatures nodales.
- `dt` : pas de temps.
- `m_mat` : matrice de masse thermique globale.
- `k_mat` : matrice de conduction globale.
- `m_unit` : matrice de masse unitaire, utilisee pour projeter les termes volumiques.
- `bc_field` : champ de condition limite de frontiere.
- `volume_loss` : pertes volumiques lineaires et radiatives.
- `burned_elements` : flag d'etat crame par element.
- `burn_times` : temps auquel chaque element est devenu crame.
- `src = _hrr_source_rhs(...)` : second membre HRR assemble element par element.
- `bc_field.rhs` : terme d'ambiance applique sur la frontiere.
- `volume_loss.rhs` : terme d'ambiance applique dans le volume.
- `radiation_loss_rhs` : perte radiative explicite calculee a partir de la temperature courante.
- `k_eff` : operateur conduction + pertes lineaires.

### Forme implicite effectivement resolue

Comme `theta = 1.0` par defaut, `theta_step(...)` applique Euler implicite :

```math
\left(\frac{M}{\Delta t} + K + B + A_v M_u \right) T^{n+1}
=
\frac{M}{\Delta t} T^n + H_{RR}(T^n) + B T_{amb} + A_v M_u T_{amb} - R(T^n)
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

