# Projet FEM - diffusion-reaction thermique 2D/3D

Ce projet simule la propagation de chaleur dans un domaine 2D ou 3D solide par elements finis. Il gere plusieurs materiaux, une source chaude initiale, l'allumage des elements combustibles, des pertes thermiques simplifiees, un transport vertical simplifie en 3D, et une visualisation animee.

Le point d'entree de reference est `main.py`. C'est la version optimisee actuelle.

## Table des matieres

- [Arborescence du projet](#arborescence-du-projet)
- [Installation](#installation)
- [Utilisation rapide](#utilisation-rapide)
- [Versions](#versions)
- [Options](#options)
- [Equation physique de notre modele](#equation-physique-de-notre-modele)
- [Equation numerique de notre modele](#equation-numerique-de-notre-modele)
- [Simplifications](#simplifications)
- [Optimisations apportées](#optimisations-apportées)
- [Ameliorations possibles](#ameliorations-possibles)
- [Sources](#sources)

## Arborescence du projet

```text
FEM-project/
|-- main.py                           # version principale optimisee
|-- materialsbank.py                  # banque des materiaux
|-- requirements.txt                  # dependances Python
|-- README.md                         # vous êtes ici
|-- calculs/                          # provient du template du code founi dans le cours
|   |-- dirichlet.py                  # schemas temporels et conditions Dirichlet
|   |-- mass.py                       # assemblage masse
|   `-- stiffness.py                  # assemblage raideur/conduction   
|-- models/
|   |-- piece.geo
|   |-- piece.msh
|   |-- immeuble.geo
|   |-- immeuble.msh
|   |-- bois-air-bois.geo
|   `-- bois-air-bois.msh
`-- old/
|   |-- mainnotopti.py                  # version de comparaison non optimisee
|   |-- maincalculmatriciel.py          # etape calcul matriciel/preassemblage
|   |-- mainfancyreassembling.py        # etape gel/degel et reassemblage selectif
|   |-- main_notelementwise_notopti.py  # version antérieur sans burn 
|   |-- diffusion_2D_fem.py             # prototype 2D
|   `-- diffusion_3D_fem.py             # prototype 3d
`-- perfs/
    `-- perfs.py                      # script testant les performances temporels des différents main
```

## Installation

Installer les dependances Python :

```bash
pip install -r requirements.txt
```

Dependances principales :

- `numpy`
- `scipy`
- `numba`
- `matplotlib`
- `meshio`
- `gmsh`

Pour exporter une animation MP4 avec `--save`, `ffmpeg` doit aussi etre disponible dans le `PATH` :

```bash
ffmpeg -version
```

## Utilisation rapide

Lancer la simulation 2D par defaut :

```bash
python main.py
```

Lancer la simulation 3D :

```bash
python main.py --3d
```

Utiliser un maillage precis :

```bash
python main.py --mesh bois-air-bois.msh --2d
```

## Versions

| Fichier | Role |
|---|---|
| `main.py` | Version principale. |
| `old/mainnotopti.py` | Version de comparaison non optimisee. Utile pour mesurer les gains. |
| `old/maincalculmatriciel.py` | Ancienne version orientee calcul matriciel, preassemblage et Numba. |
| `old/mainfancyreassembling.py` | Ancienne version orientee gel/degel et reassemblage selectif. |
| `old/main_notelementwise_notopti.py` | Ancienne version sans mise a jour element par element. (non utilisé dans perf) |
| `old/diffusion_2D_fem.py`, `old/diffusion_3D_fem.py` | Anciennes bases separees 2D/3D. |

## Options

### Options principales

| Option | Description |
|---|---|
| `--2d`, `--3d` | Force la dimension du calcul. |
| `--mesh` | Nom ou chemin du maillage `.msh`. |
| `--dt` | Pas de temps. |
| `--steps` | Nombre de frames ou pas affiches. |
| `--sub-steps` | Nombre de sous-pas de calcul par frame. |
| `--theta` | Schema theta. `1.0` correspond a Euler implicite. |
| `--h-conv` | Coefficient de convection sur la frontiere. |
| `--general-loss` | Perte volumique lineaire generale. |
| `--vent-loss` | Perte volumique de ventilation. |
| `--radiation-loss` | Coefficient radiatif volumique. |
| `--vertical-air-transfer` | Active le transfert vertical simplifie de HRR en 3D. |
| `--vertical-air-attenuation` | Attenuation du transfert vertical avec la hauteur. |
| `--vertical-air-radius` | Rayon horizontal du transfert vertical. `0` signifie automatique. |
| `--vertical-air-random-delta` | Variation aleatoire de l'attenuation verticale. |
| `--t-amb` | Temperature ambiante. |
| `--src-temp` | Temperature initiale de la source. |
| `--src-x`, `--src-y`, `--src-z` | Position de la source. |
| `--src-radius` | Rayon initial de la source. |
| `--no-plot` | Desactive l'affichage interactif. |
| `--hide-burned-elements` | Masque la couche noire des elements brules sans changer le calcul. |
| `--save` | Sauvegarde une animation MP4 et les timings. |

### Options d'optimisation

| Option | Description |
|---|---|
| `--element-freeze-steps` | Nombre de steps froids avant de ne plus retester un element. `0` desactive. |
| `--element-freeze-margin` | Marge sous `Tc` pour considerer un element froid/stable. |
| `--node-freeze-steps` | Nombre de steps quasi stationnaires avant de geler temporairement un noeud. `0` desactive. |
| `--node-freeze-delta` | Variation maximale de temperature par step pour geler un noeud. |
| `--node-freeze-margin` | Marge sous `Tc` requise pour geler un noeud. |
| `--node-thaw-delta` | Ecart de temperature avec un voisin declenchant le degel. |
| `--node-thaw-margin` | Marge sous `Tc` d'un voisin declenchant le degel. |
| `--max-frozen-node-fraction` | Fraction maximale de noeuds geles. |

### Materiaux et IDs physiques

Les materiaux sont definis dans `materialsbank.py`.

| ID | Materiau | Role typique |
|---:|---|---|
| 1 | `bois` | combustible solide |
| 2 | `beton` | mur/structure |
| 3 | `verre` | fenetre/non combustible |
| 4 | `isolation` | isolant pouvant degager de la chaleur |
| 5 | `air` | milieu de propagation thermique simplifie |
| 6 | `metal` | elements metalliques, forte conduction |
| 7 | `meche` | amorce combustible rapide |
| 8 | `explosif` | degagement thermique tres court et intense |

Chaque materiau possede aussi une variante `*_burn`. Quand un element depasse son seuil `Tc`, il passe dans son etat brule : ses proprietes thermiques sont modifiees et son HRR suit la loi du materiau.

## Equation physique de notre modele

Le modele continu approxime est une equation de diffusion-reaction thermique avec pertes :

```math
\rho c \frac{\partial T}{\partial t}
- \nabla \cdot (k \nabla T)
+ a_v (T - T_{amb})
+ a_r (T^4 - T_{amb}^4)
= HRR(x,t)
```

Sur la frontiere, on ajoute une condition de type convection :

```math
-k \nabla T \cdot n = h (T - T_{amb})
```

Explication des termes :

| Terme | Sens physique |
|---|---|
| `T(x,t)` | Temperature au point `x` et au temps `t`. |
| `rho` | Masse volumique du materiau. |
| `c` | Capacite thermique massique. |
| `rho c dT/dt` | Inertie thermique : energie necessaire pour changer la temperature. |
| `k` | Conductivite thermique du materiau. |
| `grad T` | Direction et intensite locale de la variation de temperature. |
| `-div(k grad T)` | Diffusion/conduction : transfert de chaleur dans le materiau. |
| `a_v (T - T_amb)` | Pertes volumiques lineaires, incluant pertes generales et ventilation simplifiee. |
| `a_r (T^4 - T_amb^4)` | Pertes radiatives simplifiees, importantes a haute temperature. |
| `HRR(x,t)` | Puissance volumique liberee par les elements en combustion. |
| `h (T - T_amb)` | Convection sur les bords du domaine. |
| `n` | Normale sortante de la frontiere. |
| `T_amb` | Temperature ambiante exterieure. |

L'allumage est gere par seuil : si la temperature moyenne d'un element depasse `Tc`, l'element devient brule et utilise la variante `*_burn` du materiau.

## Equation numerique de notre modele

Apres discretisation, le probleme resolu a chaque pas est :

```math
M \frac{T^{n+1} - T^n}{\Delta t}
+ (K + B + A_v M_u) T^{n+1}
= HRR(T^n) + B T_{amb} + A_v M_u T_{amb} - R(T^n)
```

Avec le schema theta general, `theta=1` donne Euler implicite. C'est le reglage par defaut.

Explication des termes numeriques :

| Terme | Sens dans le code |
|---|---|
| `T^n` | Vecteur des temperatures nodales au temps courant. |
| `T^{n+1}` | Vecteur des temperatures nodales apres le pas de temps. |
| `Delta t` | Pas de temps `dt`. |
| `M` | Matrice de masse thermique, assemblee avec `rho*c`. |
| `K` | Matrice de raideur/conduction, assemblee avec `k`. |
| `B` | Matrice diagonale de pertes de frontiere par convection. |
| `M_u` | Matrice de masse unitaire, utilisee pour projeter les pertes et sources volumiques. |
| `A_v M_u` | Contribution des pertes volumiques lineaires. |
| `HRR(T^n)` | Source thermique explicite venant des elements deja allumes. |
| `B T_amb` | Terme de retour vers la temperature ambiante sur la frontiere. |
| `A_v M_u T_amb` | Terme de retour volumique vers la temperature ambiante. |
| `R(T^n)` | Perte radiative explicite. |
| `k_eff` | Dans le code : `K + B + A_v M_u`. |
| `theta_step_fast` | Resolution du systeme lineaire pour avancer en temps. |

Dans le code, le coeur du pas de temps est de la forme :

```python
k_eff = system.k_mat + bc_field.loss_matrix + volume_loss.loss_matrix
src = _hrr_source_rhs(...)
radiation_loss_rhs = _radiation_loss_rhs(...)
rhs = src + bc_field.rhs + volume_loss.rhs - radiation_loss_rhs
t = theta_step_fast(system.m_mat, k_eff, rhs, rhs, t, dt=dt, theta=theta, ...)
```

## Simplifications

Le modele est volontairement simplifie pour rester calculable rapidement et lisible.

### HRR et combustion

- Le feu n'est pas une reaction chimique complete.
- Le HRR est une puissance volumique imposee par materiau.
- L'allumage se fait par seuil `Tc` sur la temperature moyenne de l'element.
- Les proprietes brulees sont approximees par une variante `*_burn`.

### Conditions aux frontieres

- Les bords utilisent une convection simplifiee vers `T_amb`.
- Le coefficient `h_conv` est global.
- Il n'y a pas de modele local de vent, de pression ou d'ouverture.

### Pertes volumiques

- `general_loss` et `vent_loss` sont des pertes lineaires reparties dans le volume.
- Le rayonnement est traite comme une perte volumique explicite, pas comme un vrai echange radiatif surface-surface.
- Les pertes radiatives sont limitees numeriquement pour eviter des valeurs extremes.

### Air chaud et saut d'etage 3D

- En 3D, le transfert vertical est une approximation : une partie du HRR peut chauffer les elements au-dessus.
- Le facteur utilise est attenue avec la hauteur et un rayon horizontal.

### Maillage et materiaux

- Les materiaux sont constants par element.
- Les contacts entre materiaux sont geres par le maillage, pas par une loi de contact detaillee.
- Les noeuds doublons sont fusionnes par coordonnees arrondies pour reparer certains maillages non conformes.

## Optimisations apportées

### Précréation des matrices

Pour le passage à l'état brulé, nous créons directement les matrices en deux états, non brulé et brulé, et nous prenons les données correspondant à l'état de l'élément au lieu de calculé à chaque reprise.

### Elementwise

- Le code garde un etat par element : non brule ou brule.
- Quand un element brule, on n'assemble pas tout depuis zero.
- On applique seulement le delta de matrice entre le materiau normal et sa variante brulee.
- Cela reduit fortement le cout quand peu d'elements changent a chaque pas.

### Numba

Numba utilisé dans l'assemblage des quadratures, matrice local unitaire, et les créations de matrices dans le module calcul. L'utilisation de einsum n'a pas été retenue

- Les quadratures P1 triangle et tetra sont compilees avec Numba.
- Le calcul des matrices locales unitaires est compile avec Numba.
- Les kernels evitent des boucles Python lentes et reduisent les allocations temporaires.
- Les fonctions d'assemblage optimisees des modules `calculs/` utilisent aussi Numba.

### Utilisation de kappa global par materiau

- Les coefficients de conduction `k` et d'inertie `rho*c` sont convertis en tableaux par element.
- Chaque element possede donc son coefficient effectif.
- Ce choix permet d'assembler rapidement les contributions de tous les materiaux en une passe.
- On raisonne par element car un meme maillage contient plusieurs materiaux, et un element peut changer de materiau lorsqu'il brule.

### Gel/degel

- Les elements froids et loin de leur seuil peuvent ne plus etre retestes a chaque pas.
- Ils sont reactives automatiquement s'ils redeviennent proches d'une zone chaude.
- Les noeuds quasi stationnaires peuvent etre temporairement geles par condition de Dirichlet.
- Ils sont degeles si leurs voisins changent trop ou deviennent proches de l'allumage.
- Cela reduit le nombre de degres de liberte actifs quand une grande partie du domaine ne bouge plus.

## Ameliorations possibles

### Solveurs et modèle

Comme la diffusion est localisé à un domaine en bande correspondant au front de flamme, il est envisageable de construire un solveur qui ne calcule que ce domaine local, au lieu de geler les éléments non atteind par des changements de valeurs. Dans cette optique, ammener un système de résolution à notre modèle serait considérable: au centre de ce domaine, calculer les variations avec de petits pas de temps, et au bords, avec des pas de temps plus grand afin de ce concentrer uniquement sur le front de flamme.

### Mecanique des fluides de l'air chaud

Une amelioration physique serait d'inclure l'air chaud :

- mouvement ascendant de l'air ;
- panache thermique ;
- appel d'air par les ouvertures ;
- transport convectif horizontal et vertical ;
- pression et debit entre pieces ou etages ;
- couplage temperature/vitesse de l'air.

Cela demanderait des equations de conservation de masse, quantite de mouvement et energie.

### Frontieres plus realistes

- Coefficients de convection differents selon les murs, fenetres, portes.
- Ouvertures qui changent avec la temperature.
- Echanges radiatifs entre surfaces.
- Conditions exterieures dependantes du temps.

### Combustion plus realiste

- HRR dependant de l'oxygene disponible.
- Propagation de flamme directionnelle.
- Consommation de combustible.
- Extinction possible.
- Couplage avec fumee et gaz chauds.

## sources

le template du prof

les différentes recherche sur la combustion

le hrr

un article sur les incendies ?