#test en 2D du model
import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from matplotlib.collections import PolyCollection
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
from pathlib import Path

# ---------------------------------------------------------
# 1. PARAMÈTRES PHYSIQUES
# ---------------------------------------------------------

MATERIALS = {
    1: {"name": "Bois",  "k": 5.0,  "rho": 400.0, "c": 1200.0, "Q": 1.0e6, "Tc": 480.0},
    2: {"name": "Murs",  "k": 1.0, "rho": 500.0, "c": 1000.0, "Q": 1.2e6, "Tc": 450.0},
    3: {"name": "Air",   "k": 2.0,  "rho": 1.2,   "c": 1000.0, "Q": 0.0,   "Tc": 2000.0}
}

Tc = 480.0          # Seuil de combustion [K]
T_amb = 300.0       # Température ambiante [K]
h = 500.0            # Coeff de convection (refroidissement par l'air) [W/m²K]
t_final = 50000.0    
dt = 4.0            # Pas de temps stable pour cette échelle
# ---------------------------------------------------------
# 2. CHARGEMENT DU MAILLAGE
# ---------------------------------------------------------

try:
    base_dir = Path(__file__).parent
    mesh_path = base_dir / "models" / "piece.msh"
    msh = meshio.read(str(mesh_path))
    points = msh.points[:, :2]
    n_nodes = len(points)
    elements = msh.cells_dict["triangle"]
    cell_groups = msh.cell_data_dict["gmsh:physical"]["triangle"]
    print(f"Maillage chargé : {n_nodes} nœuds.")
except Exception as e:
    print(f"Erreur : Assurez-vous que piece.msh existe. Détail : {e}")
    exit()
# ---------------------------------------------------------
# 3. ASSEMBLAGE DES MATRICES
# ---------------------------------------------------------

K = lil_matrix((n_nodes, n_nodes))
M = lil_matrix((n_nodes, n_nodes))
M_unit = lil_matrix((n_nodes, n_nodes)) # Pour le terme source et convection
Q_node = np.zeros(n_nodes)
print("Assemblage des matrices (Diffusion + Inertie + Convection)...")
for idx, tri in enumerate(elements):
    # Identification matériau
    raw_id = cell_groups[idx]
    mat_id = int(raw_id[0]) if hasattr(raw_id, "__len__") else int(raw_id)
    prop = MATERIALS.get(mat_id, MATERIALS[1])
    k_val = prop["k"]
    pc = prop["rho"] * prop["c"]
    q_val = prop["Q"]
    # Géométrie de l'élément
    nodes = tri
    xe, ye = points[nodes, 0], points[nodes, 1]
    detJ = (xe[1]-xe[0])*(ye[2]-ye[0]) - (xe[2]-xe[0])*(ye[1]-ye[0])
    area = 0.5 * abs(detJ)
    # Matrice de Rigidité (Conduction)
    b = np.array([ye[1]-ye[2], ye[2]-ye[0], ye[0]-ye[1]]) / detJ
    c_grad = np.array([xe[2]-xe[1], xe[0]-xe[2], xe[1]-xe[0]]) / detJ
    Ke = k_val * area * (np.outer(b, b) + np.outer(c_grad, c_grad))
    # Matrice de Masse (Inertie et Unitaire)
    # Formule consistante P1
    Me_block = (area / 12.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    Q_node[nodes] = q_val
    for i in range(3):
        for j in range(3):
            # K global inclut la conduction et la perte convective h
            K[nodes[i], nodes[j]] += Ke[i, j] + h * Me_block[i, j]
            M[nodes[i], nodes[j]] += pc * Me_block[i, j]
            M_unit[nodes[i], nodes[j]] += Me_block[i, j]

K = K.tocsr()
M = M.tocsr()
M_unit = M_unit.tocsr()
# Matrice de gauche pour Euler Implicite

A_lhs = (M / dt + K).tocsr()
# ---------------------------------------------------------
# 4. INITIALISATION
# ---------------------------------------------------------
T = T_amb * np.ones(n_nodes)
x_src, y_src = 0.25, 0.25
dist_source = np.sqrt((points[:,0] - x_src)**2 + (points[:,1] - y_src)**2)
T[dist_source < 0.08] = 800.0

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.tripcolor(points[:,0], points[:,1], elements, T,cmap='magma', shading='gouraud', vmin=300, vmax=1300)
# --- Affichage spécifique des murs ---
# On cherche les triangles dont l'ID physique est 2 (Béton)
murs_elements = elements[np.where(cell_groups == 2)[0]]
# On dessine une collection de polygones gris pour représenter les murs
verts = [points[tri] for tri in murs_elements]
murs_poly = PolyCollection(verts, facecolors='#808080', edgecolors='black', linewidths=0.5, zorder=3)
ax.add_collection(murs_poly)
plt.colorbar(im, label="Température [K]")
ax.set_aspect('equal')
ax.set_facecolor("#1A12BD")

# ---------------------------------------------------------
# 5. FONCTION DE MISE À JOUR ET BARRE DE PROGRESSION
# ---------------------------------------------------------
n_frames = 1000 
sub_steps = 10 
def update(frame):
    global T, _sim_time
    for _ in range(sub_steps):
        H = (T >= Tc).astype(float)
        S_combustion = M_unit.dot(Q_node * H)
        S_convection = M_unit.dot(h * T_amb * np.ones(n_nodes))
        rhs = (M / dt).dot(T) + S_combustion + S_convection
        T = spsolve(A_lhs, rhs)
    # advance cumulative simulation time
    _sim_time += dt * sub_steps
    im.set_array(T)
    ax.set_title(f"Temps : {_sim_time:.1f} s | Tmax : {np.max(T):.1f} K")
    return [im]

# ---------------------------------------------------------
# 6. ANIMATION
# ---------------------------------------------------------
print(f"Lancement de l'animation...")
_sim_time = 0.0
ani = FuncAnimation(fig, update, frames=n_frames, blit=True)

# Note : La sauvegarde MP4 nécessite ffmpeg. Décommente les lignes suivantes si ffmpeg est installé.
#writer = FFMpegWriter(fps=50, metadata=dict(artist='Gemini-Simulation'), bitrate=2000)
#ani.save("simulation_incendie.mp4", writer=writer)
plt.show()