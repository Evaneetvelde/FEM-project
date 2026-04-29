import numpy as np
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------
# 1. PARAMÈTRES PHYSIQUES RÉALISTES (Mais rapides)
# ---------------------------------------------------------
MATERIALS = {
    1: {"name": "Bois",  "k": 5.0,  "rho": 400.0, "c": 1200.0, "Q": 1.0e6, "Tc": 480.0},
    2: {"name": "Murs",  "k": 10.0, "rho": 500.0, "c": 1000.0, "Q": 1.2e6, "Tc": 450.0},
    3: {"name": "Air",   "k": 2.0,  "rho": 1.2,   "c": 1000.0, "Q": 0.0,   "Tc": 2000.0}
}

T_amb = 300.0
dt = 20.0  # Pas de temps plus court pour la stabilité
h_conv = 50.0 # On utilise une perte convective pour stabiliser Tmax

# ---------------------------------------------------------
# 2. CHARGEMENT ET ASSEMBLAGE (Version simplifiée pour stabilité)
# ---------------------------------------------------------
msh = meshio.read("models/immeuble.msh") # Assurez-vous du chemin
points = msh.points
n_nodes = len(points)
elements = msh.cells_dict["tetra"]
cell_groups = msh.cell_data_dict["gmsh:physical"]["tetra"]

rows, cols, vals_K = [], [], []
rows_m, cols_m, vals_M = [], [], []
Q_node = np.zeros(n_nodes)
Tc_node = np.full(n_nodes, 1000.0)
M_lumped = np.zeros(n_nodes)

for idx, tetra in enumerate(elements):
    mat_id = int(cell_groups[idx])
    prop = MATERIALS.get(mat_id, MATERIALS[1])
    k, pc, q = prop["k"], prop["rho"]*prop["c"], prop["Q"]
    
    nodes = tetra
    P = points[nodes]
    J = (P[1:] - P[0]).T
    vol = abs(np.linalg.det(J)) / 6.0
    invJ = np.linalg.inv(J)
    g_phys = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ invJ
    
    # Rigidité + Convection de stabilisation (h)
    Ke = k * vol * (g_phys @ g_phys.T)
    # Masse
    Me = (pc * vol / 20.0) * (np.ones((4,4)) + np.eye(4))
    
    for i in range(4):
        Q_node[nodes[i]] = max(Q_node[nodes[i]], q)
        Tc_node[nodes[i]] = min(Tc_node[nodes[i]], prop["Tc"])
        M_lumped[nodes[i]] += vol / 4.0
        for j in range(4):
            rows.append(nodes[i]); cols.append(nodes[j]); vals_K.append(Ke[i,j])
            rows_m.append(nodes[i]); cols_m.append(nodes[j]); vals_M.append(Me[i,j])

K = csr_matrix((vals_K, (rows, cols)), shape=(n_nodes, n_nodes))
M = csr_matrix((vals_M, (rows_m, cols_m)), shape=(n_nodes, n_nodes))

# Système : (M/dt + K + h*M_unitaire)
A_lhs = (M / dt + K).tocsr()

# Extraction faces pour visuel
def get_faces(elems):
    f = np.vstack([elems[:,[0,1,2]], elems[:,[0,1,3]], elems[:,[0,2,3]], elems[:,[1,2,3]]])
    return np.unique(np.sort(f, axis=1), axis=0)

surf_faces = get_faces(elements)

# ---------------------------------------------------------
# 3. INITIALISATION ET VISU
# ---------------------------------------------------------
T = T_amb * np.ones(n_nodes)
# Source initiale : on allume le bas
T[points[:,2] < 0.5] = 600.0 

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# On crée la collection une seule fois
poly3d = Poly3DCollection([points[f] for f in surf_faces], alpha=0.3)
ax.add_collection3d(poly3d)

# On utilise un scatter fixe qu'on mettra à jour
scat = ax.scatter(points[:,0], points[:,1], points[:,2], c=T, cmap='hot', s=2, vmin=300, vmax=1000)

def update(frame):
    global T
    for _ in range(5): # 5 sous-pas
        H = (T >= Tc_node).astype(float)
        S = M_lumped * Q_node * H
        rhs = (M / dt).dot(T) + S
        T = spsolve(A_lhs, rhs)
    
    # Mise à jour des couleurs des surfaces (murs)
    face_temps = np.mean(T[surf_faces], axis=1)
    colors = plt.cm.hot(np.clip((face_temps - 300) / 700, 0, 1))
    poly3d.set_facecolor(colors)
    
    # Mise à jour des points (uniquement ceux qui chauffent pour la performance)
    mask = T > 310
    scat._offsets3d = (points[mask, 0], points[mask, 1], points[mask, 2])
    scat.set_array(T[mask])
    
    ax.set_title(f"Temps: {frame*dt*5:.1f}s | Tmax: {T.max():.1f}K")
    return poly3d, scat

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
plt.show()