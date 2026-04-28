# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh


def plot_fe_solution_high_order(
    elemType, elemNodeTags, nodeCoords, U,
    M=80, show_nodes=False, ax=None, label=None
):
    """
    Plot 1D high-order FE solution by sampling each element and evaluating gmsh basis.
    Assumes U is aligned with gmsh's compact node ordering (0..nn-1).
    """
    _, _, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    u = np.linspace(-1.0, 1.0, int(M))
    pts3 = np.zeros((len(u), 3), dtype=float)
    pts3[:, 0] = u
    uvw = pts3.reshape(-1).tolist()

    _, bf, _ = gmsh.model.mesh.getBasisFunctions(elemType, uvw, "Lagrange")
    N = np.asarray(bf, dtype=float).reshape(len(u), nloc)

    if ax is None:
        fig, ax = plt.subplots()

    ne = int(len(elemNodeTags) // nloc)
    _, _, coords_flat = gmsh.model.mesh.getJacobians(elemType, uvw)
    coords = np.asarray(coords_flat, dtype=float).reshape(ne, len(u), 3)

    for e in range(ne):
        tags_e = np.asarray(elemNodeTags[e * nloc:(e + 1) * nloc], dtype=int) - 1
        Ue = U[tags_e]

        x = coords[e, :, 0]
        uh = N @ Ue

        order = np.argsort(x)
        ax.plot(x[order], uh[order], label=label if (e == 0) else None)

    if show_nodes:
        Xn = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
        ax.plot(Xn, U, "o", markersize=4)

    ax.set_xlabel("x")
    ax.set_ylabel("u_h")
    ax.grid(True)
    return ax


def setup_interactive_figure(xlim=None, ylim=None):
    plt.ion()
    fig, ax = plt.subplots()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax

def plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, tag_to_index=None):

    coords = nodeCoords.reshape(-1, 3)
    x = coords[:, 0]
    y = coords[:, 1]

    if tag_to_index is None:
        max_node_tag = int(np.max(nodeTags))
        tag_to_index = np.zeros(max_node_tag + 1, dtype=int)
        for i, tag in enumerate(nodeTags):
            tag_to_index[int(tag)] = i

    num_elements = len(elemTags)
    nodes_per_elem = len(elemNodeTags) // num_elements

    # take only the first 3 nodes (=geometric nodes that form the triangles)    
    all_nodes = elemNodeTags.reshape(num_elements, nodes_per_elem)
    corner_nodes = all_nodes[:, :3] 
    
    # Map to indices
    tri_indices = tag_to_index[corner_nodes.astype(int)]
    # ---------------------------------------

    mesh_triang = tri.Triangulation(x, y, tri_indices)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the skeleton
    ax.triplot(mesh_triang, color='black', lw=0.5, alpha=0.4)

    colors = ["red", "darkblue", "orange", "mediumpurple", "pink"]
    for i, (name, dim) in enumerate(bnds):
        tags = bnds_tags[i]
        indices = tag_to_index[tags.astype(int)]
        ax.scatter(x[indices], y[indices], label=name, s=15, zorder=3, 
                   marker="o", facecolor="None", edgecolor=colors[i % len(colors)])

    ax.set_aspect('equal')
    ax.legend(frameon=True, framealpha=1, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    plt.axis(False)
    plt.show()


def plot_fe_solution_2d(elemNodeTags, nodeCoords, nodeTags, U, tag_to_dof, show_mesh=False, ax=None, label=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Map coordinates to our compact DoF indices
    num_dofs = len(U)
    coords_mapped = np.zeros((num_dofs, 2))
    all_coords = nodeCoords.reshape(-1, 3)
    
    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]
        if dof_idx != -1:
            coords_mapped[dof_idx] = all_coords[i, :2]

    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]

    # 2. Determine nodes per element dynamically
    total_nodes_in_elems = len(elemNodeTags)
    # Standard Lagrange triangle node counts: Order 1=3, Order 2=6, Order 3=10, Order 4=15
    for possible_n in [3, 6, 10, 15, 21]:
        if total_nodes_in_elems % possible_n == 0:
            nodes_per_elem = possible_n
            break
    # 3. Reshape and extract ONLY the 3 corner nodes for Matplotlib
    conn_reshaped = elemNodeTags.reshape(-1, nodes_per_elem)
    # Map the GMSH tags to our 0...N-1 indices
    triangles = tag_to_dof[conn_reshaped[:, :3].astype(int)]
    # 4. Plotting
    U = np.array(U).flatten()
    contour = ax.tricontourf(x, y, triangles, U, levels=100, cmap='seismic', vmin=-2.0, vmax=2.0)
    
    if show_mesh:
        ax.triplot(x, y, triangles, color='white', linewidth=0.2, alpha=0.3)

    return contour


# --- Animation helpers inspired by diffusion_2D_fem.py and jsp.py ---
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_surface_faces(elems: np.ndarray) -> np.ndarray:
    """Return unique triangular faces for a tetrahedral mesh (Nx3 array)."""
    # elems is (n,4)
    f = np.vstack([elems[:, [0, 1, 2]], elems[:, [0, 1, 3]], elems[:, [0, 2, 3]], elems[:, [1, 2, 3]]])
    # sort node indices per face to canonicalize
    f_sorted = np.sort(f, axis=1)
    uniq = np.unique(f_sorted, axis=0)
    return uniq


def init_anim_2d(points: np.ndarray, elements: np.ndarray, T_init: np.ndarray, cell_groups: np.ndarray | None = None, walls_id: int | None = None, vmin: float | None = None, vmax: float | None = None):
    """Create a 2D figure and return (fig, ax, im, walls_poly, update_func).

    update_func should be called as `artists = update_func(T)` where `T` is the
    current nodal temperature vector; it returns a list/tuple of artists updated.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.tripcolor(points[:, 0], points[:, 1], elements, T_init, cmap="magma", shading="gouraud", vmin=vmin, vmax=vmax)
    walls_poly = None
    if cell_groups is not None and walls_id is not None:
        try:
            mask = np.asarray(cell_groups) == walls_id
            murs_elements = elements[mask]
            verts = [points[tri] for tri in murs_elements]
            walls_poly = PolyCollection(verts, facecolors="#808080", edgecolors="black", linewidths=0.5, zorder=3)
            ax.add_collection(walls_poly)
        except Exception:
            walls_poly = None

    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    ax.set_facecolor("white")

    def make_update(t_amb: float = 300.0, src_temp: float = 1000.0):
        def update(T: np.ndarray):
            im.set_array(T)
            # adjust color limits adaptively
            im.set_clim(vmin=t_amb, vmax=max(src_temp, float(np.max(T))))
            return (im,) if walls_poly is None else (im, walls_poly)

        return update

    return fig, ax, im, walls_poly, make_update


def init_anim_3d(points: np.ndarray, elems: np.ndarray, T_init: np.ndarray, vmin: float | None = None, vmax: float | None = None):
    """Create a 3D figure with surface faces and scatter; returns (fig, ax, poly3d, scat, faces, update_func)."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    faces = get_surface_faces(elems)
    poly3d = Poly3DCollection([points[f] for f in faces], alpha=0.3)
    ax.add_collection3d(poly3d)

    scat = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=T_init, cmap="hot", s=2, vmin=vmin, vmax=vmax)

    def make_update(t_amb: float = 300.0, vmax_local: float | None = None):
        def update(T: np.ndarray):
            face_temps = np.mean(T[faces], axis=1)
            v_max = vmax_local if vmax_local is not None else max(float(np.max(T)), t_amb + 1.0)
            colors = plt.cm.hot(np.clip((face_temps - t_amb) / max(1.0, (v_max - t_amb)), 0.0, 1.0))
            poly3d.set_facecolor(colors)

            mask = T > (t_amb + 5.0)
            try:
                scat._offsets3d = (points[mask, 0], points[mask, 1], points[mask, 2])
            except Exception:
                # fallback: recreate scatter (slower)
                scat.remove()
                new_scat = ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], c=T[mask], cmap="hot", s=2, vmin=t_amb, vmax=v_max)
                return (poly3d, new_scat)

            scat.set_array(T[mask])
            return (poly3d, scat)

        return update

    return fig, ax, poly3d, scat, faces, make_update