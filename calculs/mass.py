# mass.py
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix, lil_matrix


def assemble_mass(elemTags, conn, det, w, N, tag_to_dof):
    """
    Assemble global mass matrix:
        M_ij = sum_e ∫_e N_i N_j dx

    Parameters
    ----------
    elemTags : array-like, shape (ne,)
    conn     : flattened connectivity (ne*nloc)
    det      : flattened det(J) values (ne*ngp)
    w        : quadrature weights (ngp)
    N        : flattened basis values (ngp*nloc)

    Returns
    -------
    M : lil_matrix (nn x nn)
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)

    M = lil_matrix((nn, nn), dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            wg = w[g]
            detg = det[e, g]
            for a in range(nloc):
                Ia = int(dof_indices[a])
                Na = N[g, a]
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    M[Ia, Ib] += wg * Na * N[g, b] * detg

    return M


@njit(cache=True)
def _preassemble_pattern_kernel(conn, tag_to_dof):
    ne, nloc = conn.shape
    nnz = ne * nloc * nloc
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    p = 0
    for e in range(ne):
        for a in range(nloc):
            ia = tag_to_dof[conn[e, a]]
            for b in range(nloc):
                rows[p] = ia
                cols[p] = tag_to_dof[conn[e, b]]
                p += 1
    return rows, cols


@njit(cache=True)
def _mass_unit_data_kernel(det, w, N):
    ne, ngp = det.shape
    nloc = N.shape[1]
    data = np.zeros(ne * nloc * nloc, dtype=np.float64)
    p = 0
    for e in range(ne):
        for a in range(nloc):
            for b in range(nloc):
                value = 0.0
                for g in range(ngp):
                    value += w[g] * det[e, g] * N[g, a] * N[g, b]
                data[p] = value
                p += 1
    return data


@njit(cache=True)
def _scale_unit_data_by_elem_kernel(unit_data, coeff_by_elem, nloc):
    data = np.empty_like(unit_data)
    block = nloc * nloc
    for e in range(len(coeff_by_elem)):
        coeff = coeff_by_elem[e]
        start = e * block
        for i in range(block):
            data[start + i] = coeff * unit_data[start + i]
    return data


def preassemble_mass_unit(conn, det, w, N, tag_to_dof):
    """
    Preassemble la structure masse une fois: rows/cols COO + matrice unitaire.

    La matrice unitaire depend seulement du maillage et de la quadrature. Pour
    changer rho*c par element, on reutilise rows/cols et on multiplie seulement
    les donnees unitaires.
    """
    ne = len(det)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    w = np.asarray(w, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    tag_to_dof = np.asarray(tag_to_dof, dtype=np.int64)

    rows, cols = _preassemble_pattern_kernel(conn, tag_to_dof)
    unit_data = _mass_unit_data_kernel(det, w, N)
    return rows, cols, unit_data, int(np.max(tag_to_dof) + 1), nloc


def assemble_mass_from_preassembled(rows, cols, unit_data, n_nodes, nloc, coeff_by_elem=None):
    data = np.asarray(unit_data, dtype=np.float64)
    if coeff_by_elem is not None:
        data = _scale_unit_data_by_elem_kernel(data, np.asarray(coeff_by_elem, dtype=np.float64), int(nloc))
    return coo_matrix((data, (rows, cols)), shape=(int(n_nodes), int(n_nodes))).tocsr()


def assemble_mass_numba(elemTags, conn, det, w, N, tag_to_dof, coeff_by_elem=None):
    """
    Assemble masse via Numba + COO preassemble.

    elemTags reste dans la signature pour etre compatible avec assemble_mass.
    """
    _ = elemTags
    rows, cols, unit_data, n_nodes, nloc = preassemble_mass_unit(conn, det, w, N, tag_to_dof)
    return assemble_mass_from_preassembled(rows, cols, unit_data, n_nodes, nloc, coeff_by_elem)
