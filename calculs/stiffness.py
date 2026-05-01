# stiffness.py
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix, lil_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble global stiffness matrix and load vector for:
        -d/dx (kappa(x) du/dx) = f(x)

    K_ij = ∫ kappa * grad(N_i)·grad(N_j) dx
    F_i  = ∫ f * N_i dx

    Notes:
    - gmsh gives gN in reference coordinates; we map with inv(J).
    - For 1D line embedded in 3D, gmsh provides a 3x3 Jacobian; we keep the same approach.

    Returns
    -------
    K : lil_matrix (nn x nn)
    F : ndarray (nn,)
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            kappa_g = float(kappa_fun(xg))
            f_g = float(rhs_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                F[Ia] += wg * f_g * N[g, a] * detg

                gradNa = invjacg @ gN[g, a]
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    gradNb = invjacg @ gN[g, b]
                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

    return K, F

def assemble_rhs_neumann(F, elemTags, conn, jac, det, xphys, w, N, gN, g_neu_fun, tag_to_dof):
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]

            g_neu_g = float(g_neu_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                N_a = N[g, a]
                F[Ia] += wg * g_neu_g * N_a * detg

    return F


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
def _inv3(m):
    inv = np.empty((3, 3), dtype=np.float64)
    det_m = (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )
    inv_det = 1.0 / det_m
    inv[0, 0] = (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) * inv_det
    inv[0, 1] = (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * inv_det
    inv[0, 2] = (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * inv_det
    inv[1, 0] = (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) * inv_det
    inv[1, 1] = (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * inv_det
    inv[1, 2] = (m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]) * inv_det
    inv[2, 0] = (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]) * inv_det
    inv[2, 1] = (m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]) * inv_det
    inv[2, 2] = (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) * inv_det
    return inv


@njit(cache=True)
def _stiffness_unit_data_kernel(jac, det, w, gN):
    ne, ngp = det.shape
    nloc = gN.shape[1]
    data = np.zeros(ne * nloc * nloc, dtype=np.float64)
    grads = np.empty((nloc, 3), dtype=np.float64)
    p = 0
    for e in range(ne):
        local = np.zeros((nloc, nloc), dtype=np.float64)
        for g in range(ngp):
            inv_jac = _inv3(jac[e, g])
            for a in range(nloc):
                for i in range(3):
                    value = 0.0
                    for j in range(3):
                        value += inv_jac[i, j] * gN[g, a, j]
                    grads[a, i] = value

            wg_det = w[g] * det[e, g]
            for a in range(nloc):
                for b in range(nloc):
                    dot = 0.0
                    for i in range(3):
                        dot += grads[a, i] * grads[b, i]
                    local[a, b] += wg_det * dot

        for a in range(nloc):
            for b in range(nloc):
                data[p] = local[a, b]
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


def preassemble_stiffness_unit(conn, jac, det, w, gN, tag_to_dof):
    """
    Preassemble la structure raideur une fois: rows/cols COO + matrice unitaire.

    La matrice unitaire correspond a kappa = 1. On applique ensuite
    kappa_by_elem, car dans ce simulateur la conductivite vient du materiau de
    l'element: l'envoyer par element evite un appel Python kappa_fun a chaque
    point de quadrature et rend la mise a jour combustion beaucoup moins chere.
    """
    ne = len(det)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    w = np.asarray(w, dtype=np.float64)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)
    tag_to_dof = np.asarray(tag_to_dof, dtype=np.int64)

    rows, cols = _preassemble_pattern_kernel(conn, tag_to_dof)
    unit_data = _stiffness_unit_data_kernel(jac, det, w, gN)
    return rows, cols, unit_data, int(np.max(tag_to_dof) + 1), nloc


def assemble_stiffness_from_preassembled(rows, cols, unit_data, n_nodes, nloc, kappa_by_elem=None):
    data = np.asarray(unit_data, dtype=np.float64)
    if kappa_by_elem is not None:
        data = _scale_unit_data_by_elem_kernel(data, np.asarray(kappa_by_elem, dtype=np.float64), int(nloc))
    return coo_matrix((data, (rows, cols)), shape=(int(n_nodes), int(n_nodes))).tocsr()


def assemble_stiffness_by_elem_numba(elemTags, conn, jac, det, w, gN, tag_to_dof, kappa_by_elem):
    """
    Assemble K avec Numba et kappa_by_elem au lieu de kappa_fun.

    elemTags reste dans la signature pour faciliter le remplacement de
    assemble_stiffness_and_rhs, mais la conductivite est fournie directement
    par element.
    """
    _ = elemTags
    rows, cols, unit_data, n_nodes, nloc = preassemble_stiffness_unit(conn, jac, det, w, gN, tag_to_dof)
    return assemble_stiffness_from_preassembled(rows, cols, unit_data, n_nodes, nloc, kappa_by_elem)
