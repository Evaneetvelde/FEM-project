# dirichlet.py
import numpy as np
from numba import njit
from scipy.sparse.linalg import factorized, spsolve
from scipy.linalg import det


def apply_dirichlet_by_reduction(K, F, dirichlet_dofs, dirichlet_values):
    """
    Reduce linear system with strong Dirichlet by elimination:
        K u = F, u_D fixed
    -> K_FF u_F = F_F - K_FD u_D

    K can be sparse (csr/lil/etc).
    """
    dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
    dirichlet_values = np.asarray(dirichlet_values, dtype=float)

    n = len(F)
    mask = np.ones(n, dtype=bool)
    mask[dirichlet_dofs] = False
    free_dofs = np.nonzero(mask)[0]

    K_FF = K[free_dofs, :][:, free_dofs]
    K_FD = K[free_dofs, :][:, dirichlet_dofs]

    F_F = F[free_dofs]
    F_red = F_F - K_FD.dot(dirichlet_values)

    U_full = np.zeros(n, dtype=float)
    U_full[dirichlet_dofs] = dirichlet_values

    return K_FF, F_red, free_dofs, U_full


def solve_dirichlet(K, F, dirichlet_dofs, dirichlet_values):
    K_red, F_red, free_dofs, U_full = apply_dirichlet_by_reduction(
        K, F, dirichlet_dofs, dirichlet_values
    )
    U_free = spsolve(K_red.tocsr(), F_red)
    U_full[free_dofs] = U_free
    U_full[dirichlet_dofs] = dirichlet_values
    return U_full


def theta_step(M, K, F_n, F_np1, U_n, dt, theta, dirichlet_dofs, dir_vals_np1):
    """
    One theta-scheme step for:
        M u_t + K u = F(t)

    (M + theta dt K) u^{n+1} = (M - (1-theta) dt K) u^n + dt*(theta F^{n+1} + (1-theta) F^n)
    with Dirichlet enforced at time n+1.
    """
    A = M + theta * dt * K
    B = M - (1.0 - theta) * dt * K
    rhs = B.dot(U_n) + dt * (theta * F_np1 + (1.0 - theta) * F_n)

    A_red, rhs_red, free_dofs, U_full = apply_dirichlet_by_reduction(
        A, rhs, dirichlet_dofs, dir_vals_np1
    )
    U_free = spsolve(A_red.tocsr(), rhs_red)
    U_full[free_dofs] = U_free
    U_full[dirichlet_dofs] = dir_vals_np1
    return U_full


def solve_dirichlet_fast(K, F, dirichlet_dofs, dirichlet_values, solver=None, free_dofs=None):
    """
    Version optimisee de solve_dirichlet.

    Si solver est fourni, il doit resoudre le systeme reduit deja factorise
    (par exemple via scipy.sparse.linalg.factorized).
    """
    if free_dofs is None:
        free_dofs = precompute_dirichlet_dofs(len(F), dirichlet_dofs)
    K_red, F_red, free_dofs, U_full = apply_dirichlet_with_free_dofs(K, F, free_dofs, dirichlet_dofs, dirichlet_values)
    solve = solver if solver is not None else factorized(K_red.tocsc())
    U_full[free_dofs] = solve(F_red)
    if len(dirichlet_dofs):
        U_full[np.asarray(dirichlet_dofs, dtype=int)] = dirichlet_values
    return U_full


def theta_step_fast(M, K, F_n, F_np1, U_n, dt, theta, dirichlet_dofs, dir_vals_np1, solver=None, free_dofs=None):
    """
    Version optimisee de theta_step.

    Elle construit A/B en CSR et permet de reutiliser une factorisation via
    solver lorsque A_red ne change pas entre deux pas de temps.
    """
    A = (M + theta * dt * K).tocsr()
    B = (M - (1.0 - theta) * dt * K).tocsr()
    rhs = B.dot(U_n) + dt * (theta * F_np1 + (1.0 - theta) * F_n)

    if free_dofs is None:
        free_dofs = precompute_dirichlet_dofs(len(rhs), dirichlet_dofs)
    A_red, rhs_red, free_dofs, U_full = apply_dirichlet_with_free_dofs(A, rhs, free_dofs, dirichlet_dofs, dir_vals_np1)
    solve = solver if solver is not None else factorized(A_red.tocsc())
    U_full[free_dofs] = solve(rhs_red)
    if len(dirichlet_dofs):
        U_full[np.asarray(dirichlet_dofs, dtype=int)] = dir_vals_np1
    return U_full


@njit(cache=True)
def _free_dofs_kernel(n, dirichlet_dofs):
    fixed = np.zeros(n, dtype=np.uint8)
    for i in range(len(dirichlet_dofs)):
        fixed[dirichlet_dofs[i]] = 1

    count = 0
    for i in range(n):
        if fixed[i] == 0:
            count += 1

    free = np.empty(count, dtype=np.int64)
    p = 0
    for i in range(n):
        if fixed[i] == 0:
            free[p] = i
            p += 1
    return free


def precompute_dirichlet_dofs(n, dirichlet_dofs):
    """
    Precalcule les dofs libres avec Numba.

    A reutiliser quand les conditions de Dirichlet portent toujours sur les
    memes noeuds: on evite de reconstruire le masque booleen a chaque solve.
    """
    return _free_dofs_kernel(int(n), np.asarray(dirichlet_dofs, dtype=np.int64))


def apply_dirichlet_with_free_dofs(K, F, free_dofs, dirichlet_dofs, dirichlet_values):
    """
    Reduction Dirichlet avec dofs libres deja preassembles.
    """
    F = np.asarray(F, dtype=float)
    dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
    dirichlet_values = np.asarray(dirichlet_values, dtype=float)
    free_dofs = np.asarray(free_dofs, dtype=int)

    K_csr = K.tocsr()
    K_FF = K_csr[free_dofs][:, free_dofs]
    if len(dirichlet_dofs):
        F_red = F[free_dofs] - K_csr[free_dofs][:, dirichlet_dofs].dot(dirichlet_values)
    else:
        F_red = F[free_dofs]

    U_full = np.zeros(len(F), dtype=float)
    if len(dirichlet_dofs):
        U_full[dirichlet_dofs] = dirichlet_values
    return K_FF, F_red, free_dofs, U_full
