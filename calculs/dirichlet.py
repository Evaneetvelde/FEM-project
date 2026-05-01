# dirichlet.py
import numpy as np
from scipy.sparse.linalg import factorized, spsolve


def apply_dirichlet_by_reduction(K, F, dirichlet_dofs, dirichlet_values): # prof
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


def solve_dirichlet(K, F, dirichlet_dofs, dirichlet_values): # prof
    K_red, F_red, free_dofs, U_full = apply_dirichlet_by_reduction(
        K, F, dirichlet_dofs, dirichlet_values
    )
    U_free = spsolve(K_red.tocsr(), F_red)
    U_full[free_dofs] = U_free
    U_full[dirichlet_dofs] = dirichlet_values
    return U_full


def theta_step(M, K, F_n, F_np1, U_n, dt, theta, dirichlet_dofs, dir_vals_np1): # prof
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
    Version optimisee de solve_dirichlet. Si solver est fourni, il doit resoudre le systeme reduit deja factorise.

    param: K: matrice de rigidite (sparse)
    param: F: second membre
    param: dirichlet_dofs: liste des dofs de Dirichlet
    param: dirichlet_values: valeurs de Dirichlet correspondantes
    param: solver: fonction de resolution du systeme reduit (optionnel)
    param: free_dofs: liste des dofs libres (optionnel)
    return: solution complete avec valeurs de Dirichlet et libres
    """
    free_dofs = precompute_dirichlet_dofs(len(F), dirichlet_dofs) if free_dofs is None else free_dofs # sécurité contre code non consistant
    K_red, F_red, free_dofs, U_full = apply_dirichlet_with_free_dofs(K, F, free_dofs, dirichlet_dofs, dirichlet_values)
    solve = factorized(K_red.tocsc()) if solver is None else solver # sécurité contre code non consistant
    U_full[free_dofs] = solve(F_red)
    if len(dirichlet_dofs): # vérif présences dofs
        U_full[np.asarray(dirichlet_dofs, dtype=int)] = dirichlet_values
    return U_full


def theta_step_fast(M, K, F_n, F_np1, U_n, dt, theta, dirichlet_dofs, dir_vals_np1, solver=None, free_dofs=None):
    """
    One theta-scheme step for:
        M u_t + K u = F(t)

    (M + theta dt K) u^{n+1} = (M - (1-theta) dt K) u^n + dt*(theta F^{n+1} + (1-theta) F^n)
    with Dirichlet enforced at time n+1.

    Elle construit A/B en CSR et permet de reutiliser une factorisation via
    solver lorsque A_red ne change pas entre deux pas de temps.

    param: M: matrice de masse (sparse)
    param: K: matrice de rigidite (sparse)
    param: F_n: second membre a t^n
    param: F_np1: second membre a t^{n+1}
    param: U_n: solution a t^n
    param: dt: pas de temps
    param: theta: parametre du schema de temps
    param: dirichlet_dofs: liste des dofs de Dirichlet a t^{n+1}
    param: dir_vals_np1: valeurs de Dirichlet correspondantes a t^{n+1}
    param: solver: fonction de resolution du systeme reduit (optionnel)
    param: free_dofs: liste des dofs libres (optionnel)
    return: solution complete 
    """
    A = (M + theta * dt * K).tocsr()
    B = (M - (1.0 - theta) * dt * K).tocsr()
    rhs = B.dot(U_n) + dt * (theta * F_np1 + (1.0 - theta) * F_n)

    free_dofs = precompute_dirichlet_dofs(len(rhs), dirichlet_dofs) if free_dofs is None else free_dofs # sécurité contre code non consistant
    A_red, rhs_red, free_dofs, U_full = apply_dirichlet_with_free_dofs(A, rhs, free_dofs, dirichlet_dofs, dir_vals_np1)
    solve = factorized(A_red.tocsc()) if solver is None else solver # sécurité contre code non consistant
    U_full[free_dofs] = solve(rhs_red)
    if len(dirichlet_dofs): # vérif présences dofs
        U_full[np.asarray(dirichlet_dofs, dtype=int)] = dir_vals_np1 
    return U_full

def precompute_dirichlet_dofs(n, dirichlet_dofs):
    """
    Précalcule les dofs libres 

    param: n: nombre total de dofs
    param: dirichlet_dofs: liste des dofs de Dirichlet
    return: liste des dofs libres
    """
    mask = np.ones(n, dtype=bool)
    mask[dirichlet_dofs] = False 
    free = np.nonzero(mask)[0] # mask booléen ne rendant que les indices des dofs libres
    return free 


def apply_dirichlet_with_free_dofs(K, F, free_dofs, dirichlet_dofs, dirichlet_values):
    """
    Reduction Dirichlet avec dofs libres deja preassembles.
    param: K: matrice de rigidite (sparse)
    param: F: second membre
    param: free_dofs: liste des dofs libres (preassembles)
    param: dirichlet_dofs: liste des dofs de Dirichlet
    param: dirichlet_values: valeurs de Dirichlet correspondantes
    return: K_FF, F_red, free_dofs, U_full
     - K_FF: matrice reduite sur les dofs libres
     - F_red: second membre reduit sur les dofs libres
     - free_dofs: liste des dofs libres (identique a l'entree)
     - U_full: solution complete 
    """
    F = np.asarray(F, dtype=float)
    dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
    dirichlet_values = np.asarray(dirichlet_values, dtype=float)
    free_dofs = np.asarray(free_dofs, dtype=int)

    K_csr = K.tocsr()
    K_FF = K_csr[free_dofs][:, free_dofs]
    F_red = F[free_dofs] - K_csr[free_dofs][:, dirichlet_dofs].dot(dirichlet_values) if len(dirichlet_dofs) else F[free_dofs]

    U_full = np.zeros(len(F), dtype=float)
    if len(dirichlet_dofs): # vérif présences dofs
        U_full[dirichlet_dofs] = dirichlet_values
    return K_FF, F_red, free_dofs, U_full
