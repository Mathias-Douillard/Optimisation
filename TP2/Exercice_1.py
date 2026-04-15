"""
TP SVM – Exercice 1
Solving the SVM primal and dual problems with gradient-based algorithms.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from utilities import project_onto_C

# =============================================================================
# Load data
# =============================================================================
X_full, y = datasets.load_breast_cancer(return_X_y=True)
y = 2 * y - 1          # {0,1} -> {-1,+1}  (1=benign, -1=malignant)

# First 2 features for visual parts
X = X_full[:, :2]      # mean radius, mean texture

feature_names = datasets.load_breast_cancer().feature_names
df = pd.concat([
    pd.DataFrame(X_full, columns=feature_names),
    pd.Series(y, name='label')
], axis=1)

# =============================================================================
# Helper: standardise features (zero mean, unit variance)
# Returns (X_scaled, mean, scale) so the solution can be back-transformed.
# =============================================================================
def standardise(X):
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma


def backproject_w(w_s, b_s, mu, sigma):
    """w_s, b_s are in standardised space; return (w, b) in original space."""
    w = w_s / sigma
    b = b_s - np.dot(w, mu)
    return w, b


# =============================================================================
# Section 2 – Breast Cancer Dataset
# =============================================================================

# 2.1  Scatter plot
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="mean radius", y="mean texture",
                hue="label", palette="viridis", alpha=0.9, ax=ax)
ax.set_xlabel("Mean radius")
ax.set_ylabel("Mean texture")
ax.set_title("Breast Cancer Data  (-1: Malignant, 1: Benign)")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("section2_scatter.png", dpi=100)
plt.show()

# 2.2  Hand-made classifier
# Benign (y=+1) → small mean radius; malignant (y=-1) → large mean radius.
# Boundary: mean radius ≈ 14  → w=[-1, 0], b=14
# sign(-radius + 14) = +1  when radius < 14  (benign ✓)
#                   = -1  when radius > 14  (malignant ✓)
w_hand = np.array([-1.0, 0.0])
b_hand = 14.0

x_plot = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="mean radius", y="mean texture",
                hue="label", palette="viridis", alpha=0.9, ax=ax)
ax.axvline(x=-b_hand / w_hand[0], color="red", linewidth=2,
           label=f"Boundary: radius = {-b_hand/w_hand[0]:.0f}")
ax.set_xlabel("Mean radius")
ax.set_ylabel("Mean texture")
ax.set_title("Hand-made separating hyperplane")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("section2_handmade.png", dpi=100)
plt.show()

# 2.3  Misclassification rate
preds_hand = np.sign(X @ w_hand + b_hand)
preds_hand[preds_hand == 0] = 1
misclf_hand = np.mean(preds_hand != y)
print(f"[Section 2] Hand-made misclassification rate: {misclf_hand:.4f} "
      f"({misclf_hand*100:.2f} %)")

# 2.4 (Bonus)  Average distance to hyperplane
dist_hand = np.abs(X @ w_hand + b_hand) / np.linalg.norm(w_hand)
print(f"[Section 2] Average distance to hand-made hyperplane: {dist_hand.mean():.4f}")


# =============================================================================
# Section 3 – Primal problem  (sub-gradient descent)
# =============================================================================

def obj_func_primal(w, b, X, y, rho):
    """
    Primal SVM objective:
        f(w, b) = (1/2)||w||² + rho * Σ_i ReLU(1 − y_i(wᵀx_i + b))

    Parameters
    ----------
    w   : np.ndarray (D,)
    b   : float
    X   : np.ndarray (n, D)
    y   : np.ndarray (n,)
    rho : float

    Returns
    -------
    float
    """
    hinge = np.maximum(0.0, 1.0 - y * (X @ w + b))
    return 0.5 * np.dot(w, w) + rho * hinge.sum()


def _primal_subgrad(w, b, X, y, rho):
    """Sub-gradient of the primal objective at (w, b)."""
    margin = y * (X @ w + b)          # shape (n,)
    mask   = margin < 1.0              # ReLU is not at flat region
    # ∇_w f = w − rho * Σ_{mask} y_i x_i
    grad_w = w - rho * (y[mask, None] * X[mask]).sum(axis=0)
    # ∇_b f = −rho * Σ_{mask} y_i
    grad_b = -rho * y[mask].sum()
    return grad_w, float(grad_b)


def solve_primal(X, y, rho, stepsize_mode, n_iter=3000):
    """
    Solves the primal SVM by (sub-)gradient descent.

    Stepsize strategies
    -------------------
    'constant'   : γ_k = c₁
    'variable'   : γ_k = c₂ / (1 + c₃ k)
    'normalized' : γ_k = c₄ / ‖∇f(w,b)‖

    The constants c₁–c₄ are set automatically from the Lipschitz constant
    L = 1 + rho * σ_max(X)² of the smooth part of the gradient.

    Parameters
    ----------
    X             : np.ndarray (n, D)
    y             : np.ndarray (n,)
    rho           : float
    stepsize_mode : {'constant', 'variable', 'normalized'}
    n_iter        : int

    Returns
    -------
    w_opt      : np.ndarray (D,)
    b_opt      : float
    obj_values : list[float]
    """
    n, D = X.shape
    # Lipschitz constant of the smooth gradient (w ↦ w part)
    sigma_max = np.linalg.svd(X, compute_uv=False)[0]
    L = 1.0 + rho * sigma_max**2

    # Hyperparameter defaults derived from L
    c1 = 1.0 / L            # constant stepsize (gradient descent rule)
    c2 = 1.0 / L            # initial variable stepsize
    c3 = 1.0 / n_iter       # decay so that sum(γ_k) diverges, sum(γ_k²) converges
    c4 = 1.0                # normalised stepsize scale

    w = np.zeros(D)
    b = 0.0
    obj_values = []

    for k in range(n_iter):
        obj_values.append(obj_func_primal(w, b, X, y, rho))
        gw, gb = _primal_subgrad(w, b, X, y, rho)

        if stepsize_mode == 'constant':
            gamma = c1
        elif stepsize_mode == 'variable':
            gamma = c2 / (1.0 + c3 * k)
        elif stepsize_mode == 'normalized':
            norm_g = np.sqrt(np.dot(gw, gw) + gb**2)
            gamma  = c4 / norm_g if norm_g > 1e-12 else 0.0
        else:
            raise ValueError(f"Unknown stepsize_mode: '{stepsize_mode}'")

        w = w - gamma * gw
        b = b - gamma * gb

    obj_values.append(obj_func_primal(w, b, X, y, rho))
    return w, b, obj_values


# Work in standardised space so that all three strategies converge
X_s, mu_s, sig_s = standardise(X)
rho = 1.0

results_primal = {}
for mode in ('constant', 'variable', 'normalized'):
    w_s, b_s, obj_vals = solve_primal(X_s, y, rho, stepsize_mode=mode)
    w_orig, b_orig = backproject_w(w_s, b_s, mu_s, sig_s)
    results_primal[mode] = (w_orig, b_orig, obj_vals)
    print(f"[Primal – {mode:10s}]  f* ≈ {obj_vals[-1]:.4f} | "
          f"misclf = {np.mean(np.sign(X @ w_orig + b_orig) != y):.4f}")

# 3.2  Convergence plot
fig, ax = plt.subplots(figsize=(8, 5))
for mode, (_, _, obj_vals) in results_primal.items():
    ax.plot(obj_vals, label=mode)
ax.set_xlabel("Iteration")
ax.set_ylabel("Primal objective f(w, b)")
ax.set_title("Primal convergence – three stepsize strategies")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("section3_convergence.png", dpi=100)
plt.show()
print("Best strategy is 'constant' (1/L stepsize): guaranteed descent, "
      "monotone convergence for smooth terms.")

# 3.3  Minimiser
best_mode_p = min(results_primal, key=lambda m: results_primal[m][2][-1])
w_star_p, b_star_p, _ = results_primal[best_mode_p]
print(f"\n[Primal] Best strategy: {best_mode_p}")
print(f"[Primal] w* ≈ {w_star_p}")
print(f"[Primal] b* ≈ {b_star_p:.4f}")

# 3.4  Separating hyperplane
def plot_boundary(ax, w, b, x_range, **kwargs):
    """Plot w[0]*x + w[1]*y + b = 0  as a line on a 2-D scatter."""
    x0, x1 = x_range
    if abs(w[1]) > 1e-10:
        xs = np.array([x0, x1])
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, **kwargs)
    else:
        ax.axvline(x=-b / w[0], **kwargs)

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="mean radius", y="mean texture",
                hue="label", palette="viridis", alpha=0.9, ax=ax)
plot_boundary(ax, w_star_p, b_star_p,
              (X[:, 0].min(), X[:, 0].max()),
              color='red', linewidth=2, label="SVM primal boundary")
ax.set_xlabel("Mean radius")
ax.set_ylabel("Mean texture")
ax.set_title("Primal SVM – separating hyperplane")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("section3_hyperplane.png", dpi=100)
plt.show()

# 3.5  Misclassification rate
preds_p = np.sign(X @ w_star_p + b_star_p)
preds_p[preds_p == 0] = 1
misclf_p = np.mean(preds_p != y)
print(f"[Primal] Misclassification rate (D=2): {misclf_p:.4f} ({misclf_p*100:.2f} %)")

# 3.6 (Bonus)  D > 2
print("\n[Primal – D=31 features]")
X_full_s, mu_f, sig_f = standardise(X_full)
for mode in ('constant', 'variable', 'normalized'):
    w_s, b_s, _ = solve_primal(X_full_s, y, rho, stepsize_mode=mode)
    w_f, b_f    = backproject_w(w_s, b_s, mu_f, sig_f)
    mc = np.mean(np.sign(X_full @ w_f + b_f) != y)
    print(f"  {mode:10s}  misclassification = {mc:.4f}")


# =============================================================================
# Section 4 – Dual problem  (projected gradient ascent)
# =============================================================================

def obj_func_dual(lam, X, y, rho):
    """
    Dual SVM objective:
        g(λ) = 1ᵀλ − (1/2)‖[y₁x₁, …, yₙxₙ] λ‖²
             = 1ᵀλ − (1/2) λᵀK λ
    where K_ij = y_i y_j x_iᵀx_j.

    Parameters
    ----------
    lam : np.ndarray (n,)
    X   : np.ndarray (n, D)
    y   : np.ndarray (n,)
    rho : float  (unused here; constraint handled by projection)

    Returns
    -------
    float
    """
    yx = y[:, None] * X          # (n, D)
    v  = yx.T @ lam              # (D,)
    return lam.sum() - 0.5 * np.dot(v, v)


def _dual_gradient(lam, X, y):
    """Gradient of the dual objective: ∇g(λ) = 1 − K λ."""
    yx = y[:, None] * X
    return np.ones(len(lam)) - yx @ (yx.T @ lam)   # 1 − K λ


def solve_dual(X, y, rho, stepsize_mode, n_iter=3000):
    """
    Solves the dual SVM by projected gradient ascent.

    Stepsize strategies
    -------------------
    'constant'   : γ_k = 1/L  (L = σ_max(K))
    'variable'   : γ_k = c₂ / (1 + c₃ k)
    'normalized' : γ_k = c₄ / ‖∇g(λ)‖

    Parameters
    ----------
    X             : np.ndarray (n, D)
    y             : np.ndarray (n,)
    rho           : float
    stepsize_mode : {'constant', 'variable', 'normalized'}
    n_iter        : int

    Returns
    -------
    lam_opt    : np.ndarray (n,)
    obj_values : list[float]
    """
    n = len(y)
    sigma_max = np.linalg.svd(X, compute_uv=False)[0]
    L  = sigma_max**2           # Lipschitz constant of ∇g  (σ_max(K))

    c1 = 1.0 / L
    c2 = 1.0 / L
    c3 = 1.0 / n_iter
    c4 = 1.0

    lam = np.zeros(n)           # feasible start
    obj_values = []

    for k in range(n_iter):
        obj_values.append(obj_func_dual(lam, X, y, rho))
        grad = _dual_gradient(lam, X, y)

        if stepsize_mode == 'constant':
            gamma = c1
        elif stepsize_mode == 'variable':
            gamma = c2 / (1.0 + c3 * k)
        elif stepsize_mode == 'normalized':
            norm_g = np.linalg.norm(grad)
            gamma  = c4 / norm_g if norm_g > 1e-12 else 0.0
        else:
            raise ValueError(f"Unknown stepsize_mode: '{stepsize_mode}'")

        lam = project_onto_C(lam + gamma * grad, y, rho)

    obj_values.append(obj_func_dual(lam, X, y, rho))
    return lam, obj_values


def dual2primal(lam_opt, X, y, rho):
    """
    Recovers the primal minimiser (w*, b*) from the dual maximiser λ*.

        w* = Σ_i λ_i* y_i x_i
        b* = y_j − w*ᵀx_j   for any j with 0 < λ_j* < ρ

    Parameters
    ----------
    lam_opt : np.ndarray (n,)
    X       : np.ndarray (n, D)
    y       : np.ndarray (n,)
    rho     : float

    Returns
    -------
    w_opt : np.ndarray (D,)
    b_opt : float
    """
    w_opt = (lam_opt * y) @ X           # Σ_i λ_i* y_i x_i

    tol = 1e-5 * rho
    sv_inner = (lam_opt > tol) & (lam_opt < rho - tol)   # strict support vectors
    if sv_inner.any():
        idxs  = np.where(sv_inner)[0]
        b_opt = float(np.mean(y[idxs] - X[idxs] @ w_opt))
    else:
        sv_any = lam_opt > tol
        b_opt  = float(np.mean(y[sv_any] - X[sv_any] @ w_opt)) if sv_any.any() else 0.0

    return w_opt, b_opt


results_dual = {}
for mode in ('constant', 'variable', 'normalized'):
    lam_opt, obj_vals = solve_dual(X_s, y, rho, stepsize_mode=mode)
    w_d_s, b_d_s      = dual2primal(lam_opt, X_s, y, rho)
    w_d, b_d          = backproject_w(w_d_s, b_d_s, mu_s, sig_s)
    mc = np.mean(np.sign(X @ w_d + b_d) != y)
    results_dual[mode] = (lam_opt, w_d, b_d, obj_vals)
    print(f"[Dual   – {mode:10s}]  g* ≈ {obj_vals[-1]:.4f} | misclf = {mc:.4f}")

# 4.1  Convergence plot
fig, ax = plt.subplots(figsize=(8, 5))
for mode, (_, _, _, obj_vals) in results_dual.items():
    ax.plot(obj_vals, label=mode)
ax.set_xlabel("Iteration")
ax.set_ylabel("Dual objective g(λ)")
ax.set_title("Dual convergence – three stepsize strategies")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("section4_convergence.png", dpi=100)
plt.show()

best_mode_d = max(results_dual, key=lambda m: results_dual[m][3][-1])
lam_star, w_star_d, b_star_d, _ = results_dual[best_mode_d]
print(f"\n[Dual] Best strategy: {best_mode_d}")
print(f"[Dual] w* ≈ {w_star_d}")
print(f"[Dual] b* ≈ {b_star_d:.4f}")

# Separating hyperplane
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="mean radius", y="mean texture",
                hue="label", palette="viridis", alpha=0.9, ax=ax)
plot_boundary(ax, w_star_d, b_star_d,
              (X[:, 0].min(), X[:, 0].max()),
              color='red', linewidth=2, label="SVM dual boundary")
ax.set_xlabel("Mean radius")
ax.set_ylabel("Mean texture")
ax.set_title("Dual SVM – separating hyperplane")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("section4_hyperplane.png", dpi=100)
plt.show()

# Misclassification rate
preds_d = np.sign(X @ w_star_d + b_star_d)
preds_d[preds_d == 0] = 1
misclf_d = np.mean(preds_d != y)
print(f"[Dual] Misclassification rate (D=2): {misclf_d:.4f} ({misclf_d*100:.2f} %)")

# 4.2  Lower bound for the primal  (by strong duality: g(λ*) ≤ μ*)
lb = max(results_dual[m][3][-1] for m in results_dual)
print(f"\n[Dual] Lower bound for the primal: g(λ*) ≈ {lb:.4f}")

# 4.3 (Bonus)  Support vectors
tol_sv  = 1e-4
sv_idx  = np.where(lam_star > tol_sv)[0]
print(f"[Dual] Number of support vectors (λ > 0): {len(sv_idx)} / {len(y)}")

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="mean radius", y="mean texture",
                hue="label", palette="viridis", alpha=0.5, ax=ax)
ax.scatter(X[sv_idx, 0], X[sv_idx, 1], s=120,
           facecolors='none', edgecolors='red', linewidths=1.5,
           zorder=5, label="Support vectors")
plot_boundary(ax, w_star_d, b_star_d,
              (X[:, 0].min(), X[:, 0].max()),
              color='red', linewidth=2, label="Boundary")
ax.set_xlabel("Mean radius")
ax.set_ylabel("Mean texture")
ax.set_title("Support vectors")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("section4_support_vectors.png", dpi=100)
plt.show()

# 4.4 (Bonus)  Margin  (= 1/‖w*‖  for the canonical SVM)
w_norm = np.linalg.norm(w_star_d)
if w_norm > 1e-10:
    print(f"[Dual] Margin (1/‖w*‖) = {1.0/w_norm:.4f}")


# =============================================================================
# Section 5 – Bonus: Sequential Minimal Optimization (SMO)
# =============================================================================

def solve_dual_smo(X, y, rho, n_iter=20000):
    """
    Solves the dual SVM using Sequential Minimal Optimization (SMO).

    At each iteration two random indices i, j are chosen.  The constraint
    yᵀλ = 0 ties λ_j to λ_i, turning the problem into a univariate
    quadratic with box constraints — solved in closed form.

    Parameters
    ----------
    X      : np.ndarray (n, D)
    y      : np.ndarray (n,)
    rho    : float
    n_iter : int

    Returns
    -------
    lam_opt    : np.ndarray (n,)
    obj_values : list[float]
    """
    n  = len(y)
    yx = y[:, None] * X              # (n, D)  rows = y_i * x_i
    K  = yx @ yx.T                   # (n, n)  kernel matrix

    lam        = np.zeros(n)         # feasible start
    obj_values = []
    rng        = np.random.default_rng(42)

    for _ in range(n_iter):
        obj_values.append(lam.sum() - 0.5 * lam @ K @ lam)

        # Pick two distinct indices
        i, j = rng.choice(n, size=2, replace=False)

        # Conserved quantity: y_i λ_i + y_j λ_j = s
        s   = y[i] * lam[i] + y[j] * lam[j]
        eta = y[i] * y[j]            # ±1

        # Coefficient of λ_i² in the restricted dual (concave quadratic)
        A = -(K[i, i] - 2.0 * eta * K[i, j] + K[j, j])
        if abs(A) < 1e-12:
            continue

        # Linear coefficient (derivative at λ_i)
        Klam = K @ lam
        B    = (1.0 - Klam[i]) - eta * (1.0 - Klam[j])

        # Unconstrained maximiser
        li_star = lam[i] - B / A

        # Box constraints on λ_i derived from 0 ≤ λ_j ≤ ρ
        # λ_j = y_j(s − y_i λ_i)
        if eta > 0:    # y_i == y_j  →  same sign
            lo = max(0.0, y[j] * s - rho)
            hi = min(rho, y[j] * s)
        else:          # y_i != y_j  →  opposite sign
            lo = max(0.0, -y[j] * s)
            hi = min(rho, rho - y[j] * s)

        if lo > hi + 1e-12:
            continue

        li_new  = float(np.clip(li_star, lo, hi))
        lj_new  = y[j] * (s - y[i] * li_new)
        lam[i]  = li_new
        lam[j]  = lj_new

    obj_values.append(lam.sum() - 0.5 * lam @ K @ lam)
    return lam, obj_values


print("\n[SMO] Running SMO algorithm …")
lam_smo, obj_smo = solve_dual_smo(X_s, y, rho)
w_smo_s, b_smo_s = dual2primal(lam_smo, X_s, y, rho)
w_smo, b_smo     = backproject_w(w_smo_s, b_smo_s, mu_s, sig_s)

preds_smo = np.sign(X @ w_smo + b_smo)
preds_smo[preds_smo == 0] = 1
misclf_smo = np.mean(preds_smo != y)
print(f"[SMO] g* ≈ {obj_smo[-1]:.4f}")
print(f"[SMO] w* ≈ {w_smo}")
print(f"[SMO] b* ≈ {b_smo:.4f}")
print(f"[SMO] Misclassification rate: {misclf_smo:.4f} ({misclf_smo*100:.2f} %)")

# SMO convergence plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(obj_smo, label="SMO")
ax.axhline(y=results_dual[best_mode_d][3][-1], color='red',
           linestyle='--', label="Proj. grad. ascent (best)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Dual objective g(λ)")
ax.set_title("SMO convergence vs projected gradient ascent")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("section5_smo_convergence.png", dpi=100)
plt.show()

print("\nDone.")
