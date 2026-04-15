import numpy as np


def project_onto_C(z, y, rho=1.0):
    """
    Projects the vector z onto the set:
        𝒞 = [0, ρ]ⁿ ∩ { 𝛌 ∈ ℝⁿ | 𝒚ᵀ𝛌 = 0 }

    i.e., solves argmin_𝛌 ||𝛌 - 𝐳||²  subject to 𝛌 ∈ 𝒞.

    Parameters
    ----------
    z : np.ndarray, shape (n,)
        The input vector z to be projected.
    y : np.ndarray, shape (n,)
        The vector defining the linear equality constraint 𝒚ᵀ𝛌 = 0, with 𝒚 ∈ {−1, 1}ᵖ and 𝒚 ∉ {
        −𝟏, 𝟏}.
    rho : float
        The upper bound for the box constraint, with ρ > 0.

    Returns
    -------
    np.ndarray, shape (n,)
        The projected vector onto the constrained set.
    """
    assert np.all(np.isin(y, [-1, 1])) and not np.all(y == -1) and not np.all(y == 1)
    nodes = np.concatenate(((z - rho) * y, z * y), axis=-1)
    ind = np.argsort(nodes, axis=-1)
    nodes = np.take_along_axis(nodes, ind, axis=-1)
    startend = np.take_along_axis(
        np.concatenate((y, -y), axis=0), ind, axis=-1
    )  # start=1, end=-1
    a = np.cumsum(-startend)  # slopes
    b = rho * np.sum(y == 1) + np.cumsum(nodes * startend)  # intercepts
    im_nodes = a * nodes + b
    j = np.argmax(im_nodes < 0) - 1
    lam = -b[j] / a[j]  # solves a * lam + b = 0
    return np.clip(z - lam * y, a_min=0, a_max=rho)
