# Enable inline plots if running in a Jupyter notebook
#matplotlib inline  

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse.linalg import eigs
from scipy.integrate import trapezoid
from mpmath import hyper
from numpy import exp, sqrt, pi, heaviside, where, linspace, zeros, column_stack
from scipy.constants import g as g_earth  # Imported but not used (can remove if not needed)

# -------------------------------------------------------------------------
# Plot styling and LaTeX rendering
# -------------------------------------------------------------------------
params = {
    "backend": "MacOSX",
    "font.family": "sans-serif",
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "text.latex.preamble": "\n".join([
        r"\usepackage{amsmath}", r"\usepackage{lmodern}",
        r"\usepackage{siunitx}", r"\usepackage{physics}",
        r"\usepackage{bm}", r"\usepackage{nicefrac}", r"\usepackage{amssymb}"
    ]),
    "figure.figsize": [6, 6],
    "lines.linewidth": 3.0,
    "lines.markersize": 5.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 28,
    "axes.formatter.limits": [-4, 4],
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "hist.bins": "auto",
    "errorbar.capsize": 5.0,
}

matplotlib.rcParams.update(params)
np.set_printoptions(linewidth=400, formatter={"float_kind": lambda x: "%.5f" % x})

# -------------------------------------------------------------------------
# Normal distribution function
# -------------------------------------------------------------------------
def N_distr(x, mu, sigma2):
    """Return the normal distribution N(x; mu, sigma^2)."""
    return np.exp(-0.5 * ((x - mu) ** 2) / sigma2) / np.sqrt(2.0 * np.pi * sigma2)

# -------------------------------------------------------------------------
# Auxiliary function: mean free time as a function of R
# -------------------------------------------------------------------------
def tmfp(R):
    """Return τ_mfp(R) using a hypergeometric function."""
    z = R**2 / 2
    return float(0.5 * R**2 * hyper([1, 1], [1.5, 2], z))

# -------------------------------------------------------------------------
# Compute the propagator (Eq. S22 in the referenced work)
# -------------------------------------------------------------------------
def compute_clean_T(L, u, dg, R):
    """
    Return the propagator matrix T(L, u) for the transition
    from position u to L, given dg and R.
    """
    sigma2 = 1.0 - np.exp(-2.0 * tmfp(R))
    L_sq = L[:, None]
    delta = L_sq - R
    delta2 = L_sq**2.0 - R**2.0

    # Safe square root to avoid NaNs for negative arguments
    safe_sqrt = np.sqrt(np.where(delta > 0.0, delta2, 0.0))

    # Construct the propagator with masking for valid regions
    prop = (
        np.heaviside(delta, 0.0)
        * L[:, None]
        / np.where(delta > 0.0, safe_sqrt, 1.0)
        * (
            N_distr(u[None, :], (u[None, :] - dg) * (1.0 - np.exp(-tmfp(R))) + safe_sqrt, sigma2)
            + N_distr(u[None, :], (u[None, :] - dg) * (1.0 - np.exp(-tmfp(R))) - safe_sqrt, sigma2)
        )
    )

    return prop

# -------------------------------------------------------------------------
# Compute steady-state distribution via eigenvalue decomposition
# -------------------------------------------------------------------------
def find_clean_steady_states(out_grid, in_grid, dg, R, TOL=1e-3):
    """
    Find and return the steady-state probability distribution p(x)
    as the eigenvector of the transition operator T with eigenvalue 1.
    """
    dx = out_grid[1] - out_grid[0]

    # Compute transition matrix and normalize rows
    T_clean = compute_clean_T(out_grid, in_grid, dg, R)
    norms = trapezoid(T_clean, in_grid, axis=0)
    T_clean /= norms

    # Compute eigenvalues/eigenvectors of T
    p0 = N_distr(out_grid, 0.0, 1.0)  # initial guess (Gaussian)
    w, v = eigs(T_clean * dx, k=3, v0=p0)

    # Identify eigenvector corresponding to eigenvalue ~1
    p_ss = v[:, np.where(np.abs(w - 1.0) < TOL)[0][0]]

    # Normalize to form a proper probability distribution
    p_ss /= trapezoid(p_ss, out_grid)

    return p_ss.real

# -------------------------------------------------------------------------
# Compute mean output power as a function of radius R
# -------------------------------------------------------------------------
def compute_means(dg=0.8, nscan=40):
    """
    Run the steady-state calculation and return mean power vs R.
    """
    radii = linspace(2.5, 5.0, nscan)
    mean_powers_out = zeros(int(nscan))
    
    for idx, R in enumerate(radii):
        grid = linspace(R, 10.0, 15000)
        p_xnpr = find_clean_steady_states(grid, grid, dg, R)
        
        # Compute average power using the steady-state distribution
        mean_powers_out[idx] = dg * (trapezoid(grid * p_xnpr, grid) - dg) * (1.0 - exp(-tmfp(R))) / tmfp(R)
        
        print(f"{{ {R:.3f}, {mean_powers_out[idx]:.6f} }},")
        
    return column_stack((radii, mean_powers_out))

# -------------------------------------------------------------------------
# Run the calculation
# -------------------------------------------------------------------------
clean_results = compute_means()
