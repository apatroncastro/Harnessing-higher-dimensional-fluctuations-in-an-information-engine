# ===============================================
# Libraries and setup
# ===============================================
# Import numerical, plotting, and scientific libraries
from numpy import *
from numpy import log
from scipy.special import iv, i0e  # Modified Bessel functions of the first kind
import matplotlib.pyplot as plt
import scipy.integrate as spi
from scipy.integrate import quad, trapezoid, dblquad
from joblib import Parallel, delayed  # For parallel computing
from scipy.stats import uniform
import matplotlib
import numpy as np
import math
from scipy.sparse.linalg import eigs  # Sparse matrix eigenvalue solver
from scipy.special import erf, erfc, xlogy
from scipy.constants import Boltzmann as kB, g as g_earth
from scipy.optimize import minimize_scalar
from sympy import DiracDelta

plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering in plots
set_printoptions(linewidth=400, formatter={"float_kind": lambda x: "%.5f" % x})


# ===============================================
# Gaussian distribution helper
# ===============================================
def N_distr(x, mu, sigma2):
    """
    Normal (Gaussian) probability density function.

    Parameters
    ----------
    x : float or array
        Variable.
    mu : float
        Mean of the Gaussian.
    sigma2 : float
        Variance (sigma^2).

    Returns
    -------
    float or array
        Gaussian PDF evaluated at x.
    """
    return exp(-0.5 * ((x - mu) ** 2.0) / sigma2) / np.sqrt(2.0 * math.pi * sigma2)


# ===============================================
# Integrand for one of the transition integrals
# ===============================================
def integrand1(znpr, u, uy, dg, dt):
    """
    Integrand of the first transition kernel (T).

    znpr : float
        New state variable.
    u : float
        Old state variable.
    uy : float
        Integration variable.
    dg : float
        Drift parameter.
    dt : float
        Time increment (sampling interval).
    """
    mean = -np.sqrt(u**2.0 + uy**2.0) * exp(-dt) - dg * (1.0 - exp(-dt))
    sigma2 = 1.0 - np.exp(-2.0 * dt)
    # Product of two Gaussian PDFs
    return N_distr(znpr, mean, sigma2) * N_distr(uy, 0.0, sigma2)


# ===============================================
# Transition kernel for the “T” process
# ===============================================
def compute_clean_T(znpr, u, dg, dt, L, N):
    """
    Computes one element of the transition matrix T_clean(znpr, u),
    integrating over the auxiliary variable uy.

    znpr : float
        New variable (destination state).
    u : float
        Old variable (source state).
    dg : float
        Drift parameter.
    dt : float
        Time interval.
    L : float
        Integration limit for uy (domain is [-L, L]).
    N : int
        Number of discretization points.
    """
    phi_grid = linspace(-L, L, N)
    sigma2 = 1.0 - exp(-2.0 * dt)
    dx = 2.0 * L / N
    integral = trapezoid(integrand1(znpr, u, phi_grid, dg, dt), phi_grid)
    return integral * dx


# ===============================================
# Transition kernel for the “Ttilde” process
# ===============================================
def compute_clean_Ttilde(znr, v, dg, dt, L, N):
    """
    Computes one element of the Ttilde_clean(znr, v) transition matrix.

    znr : float
        New variable.
    v : float
        Old variable.
    dg : float
        Drift parameter.
    dt : float
        Time interval.
    L, N : float, int
        Grid parameters for numerical integration.
    """
    sigma2 = 1.0 - exp(-2.0 * dt)
    dx = 2.0 * L / N
    mean = v * np.exp(-dt) - dg * (1.0 - np.exp(-dt))
    # Logarithmic form for numerical stability when using Bessel function
    log_I0 = abs(znr * mean) / sigma2 + np.log(i0e(abs(znr * mean) / sigma2))
    log_result = -0.5 * (znr**2.0 + mean**2.0) / sigma2 + log_I0
    # Transition probability (unnormalized)
    return dx * (-v) * np.pi * np.exp(log_result) / (2.0 * np.pi * sigma2)


# ===============================================
# Helper functions to compute rows of the transition matrices
# ===============================================
def compute_Ttilde_row(i, gridTtilde, dg, dt, L, Nr):
    """Compute a single row of the Ttilde_clean transition matrix."""
    return [compute_clean_Ttilde(gridTtilde[i], v, dg, dt, L, Nr) for v in gridTtilde]


def compute_T_row(i, gridT, dg, dt, L, Npr):
    """Compute a single row of the T_clean transition matrix."""
    return [compute_clean_T(gridT[i], u, dg, dt, L, Npr) for u in gridT]


# ===============================================
# Compute steady-state distributions
# ===============================================
def find_clean_steady_state_T(gridT, gridTtilde, dg, dt, Lpr, Lr, Npr, Nr, TOL=1e-3):
    """
    Constructs transition matrices T and Ttilde, finds their stationary
    (steady-state) distributions by solving the eigenvalue problem T*p = p.

    Returns
    -------
    p_znpr, p_znr : arrays
        Steady-state distributions for the two processes.
    """
    # --- Compute transition matrices in parallel ---
    Ttilde_clean = np.array(
        Parallel(n_jobs=-1)(delayed(compute_Ttilde_row)(i, gridTtilde, dg, dt, Lr, Nr) for i in range(Nr))
    )
    T_clean = np.array(
        Parallel(n_jobs=-1)(delayed(compute_T_row)(i, gridT, dg, dt, Lpr, Npr) for i in range(Npr))
    )

    # --- Initial guesses for eigenvalue solver ---
    p0T = N_distr(gridT, 0.0, 1.0)
    p0Ttilde = N_distr(gridTtilde, 5.0, 1.0)

    # --- Find eigenvalues/vectors (largest 3) ---
    w_znpr, v_znpr = eigs(T_clean, k=3, v0=p0T)
    w_znr, v_znr = eigs(Ttilde_clean, k=3, v0=p0Ttilde)

    print("Eigenvalues =", w_znpr)
    print("Eigenvalues =", w_znr)

    # --- Identify eigenvector with eigenvalue ≈ 1 (steady state) ---
    idx_znpr = np.where(np.abs(w_znpr - 1.0) < TOL)[0][0]
    idx_znr = np.where(np.abs(w_znr - 1.0) < TOL)[0][0]

    p_znpr = v_znpr[:, idx_znpr].real
    p_znr = v_znr[:, idx_znr].real

    # --- Normalize the steady-state distributions ---
    p_znpr /= trapezoid(p_znpr, gridT)
    p_znr /= -trapezoid(p_znr * gridTtilde, gridTtilde)

    return p_znpr, p_znr


# ===============================================
# Compute mean output power as a function of sampling time
# ===============================================
def compute_means(dg=0.8, nscan=40):
    """
    Computes the steady-state average power as a function of the sampling time (dt).

    For each dt in a logarithmic range, this function:
    - Finds the stationary distributions p_znpr and p_znr.
    - Computes their corresponding power contributions.
    - Stores and returns the total output power.

    Parameters
    ----------
    dg : float
        Drift parameter.
    nscan : int
        Number of dt values to scan.

    Returns
    -------
    np.ndarray
        2-column array [dt, mean_output_power].
    """

    # Domain sizes and grid resolutions
    Lpr = 20.0
    Lr = 20.0
    Npr = 2000
    Nr = 50000

    # Grids for each variable
    gridT = linspace(-Lpr, Lpr, Npr)
    gridTtilde = linspace(-Lr, 0.0, Nr)
    times = logspace(-3.0, 2.0, nscan)
    mean_powers_out = zeros(int(nscan))

    # --- Main loop over sampling times ---
    for idx, time in enumerate(times):

        # Obtain steady-state distributions
        p_znpr, p_znr = find_clean_steady_state_T(gridT, gridTtilde, dg, time, Lpr, Lr, Npr, Nr)

        # Compute the mean power for each process
        t = time
        sigma = np.sqrt(1.0 - exp(-2.0 * t))
        power_znpr = dg * trapezoid(gridT * p_znpr, gridT) / time
        power_znr = -dg * trapezoid(gridTtilde * p_znr * gridTtilde, gridTtilde) / time

        mean_powers_out[idx] = power_znpr - power_znr
        print(power_znpr, power_znr, power_znpr - power_znr)

    return column_stack((times, mean_powers_out))


# ===============================================
# Run the full computation
# ===============================================
clean_results = compute_means()
