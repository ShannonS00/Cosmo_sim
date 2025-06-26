import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit, prange

# ------------------ Kosmologie-Parameter ------------------------------
H0      = 67.74          # km/s/Mpc  (brauchst du erst im Integrator,
Omega_m = 0.3089         #  steht hier nur der Vollständigkeit halber)
Omega_L=0.6911

# ------------------ Simulationseinstellungen --------------------------
N       = 10_000
a0      = 0.05           # Start bei z = 1/a0 - 1 ≈ 19
boxsize = 1.0            # comoving box length in beliebigen Einheiten
ngrid   = 64             # Auflösung des Gitters für die Dichte

# ------------------ Teilchenmasse & Positionen ------------------------
m       = 1.0 / N        # Gesamtmasse = 1 in Code-Units
sigma_x = 0.15           # comoving Streuungen
sigma_y = 0.10

rng = np.random.default_rng(seed=0)
Q0 = np.empty((N, 2))           # 2-D-Beispiel; 3-D ginge analog
Q0[:, 0] = (0.5 + sigma_x * rng.standard_normal(N)) % boxsize
Q0[:, 1] = (0.5 + sigma_y * rng.standard_normal(N)) % boxsize

# ------------------ Physikalische peculiar-Geschwindigkeiten ----------
sigma_v = 1e-4           # km/s oder was immer deine Code-Units sind
V0      = sigma_v * rng.standard_normal((N, 2)) 



def cic_deposit(X, Y, W, ngrid):
    """
    Deposit particle positions X, Y with weights W onto a 2-D grid
    of shape (ngrid, ngrid) using Cloud-In-Cell.
    
    Parameters
    X, Y : 1-D arrays, len = N
        Particle coordinates in [0, 1).
    W    : 1-D array, len = N
        Particle weights (e.g. masses m = 1/N).
    ngrid : int
        Grid resolution in each direction (M).

    Returns:
    rho : 2-D array (ngrid x ngrid)
        Mass density on the mesh.
    """
    rho = np.zeros((ngrid, ngrid))
    for x, y, w in zip(X, Y, W):
        # wrap into the periodic box
        x = np.fmod(1.0 + x, 1.0)
        y = np.fmod(1.0 + y, 1.0)

        # index of left / lower cell
        il = int(np.floor(x * ngrid))
        jl = int(np.floor(y * ngrid))

        # index of right / upper neighbour (periodic)
        ir = (il + 1) % ngrid
        jr = (jl + 1) % ngrid

        # fractional distances from the left / lower edges
        dx = x * ngrid - il
        dy = y * ngrid - jl

        # distribute weight to the four surrounding cells
        rho[il, jl] += (1.0 - dx) * (1.0 - dy) * w
        rho[il, jr] += (1.0 - dx) * dy * w
        rho[ir, jl] += dx * (1.0 - dy) * w
        rho[ir, jr] += dx * dy * w

    return rho

 
def poisson_solve(delta, a, Omega_m=0.3089):
    """
    Solve  ∇²Φ = (3/2 Ω_m / a) δ   on an N×N periodic grid.

    Parameters
    ----------
    delta : 2-D array
        Density contrast ρ/ρ̄ − 1 on the mesh.
    a     : float
        Scale factor of this time-slice.
    Omega_m : float
        Matter density parameter (default: Planck-15).

    Returns
    -------
    phi : 2-D array
        Gravitational potential Φ(q, a).
    ax, ay : 2-D arrays
        Comoving accelerations  −∇_q Φ  (needed in the Kick step).
    """
    N = delta.shape[0]

    # --- wave-number grid --------------------------------------------
    k  = 2.0 * np.pi * np.fft.fftfreq(N)        # 0, 1, …, N/2-1, −N/2, …, −1
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2     = kx**2 + ky**2
    k2[0, 0] = 1.0                              # avoid division by zero

    # --- forward FFT of δ --------------------------------------------
    fdelta = np.fft.fft2(delta)

    # --- Poisson solver in Fourier space -----------------------------
    prefac = -(3.0/2.0) * Omega_m / a           # RHS factor
    fphi   = prefac * fdelta / k2
    fphi[0, 0] = 0.0                            # set mean(Φ) = 0

    # --- accelerations  −∇Φ  in Fourier space ------------------------
    fax = -1j * kx * fphi
    fay = -1j * ky * fphi

    # --- back to real space ------------------------------------------
    phi = np.real(np.fft.ifft2(fphi))
    ax  = np.real(np.fft.ifft2(fax))
    ay  = np.real(np.fft.ifft2(fay))

    return phi, ax, ay

@njit(parallel=True, fastmath=True)
def cic_interpolate(X, Y, field):
    """Adjoint CIC: sample *field* at particle positions (X,Y)."""
    ngrid = field.shape[0]
    N = X.size
    out = np.empty(N, dtype=field.dtype)
    for i in prange(N):
        x = X[i] % 1.0
        y = Y[i] % 1.0
        gx = x * ngrid
        gy = y * ngrid
        il = int(gx) % ngrid
        jl = int(gy) % ngrid
        ir = (il + 1) % ngrid
        jr = (jl + 1) % ngrid
        dx = gx - il
        dy = gy - jl
        out[i] = (
            (1.0 - dx) * (1.0 - dy) * field[il, jl]
            + (1.0 - dx) * dy * field[il, jr]
            + dx * (1.0 - dy) * field[ir, jl]
            + dx * dy * field[ir, jr]
        )
    return out

# Helper functions
def E(a, Omega_m=0.3089, Omega_L=0.6911):
    return np.sqrt(Omega_m/a**3 + (1- Omega_m) ) #Omega_L + (1.0-Omega_m-Omega_L)/a**2)

def I_drift(a, da):                 # ½-Schritt reicht Mid-Point
    a_mid = a + 0.25*da
    return 0.5*da / (a_mid**3 * E(a_mid))

def I_kick(a, da):                  # ganzer Schritt
    a_mid = a + 0.5*da
    return     da / (a_mid * E(a_mid))


def leapfrog_cosmo(Q, V, m, a, da, ngrid=64,
                   Omega_m=0.3089, Omega_L=0.6911):
    """
    One DKD step in scale-factor time for a 2-D test case.

    Parameters
    ----------
    Q  : (N,2) array, comoving positions in [0,1)
    V  : (N,2) array, *physical* peculiar velocity  [length / time]
    m  : scalar,     particle mass in code units    (Σm = 1)
    a  : float,      current scale factor
    da : float,      step size in a
    """

    # -------- constants & weights ------------------------------------
    #W = np.full(Q.shape[0], m)
    W = np.full(N, 1.0 / N)  
    I_d1 = I_drift(a,        da)     # first half
    I_k  = I_kick (a,        da)
    I_d2 = I_drift(a+0.5*da, da)     # second half

    # -------- first DRIFT (Q^{n+½}) ----------------------------------
    Q += V * I_d1
    Q %= 1.0                         # periodic box

    # -------- density field on the mesh ------------------------------
    rho   = cic_deposit(Q[:,0], Q[:,1], W, ngrid)
    delta = rho / rho.mean() - 1.0

    # -------- Poisson solve (Φ, accel on mesh) -----------------------
    phi, ax_m, ay_m = poisson_solve(delta, a+0.5*da, Omega_m)

    # -------- interpolate acceleration to particles ------------------
    Ax = cic_interpolate(Q[:,0], Q[:,1], ax_m)
    Ay = cic_interpolate(Q[:,0], Q[:,1], ay_m)
    A  = np.column_stack((Ax, Ay))   # −∇_q Φ

    # -------- KICK (V^{n+1}) -----------------------------------------
    V += A * I_k                   # minus-sign already in A

    # -------- second DRIFT (Q^{n+1}) ---------------------------------
    Q += V * I_d2
    Q %= 1.0

    # -------- advance scale factor -----------------------------------
    a += da
    return Q, V, a, rho



# Time evoluiton 
# ---------- main evolution routine ------------------------------------
def visualize_scale_factor_evolution(N               = 1000000,
                                     ngrid           = 512,
                                     a0              = 0.05,   # ≈ z = 19
                                     a_final         = 0.08,    # today
                                     da              = 1e-5,
                                     sigma_x         = 0.15,
                                     sigma_y         = 0.10,
                                     sigma_v         = 1e-5,   # velocity width
                                     output_dir      = "frames_c",
                                     snapshot_every  = 30):
    """
    Make PNG snapshots of ρ(a) every `snapshot_every` steps.

    The routine assumes you already imported:
        leapfrog_cosmo, cic_deposit, poisson_solve, cic_interpolate
    """
    # ---- ICs ----------------------------------------------------------
    m   = 1.0 / N
    rng = np.random.default_rng(seed=42)

    Q = np.empty((N, 2))
    Q[:, 0] = (0.5 + sigma_x * rng.standard_normal(N)) % 1.0
    Q[:, 1] = (0.5 + sigma_y * rng.standard_normal(N)) % 1.0
    V = sigma_v * rng.standard_normal((N, 2))        # physical peculiar vel.

    a  = a0
    os.makedirs(output_dir, exist_ok=True)

    n_steps = int(np.ceil((a_final - a0) / da))
    for step in range(n_steps + 1):
        if step % snapshot_every == 0:
            # deposit & visualise BEFORE moving – cheapest place to get ρ
            rho = cic_deposit(Q[:,0], Q[:,1], np.full(N, m), ngrid)
            plt.figure(figsize=(5.5, 5))
            plt.imshow(rho, origin='lower', norm=LogNorm(),
                       cmap='viridis', extent=[0, 1, 0, 1])
            z   = 1.0/a - 1.0
            plt.title(f"a = {a:.4f}   (z = {z:.1f})")
            plt.xlabel("x");  plt.ylabel("y")
            plt.colorbar(label='mass / cell')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/frame_{step:05d}.png")
            plt.close()

        # ---------- one DKD step in scale-factor time ------------------
        Q, V, a, _ = leapfrog_cosmo(Q, V, m, a, da,
                                    ngrid=ngrid,
                                    Omega_m=Omega_m, Omega_L=Omega_L)

        if a >= a_final:        # don’t overshoot if da doesn’t divide evenly
            break

    n_saved = step // snapshot_every + 1
    print(f"Saved {n_saved} frames to “{output_dir}/”.  Final a = {a:.4f}")


visualize_scale_factor_evolution()