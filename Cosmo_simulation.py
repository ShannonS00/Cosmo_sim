# Cosmo simulation 

# WS 10

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import mpmath as mp

# parameters 
N       = 100000            # number of particles
sigma_x = 0.15
sigma_y = 0.10
m       = 1.0 / N           # particle mass
sigma_p = m * 1e-4          # momentum-space width (very narrow!)

# sample configuration space 
# draw from N(0.5, σ²) for x and y, then wrap into [0,1)
X = np.empty((N, 2))
X[:, 0] = 0.5 + sigma_x * np.random.randn(N)   # x–coordinate
X[:, 1] = 0.5 + sigma_y * np.random.randn(N)   # y–coordinate
X = np.fmod(X, 1.0)                     # periodic domain [0,1)²

# sample momentum space 
P = np.empty_like(X)
P[:]  = sigma_p * np.random.randn(N, 2)        # (p_x , p_y)

# X and P are now both shape (N, 2).  Each particle has mass m = 1/N.
print(X.shape, P.shape, m, sigma_p)



@jit(nopython=True, fastmath=True)
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

ngrid = 64
weights = np.full(N, 1.0 / N)       # each particle has mass m = 1/N

rho = cic_deposit(X[:,0], X[:,1], weights, ngrid)

# optional sanity check: total mass should be 1 (within round-off)
print("Σρ ΔxΔy =", rho.sum())       # should print ≈ 1.0


def poisson_solve(rho):
    """
    Solve ∇²ϕ = –ρ (Poisson equation) on anNxN periodic grid
    and return:
        ϕ   : potential     (NxN real array)
        a_x : acceleration x (NxN real array)
        a_y : acceleration y (NxN real array)
    """
    N = rho.shape[0]

    # 1. wave-number grid
    m  = np.concatenate((np.arange(0, N//2), np.arange(-N//2, 0)))
    k  = 2.0 * np.pi * m
    kx, ky = np.meshgrid(k, k, indexing='ij')

    # 2. forward FFT of the charge density
    frho = np.fft.fft2(rho)

    # 3. solve in Fourier space  ϕ̂ = −ρ̂ / (kx²+ky²)
    k2   = kx**2 + ky**2
    with np.errstate(divide='ignore', invalid='ignore'):
        fphi = -frho / k2
    fphi[0, 0] = 0.0                     # fix the mean (ϕ̂₀₀) to 0

    # 4. acceleration in Fourier space: â = −i k ϕ̂
    fax = -1j * kx * fphi
    fay = -1j * ky * fphi

    # 5. inverse FFT back to real space
    phi = np.real(np.fft.ifft2(fphi))
    ax  = np.real(np.fft.ifft2(fax))
    ay  = np.real(np.fft.ifft2(fay))

    return phi, ax, ay


from numba import njit, prange

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

 
# 2.  Leap-frog integrator following the requested seven steps

def leapfrog_pic(X, P, m, ngrid=64, dt=1.0, n_steps=200):
    """
    Leapfrog (drift-kick-drift) Vlasov-Poisson integrator using user-supplied
    cic_deposit, poisson_solve, cic_interpolate.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    P = np.ascontiguousarray(P, dtype=np.float64)
    N = X.shape[0]
    W = np.full(N, m, dtype=np.float64)          # equal particle masses

    for _ in range(n_steps):
        # 1. half-drift
        X += 0.5 * dt * P / m
        X %= 1.0

        # 2. deposit density ρ^{n+1/2}
        rho = cic_deposit(X[:, 0], X[:, 1], W, ngrid)

        # 3. mesh accelerations a^{n+1/2}
        _, ax, ay = poisson_solve(rho)

        # 4. inverse-interpolate to particle accelerations A^{n+1/2}
        A = np.empty_like(X)
        A[:, 0] = cic_interpolate(X[:, 0], X[:, 1], ax)
        A[:, 1] = cic_interpolate(X[:, 0], X[:, 1], ay)

        # 5. kick full step for momenta
        P += m * dt * A

        # 6. second half-drift
        X += 0.5 * dt * P / m
        X %= 1.0

        # loop proceeds to next n

    # final density for visualisation
    rho_final = cic_deposit(X[:, 0], X[:, 1], W, ngrid)
    return X, P, rho_final

# Use it
N = 1000000
sigma_x, sigma_y = 0.15, 0.10
sigma_p_factor = 1e-4
m = 1.0 / N
sigma_p = sigma_p_factor * m
ngrid = 64
dt = 1.0
n_steps = 300

np.random.seed(0)
X0 = np.empty((N, 2))
X0[:, 0] = (0.5 + sigma_x * np.random.randn(N)) % 1.0
X0[:, 1] = (0.5 + sigma_y * np.random.randn(N)) % 1.0
P0 = sigma_p * np.random.randn(N, 2)

# evolve
Xf, Pf, rho_f = leapfrog_pic(X0, P0, m, ngrid=ngrid, dt=dt, n_steps=n_steps)

# 4.  Visualise final density
plt.figure(figsize=(5, 4))
plt.imshow(
    rho_f.T,
    origin="lower",
    cmap="magma",
    interpolation="bilinear",
   # norm=LogNorm(),
   )

plt.title(fr"Particle density, N={N} at $t = {n_steps} $")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.colorbar(label="rho")
plt.tight_layout()
plt.show()