# Cosmo simulation 

# WS 10

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import mpmath as mp
import os
from matplotlib.colors import LogNorm

# Use it
N = 1000000
sigma_x, sigma_y = 0.10, 0.10 #isotropic initial conditions
sigma_p_factor = 1e-4
m = 1.0 / N
sigma_p = sigma_p_factor * m
ngrid = 64
dt = 1.0
np.random.seed(0)
X0 = np.empty((N, 2))
X0[:, 0] = (0.5 + sigma_x * np.random.randn(N)) % 1.0
X0[:, 1] = (0.5 + sigma_y * np.random.randn(N)) % 1.0
P0 = sigma_p * np.random.randn(N, 2)


'''---------NEW----------'''
# Cosmology :

def H(a, H0=0.1, Om=0.3, Ol=0.7):
    """Flat ΛCDM (radiation ignored)   units: H0 in code-time-1"""
    return H0*np.sqrt(Om/a**3 + Ol)

def advance_a(a, dt, H0, Om, Ol):
    # 2-stage (KDK) leap-frog for the scale factor
    a_half = a + 0.5*dt*a*H(a, H0, Om, Ol)   # drift
    a_new  = a +       dt*a_half*H(a_half, H0, Om, Ol)   # kick
    return a_half, a_new



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

rho = cic_deposit(X0[:,0], X0[:,1], weights, ngrid)


''' Update poisson solver to account for a '''
def poisson_solve(rho, a):
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
    ax  = np.real(np.fft.ifft2(fax)) / a**2
    ay  = np.real(np.fft.ifft2(fay)) / a**2

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

''' Update leapfrog PIC step to include the scale factor a '''
# 2.  Leap-frog integrator following the requested seven steps
def leapfrog_pic_step(X, P, m, a, ngrid=64, dt=1.0, H0 = 0.1, Om=0.3, Ol=0.7):#

    # scale factor a
    a_half, a_new = advance_a(a, dt, H0, Om, Ol)
    H_half = H(a_half, H0, Om, Ol)

    W = np.full(X.shape[0], m)

    # 1. half-drift (update with a_half)
    X += 0.5 * dt * P / (m*a_half)
    X %= 1.0 # inside boundaries 

    # 2. deposit
    rho = cic_deposit(X[:, 0], X[:, 1], W, ngrid)
    rho = rho / np.mean(rho)  # normalize to mean density

    # 3. solve field
    _, ax, ay = poisson_solve(rho, a_half)

    # 4. interpolate
    A = np.empty_like(X)
    A[:, 0] = cic_interpolate(X[:, 0], X[:, 1], ax)
    A[:, 1] = cic_interpolate(X[:, 0], X[:, 1], ay)

    # 5. kick --> updated
    P += m * dt * A - H_half * dt * P

    # 6. second half-drift
    X += 0.5 * dt * P / (m* a_new)
    X %= 1.0

    return X, P, rho, a_new


# --- initial conditions ----------------------
a0      = 0.01               # start at z = 99
H0      = 0.1
Omega_m = 0.3
Omega_L = 0.7

X  = X0.copy()
P  = P0.copy()
a  = a0

from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------
# 1) run the simulation and keep snapshots
# ---------------------------------------------
n_steps        = 300
snapshot_every = 10          # grab every 10th step
frames         = []          # list of (rho, a, t)

X, P, a = X0.copy(), P0.copy(), a0
t       = 0.0
for step in range(n_steps + 1):
    if step % snapshot_every == 0:
        rho = cic_deposit(X[:,0], X[:,1],
                          np.full(X.shape[0], m), ngrid)
        frames.append((rho.copy(), a, t))

    X, P, rho, a = leapfrog_pic_step(X, P, m, a, dt=dt,H0=H0, Om=Omega_m, Ol=Omega_L)
    t += dt                      # cosmic-time accumulator

# ---------------------------------------------
# 2) quick single-frame inspection
# ---------------------------------------------
rho0, a0_plot, t0 = frames[0]
plt.figure(figsize=(5,5))
plt.imshow(rho0, origin='lower', norm=LogNorm(),
           cmap='viridis', extent=[0,1,0,1])
plt.colorbar(label='ρ / ⟨ρ⟩')
plt.title(f'a = {a0_plot:.3f}  (z = {1/a0_plot - 1:.1f})')
plt.xlabel('x [comoving box units]')
plt.ylabel('y')
plt.tight_layout()
plt.show()

# ---------------------------------------------
fig, ax = plt.subplots(figsize=(5,5))
im  = ax.imshow(frames[0][0], origin='lower', norm=LogNorm(),
                cmap='viridis', extent=[0,1,0,1])
txt = ax.text(0.02, 0.95, '', transform=ax.transAxes,
              color='w', ha='left', va='top')

def update(i):
    rho_i, a_i, t_i = frames[i]
    im.set_data(rho_i)
    txt.set_text(f'a = {a_i:.3f}  (z = {1/a_i - 1:.1f})')
    return im, txt

ani = FuncAnimation(fig, update, frames=len(frames), blit=True)
ani.save('collapse_cosmo.gif', writer=PillowWriter(fps=8))
plt.close(fig)
print('saved collapse_cosmo.gif')
'''
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

def visualize_time_evolution(N=10**6, M=64, dt=1.0, n_steps=300, 
                             sigma_x=0.15, sigma_y=0.10, sigma_p_factor=1e-4,
                             output_dir="frames", snapshot_interval=10):
    
    # Initial conditions
    m = 1.0 / N
    sigma_p = sigma_p_factor * m
    np.random.seed(42)
    X = np.empty((N, 2))
    X[:, 0] = (0.5 + sigma_x * np.random.randn(N)) % 1.0
    X[:, 1] = (0.5 + sigma_y * np.random.randn(N)) % 1.0
    P = sigma_p * np.random.randn(N, 2)

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    for step in range(n_steps + 1):
        X, P, rho = leapfrog_pic_step(X, P, m, ngrid=M, dt=dt)

        if step % snapshot_interval == 0:
            plt.figure(figsize=(6, 5))
            plt.imshow(rho, origin='lower', cmap='viridis', norm=LogNorm(), extent=[0, 1, 0, 1])
            plt.title(f"t = {step * dt:.1f}, M = {M}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.colorbar(label='Density')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/frame_{step:04d}.png")
            plt.close()

    print(f"\nSaved {n_steps // snapshot_interval + 1} frames to '{output_dir}/'")

visualize_time_evolution(M=64, sigma_p_factor=1e-4, n_steps=300, snapshot_interval=10)
'''