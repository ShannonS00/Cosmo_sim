#!/usr/bin/env python3
# --------------------------------------------
# 2-D PIC + cosmology + Zel’dovich ICs
# --------------------------------------------
import numpy as np
from numba import jit, njit, prange
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os, pathlib, imageio

# ================================================================
# 0.  User-level parameters
# ================================================================
N        = 1024 * 1024            # MUST be a perfect square = ngrid_ic**2
ngrid    = 64                    # force-mesh for Poisson
alpha_PS = -2.5                  # slope of P(k) ∝ k^alpha  (try −2.5 … 0)
seed     = 42
# cosmology
a0       = 0.01                  # starting scale factor  (z = 99)
H0       = 0.1                   # code units: H0 = 0.1 ⇒ 1 time-unit ≈ 10 H₀⁻¹
Omega_m  = 0.3
Omega_L  = 0.7
dt       = 0.001                 # cosmic-time step  (Δa/a ≈ 1 %)
n_steps  = 600

# ================================================================
# 1.  Gaussian random field with P(k) ∝ k^alpha  (periodic box)
# ================================================================
def gaussian_random_field_2d(ngrid, alpha, seed=None):
    rng = np.random.default_rng(seed)

    # wavenumber grid
    m = np.fft.fftfreq(ngrid) * ngrid          #  0…N/2-1, -N/2…-1
    kx, ky = np.meshgrid(m, m, indexing='ij')
    k  = np.hypot(kx, ky)

    # complex white noise
    noise = rng.normal(size=(ngrid, ngrid)) + 1j*rng.normal(size=(ngrid, ngrid))

    # skip k = 0 so we never evaluate 0**negative ---
    amp = np.zeros_like(k)
    mask = k > 0.0
    amp[mask] = k[mask] ** (alpha / 2.0)

    delta_k = noise * amp

    # enforce Hermitian symmetry (so ifft is real)
    delta_k = 0.5 * (delta_k + np.conj(np.rot90(delta_k, 2)))

    return np.fft.ifft2(delta_k).real

# ================================================================
# 2.  Zel’dovich displacement / velocity
# ================================================================
def H(a, H0=H0, Om=Omega_m, Ol=Omega_L):
    return H0 * np.sqrt(Om/a**3 + Ol)

def zeldovich_displacement(delta_x, a_ic):
    Nmesh = delta_x.shape[0]
    m  = np.fft.fftfreq(Nmesh) * Nmesh
    kx, ky = np.meshgrid(m, m, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0   # avoid div/0
    delta_k = np.fft.fft2(delta_x)
    psi_kx  = 1j * kx * delta_k / k2
    psi_ky  = 1j * ky * delta_k / k2
    psi_x   = np.fft.ifft2(psi_kx).real
    psi_y   = np.fft.ifft2(psi_ky).real
    psi     = -a_ic * np.stack((psi_x, psi_y), axis=-1)         # minus sign!
    udotD_D = H(a_ic)        # dD/dt / D   (for EdS; fine at very high-z)
    vel     = udotD_D * psi
    return psi, vel

# ================================================================
# 3.  PIC utilities (deposit / interpolate)
# ================================================================
@jit(nopython=True, fastmath=True)
def cic_deposit(X, Y, W, ngrid):
    rho = np.zeros((ngrid, ngrid))
    for x, y, w in zip(X, Y, W):
        x = np.fmod(1.0 + x, 1.0)
        y = np.fmod(1.0 + y, 1.0)
        il = int(np.floor(x * ngrid))
        jl = int(np.floor(y * ngrid))
        ir = (il + 1) % ngrid
        jr = (jl + 1) % ngrid
        dx = x * ngrid - il
        dy = y * ngrid - jl
        rho[il, jl] += (1.0 - dx)*(1.0 - dy)*w
        rho[il, jr] += (1.0 - dx)*dy       *w
        rho[ir, jl] += dx       *(1.0 - dy)*w
        rho[ir, jr] += dx       *dy       *w
    return rho

@njit(parallel=True, fastmath=True)
def cic_interpolate(X, Y, field):
    ngrid = field.shape[0]
    out   = np.empty(X.size, dtype=field.dtype)
    for i in prange(X.size):
        x = X[i] % 1.0
        y = Y[i] % 1.0
        gx, gy = x*ngrid, y*ngrid
        il, jl = int(gx)%ngrid, int(gy)%ngrid
        ir, jr = (il+1)%ngrid, (jl+1)%ngrid
        dx, dy = gx - il, gy - jl
        out[i] = ((1-dx)*(1-dy)*field[il,jl] + (1-dx)*dy*field[il,jr] +
                   dx   *(1-dy)*field[ir,jl] +  dx   *dy*field[ir,jr])
    return out

# ================================================================
# 4.  Poisson solver with 1/a² factor
# ================================================================
def poisson_solve(rho, a):
    N = rho.shape[0]
    m  = np.concatenate((np.arange(0, N//2), np.arange(-N//2, 0)))
    k  = 2.0*np.pi*m
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    frho = np.fft.fft2(rho)
    with np.errstate(divide='ignore', invalid='ignore'):
        fphi = -frho / k2
    fphi[0,0] = 0.0
    fax = -1j * kx * fphi
    fay = -1j * ky * fphi
    ax  = np.fft.ifft2(fax).real / a**2
    ay  = np.fft.ifft2(fay).real / a**2
    return ax, ay

# ================================================================
# 5.  Leap-frog step with cosmology
# ================================================================
def advance_a(a, dt):
    a_half = a + 0.5*dt*a*H(a)
    a_new  = a +       dt*a_half*H(a_half)
    return a_half, a_new

def step_cosmo(X, P, mpart, a, *, ngrid=ngrid, dt=dt):
    a_half, a_new = advance_a(a, dt)
    H_half = H(a_half)
    # half-drift
    X += 0.5*dt * P/(mpart * a_half)
    X %= 1.0
    # density + field
    rho = cic_deposit(X[:,0], X[:,1], np.full(X.shape[0], mpart), ngrid)
    ax, ay = poisson_solve(rho, a_half)
    # interpolate
    A = np.empty_like(X)
    A[:,0] = cic_interpolate(X[:,0], X[:,1], ax)
    A[:,1] = cic_interpolate(X[:,0], X[:,1], ay)
    # kick   (gravity – Hubble drag)
    P += mpart*dt*A - H_half*dt*P
    # second half-drift
    X += 0.5*dt * P/(mpart * a_new)
    X %= 1.0
    return X, P, rho, a_new

# ================================================================
# 6.  Build Zel’dovich initial particle load
# ================================================================
print("Generating initial conditions …")
ng_ic = int(np.sqrt(N))
assert ng_ic**2 == N, "N must be a perfect square"
q = np.indices((ng_ic, ng_ic)).T.reshape(-1, 2) / ng_ic + 0.5/ng_ic

delta0 = gaussian_random_field_2d(ng_ic, alpha_PS, seed)
psi, vel = zeldovich_displacement(delta0, a0)

X = (q + psi.reshape(-1,2)) % 1.0
mpart = 1.0 / N
P = mpart * vel.reshape(-1,2) * a0      # canonical momenta

# ================================================================
# 7.  Time integration
# ================================================================

print("Evolving …")
frames, snapshot_every = [], 300      # grab a handful of frames
a = a0; t = 0.0
for step in range(n_steps+1):
    if step % snapshot_every == 0:
        rho = cic_deposit(X[:,0], X[:,1], np.full(N, mpart), ngrid)
        frames.append((rho.copy()/rho.mean(), a, t))       # ρ/⟨ρ⟩
        print(f"step {step:5d}/{n_steps},  a = {a:.4f}")
    X, P, rho, a = step_cosmo(X, P, mpart, a, ngrid=ngrid, dt=dt)
    t += dt
rho_last, *_ = frames[-1]          # your final snapshot
print(rho_last.min(), rho_last.max(), rho_last.std())

# ================================================================
# 8.  Quick plot of the last snapshot
# ================================================================
rho_last, a_last, _ = frames[-1]
plt.figure(figsize=(5,5))
plt.imshow(rho_last, origin='lower', cmap='magma',
           norm=LogNorm(), extent=[0,1,0,1])
plt.colorbar(label='ρ / ⟨ρ⟩')
plt.title(f'a = {a_last:.3f}   z = {1/a_last-1:.1f}')
plt.xlabel('x  [comoving units]')
plt.ylabel('y')
plt.tight_layout()
plt.show()