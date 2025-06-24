#!/usr/bin/env python3
"""
Cosmo‑PIC‑2D  ▸  a **one‑file sandbox** for cosmological N‑body concepts
===========================================================================
Key ideas
---------
* Gaussian random field → 1‑LPT (Zel’dovich) displacements  
* Leap‑frog in comoving coords with the Hubble drag term  
* Pure Particle‑Mesh forces (FFT Poisson) on a square grid  
* Tiny dependency set: `numpy`, `numba`, `matplotlib`, `imageio`, `tqdm`

Why another toy code?
--------------------
You can run a complete expansion‑era simulation in **seconds** on a laptop, 
then tweak the physics at Python speed.  Perfect for teaching or rapid‐fire
experiments before diving into 3‑D production codes.

Quick start
-----------
```bash
pip install numpy numba matplotlib imageio tqdm
python cosmo_pic_2d.py demo        # 8×8 k‑cell box, 400 steps, live plot
python cosmo_pic_2d.py compare     # compare three mesh sizes side‑by‑side
python cosmo_pic_2d.py movie       # build PNGs + timelapse.gif in ./frames
```
All tunables have short CLI flags; run `python cosmo_pic_2d.py --help`.
"""
from __future__ import annotations

# ────────────────────────────────────────────── imports ──────────────────────
import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from numba import jit, njit, prange
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio.v2 as imageio
#from tqdm import tqdm

# ─────────────────────────────────── physical + numerical defaults ───────────
@dataclass
class SimParams:
    N: int = 256 * 256                # ↳ must be a perfect square
    mesh: int = 64                    # force‑mesh size (same in x & y)
    n_steps: int = 400
    dt: float = 0.001
    a0: float = 0.01                  # start at z≈99
    alpha_PS: float = -2.5            # P(k) ∝ k^{alpha}
    seed: int = 42

    # cosmology (matter + Lambda, flat)
    H0: float = 0.1
    Omega_m: float = 0.3
    Omega_L: float = 0.7

# ───────────────────────────────── fluid helper functions ────────────────────

def H(a: float, p: SimParams) -> float:
    """Comoving H(a) in *code units* (p.H0 sets time‑scale)."""
    return p.H0 * np.sqrt(p.Omega_m / a**3 + p.Omega_L)

# ────────────────────────────────── IC generator ─────────────────────────────

def gaussian_random_field_2d(ngrid: int, alpha: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.fft.fftfreq(ngrid) * ngrid
    kx, ky = np.meshgrid(m, m, indexing="ij")
    k = np.hypot(kx, ky)

    white = rng.normal(size=(ngrid, ngrid)) + 1j * rng.normal(size=(ngrid, ngrid))
    amp = np.zeros_like(k)
    mask = k > 0
    amp[mask] = k[mask] ** (alpha / 2.0)

    delta_k = white * amp
    delta_k = 0.5 * (delta_k + np.conj(np.rot90(delta_k, 2)))  # enforce Hermitian symmetry
    return np.fft.ifft2(delta_k).real


def zeldovich_displacement(delta_x: np.ndarray, a_ic: float, p: SimParams) -> Tuple[np.ndarray, np.ndarray]:
    """Return displacement field *psi* and velocity field (1‑LPT)."""
    Nmesh = delta_x.shape[0]
    m = np.fft.fftfreq(Nmesh) * Nmesh
    kx, ky = np.meshgrid(m, m, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1  # avoid div/0

    delta_k = np.fft.fft2(delta_x)
    psi_kx = 1j * kx * delta_k / k2
    psi_ky = 1j * ky * delta_k / k2
    psi_x = np.fft.ifft2(psi_kx).real
    psi_y = np.fft.ifft2(psi_ky).real

    psi = -a_ic * np.stack((psi_x, psi_y), axis=-1)  # minus sign!
    vel = H(a_ic, p) * psi                           # 1‑LPT velocity ≈ a·Ȧ/Ȧ · psi
    return psi, vel

# ─────────────────────────────── CIC deposit & interpolate ───────────────────
@jit(nopython=True, fastmath=True)
def cic_deposit(x: np.ndarray, y: np.ndarray, w: np.ndarray, ngrid: int) -> np.ndarray:
    rho = np.zeros((ngrid, ngrid))
    for xi, yi, wi in zip(x, y, w):
        xi = np.fmod(1.0 + xi, 1.0)
        yi = np.fmod(1.0 + yi, 1.0)
        il = int(np.floor(xi * ngrid))
        jl = int(np.floor(yi * ngrid))
        ir = (il + 1) % ngrid
        jr = (jl + 1) % ngrid
        dx = xi * ngrid - il
        dy = yi * ngrid - jl
        rho[il, jl] += (1 - dx) * (1 - dy) * wi
        rho[il, jr] += (1 - dx) * dy * wi
        rho[ir, jl] += dx * (1 - dy) * wi
        rho[ir, jr] += dx * dy * wi
    return rho

@njit(parallel=True, fastmath=True)
def cic_interpolate(x: np.ndarray, y: np.ndarray, field: np.ndarray) -> np.ndarray:
    ngrid = field.shape[0]
    out = np.empty(x.size, dtype=field.dtype)
    for i in prange(x.size):
        xi = x[i] % 1.0
        yi = y[i] % 1.0
        gx, gy = xi * ngrid, yi * ngrid
        il, jl = int(gx) % ngrid, int(gy) % ngrid
        ir, jr = (il + 1) % ngrid, (jl + 1) % ngrid
        dx, dy = gx - il, gy - jl
        out[i] = ((1 - dx) * (1 - dy) * field[il, jl]
                   + (1 - dx) * dy * field[il, jr]
                   + dx * (1 - dy) * field[ir, jl]
                   + dx * dy * field[ir, jr])
    return out

# ────────────────────────────── Poisson solver (FFT) ─────────────────────────

def poisson_force(rho: np.ndarray, a: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return acceleration fields a_x, a_y given overdensity *rho* (CIC mass)."""
    N = rho.shape[0]
    m = np.concatenate((np.arange(0, N // 2), np.arange(-N // 2, 0)))
    k = 2.0 * np.pi * m
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2

    frho = np.fft.fft2(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        fphi = -frho / k2
    fphi[0, 0] = 0.0

    ax = np.fft.ifft2(-1j * kx * fphi).real / a**2
    ay = np.fft.ifft2(-1j * ky * fphi).real / a**2
    return ax, ay

# ─────────────────────────────── leap‑frog integrator ────────────────────────

def advance_a(a: float, dt: float, p: SimParams) -> Tuple[float, float]:
    a_half = a + 0.5 * dt * a * H(a, p)
    a_new = a + dt * a_half * H(a_half, p)
    return a_half, a_new

def step_cosmo(X: np.ndarray, P: np.ndarray, mp: float, a: float, p: SimParams) -> Tuple[np.ndarray, np.ndarray, float]:
    a_half, a_new = advance_a(a, p.dt, p)
    H_half = H(a_half, p)

    # half‑drift
    X += 0.5 * p.dt * P / (mp * a_half)
    X %= 1.0

    # density → force
    rho = cic_deposit(X[:, 0], X[:, 1], np.full(X.shape[0], mp), p.mesh)
    ax, ay = poisson_force(rho, a_half)

    # interpolate acceleration → particles
    A = np.empty_like(X)
    A[:, 0] = cic_interpolate(X[:, 0], X[:, 1], ax)
    A[:, 1] = cic_interpolate(X[:, 0], X[:, 1], ay)

    # kick (gravity − Hubble drag)
    P += mp * p.dt * A - H_half * p.dt * P

    # second half‑drift
    X += 0.5 * p.dt * P / (mp * a_new)
    X %= 1.0
    return X, P, a_new

# ──────────────────────────────── top‑level simulation ───────────────────────

def run_simulation(par: SimParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the full N‑step simulation and return final (X, P, rho, a)."""
    ng_ic = int(np.sqrt(par.N))
    assert ng_ic**2 == par.N, "N must be a perfect square"

    q = np.indices((ng_ic, ng_ic)).T.reshape(-1, 2) / ng_ic + 0.5 / ng_ic
    delta0 = gaussian_random_field_2d(ng_ic, par.alpha_PS, par.seed)
    psi, vel = zeldovich_displacement(delta0, par.a0, par)

    X = (q + psi.reshape(-1, 2)) % 1.0
    mp = 1.0 / par.N
    P = mp * vel.reshape(-1, 2) * par


par = SimParams(N=128*128, mesh=64, n_steps=1500)
X, P, rho, a = run_simulation(par)
show_density(rho, a) 