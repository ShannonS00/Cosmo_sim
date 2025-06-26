from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

# --- Setup ---
N = 512 * 512
ngrid = 512
boxsize = 1.0
a0 = 0.01
a_final = 0.03
da = 1e-5
snapshot_every = 30
output_dir = Path("Cosmo_frames_da1e-5")
output_dir.mkdir(exist_ok=True)

# power spectrum 
def P_k_example(k):
    k_safe = np.where(k == 0, 1.0, k)  # replace zeros with 1 temporarily
    return np.where(k > 0, k_safe ** -2.0, 0)

# --- Grid of particles ---
def generate_grid_particles(N, boxsize=1.0):
    n_side = int(np.sqrt(N))
    lin = np.linspace(0, boxsize, n_side, endpoint=False)
    x, y = np.meshgrid(lin, lin, indexing='ij')
    return np.column_stack((x.ravel(), y.ravel()))

# --- Gaussian field ---
def enforce_hermitian(fk):
    N = fk.shape[0]
    fk[0, 0] = np.real(fk[0, 0])
    for i in range(1, N):
        for j in range(1, N // 2):
            fk[-i, -j] = np.conj(fk[i, j])
    return fk

def generate_real_space_delta(N, Pk_func):
    kx = np.fft.fftfreq(N) * 2 * np.pi * N
    ky = np.fft.fftfreq(N) * 2 * np.pi * N
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(kx**2 + ky**2)
    amplitude = np.sqrt(Pk_func(k) / 2)
    phase = np.random.uniform(0, 2*np.pi, size=k.shape)
    delta_k = amplitude * (np.cos(phase) + 1j * np.sin(phase))
    return np.fft.ifft2(enforce_hermitian(delta_k)).real

# --- Zeldovich displacement ---
def compute_displacement(delta, boxsize=1.0):
    N = delta.shape[0]
    kx = np.fft.fftfreq(N, d=boxsize / N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=boxsize / N) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1
    phi_k = -np.fft.fft2(delta) / k2
    disp_x = np.fft.ifft2(1j * kx * phi_k).real
    disp_y = np.fft.ifft2(1j * ky * phi_k).real
    return disp_x, disp_y

def apply_displacement(Q, disp_x, disp_y, boxsize=1.0):
    N = disp_x.shape[0]
    grid = np.linspace(0, boxsize, N, endpoint=False)
    interp_x = RegularGridInterpolator((grid, grid), disp_x, bounds_error=False, fill_value=0)
    interp_y = RegularGridInterpolator((grid, grid), disp_y, bounds_error=False, fill_value=0)
    return (Q + np.stack([interp_x(Q), interp_y(Q)], axis=1)) % boxsize

# --- CIC and power spectrum ---
def cic_deposit(X, Y, W, ngrid):
    rho = np.zeros((ngrid, ngrid))
    for x, y, w in zip(X, Y, W):
        x %= 1.0
        y %= 1.0
        il = int(np.floor(x * ngrid)) % ngrid
        jl = int(np.floor(y * ngrid)) % ngrid
        ir = (il + 1) % ngrid
        jr = (jl + 1) % ngrid
        dx = x * ngrid - il
        dy = y * ngrid - jl
        rho[il, jl] += (1-dx)*(1-dy)*w
        rho[il, jr] += (1-dx)*dy*w
        rho[ir, jl] += dx*(1-dy)*w
        rho[ir, jr] += dx*dy*w
    return rho

def compute_power_spectrum(delta, boxsize=1.0):
    N = delta.shape[0]
    delta_k = np.fft.fft2(delta)
    P_k = np.abs(delta_k)**2 / N**4
    kx = np.fft.fftfreq(N, d=boxsize/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=boxsize/N) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(kx**2 + ky**2)
    k_flat = k.flatten()
    P_flat = P_k.flatten()
    k_bins = np.linspace(0, k.max(), N//2)
    k_centers = 0.5*(k_bins[1:] + k_bins[:-1])
    P_bin = np.zeros_like(k_centers)
    counts = np.zeros_like(k_centers)
    for i in range(len(k_flat)):
        idx = np.searchsorted(k_bins, k_flat[i]) - 1
        if 0 <= idx < len(P_bin):
            P_bin[idx] += P_flat[i]
            counts[idx] += 1
    return k_centers, P_bin / np.maximum(counts, 1)

# --- Cosmology ---
def E(a, Omega_m=0.3089, Omega_L=0.6911):
    return np.sqrt(Omega_m/a**3 + Omega_L)

def I_drift(a, da):
    a_mid = a + 0.25 * da
    return 0.5 * da / (a_mid**3 * E(a_mid))

def I_kick(a, da):
    a_mid = a + 0.5 * da
    return da / (a_mid * E(a_mid))

def poisson_solve(delta, a, Omega_m=0.3089):
    N = delta.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    fdelta = np.fft.fft2(delta)
    prefac = -(3.0 / 2.0) * Omega_m / a
    fphi = prefac * fdelta / k2
    fphi[0, 0] = 0.0
    ax = -np.fft.ifft2(1j * kx * fphi).real
    ay = -np.fft.ifft2(1j * ky * fphi).real
    return ax, ay

def cic_interpolate(X, Y, field):
    ngrid = field.shape[0]
    out = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        x, y = X[i] % 1.0, Y[i] % 1.0
        gx, gy = x * ngrid, y * ngrid
        il, jl = int(gx) % ngrid, int(gy) % ngrid
        ir, jr = (il + 1) % ngrid, (jl + 1) % ngrid
        dx, dy = gx - il, gy - jl
        out[i] = (
            (1-dx)*(1-dy)*field[il, jl] +
            (1-dx)*dy*field[il, jr] +
            dx*(1-dy)*field[ir, jl] +
            dx*dy*field[ir, jr]
        )
    return out

def leapfrog_cosmo(Q, V, m, a, da, ngrid=ngrid, Omega_m=0.3089, Omega_L=0.6911):
    W = np.full(Q.shape[0], m)
    I_d1 = I_drift(a, da)
    I_k = I_kick(a, da)
    I_d2 = I_drift(a + 0.5 * da, da)
    Q += V * I_d1
    Q %= 1.0
    rho = cic_deposit(Q[:, 0], Q[:, 1], W, ngrid)
    delta = rho / rho.mean() - 1.0
    ax, ay = poisson_solve(delta, a + 0.5 * da, Omega_m)
    Ax = cic_interpolate(Q[:, 0], Q[:, 1], ax)
    Ay = cic_interpolate(Q[:, 0], Q[:, 1], ay)
    V += np.column_stack((Ax, Ay)) * I_k
    Q += V * I_d2
    Q %= 1.0
    a += da
    return Q, V, a, rho

# --- Generate initial conditions ---
Q0 = generate_grid_particles(N)
delta0 = generate_real_space_delta(ngrid, P_k_example)
dx, dy = compute_displacement(delta0)
Q_init = apply_displacement(Q0, dx, dy)
V_init = np.zeros_like(Q_init)
Q = Q_init.copy()
V = V_init.copy()
a = a0
m = 1.0 / N

# --- Evolve with snapshots ---
step = 0
while a < a_final:
    if step % snapshot_every == 0:
        rho = cic_deposit(Q[:, 0], Q[:, 1], np.full(N, m), ngrid)
        delta = rho / rho.mean() - 1.0
        k_vals, P_vals = compute_power_spectrum(delta)

        # Save density field
        plt.figure(figsize=(5.5, 5))
        plt.imshow(rho, origin='lower', norm=LogNorm(), cmap='plasma', extent=[0, 1, 0, 1])
        plt.colorbar(label="mass / cell")
        plt.title(f"Density Field at a = {a:.4f}")
        plt.xlabel("x"); plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(output_dir / f"rho_{step:05d}.png")
        plt.close()

        # Save power spectrum
        plt.figure()
        plt.loglog(k_vals, P_vals, label="Measured")
        plt.loglog(k_vals, P_k_example(k_vals), '--', label="Input P(k) ∝ k⁻²")
        plt.xlabel("k"); plt.ylabel("P(k)")
        plt.title(f"Power Spectrum at a = {a:.4f}")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(output_dir / f"pk_{step:05d}.png")
        plt.close()

    Q, V, a, _ = leapfrog_cosmo(Q, V, m, a, da, ngrid=ngrid)
    step += 1