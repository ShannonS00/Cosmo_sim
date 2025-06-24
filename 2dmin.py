import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
import time
import pathlib

# ------------------------------------------------------------
# 1. Simulation parameters  (feel free to edit)
# ------------------------------------------------------------
Lbox      = 20.0          # comoving box size  [h^-1 Mpc]
Ngrid     = 128           # grid points per dimension  (Npart = Ngrid^2)
nsteps    = 140           # number of ln a steps (Δln a ≃ 0.03 → from a_i≈0.02 to a=1)
dln_a     = 0.03          # step in ln a  (≈2.9 % in a each step)
Omega_m   = 0.315         # Planck‑2018 ΛCDM
Omega_L   = 0.685
h         = 0.674
Gcode     = 1.0           # 4πG has been absorbed → Poisson: ∇²Φ = a² δρ
seed      = 1234
z_initial = 49.0
out_steps = [0, 70, 139]  # which leap‑frog steps to snapshot

# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------
def H(a, H0=1.0):
    """Dimension‑less Hubble parameter  H(a)/H0  (radiation neglected)."""
    return np.sqrt(Omega_m * a**(-3) + Omega_L)

def k_arrays(N, L):
    """Return 2‑D arrays kx, ky, and k^2 on an N×N FFT grid."""
    k1d = 2 * np.pi * fftfreq(N, d=L/N)  # wave‑numbers in 2π/L units
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid div/0; k=0 mode is ignored later
    return kx, ky, k2

def gaussian_random_field(N, L, Pk_func, rng):
    """Return δ_k on an FFT grid with power spectrum P(k)."""
    kx, ky, k2 = k_arrays(N, L)
    k = np.sqrt(k2)
    amp = np.sqrt(Pk_func(k) / 2.0)
    phase = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    delta_k = amp * phase
    # enforce Hermitian symmetry so δ(x) is real
    delta_k = 0.5 * (delta_k + np.conj(np.flipud(np.fliplr(delta_k))))
    delta_k[0, 0] = 0.0
    return delta_k

def simple_power_law(k, k0=0.5, n_s=-2.0):
    """Toy power spectrum  P(k) ∝ (k/k0)^{n_s}.  k, k0 in h Mpc⁻¹."""
    return (k / k0)**n_s

# CIC deposit (scalar) and interpolation (vector)
def deposit_CIC(pos, N, L):
    """Deposit masses (all 1) via CIC onto an N×N grid. Returns density field."""
    rho = np.zeros((N, N), dtype=np.float64)
    cell = N / L
    for x, y in pos:
        gx = x * cell
        gy = y * cell
        i = int(np.floor(gx)) % N
        j = int(np.floor(gy)) % N
        dx = gx - i
        dy = gy - j
        # Weights
        w00 = (1 - dx) * (1 - dy)
        w10 = dx * (1 - dy)
        w01 = (1 - dx) * dy
        w11 = dx * dy
        rho[i % N,     j % N    ] += w00
        rho[(i+1) % N, j % N    ] += w10
        rho[i % N,     (j+1) % N] += w01
        rho[(i+1) % N, (j+1) % N] += w11
    return rho * (N / L)**2   # mass / cell‑area  → density (comoving)

def interpolate_CIC(pos, grid_field, N, L):
    """CIC‑interpolate *vector* grid_field at particle positions.
       grid_field has shape (N,N,2) for forces."""
    cell = N / L
    Fx = np.empty(len(pos))
    Fy = np.empty(len(pos))
    Nx = grid_field.shape[0]
    for idx, (x, y) in enumerate(pos):
        gx = x * cell
        gy = y * cell
        i = int(np.floor(gx)) % Nx
        j = int(np.floor(gy)) % Nx
        dx = gx - i
        dy = gy - j
        w00 = (1 - dx) * (1 - dy)
        w10 = dx * (1 - dy)
        w01 = (1 - dx) * dy
        w11 = dx * dy
        Fx[idx] = (w00 * grid_field[i % Nx,     j % Nx,     0]
                 + w10 * grid_field[(i+1) % Nx, j % Nx,     0]
                 + w01 * grid_field[i % Nx,     (j+1) % Nx, 0]
                 + w11 * grid_field[(i+1) % Nx, (j+1) % Nx, 0])
        Fy[idx] = (w00 * grid_field[i % Nx,     j % Nx,     1]
                 + w10 * grid_field[(i+1) % Nx, j % Nx,     1]
                 + w01 * grid_field[i % Nx,     (j+1) % Nx, 1]
                 + w11 * grid_field[(i+1) % Nx, (j+1) % Nx, 1])
    return np.vstack((Fx, Fy)).T

def plot_particles(pos, L, title, fname, s=1.0, color='k'):
    plt.figure(figsize=(4,4))
    plt.scatter(pos[:,0] - L/2, pos[:,1] - L/2,
                s=s, lw=0, c=color)
    plt.xlim(-L/2, L/2)
    plt.ylim(-L/2, L/2)
    plt.xlabel(r'$x\;[h^{-1}\,\mathrm{Mpc}]$')
    plt.ylabel(r'$y$')
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  wrote {fname}")

# ------------------------------------------------------------
# 3. Initial conditions  (Zel’dovich @ high‑z)
# ------------------------------------------------------------
rng = np.random.default_rng(seed)
a_initial = 1.0 / (1.0 + z_initial)
Np = Ngrid * Ngrid
print(f"# Particles: {Np:,d}  grid: {Ngrid}x{Ngrid}")

# Generate δ(k) with a toy power‑law P(k)
delta_k = gaussian_random_field(Ngrid, Lbox, simple_power_law, rng)
delta_x = np.real(ifft2(delta_k))

# Solve ∇² φ = δ   →  φ_k = -δ_k/k^2
kx, ky, k2 = k_arrays(Ngrid, Lbox)
phi_k = - delta_k / k2
phi_k[0,0] = 0.0
phi_x = np.real(ifft2(phi_k))

# Displacement field  s = ∇φ
s_kx = 1j * kx * phi_k
s_ky = 1j * ky * phi_k
sx = np.real(ifft2(s_kx))
sy = np.real(ifft2(s_ky))
s_grid = np.stack((sx, sy), axis=-1)

# Regular lattice positions
grid = np.linspace(0, Lbox, Ngrid, endpoint=False)
qx, qy = np.meshgrid(grid, grid, indexing="ij")
q = np.vstack((qx.ravel(), qy.ravel())).T   # (Np,2)

# Interpolate s( q )
s_q = interpolate_CIC(q, s_grid, Ngrid, Lbox)

# Growth factor (approx)  D ≈ a  (good enough at high z)
D_i   = a_initial
H_i   = H(a_initial)
p_i = (a_initial**4 * H_i) * s_q   # canonical momentum a^2 dx/dη ; dx/dη = a^2 H ∇φ

# Apply displacement
x_i = (q + D_i * s_q) % Lbox

# ------------------------------------------------------------
# 4. Time integration  Kick‑Drift‑Kick
# ------------------------------------------------------------
positions = x_i.copy()
momenta   = p_i.copy()
a         = a_initial

# pre‑compute mesh wave‑numbers for Poisson solve
kx, ky, k2 = k_arrays(Ngrid, Lbox)
k2[0,0] = 1.0   # avoid 0

start_time = time.time()
for step in range(nsteps):
    # ---------- DRIFT part 1 ----------
    Ha   = H(a)                 # H(a)/H0, but H0 is absorbed in units
    deta = dln_a / (a * Ha)     # Δη  from  Δln a
    positions = (positions + deta * momenta / a**2) % Lbox

    # ---------- MASS DEPOSIT ----------
    rho = deposit_CIC(positions, Ngrid, Lbox)
    rho -= np.mean(rho)         # subtract mean  → δρ

    # ---------- POISSON SOLVE ----------
    delta_k = fft2(rho)
    Phi_k   = - (a**2) * delta_k / k2
    Phi_k[0,0] = 0.0
    Fx_k = 1j * kx * Phi_k
    Fy_k = 1j * ky * Phi_k
    Fx = np.real(ifft2(Fx_k))
    Fy = np.real(ifft2(Fy_k))
    force_grid = np.stack((Fx, Fy), axis=-1)

    # ---------- KICK ----------
    forces = interpolate_CIC(positions, force_grid, Ngrid, Lbox)
    momenta -= deta * a * forces
    snapshots = {}

    # finish frame?
    if step in out_steps:
        snapshots[step] = positions.copy()
        plot_particles(positions, Lbox,
                       title=f"Step {step}   a={a:.3f}",
                       fname=f"frame_{step:03d}.png",
                       s=0.3)
                    

    # update scale factor for next step (end of loop)
    pos_final = positions.copy()
    a *= np.exp(dln_a)

elapsed = time.time() - start_time
print(f"Integration done in {elapsed:.1f}s")

# ------------------------------------------------------------
# 5. Final plot – overlay selected particles
# ------------------------------------------------------------
select = rng.choice(len(positions), size=len(positions)//5, replace=False)
plot_particles(positions, Lbox,
               title="Final (sampled blue)",
               fname="frame_final_overlay.png",
               s=0.6, color='steelblue')



# ------------------------------------------------------------
# 6. Two-panel comparison figure  (like the one you showed)
# ------------------------------------------------------------
out_steps = [0, 70, 139] 
mid_step   = 70                     # left panel
pos_mid    = snapshots[mid_step]    # stored during the loop
pos_final  = positions              # final positions (a ≈ 1)

# choose ~20 % of particles for the blue overlay
overlay_idx = rng.choice(len(pos_final), size=len(pos_final)//5, replace=False)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.2))

# LEFT  – plain black
axs[0].scatter(pos_mid[:,0] - Lbox/2, pos_mid[:,1] - Lbox/2,
               s=0.25, lw=0, c='k')
axs[0].set_xlim(-Lbox/2, Lbox/2)
axs[0].set_ylim(-Lbox/2, Lbox/2)
axs[0].set_aspect('equal')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

# RIGHT – full set in grey + highlighted subset in blue
axs[1].scatter(pos_final[:,0]  - Lbox/2, pos_final[:,1]  - Lbox/2,
               s=0.25, lw=0, c='0.65')                      # light grey back-ground points
axs[1].scatter(pos_final[overlay_idx,0] - Lbox/2,
               pos_final[overlay_idx,1] - Lbox/2,
               s=0.25, lw=0, c='royalblue')                 # blue overlay
axs[1].set_xlim(-Lbox/2, Lbox/2)
axs[1].set_ylim(-Lbox/2, Lbox/2)
axs[1].set_aspect('equal')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

plt.tight_layout()
plt.savefig("two_panel_web.png", dpi=150)
plt.close()
print("  wrote two_panel_web.png")

print("All done.")