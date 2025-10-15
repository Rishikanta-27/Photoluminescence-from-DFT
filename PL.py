import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# --- Physical Constants ---
hbar = 1.0545718e-34  # Planck constant over 2pi (J·s)
c = 3e8               # Speed of light (m/s)
kB = 1.380649e-23     # Boltzmann constant (J/K)
q = 1.60218e-19       # Elementary charge (C)
T = 300               # Temperature (K)

# --- Thickness of material (in meters) ---
d = 3e-9  # 3 nm thickness

# --- System: TPBi SQW ---
system = {
    "file": "absorption.dat",
    "QFLc": 0.98,
    "QFLv": 0.12,
    "color_PL": "tab:purple",
    "color_alpha": "tab:brown"
}

# --- Helper: Compute Y_PL with absorptivity multiplication ---
def compute_Y_PL(E_eV, delta_mu_eV, absorptivity):
    E_J = E_eV * q
    delta_mu_J = delta_mu_eV * q
    exp_arg = (E_J - delta_mu_J) / (kB * T)
    exp_arg = np.clip(exp_arg, -700, 700)  # avoid overflow

    exp_term = np.exp(exp_arg)
    denominator = (4 * np.pi ** 2 * hbar ** 3 * c ** 2) * (exp_term - 1)
    denominator = np.where(denominator <= 0, 1e-30, denominator)

    numerator = E_J ** 2 * absorptivity
    Y_PL = numerator / denominator
    return Y_PL

# --- Load Data ---
data = np.loadtxt(system["file"], skiprows=1)
E = data[:, 0]
alpha = data[:, 1] * 100  # cm^-1 → m^-1

# --- Energy filtering ---
mask = (E >= 0) & (E <= 10)
E_filtered = E[mask]
alpha_filtered = alpha[mask]

# --- Absorptivity ---
absorptivity = 1 - np.exp(-alpha_filtered * d)

# --- Quasi-Fermi level difference ---
delta_mu = system["QFLc"] - system["QFLv"]

# --- Compute PL weighted by absorptivity ---
Y_PL = compute_Y_PL(E_filtered, delta_mu, absorptivity)

# --- Gaussian smoothing ---
sigma_eV = 0.4
sigma_pts = sigma_eV / np.mean(np.diff(E_filtered))
Y_PL_smooth = gaussian_filter1d(Y_PL, sigma=sigma_pts)

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ax1.plot(E_filtered, Y_PL_smooth, label="TPBi SQW PL", color=system["color_PL"])
ax2.plot(E_filtered, alpha_filtered, '--', label="TPBi SQW Absorption", color=system["color_alpha"])

ax1.set_xlabel("Energy (eV)", fontsize=12)
ax1.set_ylabel(r"$Y_{PL}(E)$ [arb. units, unnormalized]", fontsize=12)
ax2.set_ylabel("Absorption coefficient (m$^{-1}$)", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

plt.title("PL and Absorption for TPBi SQW", fontsize=14)
plt.tight_layout()
plt.show()

# --- Save PL data ---
output_file = "TPBi_SQW_PL.dat"
np.savetxt(output_file, np.column_stack((E_filtered, Y_PL_smooth)),
           header="Energy (eV)   Y_PL (arb. units)",
           fmt="%.6f")
print(f"Saved PL data to: {output_file}")
