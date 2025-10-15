import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simps

# --- Load data (only cbm1 and vbm1) ---
cbm1 = np.loadtxt("PLANAR_AVERAGE_cbm1.dat")
vbm1 = np.loadtxt("PLANAR_AVERAGE_vbm1.dat")

# Use CBM1 z-grid as reference
z_c = cbm1[:,0]
rho_cbm1 = cbm1[:,1]
z_v1, rho_vbm1 = vbm1[:,0], vbm1[:,1]

# Interpolate VBM data onto CBM grid if needed
if not np.allclose(z_c, z_v1):
    rho_vbm1 = np.interp(z_c, z_v1, rho_vbm1)

# --- Assign CBM and VBM contributions (only cbm1 & vbm1) ---
rho_c = rho_cbm1
rho_v = rho_vbm1

# --- Normalize ---
rho_c /= simps(rho_c, z_c)
rho_v /= simps(rho_v, z_c)

# --- Overlap integral ---
overlap = simps(rho_c * rho_v, z_c)
overlap_percent = overlap * 100

# --- Extra metrics ---
z_c_mean = simps(z_c * rho_c, z_c)
z_v_mean = simps(z_c * rho_v, z_c)

z_c_var = simps((z_c - z_c_mean)**2 * rho_c, z_c)
z_v_var = simps((z_c - z_v_mean)**2 * rho_v, z_c)

z_c_std, z_v_std = np.sqrt(z_c_var), np.sqrt(z_v_var)
centroid_sep = abs(z_c_mean - z_v_mean)

# --- Print ---
print(f"Wavefunction overlap integral (relative): {overlap:.4f}")
print(f"Wavefunction overlap integral (percentage): {overlap_percent:.2f}%")
print(f"CBM centroid: {z_c_mean:.3f} Å, spread: {z_c_std:.3f} Å")
print(f"VBM centroid: {z_v_mean:.3f} Å, spread: {z_v_std:.3f} Å")
print(f"Centroid separation: {centroid_sep:.3f} Å")

# --- Plot ---
plt.figure(figsize=(8,5))
plt.plot(z_c, rho_c, label="CBM1 (normalized)", color="blue", linewidth=2)
plt.plot(z_c, rho_v, label="VBM1 (normalized)", color="red", linewidth=2)
plt.fill_between(z_c, 0, rho_c*rho_v, color="purple", alpha=0.3, label="Overlap")

plt.axvline(z_c_mean, color="blue", linestyle="--", alpha=0.7, label="CBM centroid")
plt.axvline(z_v_mean, color="red", linestyle="--", alpha=0.7, label="VBM centroid")

info_text = (
    f"Overlap: {overlap_percent:.2f}%\n"
    f"CBM σ: {z_c_std:.2f} Å\n"
    f"VBM σ: {z_v_std:.2f} Å\n"
    f"Δz: {centroid_sep:.2f} Å"
)
plt.text(0.98, 0.95, info_text, transform=plt.gca().transAxes,
         fontsize=12, color="black", ha="right", va="top",
         bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

plt.xlabel("Distance (Å)", fontsize=14)
plt.ylabel("Normalized density", fontsize=14)
plt.title("CBM1–VBM1 Planar Charge Density Overlap", fontsize=16)
plt.legend(fontsize=12)
plt.tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.savefig("CBM1_VBM1_overlap.png", dpi=300)
plt.show()

# --- Save raw (unnormalized) data ---
np.savetxt("PLANAR_AVERAGE_cbm1_raw.dat", np.column_stack((z_c, cbm1[:,1])),
           header="z(A)  rho_cbm1", fmt="%.6e")

np.savetxt("PLANAR_AVERAGE_vbm1_raw.dat", np.column_stack((z_c, rho_vbm1)),
           header="z(A)  rho_vbm1", fmt="%.6e")

print("Raw CBM1 and VBM1 planar averages saved to text files.")
