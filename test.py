import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("Generating complex permeability field...")

# --- 1. Define Grid and Layer Parameters ---
NX_FINE, NY_FINE = 50, 50
NCY_COARSE = 10 # Assuming 10 coarse cells in Y-direction
fine_per_coarse_y = NY_FINE // NCY_COARSE

n_coarse_bottom = 4
n_coarse_middle = 3
n_fine_bottom = n_coarse_bottom * fine_per_coarse_y
n_fine_middle = n_coarse_middle * fine_per_coarse_y

# --- 2. Define Statistical Properties for the Three Rock Types ---
def get_lognormal_params(mean, std_dev):
    """Helper function to get parameters for the log-normal distribution."""
    mean_log = np.log(mean) - 0.5 * np.log(1 + (std_dev**2 / mean**2))
    sigma_log = np.sqrt(np.log(1 + (std_dev**2 / mean**2)))
    return mean_log, sigma_log

# Middle "Aquifer" Layer (High Perm, More Homogeneous)
mean_aq, sigma_aq = get_lognormal_params(1000, 100)
anisotropy_aq = 0.5

# Background Shale (Very Low Perm "Mud")
mean_sh, sigma_sh = get_lognormal_params(2, 1)
anisotropy_sh = 0.01

# Sand Stringers (High-Perm Pockets in the Jumbled Layers)
mean_ss, sigma_ss = get_lognormal_params(500, 50)
anisotropy_ss = 0.3
sand_stringer_fraction = 0.15 # 15% of the shale layers will be sand stringers

# --- 3. Generate the kx Field using a "Stamping" Method ---
np.random.seed(42) # Set a seed for reproducibility

# Start with a field of 100% background shale
kx_field_2d = np.random.lognormal(mean=mean_sh, sigma=sigma_sh, size=(NY_FINE, NX_FINE))

# "Stamp" the main aquifer channel in the middle
kx_field_2d[n_fine_bottom : n_fine_bottom + n_fine_middle, :] = np.random.lognormal(
    mean=mean_aq, sigma=sigma_aq, size=(n_fine_middle, NX_FINE)
)

# "Sprinkle" high-perm sand stringers into the top and bottom shale layers
shale_indices = []
for j in range(NY_FINE):
    if j < n_fine_bottom or j >= n_fine_bottom + n_fine_middle:
        for i in range(NX_FINE):
            shale_indices.append((j, i))

num_to_replace = int(len(shale_indices) * sand_stringer_fraction)
indices_to_replace_idx = np.random.choice(len(shale_indices), size=num_to_replace, replace=False)
indices_to_replace = [shale_indices[i] for i in indices_to_replace_idx]
new_sand_values = np.random.lognormal(mean=mean_ss, sigma=sigma_ss, size=num_to_replace)

for idx, (j, i) in enumerate(indices_to_replace):
    kx_field_2d[j, i] = new_sand_values[idx]

# --- 4. Generate the ky Field by Applying Correct Anisotropy ---
ky_field_2d = np.zeros_like(kx_field_2d)
ky_field_2d[:, :] = kx_field_2d[:, :] * anisotropy_sh
ky_field_2d[n_fine_bottom : n_fine_bottom + n_fine_middle, :] = \
    kx_field_2d[n_fine_bottom : n_fine_bottom + n_fine_middle, :] * anisotropy_aq
for j, i in indices_to_replace:
    ky_field_2d[j, i] = kx_field_2d[j, i] * anisotropy_ss

# --- 5. Flatten the 2D fields into 1D arrays for saving ---
# This uses 'C' order (row-major), which matches the standard (j * NX + i) indexing.
perm_field_x = kx_field_2d.flatten(order='C')
perm_field_y = ky_field_2d.flatten(order='C')

# --- 6. SAVE THE DATA TO TEXT FILES ---
np.savetxt('kx_values.txt', perm_field_x, fmt='%.6f')
np.savetxt('ky_values.txt', perm_field_y, fmt='%.6f')

print("\nSUCCESS!")
print("Created two files: 'kx_values.txt' and 'ky_values.txt'")
print("Each file contains 2500 permeability values.\n")


# --- 7. Visualize the Fields so you can see what was created ---
print("Displaying plot of the generated fields...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
vmin, vmax = 0.01, kx_field_2d.max()
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

im1 = ax1.imshow(kx_field_2d, cmap='jet', norm=norm, origin='lower', interpolation='none')
ax1.set_title("Horizontal Permeability (kx)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Grid Block Index (i)")
ax1.set_ylabel("Grid Block Index (j)")

im2 = ax2.imshow(ky_field_2d, cmap='jet', norm=norm, origin='lower', interpolation='none')
ax2.set_title("Vertical Permeability (ky)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Grid Block Index (i)")

fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', label='Permeability (mD)')
fig.suptitle("Generated Heterolithic Permeability Field", fontsize=20)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
