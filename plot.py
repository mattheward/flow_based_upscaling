'''
Energy 223 - Phase II
04/25/2025

==========
This file is to create all the matrices (A, p, and b)
==========
'''

# Packages
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Files
from config import Rock, Grid, Field, Simulation, Upscaling
import calculations as Calc
import connections as Con

def fine_scale_pressure_map(p_new_f):

    # Shape as (nrows, ncols) = (NY_total, NX_total) so first index is y (rows)
    Pressure_Grid = p_new_f.reshape((Grid.NY_total, Grid.NX_total))

    # Creates plot for pressure map
    plt.figure()
    plt.imshow(Pressure_Grid, cmap = 'viridis', interpolation = 'nearest', origin = 'lower')
    plt.colorbar(label = 'Pressure (psi)' )
    plt.title('Pressure Distribution in Reservoir')
    plt.xlabel('X Direction')
    plt.ylabel('Y Direction')

    plt.show()


def coarse_scale_pressure_map(p_new_c):

    # Shape coarse as (NCy, NCx): rows = y, cols = x
    Pressure_Grid_Coarse = p_new_c.reshape((Upscaling.NCy, Upscaling.NCx))

    # Creates plot for pressure map
    plt.figure()
    plt.imshow(Pressure_Grid_Coarse, cmap = 'viridis', interpolation = 'nearest', origin = 'lower')
    plt.colorbar(label = 'Pressure (psi)' )
    plt.title('Coarse Scale Pressure Distribution in Reservoir')
    plt.xlabel('X Direction')
    plt.ylabel('Y Direction')

    plt.show()


def compare_pressure_fields(p_new_f, p_new_c, coarse_map):
    """
    Compares the final pressure fields by averaging the fine grid and plotting
    it alongside the coarse grid result and an error map.
    """
    NCx = Upscaling.NCx
    NCy = Upscaling.NCy

    # Ensure coarse grid is shaped (NCy, NCx): rows = y, cols = x
    Pressure_Grid_Coarse = p_new_c.reshape((NCy, NCx))

    # We'll compute averaged fine pressures directly from the 1D fine vector
    # coarse_map uses 0-based fine IDs (created by `create_upscaled_grid`).
    fine_pressure_averaged = np.zeros((NCy, NCx))

    for coarse_id, fine_cell_ids in coarse_map.items():
        # coarse_map currently stores 1-based fine IDs (created elsewhere),
        # convert to 0-based indices for direct indexing into p_new_f
        pressures_in_block = [p_new_f[fine_id - 1] for fine_id in fine_cell_ids]
        avg_pressure = np.mean(pressures_in_block)

        # Map 1-based coarse_id -> 0-based (jc, ic)
        zero_based_coarse = coarse_id - 1
        jc = zero_based_coarse // NCx
        ic = zero_based_coarse % NCx
        fine_pressure_averaged[jc, ic] = avg_pressure
        
    # 2. Calculate the Error Map
    # Percentage error (guard division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_map = np.abs(fine_pressure_averaged - Pressure_Grid_Coarse) / fine_pressure_averaged * 100
        error_map = np.nan_to_num(error_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Determine a common color scale for pressures
    p_min = min(fine_pressure_averaged.min(), Pressure_Grid_Coarse.min())
    p_max = max(fine_pressure_averaged.max(), Pressure_Grid_Coarse.max())
    
    # Plot 1: Averaged Fine Grid ("Truth")
    ax1 = axes[0]
    im1 = ax1.imshow(fine_pressure_averaged, cmap='viridis', interpolation='nearest', origin='lower', vmin=p_min, vmax=p_max)
    ax1.set_title('Averaged Fine Grid Pressure ("Truth")')
    ax1.set_xlabel('Coarse Cell X')
    ax1.set_ylabel('Coarse Cell Y')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot 2: Upscaled Coarse Grid ("Model")
    ax2 = axes[1]
    im2 = ax2.imshow(Pressure_Grid_Coarse, cmap='viridis', interpolation='nearest', origin='lower', vmin=p_min, vmax=p_max)
    ax2.set_title('Upscaled Coarse Grid Pressure ("Model")')
    ax2.set_xlabel('Coarse Cell X')
    ax2.set_ylabel('Coarse Cell Y')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Plot 3: Error Map
    ax3 = axes[2]
    # Use a diverging colormap for errors (e.g., 'coolwarm' or 'bwr')
    # This centers the "zero error" color (white) and shows positive/negative errors
    error_max_abs = np.max(np.abs(error_map))
    im3 = ax3.imshow(error_map, cmap='coolwarm', interpolation='nearest', origin='lower', vmin=0, vmax=error_max_abs)
    ax3.set_title('Error Map (% Difference)')
    ax3.set_xlabel('Coarse Cell X')
    ax3.set_ylabel('Coarse Cell Y')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Pressure Difference (psi)')
    
    plt.tight_layout()
    plt.show()


def plot_coarse_grid(coarse_map):
    """
    Visualizes how the fine grid is partitioned into the coarse grid.
    This function does not need to change.
    """
    Nx = Grid.NX_total
    Ny = Grid.NY_total
    NCx = Upscaling.NCx
    NCy = Upscaling.NCy
    
    visualization_array = np.zeros((Ny, Nx))
    
    for coarse_id, fine_cells in coarse_map.items():
        for fine_id in fine_cells:

            # `fine_id` in `coarse_map` is 1-based; convert to 0-based index
            idx = fine_id - 1
            j = idx // Nx  # Cartesian row (0 = bottom)
            i = idx % Nx   # Matrix col (from left)

            # With `origin='lower'` in imshow we can store directly using Cartesian j
            visualization_array[j, i] = coarse_id
            
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('viridis', NCx * NCy)
    # Use origin='lower' so row 0 appears at the bottom (Cartesian convention)
    mat = ax.imshow(visualization_array, cmap=cmap, interpolation='none', aspect='equal', origin='lower')
    
    block_size_x = Nx // NCx
    block_size_y = Ny // NCy

    ax.set_xticks(np.arange(-.5, Nx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, Ny, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(-.5, Nx, block_size_x))
    ax.set_yticks(np.arange(-.5, Ny, block_size_y))
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for coarse_id in coarse_map.keys():
        zero_based = coarse_id - 1
        jc = zero_based // NCx
        ic = zero_based % NCx
        
        # Calculate text position in Cartesian coordinates
        center_x = ic * block_size_x + block_size_x / 2 - 0.5
        center_y = jc * block_size_y + block_size_y / 2 - 0.5
        
        # center_y is already in Cartesian coordinates (0 = bottom), matching origin='lower'
        ax.text(center_x, center_y, str(coarse_id), 
                ha='center', va='center', color='white', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc='black', ec='black', lw=1, alpha=0.4))

    ax.set_title(f"Corrected Cartesian Grid ({Nx}x{Ny}) to ({NCx}x{NCy})", fontsize=16)
    ax.set_xlabel("Fine Cell Index (i)")
    ax.set_ylabel("Fine Cell Index (j)")
    
    # origin='lower' already places 0 at the bottom; no inversion is needed.
    
    plt.colorbar(mat, ticks=range(NCx * NCy), label='Coarse Cell ID')
    plt.tight_layout()
    plt.show()


def perm_field_plot(perm_field):

    perm_x_field = perm_field['x'].reshape(Grid.NY_total, Grid.NX_total)
    perm_y_field = perm_field['y'].reshape(Grid.NY_total, Grid.NX_total)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    k_min = min(perm_x_field.min(), perm_y_field.min())
    k_max = max(perm_x_field.max(), perm_y_field.max())

    ax1 = axes[0]
    im1 = ax1.imshow(perm_x_field, cmap='tab20b', interpolation='nearest', origin='lower', vmin=k_min, vmax=k_max)
    ax1.set_title('X Permeability Values')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    im2 = ax2.imshow(perm_y_field, cmap='tab20b', interpolation='nearest', origin='lower', vmin=k_min, vmax=k_max)
    ax2.set_title('Y Permeability Values')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    