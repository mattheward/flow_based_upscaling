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

    Pressure_Grid = p_new_f.reshape(Grid.NX_total, Grid.NY_total)

    # Creates plot for pressure map
    plt.figure()
    plt.imshow(Pressure_Grid, cmap = 'viridis', interpolation = 'nearest', origin = 'lower')
    plt.colorbar(label = 'Pressure (psi)' )
    plt.title('Pressure Distribution in Reservoir')
    plt.xlabel('X Direction')
    plt.ylabel('Y Direction')

    plt.show()


def coarse_scale_pressure_map(p_new_c):

    Pressure_Grid_temp = p_new_c.reshape(Upscaling.NCx, Upscaling.NCy)
    Pressure_Grid_Coarse = np.rot90(Pressure_Grid_temp, k=2) # Flips matrix to match fine grid orientation

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

    Pressure_Grid_temp = p_new_c.reshape(NCx, NCy)
    Pressure_Grid_Coarse = np.rot90(Pressure_Grid_temp, k=2) # Flips matrix to match fine grid orientation

    Pressure_Grid_fine = p_new_f.reshape(Grid.NX_total, Grid.NY_total)


    # 1. Create the "Averaged-Down" Fine Grid Pressure Field
    fine_pressure_averaged = np.zeros((NCy, NCx))
    
    for coarse_id, fine_cell_ids in coarse_map.items():
        # Get the pressures for all fine cells in this coarse block
        pressures_in_block = []
        for fine_id in fine_cell_ids:
            # Convert 1D fine_id to 2D (j, i) coordinates
            j = fine_id // Grid.NX_total
            i = fine_id % Grid.NX_total
            pressures_in_block.append(Pressure_Grid_fine[j, i])
        
        # Calculate the average pressure for the block
        avg_pressure = np.mean(pressures_in_block)
        
        # Store this average value in the corresponding coarse cell location
        coarse_j = coarse_id // NCx
        coarse_i = coarse_id % NCx
        fine_pressure_averaged[coarse_j, coarse_i] = avg_pressure
        
    # 2. Calculate the Error Map
    error_map = abs(fine_pressure_averaged - Pressure_Grid_Coarse) / fine_pressure_averaged * 100  # Percentage error
    
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
    im3 = ax3.imshow(error_map, cmap='coolwarm', interpolation='nearest', origin='lower', vmin=-error_max_abs, vmax=error_max_abs)
    ax3.set_title('Error Map (% Difference)')
    ax3.set_xlabel('Coarse Cell X')
    ax3.set_ylabel('Coarse Cell Y')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Pressure Difference (psi)')
    
    plt.tight_layout()
    plt.show()
