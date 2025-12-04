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
from scipy.spatial import ConvexHull

# Files
from config import Rock, Grid, Field, Simulation, Upscaling
import calculations as Calc
import connections as Con
import upscaling as Up

def fine_scale_pressure_map(p_new_f):

    # Shape as (nrows, ncols) = (NY_total, NX_total) so first index is y (rows)
    Pressure_Grid = p_new_f.reshape((Grid.NY_total, Grid.NX_total))

    # Creates plot for pressure map
    plt.figure()
    plt.imshow(Pressure_Grid, cmap = 'viridis', interpolation = 'nearest', origin = 'lower')
    plt.colorbar(label = 'Pressure (psi)' )
    plt.title('Fine Grid Pressure Distribution')
    plt.xlabel('X Direction')
    plt.ylabel('Y Direction')

    plt.show()


def reshape_coarse_pressure_vector(p_coarse, coarse_map):

    # Shape coarse as (NCy, NCx): rows = y, cols = x
    """
    Creates a 2D plottable array from a 1D pressure vector of an unstructured grid.

    Args:
        p_coarse (np.array): The 1D vector of coarse cell pressures.
        coarse_map (dict): The map from coarse_id to a list of its fine_ids.
        grid_config (class): Your grid configuration object (e.g., Grid).

    Returns:
        np.array: A 2D array ready for plotting with imshow.
    """

    Nx = Upscaling.NCx
    Ny = Upscaling.NCy
    
    structured_grid = Up.create_structured_grid()

    # --- Step 1: Create an empty canvas based on the FINE grid dimensions ---
    reshaped_coarse_pressure = np.full((Ny, Nx), np.nan) # Use np.nan as a placeholder    

    for cell_structured in structured_grid.keys():

        location_found = False

        fine_cells_structured = set(structured_grid[cell_structured])

        for cell_unstructured in coarse_map.keys():

            fine_cells_unstructured = set(coarse_map[cell_unstructured])

            for cell in fine_cells_structured:
                if cell in fine_cells_unstructured:
                    cell_id = cell_structured - 1
                    j = cell_id // Nx
                    i = cell_id % Nx

                    reshaped_coarse_pressure[j, i] = p_coarse[cell_unstructured - 1]

                    location_found = True
                    break
            
            if location_found:
                break

    return reshaped_coarse_pressure


def coarse_scale_pressure_map(reshaped_coarse_pressure):
            
    # Creates plot for pressure map
    plt.figure()
    plt.imshow(reshaped_coarse_pressure, cmap = 'viridis', interpolation = 'nearest', origin = 'lower')
    plt.colorbar(label = 'Pressure (psi)' )
    plt.title('Coarse Scale Pressure Distribution in Reservoir')
    plt.xlabel('X Direction')
    plt.ylabel('Y Direction')

    plt.show()

    
def compare_pressure_fields(p_new_f, Pressure_Grid_Coarse, coarse_map):
    """
    Compares the final pressure fields by averaging the fine grid and plotting
    it alongside the coarse grid result and an error map.
    """
    NCx = Upscaling.NCx
    NCy = Upscaling.NCy

    structure_grid = Up.create_structured_grid()

    # # Ensure coarse grid is shaped (NCy, NCx): rows = y, cols = x
    # Pressure_Grid_Coarse = p_new_c.reshape((NCy, NCx))

    # We'll compute averaged fine pressures directly from the 1D fine vector
    # coarse_map uses 0-based fine IDs (created by `create_upscaled_grid`).
    fine_pressure_averaged = []

    for coarse_id, fine_cell_ids in structure_grid.items():
        # coarse_map currently stores 1-based fine IDs (created elsewhere),
        # convert to 0-based indices for direct indexing into p_new_f
        pressures_in_block = [p_new_f[fine_id - 1] for fine_id in fine_cell_ids]
        avg_pressure = np.mean(pressures_in_block)
        fine_pressure_averaged.append(avg_pressure)

    if Upscaling.GRID_TYPE == 'unstructured':
        coarse_pressure_1 = fine_pressure_averaged[0]
        coarse_pressure_2 = fine_pressure_averaged[1]
        merged_pressure = np.mean([coarse_pressure_1, coarse_pressure_2])

        fine_pressure_averaged[0] = merged_pressure
        fine_pressure_averaged[1] = merged_pressure

    fine_pressure_averaged = np.array(fine_pressure_averaged)
    fine_pressure_averaged_grid = fine_pressure_averaged.reshape((NCy, NCx))
        
    # 2. Calculate the Error Map
    # Percentage error (guard division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_map = np.abs(fine_pressure_averaged_grid - Pressure_Grid_Coarse) / fine_pressure_averaged_grid * 100
        error_map = np.nan_to_num(error_map, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Determine a common color scale for pressures
    p_min = min(fine_pressure_averaged.min(), Pressure_Grid_Coarse.min())
    p_max = max(fine_pressure_averaged.max(), Pressure_Grid_Coarse.max())

    # --- Plot 1 ---
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(fine_pressure_averaged_grid, cmap='viridis', interpolation='nearest', origin='lower', vmin=p_min, vmax=p_max)
    ax.set_title('Averaged Fine Grid Pressure')
    ax.set_xlabel('Coarse Cell X')
    ax.set_ylabel('Coarse Cell Y')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label='Pressure (Psi)', labelpad=15, size=12) # 'size' can also be useful
    plt.show()

    # --- Plot 2 ---
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(Pressure_Grid_Coarse, cmap='viridis', interpolation='nearest', origin='lower', vmin=p_min, vmax=p_max)
    ax.set_title('Upscaled Coarse Grid Pressure')
    ax.set_xlabel('Coarse Cell X')
    ax.set_ylabel('Coarse Cell Y')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label='Pressure (Psi)', labelpad=15, size=12) # 'size' can also be useful
    plt.show()

    # --- Plot 3 ---
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(error_map, cmap='coolwarm', interpolation='nearest', origin='lower')
    ax.set_title('Error Map (Relative Percent Difference)')
    ax.set_xlabel('Coarse Cell X')
    ax.set_ylabel('Coarse Cell Y')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pressure Difference (%)', labelpad=15, size=12) # 'size' can also be useful
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
    mat = ax.imshow(visualization_array, cmap=cmap, interpolation='none', aspect='equal', origin='lower')
    
    for coarse_id, fine_cells in coarse_map.items():
        
        # For each fine cell, find its four corners.
        all_corners = []
        for fine_id in fine_cells:
            idx = fine_id - 1
            j = idx // Nx
            i = idx % Nx
            
            # Corners are at (i-0.5, j-0.5), (i+0.5, j+0.5), etc. for plotting
            all_corners.append([i - 0.5, j - 0.5])
            all_corners.append([i + 0.5, j - 0.5])
            all_corners.append([i - 0.5, j + 0.5])
            all_corners.append([i + 0.5, j + 0.5])
        
        all_corners = np.array(all_corners)
        
        # Use ConvexHull to find the points that form the outer boundary.
        # This is a robust way to find the "outline" of a shape made of squares.
        hull = ConvexHull(all_corners)
        
        # Plot the lines of the convex hull
        for simplex in hull.simplices:
            ax.plot(all_corners[simplex, 0], all_corners[simplex, 1], 'k-', linewidth=2)

    for coarse_id, fine_cells in coarse_map.items():
        
        i_coords = []
        j_coords = []
        for fine_id in fine_cells:
            idx = fine_id - 1
            j_coords.append(idx // Nx)
            i_coords.append(idx % Nx)
        
        # The center is the average of the fine cell coordinates.
        center_x = np.mean(i_coords)
        center_y = np.mean(j_coords)
        
        ax.text(center_x, center_y, str(coarse_id), 
                ha='center', va='center', color='white', fontsize=12, weight='bold',
                bbox=dict(boxstyle="circle,pad=0.3", fc='black', ec='white', lw=1, alpha=0.6))

    # --- Part 4: Final Plot Formatting ---
    ax.set_title(f"Unstructured Coarse Grid Visualization", fontsize=16)
    # ax.set_xlabel("Fine Cell Index (i)")
    # ax.set_ylabel("Fine Cell Index (j)")
    # ax.set_xticks(np.arange(0, Nx, 1))
    # ax.set_yticks(np.arange(0, Ny, 1))
    # ax.set_xlim(-0.5, Nx - 0.5)
    # ax.set_ylim(-0.5, Ny - 0.5)
    
    # origin='lower' already places 0 at the bottom; no inversion is needed.
    
    plt.colorbar(mat, ticks=range(NCx * NCy), label='Coarse Cell ID')
    plt.tight_layout()
    plt.show()


def perm_field_plot(perm_field):

    perm_x_field = perm_field['x'].reshape(Grid.NY_total, Grid.NX_total)
    perm_y_field = perm_field['y'].reshape(Grid.NY_total, Grid.NX_total)

    k_min = min(perm_x_field.min(), perm_y_field.min())
    k_max = max(perm_x_field.max(), perm_y_field.max())

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(perm_x_field, cmap='tab20b', interpolation='nearest', origin='lower', vmin=k_min, vmax=k_max)
    ax.set_title('Permeability Field in X Direction')
    ax.set_xlabel('Cell X')
    ax.set_ylabel('Cell Y')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label='Permeability (mD)', labelpad=15, size=12) # 'size' can also be useful
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(perm_y_field, cmap='tab20b', interpolation='nearest', origin='lower', vmin=k_min, vmax=k_max)
    ax.set_title('Permeability Field in Y Direction')
    ax.set_xlabel('Cell X')
    ax.set_ylabel('Cell Y')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label='Permeability (mD)', labelpad=15, size=12) # 'size' can also be useful
    plt.show()
    

def porosity_field_plot(porosity_field):

    porosity_field = porosity_field.reshape((Grid.NY_total, Grid.NX_total))

    porosity_min = porosity_field.min()
    porosity_max = porosity_field.max()

    # Creates plot for pressure map
    plt.figure()
    plt.imshow(porosity_field, cmap = 'tab20', interpolation = 'nearest', origin = 'lower', vmin=porosity_min, vmax=porosity_max)
    plt.colorbar(label = 'Porosity' )
    plt.title('Reservior Porosity Map')
    plt.tight_layout()
    plt.show()


def bhp_pressure_plots(WBHP_vals_fine, WBHP_vals_Coarse):

    # Creates plot for pressure at bottom of well(s)
    for w in Grid.WELLS.keys(): # Creates plot for each well pressure
        if Grid.WELLS[w]['type'] == 'injector':
            plt.figure() 
            plt.plot(np.arange(Simulation.number_of_steps), WBHP_vals_fine[w], label='Fine Grid Simulation')
            plt.plot(np.arange(Simulation.number_of_steps), WBHP_vals_Coarse[w], label=f'Coarse Grid Simulation')
            plt.xlabel("Time (days)")
            plt.ylabel("Well Bottom-Hole Pressure (psi)")
            plt.title(f"Change in Well Bottom-Hole Pressure Over Time for {w}")
            plt.legend()
            plt.grid()
            plt.show()

def producer_rate_plot(producer_rate_fine, producer_rate_coarse):

    plt.figure() 
    plt.plot(np.arange(Simulation.number_of_steps), producer_rate_fine, label='Fine Grid Simulation')
    plt.plot(np.arange(Simulation.number_of_steps), producer_rate_coarse, label='Coarse Grid Simulation')
    plt.xlabel("Time (days)")
    plt.ylabel("Well Production Rate (STB/day)")
    plt.title('Chnage in Well Production Rate')
    plt.legend()
    plt.grid()
    plt.show()