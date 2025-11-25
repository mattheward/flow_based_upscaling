'''
Matthew Ard 
Energy 224 - Flow Based Upscaling
04/25/2025

==========
This code contains the logic for the flow based upscaling.
==========
'''

from config import Grid, Upscaling
import matplotlib.pyplot as plt
import numpy as np

def create_upscaled_grid():

    Nx = Grid.NX_total
    Ny = Grid.NY_total
    NCx = Upscaling.NCx
    NCy = Upscaling.NCy

    assert Nx % NCx == 0, "Fine grid cells can't be evenly divided."
    assert Ny % NCy == 0, "Fine grid cells can't be evenly divided."

    block_size_x = Nx // NCx
    block_size_y = Ny // NCy

    coarse_grid_map = {}

    for jc in range(NCy): # 0, 1, 2... from bottom
        for ic in range(NCx): # 0, 1, 2... from left
            
            # Calculate coarse cell ID based on the Cartesian layout
            coarse_cell_id = jc * NCx + ic
            
            fine_cells_in_block = []
            
            # Find the starting bottom-left corner of the coarse block
            start_i = ic * block_size_x
            start_j = jc * block_size_y
            
            # Loop over the fine cells within this block's boundaries
            for j_fine in range(start_j, start_j + block_size_y):
                for i_fine in range(start_i, start_i + block_size_x):
                    
                    # Convert Cartesian (i, j) coordinates (0-based) to your
                    # 1-based global fine ID.
                    fine_cell_id = (j_fine * Nx + i_fine)
                    fine_cells_in_block.append(fine_cell_id)
            
            coarse_grid_map[coarse_cell_id] = fine_cells_in_block
            
    return coarse_grid_map


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

            j = fine_id // Nx  # Cartesian row (0 = bottom)
            i = fine_id % Nx   # Matrix col (from left)

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
        jc = coarse_id // NCx
        ic = coarse_id % NCx
        
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