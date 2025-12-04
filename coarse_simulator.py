'''
Matthew Ard 
Energy 224 - Flow Based Upscaling
04/25/2025

==========
This code contains the logic for the flow based upscaling.
==========
'''

# Import Packages
import numpy as np
import scipy.sparse as sp
import json
from ast import literal_eval

# Import Files
from config import Field, Grid, Simulation, Upscaling
import connections as Con
import calculations as Calc

def Total_Coarse_Cells_2D(num_merged_cells):

    return Upscaling.NCx * Upscaling.NCy - num_merged_cells # [cells]

def Delta_Coarse_Values():

    delta_x_coarse = Field.Lx / Upscaling.NCx
    delta_y_coarse = Field.Ly / Upscaling.NCy
    delta_z_coarse = Field.Lz / Grid.NZ
    delta_t_coarse = int(Simulation.run_time / Simulation.number_of_steps)

    return [delta_x_coarse, delta_y_coarse, delta_z_coarse, delta_t_coarse]


def Coarse_Cell_Volume(delta_coarse_vals):

    return delta_coarse_vals[0] * delta_coarse_vals[1] * delta_coarse_vals[2] # [ft^3]


def Form_A_Matrix_Coarse(upscaled_transmissibility, coarse_accumulation, num_coarse_cells):

    a_matrix = sp.lil_matrix((num_coarse_cells, num_coarse_cells), dtype=float)
        
    # We create a temporary array to hold the sum of T* for each cell's diagonal
    diag_T_sum = np.zeros(num_coarse_cells)

    # --- This is the new, robust loop ---
    # We iterate directly over the calculated T* values. This is the "source of truth".
    for (i, j), T_star in upscaled_transmissibility.items():
        
        # Convert 1-based IDs from your dictionary keys to 0-based indices
        idx_i = i - 1
        idx_j = j - 1
        
        # Set the off-diagonal terms. The matrix is symmetric.
        a_matrix[idx_i, idx_j] = T_star
        a_matrix[idx_j, idx_i] = T_star
        
        # Add this T* value to the sum for each cell's diagonal calculation
        diag_T_sum[idx_i] += T_star
        diag_T_sum[idx_j] += T_star

    # --- Now, build the diagonal ---
    is_scalar_acc = np.isscalar(coarse_accumulation)
    
    # Loop through all cells to set their diagonal value
    for k in range(num_coarse_cells):
        
        acc_val = coarse_accumulation if is_scalar_acc else coarse_accumulation[k]
        
        # Get the sum of T* for this cell
        T_sum_for_cell_k = diag_T_sum[k]
        
        # Your Calc function expects a list of T_vals and an accumulation term.
        # We pass it the single summed value in a list for compatibility.
        # It also applies the negative sign.
        a_matrix[k, k] = -Calc.Diag_Transmissibility_Calc([T_sum_for_cell_k], acc_val)
        
        # A more direct way, if your Calc function is just sum(T) + acc:
        # a_matrix[k, k] = -(diag_T_sum[k] + acc_val)

    return a_matrix.tocsr()

