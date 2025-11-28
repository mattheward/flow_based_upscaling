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

def Total_Coarse_Cells_2D():

    return Upscaling.NCx * Upscaling.NCy # [cells]

def Delta_Coarse_Values():

    delta_x_coarse = Field.Lx / Upscaling.NCx
    delta_y_coarse = Field.Ly / Upscaling.NCy
    delta_z_coarse = Field.Lz / Grid.NZ
    delta_t_coarse = int(Simulation.run_time / Simulation.number_of_steps)

    return [delta_x_coarse, delta_y_coarse, delta_z_coarse, delta_t_coarse]


def Coarse_Cell_Volume(delta_coarse_vals):

    return delta_coarse_vals[0] * delta_coarse_vals[1] * delta_coarse_vals[2] # [ft^3]


def Form_A_Matrix_Coarse(coarse_connections_x, coarse_connections_y, upscaled_transmissbility, coarse_accumulation, num_coarse_cells):
    a_matrix = sp.lil_matrix((num_coarse_cells, num_coarse_cells), dtype=float)

    # Accept either scalar or per-cell accumulation
    is_scalar_acc = np.isscalar(coarse_accumulation)

    for i in range(num_coarse_cells):

        t_vals_for_diag = []

        # neighbors in coarse_connections are 1-based IDs or 0 placeholders
        all_neighbors = list(coarse_connections_x[i, 1:]) + list(coarse_connections_y[i, 1:])

        for neighbor in all_neighbors:
            # skip placeholders
            if neighbor == 0:
                continue

            # neighbor is 1-based; convert to 0-based index for matrix placement
            neigh_idx = int(neighbor) - 1

            # Look up T* using the upscaling dict keys which are 1-based in your upscaling module
            key1 = (i + 1, int(neighbor))
            key2 = (int(neighbor), i + 1)

            if key1 in upscaled_transmissbility:
                T_star = upscaled_transmissbility[key1]
            elif key2 in upscaled_transmissbility:
                T_star = upscaled_transmissbility[key2]
            else:
                print(f"Warning: No T* found for connection ({i+1}, {neighbor}) - skipping")
                continue

            # set off-diagonal (i, neigh_idx)
            a_matrix[i, neigh_idx] = T_star
            t_vals_for_diag.append(T_star)

        # diagonal accumulation value per cell
        acc_val = coarse_accumulation if is_scalar_acc else coarse_accumulation[i]
        a_matrix[i, i] = -Calc.Diag_Transmissibility_Calc(t_vals_for_diag, acc_val)

    return a_matrix.tocsr()

