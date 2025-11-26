'''
Matthew Ard 
Energy 223 - Phase II
04/25/2025

==========
This file is to create all the matrices (A, p, and b)
==========
'''

# Packages
import numpy as np
import scipy.sparse as sp

# Files
from config import Grid, Field, Simulation
import calculations as Calc
import connections as Con

# Initialize flow rate vector (which goes inside of b vector)
def Initilalize_Q_Vectors(total_cells, well_location, stop_injection):

    # Initialize the source vector
    q = np.zeros(total_cells, dtype=float) # 1D vector [STB/day]

    if not stop_injection:
        for w in range(len(Grid.well_x)):
            q[well_location[w]-1] = Grid.well_rate[w] # [STB/day]

    return q

# Initalizes P vector
def Initalize_P_Vector(total_cells):
    p = np.full(total_cells, Field.P_init) # 1D pressure vector [psi]

    return p

# Creates b vector (RHS) 
def RHS_Vector(accumulation, p, q):
    # p and q are 1D arrays; return 1D RHS
    return -(accumulation * p) - q


# Gets transmissibility values needed for A matrix of nondiagonal values
def Get_Transmissibility_Values(a_matrix, i, p_n_vector, cell_number, direction, b_star, t_values, delta_vals):

    if cell_number != 0: # Only runs for cell references numbers that aren't zero

        # Finds formation volume factor
        b1 = Calc.B_Calc(p_n_vector[cell_number - 1])
        b_int = Calc.B_Interface_Calc(b1, b_star)

        # Finds transmissibility for each direction
        if direction == 'x':
            transmissibility = Calc.Transmissibility_Calc(delta_vals, direction, b_int)
        else:
            transmissibility = Calc.Transmissibility_Calc(delta_vals, direction, b_int)

        t_values.append(transmissibility) # Adds to list storing transmissibility values
        a_matrix[i - 1, cell_number-1] = transmissibility # Puts transmissibility value in A matrix (works for LIL)


# Create A Matrix
def Form_A_Matrix(connections_x, connections_y, p_n_vector, total_cells, accumulation, delta_vals):
    # Use sparse LIL matrix for fast construction
    a_matrix = sp.lil_matrix((total_cells, total_cells), dtype=float)

    for i in range(1, total_cells + 1): # loop through each cell

        t_values = [] # Initialize a list to hold the transmissibility values

        b_star = Calc.B_Calc(p_n_vector[i - 1]) # Calculate the B value for the current cell

        # process up to two x-neighbors and two y-neighbors
        Get_Transmissibility_Values(a_matrix, i, p_n_vector, connections_x[i - 1, 1], 'x', b_star, t_values, delta_vals)
        Get_Transmissibility_Values(a_matrix, i, p_n_vector, connections_x[i - 1, 2], 'x', b_star, t_values, delta_vals)
        Get_Transmissibility_Values(a_matrix, i, p_n_vector, connections_y[i - 1, 1], 'y', b_star, t_values, delta_vals)
        Get_Transmissibility_Values(a_matrix, i, p_n_vector, connections_y[i - 1, 2], 'y', b_star, t_values, delta_vals)

        # Calculate diagonal value and set it
        a_matrix[i - 1, i - 1] = -Calc.Diag_Transmissibility_Calc(t_values, accumulation)

    # Apply constant boundary condition efficiently on sparse LIL
    if Grid.Boundary_Condition == 1:
        boundary_values = Con.Find_Boundary_Values()

        for i in boundary_values:
            idx = i - 1
            a_matrix.rows[idx] = [idx]
            a_matrix.data[idx] = [-accumulation]

    return a_matrix.tocsr()
