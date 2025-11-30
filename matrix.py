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
from config import Rock, Grid, Field, Simulation
import calculations as Calc
import connections as Con


# Initialize flow rate vector (which goes inside of b vector)
# def Initilalize_Q_Vectors(total_cells, well_location, stop_injection):

#     # Initialize the source vector
#     q = np.zeros(total_cells, dtype=float) # 1D vector [STB/day]

#     if not stop_injection:
#         for w in range(len(Grid.well_x)):
#             q[well_location[w]-1] = Grid.well_rate[w] # [STB/day]

#     return q


# Initalizes P vector
def Initalize_P_Vector(total_cells):

    p = np.full(total_cells, Field.P_init) # 1D pressure vector [psi]

    return p


def grid_permeability():

    perm_field = {}

    # # Generate Perm Field

    # perm_x_field = np.full((Grid.NX_total, Grid.NY_total), Rock.k_high)

    # mid_point = Grid.NY_total // 2

    # perm_x_field[mid_point:, :] = Rock.k_low

    # perm_y_field = perm_x_field * Rock.k_xy_ratio


    # Keep Perm isentropic

    # perm_x_field = np.full((Grid.NX_total, Grid.NY_total), Rock.k_x)
    # perm_y_field = np.full((Grid.NX_total, Grid.NY_total), Rock.k_y)


    # Read file

    perm_x_field = []
    file_name_x = Rock.K_Y_FILE

    try:
        with open(file_name_x, 'r') as f:
            for line in f:
                try:
                    num = float(line.strip())
                    perm_x_field.append(num)
                except ValueError:
                    pass

    except FileNotFoundError:
        print(f"Error: The file '{file_name_x}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    perm_x_field = np.array(perm_x_field)

    perm_y_field = []
    file_name_y = Rock.K_Y_FILE

    try:
        with open(file_name_y, 'r') as f:
            for line in f:
                try:
                    num = float(line.strip())
                    perm_y_field.append(num * 1.4)
                except ValueError:
                    pass

    except FileNotFoundError:
        print(f"Error: The file '{file_name_y}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    perm_y_field = np.array(perm_y_field)

    perm_field['x'] = perm_x_field.ravel()
    perm_field['y'] = perm_y_field.ravel()

    return perm_field # Returns 1D arrays of permeability values


# Creates b vector (RHS) 
def RHS_Vector(accumulation, p):
    # p and q are 1D arrays; return 1D RHS
    return -(accumulation * p)


# Gets transmissibility values needed for A matrix of nondiagonal values
def Get_Transmissibility_Values(a_matrix, i, p_n_vector, cell_number, direction, b_star, t_values, delta_vals, perm_field):

    if cell_number != 0: # Only runs for cell references numbers that aren't zero

        # Finds formation volume factor
        b1 = Calc.B_Calc(p_n_vector[cell_number - 1])
        b_int = Calc.B_Interface_Calc(b1, b_star)

        k1 = perm_field[direction][i - 1]
        k2 = perm_field[direction][cell_number - 1]
        k_int = Calc.permeability_average(k1, k2)

        # Finds transmissibility for each direction
        if direction == 'x':
            transmissibility = Calc.Transmissibility_Calc(delta_vals, direction, k_int, b_int)
        else:
            transmissibility = Calc.Transmissibility_Calc(delta_vals, direction, k_int, b_int)

        t_values.append(transmissibility) # Adds to list storing transmissibility values
        a_matrix[i - 1, cell_number-1] = transmissibility # Puts transmissibility value in A matrix (works for LIL)


# Create A Matrix
def Form_A_Matrix(connections_x, connections_y, p_n_vector, total_cells, accumulation, delta_vals, perm_field):
    # Use sparse LIL matrix for fast construction
    a_matrix = sp.lil_matrix((total_cells, total_cells), dtype=float)

    for center_cell in range(1, total_cells + 1): # loop through each cell

        t_values = [] # Initialize a list to hold the transmissibility values

        b_star = Calc.B_Calc(p_n_vector[center_cell - 1]) # Calculate the B value for the current cell

        neighbor_x1 = connections_x[center_cell - 1, 1]
        neighbor_x2 = connections_x[center_cell - 1, 2]
        neighbor_y1 = connections_y[center_cell - 1, 1]
        neighbor_y2 = connections_y[center_cell - 1, 2]

        # process up to two x-neighbors and two y-neighbors
        Get_Transmissibility_Values(a_matrix, center_cell, p_n_vector, neighbor_x1, 'x', b_star, t_values, delta_vals, perm_field)
        Get_Transmissibility_Values(a_matrix, center_cell, p_n_vector, neighbor_x2, 'x', b_star, t_values, delta_vals, perm_field)
        Get_Transmissibility_Values(a_matrix, center_cell, p_n_vector, neighbor_y1, 'y', b_star, t_values, delta_vals, perm_field)
        Get_Transmissibility_Values(a_matrix, center_cell, p_n_vector, neighbor_y2, 'y', b_star, t_values, delta_vals, perm_field)

        # Calculate diagonal value and set it
        a_matrix[center_cell - 1, center_cell - 1] = -Calc.Diag_Transmissibility_Calc(t_values, accumulation)

    # Apply constant boundary condition efficiently on sparse LIL
    if Grid.Boundary_Condition == 1:
        
        reservior = Con.Initialize_Arrays()[0]
        boundary_values = Con.Find_Boundary_Values(reservior)

        for i in boundary_values:
            idx = i - 1
            a_matrix.rows[idx] = [idx]
            a_matrix.data[idx] = [-accumulation]

    return a_matrix.tocsr()


def well_treatment(well_index, a_matrix, b_vector, delta_values, perm_field, current_time):
    """
    Apply well treatments (modify matrix and RHS) and return (a_matrix, b_vector).

    Note: `a_matrix` may be CSR on entry; convert to LIL for efficient assignment,
    then return CSR for the solver.
    """
    # Convert matrix to LIL for efficient element/row assignment
    try:
        a_matrix = a_matrix.tolil()
    except Exception:
        # If a_matrix is already a dense ndarray or similar, leave it
        pass

    if current_time in Simulation.RATE_SCHEDULE:
        print(' =============== Rate Update =============== ')

    # for w in Grid.WELLS:
    #     well_id = well_index[w]

    #     b_vector[well_id - 1] -= Grid.WELLS[w]['rates'][0]

    for w in Grid.WELLS:
        if Grid.WELLS[w]['type'] == 'injector':
            rate_index = 0
            for j, start_time in enumerate(Simulation.RATE_SCHEDULE):
                if current_time >= start_time:
                    rate_index = j
                else:
                    break

            if current_time in Simulation.RATE_SCHEDULE:
                print(f'Updating injection rate for {w} at time step {current_time} to {Grid.WELLS[w]["rates"][rate_index]} STB/day')

            b_vector[well_index[w] - 1] -= Grid.WELLS[w]['rates'][rate_index]  # [STB/day]

        elif Grid.WELLS[w]['type'] == 'producer':
            well_transmissibility = Calc.well_transmissibility(delta_values, perm_field, well_index[w])
            b_vector[well_index[w] - 1] -= well_transmissibility * Grid.BHP
            # LIL supports item assignment
            a_matrix[well_index[w] - 1, well_index[w] - 1] -= well_transmissibility

    # Convert back to CSR for efficient solves
    try:
        a_matrix = a_matrix.tocsr()
    except Exception:
        pass

    return a_matrix, b_vector