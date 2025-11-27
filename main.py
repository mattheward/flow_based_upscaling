'''
Matthew Ard 
Energy 223 - Phase II
04/25/2025

==========
This is the main file where the user will run the code after inputing the parameters. This code initializes the program, runs
the simulation (finding pressure at each time step), plots/prints the results, and times the execution.
==========
'''

# Packages
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.sparse.linalg as spla

# Files
from config import Grid, Field, Simulation, Upscaling
import connections as Con
import matrix as M
import calculations as Calc
import upscaling as Up
import coarse_simulator as Cs

def main():

    # Start Timer
    start_time = time.time()

    # Define Values
    total_cells = Calc.Total_Cells_2D()
    delta_vals = Calc.Delta_Values()
    cell_volume = Calc.Cell_Volume(delta_vals)
    accumulation = Calc.Accumulation(cell_volume, delta_vals[3])
    well_index = Con.Well_Index()

    # Intialize matrices and connections
    p0 = M.Initalize_P_Vector(total_cells)
    perm_field = M.grid_permeability()
    connections_x, connections_y = Con.Initialize_Connections(total_cells)
    WBHP_vals = [[] for _ in range(len(Grid.WELLS))] # Creates list of list to create multiple plots if have multiple wells

    # Initialize matrices
    b_vector = M.RHS_Vector(accumulation, p0)
    a_matrix = M.Form_A_Matrix(connections_x, connections_y, p0, total_cells, accumulation, delta_vals, perm_field)

    a_matrix, b_vector = M.well_treatment(well_index, a_matrix, b_vector, delta_vals, perm_field, current_time=0)

    # Solve for pressures using sparse solver
    p_new = spla.spsolve(a_matrix, b_vector)

    # If constant pressure boundary condition is true then this will add a ring with constant pressure
    if Grid.Boundary_Condition == 1:

        Pressure_Grid = p_new.reshape(Grid.NX_total, Grid.NY_total) # Convert P_new vector into matrix

        # Create ring
        Pressure_Grid[0, :] = Field.P_boundary
        Pressure_Grid[-1, :] = Field.P_boundary
        Pressure_Grid[:, 0] = Field.P_boundary
        Pressure_Grid[:, -1] = Field.P_boundary

        p_new = Pressure_Grid.ravel() # Convert back to 1D vector

    # Add pressure at each well location to respective WBHP list for plotting
    for w in range(len(Grid.well_x)):
        WBHP_vals[w].append(p_new[well_index[w]-1])

    # Loop through time steps
    for t in range(Simulation.number_of_steps - 1):

        p_n = p_new # Defines previous future pressure as current pressure
        b_vector = M.RHS_Vector(accumulation, p_n) # Recalculates b vector
        a_matrix = M.Form_A_Matrix(connections_x, connections_y, p_n, total_cells, accumulation, delta_vals, perm_field) # Recalculates a matrix
        
        a_matrix, b_vector = M.well_treatment(well_index, a_matrix, b_vector, delta_vals, perm_field, t+1)
        
        p_new = spla.spsolve(a_matrix, b_vector) # Finds new pressure vector (1D)

        for w in range(len(Grid.well_x)):
            WBHP_vals[w].append(p_new[well_index[w]-1])

        if Grid.Boundary_Condition == 1:
            Pressure_Grid = p_new.reshape(Grid.NX_total, Grid.NY_total)
            Pressure_Grid[0, :] = Field.P_boundary
            Pressure_Grid[-1, :] = Field.P_boundary
            Pressure_Grid[:, 0] = Field.P_boundary
            Pressure_Grid[:, -1] = Field.P_boundary

            p_new = Pressure_Grid.ravel()
    
    # Calculates the mass flow rate if there is a constant pressure boundary layer
    if Grid.Boundary_Condition == 1:
        flow_rate = Calc.Flow_Rate(p_new)
        print(f'Final Total Flow Rate: {flow_rate:.3f} STB/day')

    # Prints final well pressure for each well
    for w in range(len(Grid.well_x)):
        print(f"Final pressure of well {w + 1}:", p_new[well_index[w] -1 ])

    # Converts final pressure vector to matrix for plotting
    Pressure_Grid = p_new.reshape(Grid.NX_total, Grid.NY_total)



    # # ========== Coarse Simulator ==========

    # num_coarse_cells = Cs.Total_Coarse_Cells_2D()
    # delta_coarse_vals = Cs.Delta_Coarse_Values()
    # coarse_cell_volume = Cs.Coarse_Cell_Volume(delta_coarse_vals)
    # coarse_accumulation = Calc.Accumulation(coarse_cell_volume, delta_coarse_vals[3])
    # upscaled_transmissbility = Up.solve_local_problems(coarse_map, connections_x, connections_y, delta_vals, perm_field)

    # p0_c = M.Initalize_P_Vector(num_coarse_cells)
    # q0_c = M.Initilalize_Q_Vectors(num_coarse_cells, Cs.Well_Location_Coarse(), stop_injection)
    # coarse_map = Up.create_upscaled_grid()
    # connections_x, connections_y =  Up.upscaled_connections()

    # b_vector_c = M.RHS_Vector(coarse_accumulation, p0_c, q0_c)
    # a_matrix_c = Cs.Form_A_Matrix_Coarse(connections_x, connections_y, upscaled_transmissbility, coarse_accumulation, num_coarse_cells)

    # p_new_c = spla.spsolve(a_matrix_c, b_vector_c)

    # for t in range(Simulation.number_of_steps - 1):

    #     p_n_c = p_new_c
    #     b_vector_c = M.RHS_Vector(coarse_accumulation, p_n_c, q0_c)
    #     a_matrix_c = Cs.Form_A_Matrix_Coarse(connections_x, connections_y, upscaled_transmissbility, coarse_accumulation, num_coarse_cells)
    #     p_new_c = spla.spsolve(a_matrix_c, b_vector_c)

    # Pressure_Grid_Coarse = p_new_c.reshape(Upscaling.NCx, Upscaling.NCy)

    time.sleep(1) # Stops timer
    stop_time = time.time() # Captures stop time
    elapsed_time = stop_time - start_time # Finds time taken to run
    print(f"Elapsed time: {elapsed_time:.2f} seconds") # Prints time taken to run


    # # Creates plot for pressure at bottom of well(s)
    # plt.figure() 
    # for w, line in enumerate(WBHP_vals): # Creates plot for each well pressure
    #     plt.plot(np.arange(Simulation.number_of_steps), line, label=f'Well {w+1} ({Grid.well_x_location[w]}, {Grid.well_y_location[w]})')
    # plt.xlabel("Time (days)")
    # plt.ylabel("Well Bottom-Hole Pressure (psi)")
    # plt.title("Change in Well Bottom-Hole Pressure Over Time")
    
    # if len(Grid.well_x) > 1: # Creates legend if there are multiple well graphed
    #     plt.legend()

    # plt.grid()


    # # Creates plot for pressure map
    # plt.figure()
    # plt.imshow(Pressure_Grid, cmap = 'viridis', interpolation = 'nearest', origin = 'lower')
    # plt.colorbar(label = 'Pressure (psi)' )
    # plt.title('Pressure Distribution in Reservoir')
    # plt.xlabel('X Direction')
    # plt.ylabel('Y Direction')

    # plt.show()


# Runs program
if __name__ == "__main__":

    print("Running program...")

    main()

    print("End of program")
