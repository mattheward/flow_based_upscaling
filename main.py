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
import plot as Pl

def main():

    # Start Timer
    start_time = time.time()
  
    # Define Values
    total_cells = Calc.Total_Cells_2D()
    delta_vals = Calc.Delta_Values()
    cell_volume = Calc.Cell_Volume(delta_vals)
    perm_field = M.grid_permeability()
    Pl.perm_field_plot(perm_field)
    porosity_field = M.grid_porosity()
    accumulation = Calc.Accumulation(cell_volume, delta_vals[3], porosity_field)
    connections_x_fine, connections_y_fine = Con.Initialize_Connections(total_cells)
    well_index_fine_vals = Con.Well_Index()

    # Intialize matrices and connections
    p0 = M.Initalize_P_Vector(total_cells)
    WBHP_vals = [[] for _ in range(len(Grid.WELLS))] # Creates list of list to create multiple plots if have multiple wells

    # Initialize matrices
    b_vector = M.RHS_Vector(accumulation, p0)
    a_matrix = M.Form_A_Matrix(connections_x_fine, connections_y_fine, p0, total_cells, accumulation, delta_vals, perm_field)

    a_matrix, b_vector = M.well_treatment(well_index_fine_vals, a_matrix, b_vector, delta_vals, perm_field, current_time=0)

    # Solve for pressures using sparse solver
    p_new_f = spla.spsolve(a_matrix, b_vector)

    # If constant pressure boundary condition is true then this will add a ring with constant pressure
    if Grid.Boundary_Condition == 1:

        Pressure_Grid = p_new_f.reshape(Grid.NX_total, Grid.NY_total) # Convert P_new vector into matrix

        # Create ring
        Pressure_Grid[0, :] = Field.P_boundary
        Pressure_Grid[-1, :] = Field.P_boundary
        Pressure_Grid[:, 0] = Field.P_boundary
        Pressure_Grid[:, -1] = Field.P_boundary

        p_new_f = Pressure_Grid.ravel() # Convert back to 1D vector

    # Add pressure at each well location to respective WBHP list for plotting
    for n, w in enumerate(Grid.WELLS):
        WBHP_vals[n].append(p_new_f[well_index_fine_vals[w]-1])

    # Loop through time steps
    for t in range(Simulation.number_of_steps - 1):

        current_time = t + 1

        p_n = p_new_f # Defines previous future pressure as current pressure
        b_vector = M.RHS_Vector(accumulation, p_n) # Recalculates b vector
        a_matrix = M.Form_A_Matrix(connections_x_fine, connections_y_fine, p_n, total_cells, accumulation, delta_vals, perm_field) # Recalculates a matrix
        
        a_matrix, b_vector = M.well_treatment(well_index_fine_vals, a_matrix, b_vector, delta_vals, perm_field, current_time)
        
        p_new_f = spla.spsolve(a_matrix, b_vector) # Finds new pressure vector (1D)

        for n, w in enumerate(Grid.WELLS):
            WBHP_vals[n].append(p_new_f[well_index_fine_vals[w]-1])

        if Grid.Boundary_Condition == 1:
            Pressure_Grid = p_new_f.reshape(Grid.NX_total, Grid.NY_total)
            Pressure_Grid[0, :] = Field.P_boundary
            Pressure_Grid[-1, :] = Field.P_boundary
            Pressure_Grid[:, 0] = Field.P_boundary
            Pressure_Grid[:, -1] = Field.P_boundary

            p_new_f = Pressure_Grid.ravel()
    
    # Calculates the mass flow rate if there is a constant pressure boundary layer
    if Grid.Boundary_Condition == 1:
        flow_rate = Calc.Flow_Rate(p_new_f)
        print(f'Final Total Flow Rate: {flow_rate:.3f} STB/day')

    # Prints final well pressure for each well
    for w in Grid.WELLS:
        print(f"Final pressure of {w}:", p_new_f[well_index_fine_vals[w] - 1])

    time.sleep(1) # Stops timer
    stop_time = time.time() # Captures stop time
    elapsed_time = stop_time - start_time # Finds time taken to run
    print(f"Elapsed time fine simulation: {elapsed_time:.2f} seconds") # Prints time taken to run

    start_time2 = time.time()

    # ========== Coarse Simulator ==========

    coarse_map, num_merged_cells = Up.create_coarse_grid()
    # print("Coarse Map:", coarse_map)
    # Pl.plot_coarse_grid(coarse_map)
    num_coarse_cells = Cs.Total_Coarse_Cells_2D()
    delta_coarse_vals = Cs.Delta_Coarse_Values()
    coarse_cell_volume = Cs.Coarse_Cell_Volume(delta_coarse_vals)
    coarse_porosity_field = Up.upscaled_porosity_field(coarse_map, porosity_field)
    coarse_accumulation = Calc.Accumulation(coarse_cell_volume, delta_coarse_vals[3], coarse_porosity_field)
    well_index_coarse_vals = Up.coarse_well_locations(coarse_map, well_index_fine_vals)
    connections_x_coarse = Up.coarse_connections(coarse_map, connections_x_fine)
    connections_y_coarse = Up.coarse_connections(coarse_map, connections_y_fine)

    upscaled_well_transmissibility = Up.coarse_well_transmissibilities(coarse_map, well_index_coarse_vals, well_index_fine_vals, delta_vals, delta_coarse_vals, perm_field)
    upscaled_transmissbility = Up.solve_local_problems(coarse_map, connections_x_coarse, connections_y_coarse, delta_vals, perm_field)

    p0_c = M.Initalize_P_Vector(num_coarse_cells)

    b_vector_c = M.RHS_Vector(coarse_accumulation, p0_c)
    a_matrix_c = Cs.Form_A_Matrix_Coarse(upscaled_transmissbility, coarse_accumulation, num_coarse_cells)

    a_matrix_c, b_vector_c = Up.coarse_well_treamtment(upscaled_well_transmissibility, well_index_coarse_vals, a_matrix_c, b_vector_c, current_time=0)

    p_new_c = spla.spsolve(a_matrix_c, b_vector_c)

    for t in range(Simulation.number_of_steps - 1):

        current_time = t + 1

        p_n_c = p_new_c
        b_vector_c = M.RHS_Vector(coarse_accumulation, p_n_c)
        a_matrix_c = Cs.Form_A_Matrix_Coarse(upscaled_transmissbility, coarse_accumulation, num_coarse_cells)

        a_matrix_c, b_vector_c = Up.coarse_well_treamtment(upscaled_well_transmissibility, well_index_coarse_vals, a_matrix_c, b_vector_c, current_time)

        p_new_c = spla.spsolve(a_matrix_c, b_vector_c)

    for w in Grid.WELLS:
        # print(f"Final pressure of {w}:", p_new_c[well_index_coarse_vals[w] - 1])
        # print('\n ----------------------- \n')

        coarse_well_index = well_index_coarse_vals[w] - 1
        p_block_avg = p_new_c[coarse_well_index]

        WI = upscaled_well_transmissibility[w]

        if Grid.WELLS[w]['type'] == 'injector':
            well_rate = Grid.WELLS[w]['rates'][0]
        elif Grid.WELLS[w]['type'] == 'producer':
            # For a pressure-controlled producer, we calculate the rate using the results.
            # This is the correct formula: q = WI * (p_block - p_bhp)
            well_rate = WI * (p_block_avg - Grid.BHP)
        
        else:
            raise ValueError('Improper well type given')

        # --- Now, perform the VALIDATION CHECK ---
        # We back-calculate the BHP using the results we just got.
        # This value SHOULD be very close to the target BHP.

        if Grid.WELLS[w]['type'] == 'injector':
            # For an injector: p_bhp = p_block + (q / WI)
            # Note the '+' sign because injection pressure is higher than block pressure.
            back_calculated_BHP = p_block_avg + (well_rate / WI)
            print(f'Injector "{w}" Back-Calculated BHP: {back_calculated_BHP:.2f} psi')

        elif Grid.WELLS[w]['type'] == 'producer':
            # For a producer: p_bhp = p_block - (q / WI)
            back_calculated_BHP = p_block_avg - (well_rate / WI)
            print(f"Producer '{w}' Target BHP was: {Grid.BHP:.2f} psi")
            print(f"Producer '{w}' Back-Calculated BHP is: {back_calculated_BHP:.2f} psi")


    time.sleep(1) # Stops timer
    stop_time2 = time.time() # Captures stop time
    elapsed_time = stop_time2 - start_time2 # Finds time taken to run
    print(f"Elapsed time Coarse Simulation: {elapsed_time:.2f} seconds") # Prints time taken to run



    # Pl.fine_scale_pressure_map(p_new_f)
    reshaped_coarse_pressure = Pl.reshape_coarse_pressure_vector(p_new_c, coarse_map)
    Pl.coarse_scale_pressure_map(reshaped_coarse_pressure)
    Pl.compare_pressure_fields(p_new_f, reshaped_coarse_pressure, coarse_map)
    # Pl.plot_coarse_grid(coarse_map)
    # Pl.perm_field_plot(perm_field)
    # Pl.porosity_field_plot(porosity_field)

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


# Runs program
if __name__ == "__main__":

    print("Running program...")

    main()

    print("End of program")
