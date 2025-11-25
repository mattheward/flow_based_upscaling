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
from config import Grid, Field, Simulation
import connections as Con
import matrix as M
import calculations as Calc
import upscaling as Up

def main():

    # Start Timer
    start_time = time.time()

    # Define Values
    total_cells = Calc.Total_Cells_2D()
    delta_vals = Calc.Delta_Values()
    cell_volume = Calc.Cell_Volume(delta_vals)
    accumulation = Calc.Accumulation(cell_volume, delta_vals[3])
    well_location = Con.Well_Location()
    stop_injection = False # Set False by default, if injection is to be stopped, will become True

    # Find time step to stop injection if true
    if Simulation.stop_injection:
        t_stop = Calc.Time_To_Stop(delta_vals[3])

    # Intialize matrices and connections
    q0 = M.Initilalize_Q_Vectors(total_cells, well_location, stop_injection)
    p0 = M.Initalize_P_Vector(total_cells)
    connections_x, connections_y = Con.Initialize_Connections(total_cells)
    WBHP_vals = [[] for _ in range(len(Grid.well_x))] # Creates list of list to create multiple plots if have multiple wells

    # Initialize matrices
    b_vector = M.RHS_Vector(accumulation, p0, q0)
    a_matrix = M.Form_A_Matrix(connections_x, connections_y, p0, total_cells, accumulation, delta_vals)

    # Attempt to build an ILU preconditioner once and reuse it for time steps.
    # This avoids a full sparse LU factorization each timestep and speeds up repeated solves.
    try:
        ilu = spla.spilu(a_matrix.tocsc())
        M_x = spla.LinearOperator(a_matrix.shape, ilu.solve)
        p_new, info = spla.gmres(a_matrix, b_vector, M=M_x, atol=1e-10)
        if info != 0:
            # GMRES did not converge; fall back to direct solve
            p_new = spla.spsolve(a_matrix, b_vector)
    except Exception:
        # If ILU/preconditioner construction fails (very small grids or singular), use direct solver
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
        WBHP_vals[w].append(p_new[well_location[w]-1])

    # Loop through time steps
    for t in range(Simulation.number_of_steps - 1):

        # Stops injection by setting q to zero when at time step specified
        if Simulation.stop_injection and t == t_stop - 1:
            print('stopped!')
            stop_injection = True
            q0 = M.Initilalize_Q_Vectors(total_cells, well_location, stop_injection)

        p_n = p_new # Defines previous future pressure as current pressure
        b_vector = M.RHS_Vector(accumulation, p_n, q0) # Recalculates b vector
        a_matrix = M.Form_A_Matrix(connections_x, connections_y, p_n, total_cells, accumulation, delta_vals) # Recalculates a matrix
        # Try iterative solve with the preconditioner first (reuse `ilu` if available)
        try:
            p_new, info = spla.gmres(a_matrix, b_vector, M=M_x, atol=1e-10)
            if info != 0:
                p_new = spla.spsolve(a_matrix, b_vector)
        except Exception:
            p_new = spla.spsolve(a_matrix, b_vector)

        for w in range(len(Grid.well_x)):
            WBHP_vals[w].append(p_new[well_location[w]-1])

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
        print(f"Final pressure of well {w + 1}:", p_new[well_location[w] -1 ])

    # Converts final pressure vector to matrix for plotting
    Pressure_Grid = p_new.reshape(Grid.NX_total, Grid.NY_total)

    time.sleep(1) # Stops timer
    stop_time = time.time() # Captures stop time
    elapsed_time = stop_time - start_time # Finds time taken to run
    print(f"Elapsed time: {elapsed_time:.2f} seconds") # Prints time taken to run

    coarse_map = Up.create_upscaled_grid()
    Up.plot_coarse_grid(coarse_map)


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
