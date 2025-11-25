'''
Matthew Ard 
Energy 223 - Phase II
04/25/2025

==========
This file is for the user to input the parameters
==========
'''

# Packages
import numpy as np

class Field: # Reservoir parameters
    Lx = 5000 # [ft]
    Ly = 5000 # [ft]
    Lz = 130 # [ft]
    D_top = 5000 # [ft]
    D_bot = 5140 # [ft]
    P_init = 3250 # [psi, lbf/in^2]
    P_ref = 14.7 # [psi, lbf/in^2]
    P_boundary = P_init # [psi, lbf/in^2]

class Rock: # Rock properties
    k_x = 85 # [mD]
    k_y = 40 # [mD]
    porosity = 0.23 # [fraction]
    c_R = 0 # [psi^-1] 

class Fluid: # Fluid properties
    c_f = 1.9e-5 # [psi^-1]
    density = 58 # [lbm/ft^3]
    viscosity = 0.75 # [cp]

class Grid: # Simulation grid properties
    
    # Set the boundary condition
    Boundary_Condition = 0 # 0 = no flux B.C., 1 = Const Pressure B.C.

    # number of cells in x-direction
    NX = 50 
    NY = 50 
    NZ = 1 

    # Well properties. In a list to add multiple wells
    well_x = np.array([0])
    well_y = np.array([0])
    well_rate = np.array([900]) # [STB/day]

    '''
    DO NOT CHANGE THESE VALUES

    Adjusts the values for different boundary conditions.
    If you have a constant boundary pressure condition then it will add a
    ring around the reservoir grid and adjust the well location to account for the
    wider grid.
    '''
    NX_total = NX + 2 * Boundary_Condition
    NY_total = NY + 2 * Boundary_Condition
    well_y_location = well_x + 1 * Boundary_Condition
    well_x_location = well_y + 1 * Boundary_Condition

    
class Simulation: # Simulation perameters
    
    run_time = 800 # [days]
    number_of_steps = 80 # [days]
    stop_injection = False # Set to 'True' if you want the injection to stop
    stop_time = 400 # [days] (Set day when injection stops)

class Upscaling: # Upscaling parameters

    NCy = 10
    NCx = 10