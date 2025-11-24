'''
Matthew Ard 
Energy 223 - Phase II
04/25/2025

==========
This file runs all the numerical calculations used in the other files.
==========
'''

# Packages
import numpy as np

# Files
from config import Grid, Field, Fluid, Rock, Simulation

# Calculate total amount of cells in reservoir grid
def Total_Cells_2D():

    return Grid.NX_total * Grid.NY_total # [cells]


# Calculate the delta values (x, y, z, t) and return a list for indexing
def Delta_Values():

    delta_x = Field.Lx / Grid.NX_total
    delta_y = Field.Ly / Grid.NY_total
    delta_z = Field.Lz / Grid.NZ
    delta_t = int(Simulation.run_time / Simulation.number_of_steps)

    return [delta_x, delta_y, delta_z, delta_t]


# Calculates volume of cells using delta_vals list
def Cell_Volume(delta_vals):

    return delta_vals[0] * delta_vals[1] * delta_vals[2] # [ft^3]


# Calculates formation volume factor for given pressure value
def B_Calc(P):

    return (1 + Fluid.c_f*(P + Field.P_ref))**(-1) # [-]


# Finds the value of the formation value factor at the interface of two connecting cells
def B_Interface_Calc(B1, B2):

    return (B1 + B2) / 2 # [-]


# Calculates the accumulation number
def Accumulation(Cell_Volume, delta_t):

    return (Cell_Volume * Rock.porosity * Fluid.c_f)/(5.615 * delta_t) # [ft^3/psi/day]


# Calculates transissibility for the off-diagonal values
def Transmissibility_Calc(b_int, delta_vals, direction):

    # Checks which direction the cells interface each other to use correct equation
    if direction == 'x':
        T = (1 / (b_int * Fluid.viscosity)) * ((delta_vals[1] * delta_vals[2] * Rock.k_x) / delta_vals[0]) / 887.5 # [ft^2/psi/day]
    else:
        T = (1 / (b_int * Fluid.viscosity)) * ((delta_vals[0] * delta_vals[2] * Rock.k_y) / delta_vals[1]) / 887.5 # [ft^2/psi/day]

    return float(T) # [ft^2/psi/day]

# Calculates transmissibility of the diagonals using transmissbility values calculated before (inputed as a list)
def Diag_Transmissibility_Calc(t_values, Accumulation):

    return float(sum(t_values) + Accumulation) # [ft^2/psi/day]
    

# Calculates the mass flow rate of the edges of the grid
def Flow_Rate(p_vals):

    delta_values = Delta_Values() # Finds delta_values
    flow_rate = np.array([]) # Initializes list to hold flow_rates

    b_star = B_Calc(Field.P_boundary) # Calculates b of the boundary with constant pressure, since its the safe value for all boundary cells

    for j in [1, Grid.NX]: # Index edge columns in grid
        for i in range(Grid.NX): # Index each cell in the edge column

            l = ((j + 1) - 1) * Grid.NX_total + (i + 1) # Find reference value of edge cell for indexing
            b1 = B_Calc(p_vals[l]) # Calculates formulation flow factor for edge cell
            b_int = B_Interface_Calc(b1, b_star) # Calculates flow factor at interface

            # Calculates the mass flow rate at edge cell
            f = ((Rock.k_y * delta_values[0] * delta_values[2]) / (delta_values[1])) * ((p_vals[l] - Field.P_boundary) / (b_int * Fluid.viscosity))
            flow_rate = np.append(flow_rate, f) # adds flow rate to list

    # Perform same steps but for the edge rows
    for i in [1, Grid.NY]: 
        for j in range(Grid.NX):

            l = ((j + 2) - 1) * Grid.NX_total + (i + 1)
            #print(p_vals[l-1])
            b2 = B_Calc(p_vals[l])
            b_int = B_Interface_Calc(b2, b_star)
            f = ((Rock.k_x * delta_values[1] * delta_values[2]) / (delta_values[0])) * ((p_vals[l-1] - Field.P_boundary)/ (b_int * Fluid.viscosity))
            flow_rate = np.append(flow_rate, f)

    return np.sum(flow_rate) / 887 # Divide by 887 to convert to STB/day


# Calculates time step to stop injection
def Time_To_Stop(delta_t):

    return Simulation.number_of_steps - ((Simulation.run_time - Simulation.stop_time) / delta_t)
