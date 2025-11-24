'''
Matthew Ard 
Energy 223 - Phase II
04/25/2025

==========
This file is an algorithm to determine which cells are neighboring each other.
It uses Method 2 from the TA Help Session slides (Connection Lists)
==========
'''

# Packges
import numpy as np

# Files
from config import Grid

# Create empty lists to hold the pairs of connected reservoir cells
def Initialize_Arrays():

    '''
    Set up reservoir grid by creating a numpy matrix with reference numbers in each cell

                                      [1 2 3]
    For a 3x3 it would look like this [4 5 6]
                                      [7 8 9]

    '''
    reservior = np.arange(1, Grid.NX_total * Grid.NY_total + 1).reshape(Grid.NX_total, Grid.NY_total)

    # connection lists will be created vectorized when needed; return placeholders for compatibility
    connection_list_x = None
    connection_list_y = None

    return reservior, connection_list_x, connection_list_y


# Determine the location (reference cell number) of the well(s)
def Well_Location():
    reservior = Initialize_Arrays()[0] # Gets reservoir from Initialize_Arrays function

    # Creates list of locations (reference cell numbers) for the wells. Creates a list for when there is multiple wells
    well_location = [reservior[Grid.well_x_location[w]-1, Grid.well_y_location[w]-1] for w in range(len(Grid.well_x))]

    return well_location

# Function to create matrix of pairs of connected reservior cells in the x direction
def Reservior_X_Connection(reservior, connection_list_x):
    # Vectorized creation: pair each cell with the one next in the second axis
    a = reservior[:, :-1].ravel()
    b = reservior[:, 1:].ravel()
    connection_list_x = np.column_stack((a, b)).astype(int)

    return connection_list_x # Return the list of x connections


# Function to create matrix of pairs of connected reservior cells in the y direction
def Reservior_Y_Connection(reservior, connection_list_y):
    # Vectorized creation: pair each cell with the one next in the first axis
    a = reservior[:-1, :].ravel()
    b = reservior[1:, :].ravel()
    connection_list_y = np.column_stack((a, b)).astype(int)

    return connection_list_y # Return the list of y connections


# Find the location (reference cell number) of the neighboring cells in the x direction using connections_list
def Neighbor_X_Values_Matrix(connection_list_x, total_cells):
    # Build neighbor lists in O(number_of_connections) rather than O(n^2)
    connections_x = np.zeros((total_cells, 3), dtype=int)
    connections_x[:, 0] = np.arange(1, total_cells + 1)

    # iterate through each connection and append neighbors to both cells
    # connection_list_x is shape (M,2)
    for a, b in connection_list_x:
        # append b to a's neighbor slots
        row_a = a - 1
        if connections_x[row_a, 1] == 0:
            connections_x[row_a, 1] = b
        else:
            connections_x[row_a, 2] = b

        # append a to b's neighbor slots
        row_b = b - 1
        if connections_x[row_b, 1] == 0:
            connections_x[row_b, 1] = a
        else:
            connections_x[row_b, 2] = a

    return connections_x # Returns array with each neighboring cell for each cell in x direction


# Does the same thing as the above function but for the y direction
def Neighbor_Y_Values_Matrix(connection_list_y, total_cells):
    connections_y = np.zeros((total_cells, 3), dtype=int)
    connections_y[:, 0] = np.arange(1, total_cells + 1)

    for a, b in connection_list_y:
        row_a = a - 1
        if connections_y[row_a, 1] == 0:
            connections_y[row_a, 1] = b
        else:
            connections_y[row_a, 2] = b

        row_b = b - 1
        if connections_y[row_b, 1] == 0:
            connections_y[row_b, 1] = a
        else:
            connections_y[row_b, 2] = a

    return connections_y


# Function calls all previous functions to obtain the connections list in the x and y direction and return them both by calling one function
def Initialize_Connections(total_cells):
    reservior, _, _ = Initialize_Arrays()
    connection_list_x = Reservior_X_Connection(reservior, None)
    connection_list_y = Reservior_Y_Connection(reservior, None)

    connections_x = Neighbor_X_Values_Matrix(connection_list_x, total_cells)
    connections_y = Neighbor_Y_Values_Matrix(connection_list_y, total_cells)

    return connections_x, connections_y


# Finds the values of the boundary cells (for when we have constant pressure ring)
def Find_Boundary_Values():

    reservior = Initialize_Arrays()[0]
    boundary_values = []

    i_values = [0, Grid.NX_total - 1]

    # Finds the boundary cells locations (reference number) for the boundary columns
    for i in i_values:
        for j in range(Grid.NY_total):
            boundary_values.append(reservior[i, j])

    # # Finds the boundary cells locations (reference number) for the boundary rows
    for j in i_values:
        for i in range(Grid.NX_total):
            boundary_values.append(reservior[i, j])

    boundary_values = list(set(boundary_values)) # Removes any duplicate cells reference numbers

    return boundary_values
