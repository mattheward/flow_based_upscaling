'''
Matthew Ard 
Energy 224 - Flow Based Upscaling
04/25/2025

==========
This code contains the logic for the flow based upscaling.
==========
'''

# Import Packages
import matplotlib.pyplot as plt
import numpy as np
import math

# Import Files
from config import Grid, Upscaling, Simulation
import connections as Con
import calculations as Calc


def create_upscaled_grid():

    Nx = Grid.NX_total
    Ny = Grid.NY_total
    NCx = Upscaling.NCx
    NCy = Upscaling.NCy

    assert Nx % NCx == 0, "Fine grid cells can't be evenly divided."
    assert Ny % NCy == 0, "Fine grid cells can't be evenly divided."

    block_size_x = Nx // NCx
    block_size_y = Ny // NCy

    coarse_grid_map = {}

    for jc in range(NCy): # 0, 1, 2... from bottom
        for ic in range(NCx): # 0, 1, 2... from left
            
            # Calculate coarse cell ID based on the Cartesian layout
            coarse_cell_id = jc * NCx + ic + 1
            
            fine_cells_in_block = []
            
            # Find the starting bottom-left corner of the coarse block
            start_i = ic * block_size_x
            start_j = jc * block_size_y
            
            # Loop over the fine cells within this block's boundaries
            for j_fine in range(start_j, start_j + block_size_y):
                for i_fine in range(start_i, start_i + block_size_x):
                    
                    # Convert Cartesian (i, j) coordinates (0-based) to your
                    # 1-based global fine ID.
                    fine_cell_id = (j_fine * Nx + i_fine + 1)
                    fine_cells_in_block.append(fine_cell_id)
            
            coarse_grid_map[coarse_cell_id] = fine_cells_in_block
            
    return coarse_grid_map


def upscaled_connections():

    total_cells = Upscaling.NCx * Upscaling.NCy

    upscaled_reservior = np.arange(1, Upscaling.NCx * Upscaling.NCy + 1).reshape(Upscaling.NCx, Upscaling.NCy)
    
    connection_list_x = None
    connection_list_y = None

    connection_list_x = Con.Reservior_X_Connection(upscaled_reservior, connection_list_x)
    connection_list_y = Con.Reservior_Y_Connection(upscaled_reservior, connection_list_y)

    connections_x = Con.Neighbor_X_Values_Matrix(connection_list_x, total_cells)
    connections_y = Con.Neighbor_Y_Values_Matrix(connection_list_y, total_cells)

    return connections_x, connections_y


def coarse_well_locations(coarse_grid_map, well_index):

    well_index_coarse = {}


    for w in Grid.WELLS:

        for coarse_id, fine_cells in coarse_grid_map.items():
            if well_index[w] in fine_cells:
                well_index_coarse[w] = coarse_id

    return well_index_coarse


def solve_local_problems(coarse_grid_map, connections_x, connections_y, delta_vals, perm_field):

    upscaled_transmissbility = {}

    for i in range(1, Upscaling.NCx * Upscaling.NCy + 1):
        
        center_cell = i
        x_neighbor = next((num for num in connections_x[center_cell - 1, 1:] if num > i), None)
        y_neighbor = next((num for num in connections_y[center_cell - 1, 1:] if num > i), None)

        # If neighbor == None, then there is no neighbor in that direction

        if x_neighbor is not None:
            T_x_upscaled = upscaled_transmissibility(coarse_grid_map, center_cell, x_neighbor, delta_vals, 'x', perm_field)
            upscaled_transmissbility[(center_cell, x_neighbor)] = T_x_upscaled

        if y_neighbor is not None:
            T_y_upscaled = upscaled_transmissibility(coarse_grid_map, center_cell, y_neighbor, delta_vals, 'y', perm_field)
            upscaled_transmissbility[(center_cell, y_neighbor)] = T_y_upscaled


    return upscaled_transmissbility


def upscaled_transmissibility(coarse_grid_map, coarse_cell_i, coarse_cell_j, delta_vals, direction, perm_field):
    
    # defensive lookup: coarse_grid_map keys should be 0-based ints
    key_i = int(coarse_cell_i)
    key_j = int(coarse_cell_j)

    fine_cells_i = coarse_grid_map.get(key_i)
    if fine_cells_i is None:
        fine_cells_i = coarse_grid_map.get(int(coarse_cell_i))
    if fine_cells_i is None:
        raise KeyError(f"Coarse cell {coarse_cell_i} not found in coarse_grid_map keys {list(coarse_grid_map.keys())[:10]}...")

    fine_cells_j = coarse_grid_map.get(key_j)
    if fine_cells_j is None:
        fine_cells_j = coarse_grid_map.get(int(coarse_cell_j))
    if fine_cells_j is None:
        raise KeyError(f"Coarse cell {coarse_cell_j} not found in coarse_grid_map keys {list(coarse_grid_map.keys())[:10]}...")
    
    local_domain = set(fine_cells_i + fine_cells_j)

    global_to_local_map = {global_id: local_id for local_id, global_id in enumerate(local_domain)}
    num_local_cells = len(local_domain)
    # print(f"Isolated {num_local_cells} fine cells for this local problem.")

    A_local = np.zeros((num_local_cells, num_local_cells))
    b_local = np.zeros(num_local_cells)

    for local_idx, global_id in enumerate(local_domain):
        
        center_global_id = global_id

        if (global_id + 1) in global_to_local_map:

            neighbor_global_id = global_id + 1

            neighbor_local_idx = global_to_local_map[neighbor_global_id]

            k1 = perm_field['x'][center_global_id - 1]
            k2 = perm_field['x'][neighbor_global_id - 1]
            k_int = Calc.permeability_average(k1, k2)

            T_fine_x = Calc.Transmissibility_Calc(delta_vals, 'x', k_int)

            A_local[local_idx, neighbor_local_idx] = -T_fine_x
            A_local[local_idx, local_idx] += T_fine_x

        if (global_id - 1) in global_to_local_map:

            neighbor_global_id = global_id - 1

            neighbor_local_idx = global_to_local_map[neighbor_global_id]

            k1 = perm_field['x'][center_global_id - 1]
            k2 = perm_field['x'][neighbor_global_id - 1]
            k_int = Calc.permeability_average(k1, k2)

            T_fine_x = Calc.Transmissibility_Calc(delta_vals, 'x', k_int)

            A_local[local_idx, neighbor_local_idx] = -T_fine_x
            A_local[local_idx, local_idx] += T_fine_x

        if (global_id + Grid.NX_total) in global_to_local_map: 

            neighbor_global_id = global_id + Grid.NX_total

            neighbor_local_idx = global_to_local_map[neighbor_global_id]

            k1 = perm_field['y'][center_global_id - 1]
            k2 = perm_field['y'][neighbor_global_id - 1]
            k_int = Calc.permeability_average(k1, k2)   

            T_fine_y = Calc.Transmissibility_Calc(delta_vals, 'y', k_int)

            A_local[local_idx, neighbor_local_idx] = -T_fine_y
            A_local[local_idx, local_idx] += T_fine_y

        if (global_id - Grid.NX_total) in global_to_local_map: 

            neighbor_global_id = global_id - Grid.NX_total

            neighbor_local_idx = global_to_local_map[neighbor_global_id]

            k1 = perm_field['y'][center_global_id - 1]
            k2 = perm_field['y'][neighbor_global_id - 1]
            k_int = Calc.permeability_average(k1, k2)

            T_fine_y = Calc.Transmissibility_Calc(delta_vals, 'y', k_int)

            A_local[local_idx, neighbor_local_idx] = -T_fine_y
            A_local[local_idx, local_idx] += T_fine_y
        
    # Apply Dirichlet BCs by replacing rows (robust and avoids ill-conditioning)
    for local_idx, global_id in enumerate(local_domain):
        is_inflow, is_outflow = False, False
        if direction == 'x':
            if global_id in fine_cells_i and (global_id % Grid.NX_total == 0 or (global_id - 1) not in local_domain): is_inflow = True
            if global_id in fine_cells_j and ((global_id + 1) % Grid.NX_total == 0 or (global_id + 1) not in local_domain): is_outflow = True
        else: # 'y' direction
            if global_id in fine_cells_i and (global_id < Grid.NX_total or (global_id - Grid.NX_total) not in local_domain): is_inflow = True
            if global_id in fine_cells_j and (global_id >= (Grid.NY_total-1)*Grid.NX_total or (global_id + Grid.NX_total) not in local_domain): is_outflow = True

        if is_inflow:
            # enforce p = 1.0
            A_local[local_idx, :] = 0.0
            A_local[local_idx, local_idx] = 1.0
            b_local[local_idx] = 1.0
        if is_outflow:
            # enforce p = 0.0
            A_local[local_idx, :] = 0.0
            A_local[local_idx, local_idx] = 1.0
            b_local[local_idx] = 0.0

    p_local = np.linalg.solve(A_local, b_local)
    
    p_avg_i = np.mean([p_local[global_to_local_map[gid]] for gid in fine_cells_i])
    p_avg_j = np.mean([p_local[global_to_local_map[gid]] for gid in fine_cells_j])

    Q_total = 0.0

    for global_id_i in fine_cells_i:

        if direction == 'x':
            neighbor_j = global_id_i + 1
            if neighbor_j in fine_cells_j:
                pi = p_local[global_to_local_map[global_id_i]]
                pj = p_local[global_to_local_map[neighbor_j]]
                Q_total += T_fine_x * (pi - pj)
        else: # 'y' direction
            neighbor_j = global_id_i + Grid.NX_total
            if neighbor_j in fine_cells_j:
                pi = p_local[global_to_local_map[global_id_i]]
                pj = p_local[global_to_local_map[neighbor_j]]
                Q_total += T_fine_y * (pi - pj)

    delta_p_avg = p_avg_i - p_avg_j
    if abs(delta_p_avg) < 1e-9: return 0.0

    T_upscaled = Q_total / delta_p_avg

    # print(f"  -> Avg P_I={p_avg_i:.4f}, Avg P_J={p_avg_i:.4f}, Q_total={Q_total:.4f}")
    # print(f"  -> Upscaled T* = {T_upscaled:.4f}")

    return T_upscaled


def upscaled_well_transmissibility(coarse_grid_map, well_index_coarse, well_index_fine, well_rate, delta_vals, perm_field):

    core_domain = set(coarse_grid_map[well_index_coarse])

    buffer_ring = set()
    Nx = Grid.NX_total 
    Ny = Grid.NY_total 

    for global_id in core_domain:
        
        # First, convert the global ID to its (i, j) coordinates
        i = global_id % Nx
        j = global_id // Nx
        
        # Now, check for neighbors using the coordinates to prevent wraparound
        
        # Right neighbor: Only exists if we are not on the far-right edge
        if i + 1 < Nx:
            neighbor_id = global_id + 1
            if neighbor_id not in core_domain: 
                buffer_ring.add(neighbor_id)
            
        # Left neighbor: Only exists if we are not on the far-left edge
        if i - 1 >= 0:
            neighbor_id = global_id - 1
            if neighbor_id not in core_domain: 
                buffer_ring.add(neighbor_id)
            
        # Top neighbor: Only exists if we are not on the top edge
        if j + 1 < Ny:
            neighbor_id = global_id + Nx
            if neighbor_id not in core_domain: 
                buffer_ring.add(neighbor_id)

        # Bottom neighbor: Only exists if we are not on the bottom edge
        if j - 1 >= 0:
            neighbor_id = global_id - Nx
            if neighbor_id not in core_domain: 
                buffer_ring.add(neighbor_id)

    padding_domain = sorted(list(core_domain.union(buffer_ring)))
    global_to_local_map = {global_id: local_id for local_id, global_id in enumerate(padding_domain)}
    num_local_cells = len(padding_domain)
    print(f"Core domain: {len(core_domain)} cells. Padded domain: {num_local_cells} cells.")

    A_local = np.zeros((num_local_cells, num_local_cells))
    b_local = np.zeros(num_local_cells)

    for local_idx, global_id in enumerate(padding_domain):

        center_global_id = global_id
        
        if (global_id + 1) in global_to_local_map:

            neighbor_global_id = global_id + 1

            neighbor_local_idx = global_to_local_map[neighbor_global_id]

            k1 = perm_field['x'][center_global_id - 1]
            k2 = perm_field['x'][neighbor_global_id - 1]
            k_int = Calc.permeability_average(k1, k2)

            T_fine_x = Calc.Transmissibility_Calc(delta_vals, 'x', k_int)

            A_local[local_idx, neighbor_local_idx] = -T_fine_x
            A_local[local_idx, local_idx] += T_fine_x

        if (global_id - 1) in global_to_local_map:

            neighbor_global_id = global_id - 1

            neighbor_local_idx = global_to_local_map[neighbor_global_id]

            k1 = perm_field['x'][center_global_id - 1]
            k2 = perm_field['x'][neighbor_global_id - 1]
            k_int = Calc.permeability_average(k1, k2)

            T_fine_x = Calc.Transmissibility_Calc(delta_vals, 'x', k_int)

            A_local[local_idx, neighbor_local_idx] = -T_fine_x
            A_local[local_idx, local_idx] += T_fine_x

        if (global_id + Grid.NX_total) in global_to_local_map: 

            neighbor_local_idx = global_to_local_map[global_id + Grid.NX_total]

            k1 = perm_field['y'][center_global_id - 1]
            k2 = perm_field['y'][neighbor_local_idx - 1]
            k_int = Calc.permeability_average(k1, k2)   

            T_fine_y = Calc.Transmissibility_Calc(delta_vals, 'y', k_int)

            A_local[local_idx, neighbor_local_idx] = -T_fine_y
            A_local[local_idx, local_idx] += T_fine_y

        if (global_id - Grid.NX_total) in global_to_local_map: 

            neighbor_local_idx = global_to_local_map[global_id - Grid.NX_total]

            k1 = perm_field['y'][center_global_id - 1]
            k2 = perm_field['y'][neighbor_local_idx - 1]
            k_int = Calc.permeability_average(k1, k2)

            T_fine_y = Calc.Transmissibility_Calc(delta_vals, 'y', k_int)

            A_local[local_idx, neighbor_local_idx] = -T_fine_y
            A_local[local_idx, local_idx] += T_fine_y

    # Accept either 0-based or 1-based fine indices for the well
    if well_index_fine in global_to_local_map:
        local_well_idx = global_to_local_map[well_index_fine]
    elif (well_index_fine - 1) in global_to_local_map:
        local_well_idx = global_to_local_map[well_index_fine - 1]
    elif (well_index_fine + 1) in global_to_local_map:
        # defensive: sometimes callers pass 0-based when others pass 1-based
        local_well_idx = global_to_local_map[well_index_fine + 1]
    else:
        raise KeyError(f"Well fine-index {well_index_fine} not found in padded domain keys {list(global_to_local_map.keys())}")

    b_local[local_well_idx] += well_rate

    # Enforce Dirichlet P=0 on outer boundary using row replacement (robust)
    for local_idx, global_id in enumerate(padding_domain):

        is_outer_boundary = False
        if (global_id + 1) not in global_to_local_map: is_outer_boundary = True
        if (global_id - 1) not in global_to_local_map: is_outer_boundary = True
        if (global_id + Grid.NX_total) not in global_to_local_map: is_outer_boundary = True
        if (global_id - Grid.NX_total) not in global_to_local_map: is_outer_boundary = True

        if is_outer_boundary:
            # Do not overwrite the well row if the well happens to lie on the
            # physical/padded outer boundary; keep the source term in place.
            if local_idx == local_well_idx:
                continue
            # set row to enforce p = 0.0
            A_local[local_idx, :] = 0.0
            A_local[local_idx, local_idx] = 1.0
            b_local[local_idx] = 0.0

    p_local = np.linalg.solve(A_local, b_local)
    
    p_well = p_local[local_well_idx]


    coarse_pressures = [p_local[global_to_local_map[gid]] for gid in core_domain]
    p_block_avg = np.mean(coarse_pressures)

    print(f"  -> Solved local problem:")
    print(f"  -> P_well (fine cell pressure) = {p_well:.4f}")
    print(f"  -> P_block_avg (avg pressure of CORE)  = {p_block_avg:.4f}")

    delta_p = p_well - p_block_avg

    if abs(delta_p) < 1e-9:
        print("Warning: Delta P is near zero.")
        return 1e12 

    WI_star = well_rate / delta_p

    print(f"  -> Calculated Upscaled Well Index WI* = {WI_star:.4f}")
    
    return WI_star


def coarse_well_transmissibilities(coarse_map, well_index_coarse_vals, well_index_fine_vals, delta_vals, perm_field):

    upscaled_well_transmissibilities = {}

    for well, well_index_c in well_index_coarse_vals.items():
        well_rate = Grid.WELLS[well]['rates'][0]
        well_index_fine = well_index_fine_vals[well]
        WI_star = upscaled_well_transmissibility(coarse_map, well_index_c, well_index_fine, well_rate, delta_vals, perm_field)
        upscaled_well_transmissibilities[well] = WI_star

    return upscaled_well_transmissibilities


def plot_padded_domain(core_domain, buffer_ring):
    """
    Visualizes the core domain and its surrounding buffer ring.
    """
    Nx, Ny = Grid.NX_total, Grid.NY_total
    
    # Create a 2D array to represent the grid. We'll use different numbers
    # to represent different regions.
    # 0 = Rest of the reservoir
    # 1 = Buffer Ring
    # 2 = Core Domain
    visualization_array = np.zeros((Ny, Nx))
    
    # Mark the buffer ring cells with the value 1
    for fine_id in buffer_ring:
        j = fine_id // Nx  # Cartesian row
        i = fine_id % Nx   # Cartesian col
        visualization_array[j, i] = 1
        
    # Mark the core domain cells with the value 2 (this will overwrite any buffer cells if there's an error)
    for fine_id in core_domain:
        j = fine_id // Nx  # Cartesian row
        i = fine_id % Nx   # Cartesian col
        visualization_array[j, i] = 2
        
    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a custom colormap with 3 distinct colors
    cmap = plt.cm.colors.ListedColormap(['#d3d3d3', '#6495ed', '#ff6347']) # Gray, Blue, Red
    
    # Use origin='lower' to match your Cartesian grid convention
    mat = ax.imshow(visualization_array, cmap=cmap, interpolation='none', aspect='equal', origin='lower')
    
    # Add grid lines for all fine cells
    ax.set_xticks(np.arange(-.5, Nx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, Ny, 1), minor=True)
    ax.grid(which='both', color='white', linestyle='-', linewidth=1)
    
    # Add labels for cell IDs
    for j in range(Ny):
        for i in range(Nx):
            ax.text(i, j, str(j * Nx + i), ha='center', va='center', color='black', fontsize=8)

    ax.set_title("Visualization of Padded Domain for Well Index Upscaling", fontsize=16)
    ax.set_xlabel("Fine Cell Index (i)")
    ax.set_ylabel("Fine Cell Index (j)")

    # Create a colorbar with custom labels
    cbar = plt.colorbar(mat, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Reservoir', 'Buffer Ring', 'Core Domain'])
    
    plt.tight_layout()
    plt.show()


def coarse_well_treamtment(upscaled_well_transmissibility, well_index_coarse_vals, a_matrix_c, b_vector_c, current_time):
    """
    Applies the upscaled well transmissibility to the coarse A matrix and b vector.
    """

    # for w, WI_star in upscaled_well_transmissibility.items():
    #     coarse_well_idx = well_index_coarse_vals[w]

    #     b_vector_c[coarse_well_idx - 1] -= Grid.WELLS[w]['rates'][0]
    
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

            b_vector_c[well_index_coarse_vals[w] - 1] -= Grid.WELLS[w]['rates'][rate_index]  # [STB/day]

        elif Grid.WELLS[w]['type'] == 'producer':
            b_vector_c[well_index_coarse_vals[w] - 1] -= upscaled_transmissibility[w] * Grid.BHP
            # LIL supports item assignment
            a_matrix_c[well_index_coarse_vals[w] - 1, well_index_coarse_vals[w] - 1] -= upscaled_transmissibility[w]

    # Convert back to CSR for efficient solves
    try:
        a_matrix = a_matrix.tocsr()
    except Exception:
        pass

        # # Match fine-grid sign/convention from `matrix.well_treatment`:
        # # - Injector: apply rate as a source (b -= rate)
        # # - Producer: apply well transmissibility (diag -= WI*) and b -= WI* * BHP
        # well_def = Grid.WELLS[w]

        # if well_def.get('type') == 'injector':
        #     # handle rate schedule if present (use first entry by default)
        #     rate_index = 0
        #     for j, start_time in enumerate(Simulation.RATE_SCHEDULE):
        #         if 0 >= start_time:  # calling outside time loop; assume initial rate
        #             rate_index = j
        #         else:
        #             break

        #     rate = well_def.get('rates', [0])[rate_index]
        #     b_vector_c[coarse_well_idx - 1] -= rate

        # elif well_def.get('type') == 'producer':
        #     a_matrix_c[coarse_well_idx - 1, coarse_well_idx - 1] -= WI_star
        #     b_vector_c[coarse_well_idx - 1] -= WI_star * Grid.BHP

        # else:
        #     # Unknown type: fall back to applying as injector rate if provided
        #     if 'rates' in well_def:
        #         b_vector_c[coarse_well_idx - 1] -= well_def['rates'][0]

    return a_matrix_c, b_vector_c