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
import os
import pickle

# Import Files
from config import Grid, Upscaling
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
            coarse_cell_id = jc * NCx + ic
            
            fine_cells_in_block = []
            
            # Find the starting bottom-left corner of the coarse block
            start_i = ic * block_size_x
            start_j = jc * block_size_y
            
            # Loop over the fine cells within this block's boundaries
            for j_fine in range(start_j, start_j + block_size_y):
                for i_fine in range(start_i, start_i + block_size_x):
                    
                    # Convert Cartesian (i, j) coordinates (0-based) to your
                    # 1-based global fine ID.
                    fine_cell_id = (j_fine * Nx + i_fine)
                    fine_cells_in_block.append(fine_cell_id)
            
            coarse_grid_map[coarse_cell_id] = fine_cells_in_block
            
    return coarse_grid_map


def plot_coarse_grid(coarse_map):
    """
    Visualizes how the fine grid is partitioned into the coarse grid.
    This function does not need to change.
    """
    Nx = Grid.NX_total
    Ny = Grid.NY_total
    NCx = Upscaling.NCx
    NCy = Upscaling.NCy
    
    visualization_array = np.zeros((Ny, Nx))
    
    for coarse_id, fine_cells in coarse_map.items():
        for fine_id in fine_cells:

            j = fine_id // Nx  # Cartesian row (0 = bottom)
            i = fine_id % Nx   # Matrix col (from left)

            # With `origin='lower'` in imshow we can store directly using Cartesian j
            visualization_array[j, i] = coarse_id
            
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('viridis', NCx * NCy)
    # Use origin='lower' so row 0 appears at the bottom (Cartesian convention)
    mat = ax.imshow(visualization_array, cmap=cmap, interpolation='none', aspect='equal', origin='lower')
    
    block_size_x = Nx // NCx
    block_size_y = Ny // NCy

    ax.set_xticks(np.arange(-.5, Nx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, Ny, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(-.5, Nx, block_size_x))
    ax.set_yticks(np.arange(-.5, Ny, block_size_y))
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for coarse_id in coarse_map.keys():
        jc = coarse_id // NCx
        ic = coarse_id % NCx
        
        # Calculate text position in Cartesian coordinates
        center_x = ic * block_size_x + block_size_x / 2 - 0.5
        center_y = jc * block_size_y + block_size_y / 2 - 0.5
        
        # center_y is already in Cartesian coordinates (0 = bottom), matching origin='lower'
        ax.text(center_x, center_y, str(coarse_id), 
                ha='center', va='center', color='white', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc='black', ec='black', lw=1, alpha=0.4))

    ax.set_title(f"Corrected Cartesian Grid ({Nx}x{Ny}) to ({NCx}x{NCy})", fontsize=16)
    ax.set_xlabel("Fine Cell Index (i)")
    ax.set_ylabel("Fine Cell Index (j)")
    
    # origin='lower' already places 0 at the bottom; no inversion is needed.
    
    plt.colorbar(mat, ticks=range(NCx * NCy), label='Coarse Cell ID')
    plt.tight_layout()
    plt.show()


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


def solve_local_problems(coarse_grid_map, connections_x, connections_y, delta_vals, cache_path=None, force_recompute=False):
    """
    Compute upscaled transmissibilities for each coarse interface.

    Parameters:
    - coarse_grid_map, connections_x, connections_y, delta_vals: as before
    - cache_path: optional path to a pickle file to load/save the transmissibility dict
    - force_recompute: if True, ignore any existing cache and recompute

    Returns: dict mapping (i, j) tuples to upscaled transmissibility values
    """

    # Try loading from cache if provided
    if cache_path is not None and os.path.exists(cache_path) and not force_recompute:
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            print(f"Loaded upscaled transmissibilities from cache: {cache_path}")
            return cached
        except Exception as e:
            print(f"Warning: failed to load cache '{cache_path}': {e} -- continuing to recompute")

    upscaled_transmissbility = {}

    total_coarse = Upscaling.NCx * Upscaling.NCy
    for i in range(1, total_coarse + 1):
        center_cell = i
        x_neighbor = next((num for num in connections_x[center_cell - 1, 1:] if num > i), None)
        y_neighbor = next((num for num in connections_y[center_cell - 1, 1:] if num > i), None)

        # If neighbor == None, then there is no neighbor in that direction
        if x_neighbor is not None:
            T_x_upscaled = upscaled_transmissibility(coarse_grid_map, center_cell, x_neighbor, delta_vals, 'x')
            upscaled_transmissbility[(center_cell, x_neighbor)] = T_x_upscaled

        if y_neighbor is not None:
            T_y_upscaled = upscaled_transmissibility(coarse_grid_map, center_cell, y_neighbor, delta_vals, 'y')
            upscaled_transmissbility[(center_cell, y_neighbor)] = T_y_upscaled

    # Save cache if requested
    if cache_path is not None:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(upscaled_transmissbility, f)
            print(f"Saved upscaled transmissibilities to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: failed to save cache '{cache_path}': {e}")

    return upscaled_transmissbility


def upscaled_transmissibility(coarse_grid_map, coarse_cell_i, coarse_cell_j, delta_vals, direction):
    # defensive lookup: coarse_grid_map keys should be 0-based ints
    key_i = int(coarse_cell_i - 1)
    key_j = int(coarse_cell_j - 1)

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
    print(f"Isolated {num_local_cells} fine cells for this local problem.")

    A_local = np.zeros((num_local_cells, num_local_cells))
    b_local = np.zeros(num_local_cells)

    T_fine_x = Calc.Transmissibility_Calc(delta_vals, 'x')
    T_fine_y = Calc.Transmissibility_Calc(delta_vals, 'y')

    for local_idx, global_id in enumerate(local_domain):
        
        if (global_id + 1) in global_to_local_map:
            neighbor_local_idx = global_to_local_map[global_id + 1]
            A_local[local_idx, neighbor_local_idx] = -T_fine_x
            A_local[local_idx, local_idx] += T_fine_x
        if (global_id - 1) in global_to_local_map:
            neighbor_local_idx = global_to_local_map[global_id - 1]
            A_local[local_idx, neighbor_local_idx] = -T_fine_x
            A_local[local_idx, local_idx] += T_fine_x
        if (global_id + Grid.NX_total) in global_to_local_map: 
            neighbor_local_idx = global_to_local_map[global_id + Grid.NX_total]
            A_local[local_idx, neighbor_local_idx] = -T_fine_y
            A_local[local_idx, local_idx] += T_fine_y
        if (global_id - Grid.NX_total) in global_to_local_map: 
            neighbor_local_idx = global_to_local_map[global_id - Grid.NX_total]
            A_local[local_idx, neighbor_local_idx] = -T_fine_y
            A_local[local_idx, local_idx] += T_fine_y
        
    BIG_NUM = 1e20

    for local_idx, global_id in enumerate(local_domain):
        is_inflow, is_outflow = False, False
        if direction == 'x':
            if global_id in fine_cells_i and (global_id % Grid.NX_total == 0 or (global_id - 1) not in local_domain): is_inflow = True
            if global_id in fine_cells_j and ((global_id + 1) % Grid.NX_total == 0 or (global_id + 1) not in local_domain): is_outflow = True
        else: # 'y' direction
            if global_id in fine_cells_i and (global_id < Grid.NX_total or (global_id - Grid.NX_total) not in local_domain): is_inflow = True
            if global_id in fine_cells_j and (global_id >= (Grid.NY_total-1)*Grid.NX_total or (global_id + Grid.NX_total) not in local_domain): is_outflow = True

        if is_inflow:
            A_local[local_idx, local_idx] += BIG_NUM
            b_local[local_idx] += BIG_NUM * 1.0  # Set infl
        if is_outflow:
            A_local[local_idx, local_idx] += BIG_NUM
            b_local[local_idx] += BIG_NUM * 0.0  # Set outflow to 0 pressure

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

    print(f"  -> Avg P_I={p_avg_i:.4f}, Avg P_J={p_avg_i:.4f}, Q_total={Q_total:.4f}")
    print(f"  -> Upscaled T* = {T_upscaled:.4f}")

    return T_upscaled