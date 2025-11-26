"""
Run the fine-scale simulation with no wells and aggregate fine-cell pressures
into coarse blocks using `upscaling.create_upscaled_grid()`.

Output:
- saves `fine_p_history.npy` (timesteps x fine_cells)
- saves `coarse_p_history.npy` (timesteps x coarse_cells)
- shows a plot of coarse block average pressures over time

Usage:
    python3 scripts/run_fine_aggregate.py

Note: this script runs the existing solver code (same math as `main.py`) but
forces zero sources (no wells). It is intended to produce a reference coarse
time-series by averaging fine-scale pressures inside each coarse block.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os, sys

# Ensure project root is on sys.path when running this script from the `scripts/` folder
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Grid, Field, Simulation
import calculations as Calc
import connections as Con
import matrix as M
import upscaling as Ups
import scipy.sparse.linalg as spla


def run_fine_no_wells():
    total_cells = Calc.Total_Cells_2D()
    delta_vals = Calc.Delta_Values()
    cell_volume = Calc.Cell_Volume(delta_vals)
    accumulation = Calc.Accumulation(cell_volume, delta_vals[3])

    # No wells: provide empty well_location and zero q vector
    well_location = []
    stop_injection = True
    q0 = M.Initilalize_Q_Vectors(total_cells, well_location, stop_injection)

    p0 = M.Initalize_P_Vector(total_cells)
    connections_x, connections_y = Con.Initialize_Connections(total_cells)

    # Pre-allocate storage
    nsteps = Simulation.number_of_steps
    p_history = np.zeros((nsteps, total_cells), dtype=float)

    # initial
    b_vector = M.RHS_Vector(accumulation, p0, q0)
    a_matrix = M.Form_A_Matrix(connections_x, connections_y, p0, total_cells, accumulation, delta_vals)

    # use sparse direct solver for robustness here
    p_new = spla.spsolve(a_matrix, b_vector)
    p_history[0, :] = p_new.ravel()

    # time loop
    for t in range(1, nsteps):
        p_n = p_new
        b_vector = M.RHS_Vector(accumulation, p_n, q0)
        a_matrix = M.Form_A_Matrix(connections_x, connections_y, p_n, total_cells, accumulation, delta_vals)
        p_new = spla.spsolve(a_matrix, b_vector)
        p_history[t, :] = p_new.ravel()

    return p_history


def aggregate_to_coarse(p_history):
    coarse_map = Ups.create_upscaled_grid()
    NC = len(coarse_map)
    nsteps = p_history.shape[0]

    coarse_history = np.zeros((nsteps, NC), dtype=float)

    for cid, fine_ids in coarse_map.items():
        # fine cell ids in `coarse_map` use 0-based indexing
        fine_idxs = np.array(fine_ids, dtype=int)
        # average over those fine cells for each timestep
        coarse_history[:, cid] = p_history[:, fine_idxs].mean(axis=1)

    return coarse_history


def main():
    print("Running fine-scale simulation (no wells) and aggregating to coarse blocks...")
    start = time.time()
    p_history = run_fine_no_wells()
    coarse_history = aggregate_to_coarse(p_history)
    np.save('fine_p_history.npy', p_history)
    np.save('coarse_p_history.npy', coarse_history)
    end = time.time()
    print(f"Done. Simulation time: {end-start:.2f}s. Saved histories to npy files.")

    # Quick plot of a few coarse blocks
    NC = coarse_history.shape[1]
    plt.figure(figsize=(8, 5))
    for cid in range(min(6, NC)):
        plt.plot(coarse_history[:, cid], label=f'coarse {cid}')
    plt.xlabel('Timestep')
    plt.ylabel('Pressure')
    plt.title('Coarse-block averaged pressure (from fine-scale sim)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
