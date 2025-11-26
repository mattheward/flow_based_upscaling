"""
Run fine-scale simulation with no wells, run the coarse simulator (no wells),
and compare coarse-averaged fine solution to the coarse simulator solution.

Usage:
    python3 scripts/compare_fine_coarse.py

Notes:
- Assumes your coarse transmissibilities can be produced by `upscaling.solve_local_problems`.
- If that function fails, the script will stop and report the exception so you can
  provide the upscaled transmissibility dictionary manually.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import os, sys

# Ensure project root is on sys.path when running this script from the `scripts/` folder
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Grid, Field, Simulation, Upscaling
import calculations as Calc
import connections as Con
import matrix as M
import upscaling as Up
import coarse_simulator as Cs


def run_fine_no_wells():
    total_cells = Calc.Total_Cells_2D()
    delta_vals = Calc.Delta_Values()
    cell_volume = Calc.Cell_Volume(delta_vals)
    accumulation = Calc.Accumulation(cell_volume, delta_vals[3])

    # No wells: empty well list and zero q
    well_location = []
    stop_injection = True
    q0 = M.Initilalize_Q_Vectors(total_cells, well_location, stop_injection)

    p0 = M.Initalize_P_Vector(total_cells)
    connections_x, connections_y = Con.Initialize_Connections(total_cells)

    nsteps = Simulation.number_of_steps
    p_history = np.zeros((nsteps, total_cells), dtype=float)

    b_vector = M.RHS_Vector(accumulation, p0, q0)
    a_matrix = M.Form_A_Matrix(connections_x, connections_y, p0, total_cells, accumulation, delta_vals)
    p_new = spla.spsolve(a_matrix, b_vector)
    p_history[0, :] = p_new.ravel()

    for t in range(1, nsteps):
        p_n = p_new
        b_vector = M.RHS_Vector(accumulation, p_n, q0)
        a_matrix = M.Form_A_Matrix(connections_x, connections_y, p_n, total_cells, accumulation, delta_vals)
        p_new = spla.spsolve(a_matrix, b_vector)
        p_history[t, :] = p_new.ravel()

    return p_history, connections_x, connections_y, delta_vals


def run_coarse_no_wells(upscaled_T, connections_x_c, connections_y_c, delta_coarse_vals):
    num_coarse = Cs.Total_Coarse_Cells_2D()
    coarse_cell_volume = Cs.Coarse_Cell_Volume(delta_coarse_vals)
    # coarse_accumulation can be a scalar or array; make it an array for the coarse solver
    acc_scalar = Calc.Accumulation(coarse_cell_volume, delta_coarse_vals[3])
    coarse_accumulation = np.full(num_coarse, acc_scalar)

    # no wells
    q0_c = M.Initilalize_Q_Vectors(num_coarse, [], True)
    p0_c = M.Initalize_P_Vector(num_coarse)

    nsteps = Simulation.number_of_steps
    p_history_c = np.zeros((nsteps, num_coarse), dtype=float)

    b_vector_c = M.RHS_Vector(coarse_accumulation, p0_c, q0_c)
    a_matrix_c = Cs.Form_A_Matrix_Coarse(connections_x_c, connections_y_c, upscaled_T, coarse_accumulation, num_coarse)
    p_new_c = spla.spsolve(a_matrix_c, b_vector_c)
    p_history_c[0, :] = p_new_c.ravel()

    for t in range(1, nsteps):
        p_n_c = p_new_c
        b_vector_c = M.RHS_Vector(coarse_accumulation, p_n_c, q0_c)
        a_matrix_c = Cs.Form_A_Matrix_Coarse(connections_x_c, connections_y_c, upscaled_T, coarse_accumulation, num_coarse)
        p_new_c = spla.spsolve(a_matrix_c, b_vector_c)
        p_history_c[t, :] = p_new_c.ravel()

    return p_history_c


def aggregate_fine_to_coarse(p_history_fine, coarse_map):
    nsteps = p_history_fine.shape[0]
    NC = len(coarse_map)
    coarse_history = np.zeros((nsteps, NC), dtype=float)

    for cid, fine_ids in coarse_map.items():
        idxs = np.array(fine_ids, dtype=int)
        coarse_history[:, cid] = p_history_fine[:, idxs].mean(axis=1)

    return coarse_history


def compare_series(ref, sim):
    # Both shape (timesteps, cells)
    assert ref.shape == sim.shape
    err = ref - sim
    rmse_t = np.sqrt(np.mean(err**2, axis=1))
    overall_rmse = np.sqrt(np.mean(err**2))
    return rmse_t, overall_rmse


def main():
    print("Running fine-scale (no wells)...")
    start = time.time()
    p_fine, fx, fy, delta_vals = run_fine_no_wells()
    coarse_map = Up.create_upscaled_grid()
    # create coarse connections
    cx, cy = Up.upscaled_connections()

    # compute upscaled transmissibilities (user's code)
    print("Computing upscaled transmissibilities (may take a while)...")
    # use coarse connections (cx, cy) when solving local problems
    upscaled_T = Up.solve_local_problems(coarse_map, cx, cy, delta_vals)

    # run coarse sim
    delta_coarse_vals = Cs.Delta_Coarse_Values()
    p_coarse_sim = run_coarse_no_wells(upscaled_T, cx, cy, delta_coarse_vals)

    # aggregate fine to coarse
    p_coarse_ref = aggregate_fine_to_coarse(p_fine, coarse_map)

    # compare
    rmse_t, overall_rmse = compare_series(p_coarse_ref, p_coarse_sim)
    print(f"Overall RMSE between aggregated fine and coarse sim: {overall_rmse:.6f}")

    # quick plots
    timesteps = np.arange(p_coarse_ref.shape[0])
    plt.figure(figsize=(9, 5))
    for cid in range(min(6, p_coarse_ref.shape[1])):
        plt.plot(timesteps, p_coarse_ref[:, cid], '--', label=f'ref {cid}')
        plt.plot(timesteps, p_coarse_sim[:, cid], '-', label=f'coarse {cid}')
    plt.xlabel('Timestep')
    plt.ylabel('Pressure')
    plt.title(f'Coarse comparison (overall RMSE={overall_rmse:.4g})')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7, 3))
    plt.plot(timesteps, rmse_t)
    plt.xlabel('Timestep')
    plt.ylabel('RMSE')
    plt.title('RMSE over time')
    plt.grid(True)
    plt.show()

    end = time.time()
    print(f"Done in {end-start:.2f}s")


if __name__ == '__main__':
    main()
