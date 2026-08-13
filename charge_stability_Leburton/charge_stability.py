'''
This example computes the charge stability diagram of a double quantum dot.

Ref: https://journals.aps.org/prb/pdf/10.1103/PhysRevB.110.205428
'''


from pathlib import Path
from time import time
import atexit

import numpy as np

from pyscf import fci
import frayedends as fe

from solvers import closed_shell_calculation, open_shell_calculation
from utils import DataHelper, LoggerSetup, cleanup_matching_files, format_elapsed_time

logging_setup = LoggerSetup(
    __name__,
    "charge_stability.pkl",
    "charge_stability.log",
    env_log_level="QD_LOG_LEVEL",
)
logger = logging_setup.logger
data_helper = DataHelper(
    data_dir=logging_setup.data_dir,
    results_filename=logging_setup.results_path.name,
    checkpoint_glob="charge_stability_runs_*.pkl",
    logger=logger,
    fsync_env="QD_CHECKPOINT_FSYNC",
)
data_dir = data_helper.data_dir
results_path = data_helper.results_path
log_path = logging_setup.log_path
script_dir = Path(__file__).resolve().parent
temp_file_patterns = ("mad*.log", "potential.dat")
temp_file_dirs = (Path.cwd(), script_dir)


def cleanup_temp_files():
    removed = cleanup_matching_files(temp_file_patterns, temp_file_dirs, logger=logger)
    if removed:
        logger.debug("Removed %s temporary files", len(removed))
    return removed


atexit.register(cleanup_temp_files)
cleanup_temp_files()

# -----------------------------------------------------------------------------
# Paths and physical constants
# -----------------------------------------------------------------------------
start_time = time()

# SI units
hbar = 1.054571817e-34  # J.s
m_e = 9.10938356e-31    # kg
m_GaAs = 0.067 * m_e       # kg
e = 1.602176634e-19     # C
epsilon_0 = 8.854187817e-12  # F/m (vacuum permittivity)
epsilon_r = 12.9  # Relative permittivity for GaAs

l0 = 30e-9  # 50 nm in meters
E0 = hbar**2 / (m_GaAs * l0**2) 

eVtoE0 = e * 1 / E0  # 1 eV in effective units
C_E0 = e**2 / (4 * np.pi * epsilon_0 * epsilon_r * l0) * 1 / E0  # Coulomb energy in effective units

logger.info("1 eV in effective units: %.6f", eVtoE0)
logger.info("Coulomb energy in effective units: %.6f", C_E0)

# -----------------------------------------------------------------------------
# Device geometry and sweep configuration
# -----------------------------------------------------------------------------
L = 1 # l0 units
R = 30 / (l0 * 1e9)  # Dot radius
d = 60 / (l0 * 1e9)  # Dot separation

JtoeV = e * 1 # J to eV conversion factor

electrons_configurations = {
    1: [(1, 0), 2],
    2: [2, 6],
    #3: [(2, 1), 8],
    #4: [4, 10],
    #5: [(3, 2), 12],
    #6: [6, 14],
}

Vs = np.linspace(21, 29, 10) * 1e-3  # V

# -----------------------------------------------------------------------------
# Optimization parameters
# -----------------------------------------------------------------------------

eV_error = 1e-7

thresh = min(eV_error * JtoeV / E0 / 10, 1e-5)  # Set MADNESS threshold to be an order of magnitude smaller than the desired energy error
e_conv = eV_error * JtoeV / E0  # Convert eV error to effective units

opt_thresh = e_conv / 10  # Set orbital optimization threshold to be an order of magnitude smaller than the desired energy error

max_iter_orbital_optimization = 5
max_iter = 100

save_potential = True  # Set to True to save the potential along the x-axis for each voltage point

# -----------------------------------------------------------------------------
# Checkpoint and stored-output configuration
# -----------------------------------------------------------------------------
energy_plot_electron_counts = tuple(n for n in (1, 2) if n in electrons_configurations)
full_save_interval_points = data_helper.read_positive_int_from_env("QD_FULL_SAVE_INTERVAL", 50)
checkpoint_fsync = data_helper.should_fsync_checkpoints()

sweep_config = {
    "L": float(L),
    "R": float(R),
    "d": float(d),
    "voltages": [float(v) for v in Vs],
    "electrons_configurations": {
        str(n): [list(electrons) if isinstance(electrons, tuple) else int(electrons), int(n_orbitals)]
        for n, (electrons, n_orbitals) in electrons_configurations.items()
    },
    "energy_plot_electron_counts": [int(n) for n in energy_plot_electron_counts],
    "eV_error": float(eV_error),
    "e_conv": float(e_conv),
    "thresh": float(thresh),
    "opt_thresh": float(opt_thresh),
    "max_iter": int(max_iter),
    "max_iter_orbital_optimization": int(max_iter_orbital_optimization),
}
sweep_hash = data_helper.build_hash(sweep_config)
checkpoint_path = data_dir / f"charge_stability_runs_{sweep_hash}.pkl"

if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
    data_helper.append_checkpoint_record(
        checkpoint_path,
        {
            "kind": "header",
            "sweep_hash": sweep_hash,
            "sweep_config": sweep_config,
        },
        fsync=checkpoint_fsync,
    )

# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------
results = {
    "metadata": {
        "description": "Charge stability sweep for a double quantum dot.",
        "length_scale_nm": l0 * 1e9,
        "effective_units_reference_energy": float(E0),
        "coulomb_energy_effective_units": float(C_E0),
        "eV_error_tolerance": float(eV_error),
        "e_conv_tolerance_effective_units": float(e_conv),
        "voltage_grid": [float(v) for v in Vs],
        "voltage_unit": "V",
        "computed_voltage_region": "vL >= vR; use symmetry for vL < vR",
        "energy_plot_electron_counts": [int(n) for n in energy_plot_electron_counts],
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_sweep_hash": sweep_hash,
        "full_save_interval_points": int(full_save_interval_points),
        "log_file": str(log_path),
    },
    "runs": [],
}

loaded_data = data_helper.load_data_with_checkpoints(
    default={"metadata": {}, "runs": [], "charge_stability": {}},
    checkpoint_paths=[checkpoint_path],
)
full_results = loaded_data["data"] or {"metadata": {}, "runs": [], "charge_stability": {}}
completed_runs_by_grid_point = {}


def grid_key_from_run(run):
    v_l = float(run["vL"])
    v_r = float(run["vR"])
    i = int(np.argmin(np.abs(Vs - v_l)))
    j = int(np.argmin(np.abs(Vs - v_r)))
    return i, j


full_snapshot_sweep_hash = full_results.get("metadata", {}).get("checkpoint_sweep_hash")
if full_snapshot_sweep_hash == sweep_hash:
    for run in full_results.get("runs", []):
        completed_runs_by_grid_point[grid_key_from_run(run)] = run
elif full_results.get("runs"):
    logger.warning("Ignoring full results snapshot with a different sweep hash in %s", results_path)
completed_full_snapshot_points = len(completed_runs_by_grid_point)

checkpoint_runs_by_grid_point = {}
for record in loaded_data["checkpoint_records"]:
    if record.get("sweep_hash") != sweep_hash:
        logger.warning("Ignoring checkpoint record with a different sweep hash in %s", checkpoint_path)
        continue

    if record.get("kind") != "run":
        continue

    key = (int(record["i"]), int(record["j"]))
    checkpoint_runs_by_grid_point[key] = record["run"]

completed_runs_by_grid_point.update(checkpoint_runs_by_grid_point)

n_stable_grid = {(0, 0): (0, 0.0)}  # Dictionary to store the stable configuration for each voltage point

for (i, j), run in sorted(completed_runs_by_grid_point.items()):
    results["runs"].append(run)
    n_stable_grid[(i, j)] = (int(run["n_electrons"]), float(run["energy"]))

results["charge_stability"] = n_stable_grid

logger.info("Starting charge stability sweep for a double quantum dot, error: %.2e eV or %.2e effective units", eV_error, e_conv)
logger.info("Incremental checkpoint: %s", checkpoint_path)
logger.info("Full results snapshot: %s (every %s completed points and at finish)", results_path, full_save_interval_points)
if completed_runs_by_grid_point:
    logger.info(
        "Loaded %s completed voltage points (%s from full snapshot, %s from checkpoint)",
        len(completed_runs_by_grid_point),
        completed_full_snapshot_points,
        len(checkpoint_runs_by_grid_point),
    )

# -----------------------------------------------------------------------------
# Main sweep loop
# -----------------------------------------------------------------------------

completed_grid_points = set(completed_runs_by_grid_point)
new_points_since_snapshot = 0

for i, v_x in enumerate(Vs):
    for j, v_y in enumerate(Vs):
        # Since the results will be simetrical, we can skip half of the points
        if v_x < v_y:
            continue

        grid_key = (i, j)
        if grid_key in completed_grid_points:
            continue

        start_time_per_point = time()

        run = {
            "vL": float(v_x),
            "vR": float(v_y),
            "points" : {}
        }

        logger.info("")
        logger.info("Voltage sweep")
        logger.info("voltage=%.3f V, %.3f V", v_x, v_y)

        if j > 0:
            n_electrons_stable, _ = n_stable_grid[(i, j - 1)]
        elif i > 0:
            n_electrons_stable, _ = n_stable_grid[(i - 1, j)]
        else:
            n_electrons_stable, _ = 0, 0

        def potential(x, y):
            return - v_x * eVtoE0 * np.exp(-((x + d/2)**2 + y**2) / (R**2)) - v_y * eVtoE0 * np.exp(-((x - d/2)**2 + y**2) / (R**2))

        candidate_ns = {
            n_electrons_stable - 1,
            n_electrons_stable,
            n_electrons_stable + 1,
        }
        candidate_ns.update(energy_plot_electron_counts)
        candidate_ns = sorted(
            n for n in candidate_ns
            if n == 0 or n in electrons_configurations
        )

        for n_electrons in candidate_ns:

            logger.info("  electrons=%s", n_electrons)
            
            if n_electrons == 0:
                result = {
                    "n_electrons": 0,
                    "n_alpha": 0,
                    "n_beta": 0,
                    "n_orbitals": 0,
                    "energy": 0.0,
                }

                run["points"][n_electrons] = result
                continue
            
            electrons, n_orbitals = electrons_configurations[n_electrons]

            result = {
                "n_electrons": int(n_electrons),
                "n_orbitals": int(n_orbitals),
            }

            if n_electrons % 2 == 1:
                n_alpha, n_beta = electrons
                result.update({
                    "n_alpha": int(n_alpha),
                    "n_beta": int(n_beta),
                })
                result_calculation = open_shell_calculation(potential, L, n_alpha, n_beta, n_orbitals, C_E0=C_E0, econv=e_conv, nuc_repulsion=0.0, max_iter=max_iter, logger=logger, get_potential=save_potential, opt_thresh=opt_thresh, max_iter_orbital_optimization=max_iter_orbital_optimization, thresh=thresh)

            else:
                result.update({
                    "n_alpha": int(n_electrons // 2),
                    "n_beta": int(n_electrons // 2),
                })
                result_calculation = closed_shell_calculation(potential, L, n_electrons, n_orbitals, C_E0=C_E0, econv=e_conv, max_iter=max_iter, logger=logger, get_potential=save_potential, opt_thresh=opt_thresh, max_iter_orbital_optimization=max_iter_orbital_optimization, thresh=thresh)

            result.update(result_calculation)
            run["points"][n_electrons] = result

            cleanup_temp_files()

            if not result["converged"]:
                logger.warning("    reached max_iter without convergence | energy=%+.10f", result["energy"])

            logger.info("  result | energy=%+.10f | iterations=%s | converged=%s", result["energy"], result["n_iterations"], result["converged"])

        best_result = min(
            run["points"].values(),
            key=lambda result: result["energy"]
        )

        energy_stable = best_result["energy"]
        n_electrons_stable = best_result["n_electrons"]

        run["energy"] = float(energy_stable)
        run["n_electrons"] = int(n_electrons_stable)

        results["runs"].append(run)
        n_stable_grid[(i, j)] = (n_electrons_stable, energy_stable)
        results["charge_stability"] = n_stable_grid
        completed_grid_points.add(grid_key)

        logger.info("  stable configuration | electrons=%s | energy=%+.10f", n_electrons_stable, energy_stable)

        data_helper.append_checkpoint_record(
            checkpoint_path,
            {
                "kind": "run",
                "sweep_hash": sweep_hash,
                "i": int(i),
                "j": int(j),
                "run": run,
            },
            fsync=checkpoint_fsync,
        )
        new_points_since_snapshot += 1
        logger.info("  checkpoint | appended completed point to %s", checkpoint_path)

        if new_points_since_snapshot >= full_save_interval_points:
            data_helper.save(results, fsync=checkpoint_fsync)
            new_points_since_snapshot = 0
            logger.info("  snapshot | refreshed %s", results_path)

        end_time_per_point = time()
        elapsed_time_per_point = end_time_per_point - start_time_per_point
        logger.info("  elapsed | %s", format_elapsed_time(elapsed_time_per_point))
        total_elapsed_time = end_time_per_point - start_time
        logger.info("  total elapsed | %s", format_elapsed_time(total_elapsed_time))

data_helper.save(results, fsync=checkpoint_fsync)
# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
end_time = time()
elapsed_time = end_time - start_time
logger.info("")
logger.info("Finished")
logger.info("  total time | %s", format_elapsed_time(elapsed_time))
logger.info("  data saved | %s", results_path)
logger.info("  log file   | %s", log_path)
