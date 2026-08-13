'''
This example computes the charge stability diagram of a double quantum dot.

Ref: https://journals.aps.org/prb/pdf/10.1103/PhysRevB.110.205428
'''

from pathlib import Path
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pickle import HIGHEST_PROTOCOL, dump, load
from time import time
import hashlib
import json
import logging
import os
import faulthandler

import numpy as np

from pyscf import fci
import frayedends as fe

from solvers import closed_shell_calculation, open_shell_calculation

# -----------------------------------------------------------------------------
# Paths, logging, and helpers
# -----------------------------------------------------------------------------
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
results_path = data_dir / "charge_stability.pkl"
log_path = data_dir / "charge_stability.log"

log_level_name = os.getenv("QD_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(logging.Formatter("%(message)s"))

file_handler = logging.FileHandler(log_path, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
faulthandler.enable()


def format_elapsed_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


def read_positive_int_from_env(name, default):
    try:
        value = int(os.getenv(name, default))
    except ValueError:
        logger.warning("%s must be an integer; using %s", name, default)
        return default

    if value < 1:
        logger.warning("%s must be positive; using %s", name, default)
        return default

    return value


def should_fsync_checkpoints():
    return os.getenv("QD_CHECKPOINT_FSYNC", "1").lower() not in {"0", "false", "no"}


def build_sweep_hash(sweep_config):
    payload = json.dumps(sweep_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def append_pickle_record(path, record, fsync=True):
    with path.open("ab") as f:
        dump(record, f, protocol=HIGHEST_PROTOCOL)
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def save_results_snapshot(results, path, fsync=True):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        dump(results, f, protocol=HIGHEST_PROTOCOL)
        f.flush()
        if fsync:
            os.fsync(f.fileno())

    os.replace(tmp_path, path)


def load_checkpoint_runs(path, sweep_hash):
    runs_by_grid_point = {}

    if not path.exists():
        return runs_by_grid_point

    with path.open("rb") as f:
        while True:
            try:
                record = load(f)
            except EOFError:
                break
            except Exception as exc:
                logger.warning("Stopped reading checkpoint %s after a partial/corrupt record: %s", path, exc)
                break

            if record.get("sweep_hash") != sweep_hash:
                logger.warning("Ignoring checkpoint record with a different sweep hash in %s", path)
                continue

            if record.get("kind") != "run":
                continue

            key = (int(record["i"]), int(record["j"]))
            runs_by_grid_point[key] = record["run"]

    return runs_by_grid_point

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

eVtoE0 = e * 1 / epsilon_r / E0  # 1 eV in effective units
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

Vs = np.linspace(21, 29, 100) * 1e-3 # V

# -----------------------------------------------------------------------------
# Optimization parameters
# -----------------------------------------------------------------------------

eV_error = 1e-6

thresh = min(eV_error * JtoeV / E0 / 10, 1e-5)  # Set MADNESS threshold to be an order of magnitude smaller than the desired energy error
e_conv = eV_error * JtoeV / E0  # Convert eV error to effective units

opt_thresh = e_conv / 10  # Set orbital optimization threshold to be an order of magnitude smaller than the desired energy error

max_iter_orbital_optimization = 5
max_iter = 100

# -----------------------------------------------------------------------------
# Checkpoint and stored-output configuration
# -----------------------------------------------------------------------------
energy_plot_electron_counts = tuple(n for n in (1, 2) if n in electrons_configurations)
full_save_interval_points = read_positive_int_from_env("QD_FULL_SAVE_INTERVAL", 10)
checkpoint_fsync = should_fsync_checkpoints()

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
sweep_hash = build_sweep_hash(sweep_config)
checkpoint_path = data_dir / f"charge_stability_runs_{sweep_hash}.pkl"

if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
    append_pickle_record(
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

completed_runs_by_grid_point = load_checkpoint_runs(checkpoint_path, sweep_hash)
n_stable_grid = {(0, 0): (0, 0.0)}  # Dictionary to store the stable configuration for each voltage point

for (i, j), run in sorted(completed_runs_by_grid_point.items()):
    results["runs"].append(run)
    n_stable_grid[(i, j)] = (int(run["n_electrons"]), float(run["energy"]))

results["charge_stability"] = n_stable_grid

logger.info("Starting charge stability sweep for a double quantum dot, error: %.2e eV or %.2e effective units", eV_error, e_conv)
logger.info("Incremental checkpoint: %s", checkpoint_path)
logger.info("Full results snapshot: %s (every %s completed points and at finish)", results_path, full_save_interval_points)
if completed_runs_by_grid_point:
    logger.info("Loaded %s completed voltage points from checkpoint", len(completed_runs_by_grid_point))

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
            return - v_x * np.exp(-((x + d/2)**2 + y**2) / (R**2)) - v_y * np.exp(-((x - d/2)**2 + y**2) / (R**2))

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
                result_calculation = open_shell_calculation(potential, L, n_alpha, n_beta, n_orbitals, C_E0=C_E0, econv=e_conv, nuc_repulsion=0.0, max_iter=max_iter, logger=logger, get_potential=False,
                                                    opt_thresh=opt_thresh, max_iter_orbital_optimization=max_iter_orbital_optimization, thresh=thresh)

            else:
                result.update({
                    "n_alpha": int(n_electrons // 2),
                    "n_beta": int(n_electrons // 2),
                })
                result_calculation = closed_shell_calculation(potential, L, n_electrons, n_orbitals, C_E0=C_E0, econv=e_conv, max_iter=max_iter, logger=logger, get_potential=False, opt_thresh=opt_thresh, max_iter_orbital_optimization=max_iter_orbital_optimization, thresh=thresh)

            result.update(result_calculation)
            run["points"][n_electrons] = result

            # Clean temp logs files:
            for temp_file in Path('.').glob('mad*.log'):
                temp_file.unlink()

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

        append_pickle_record(
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
            save_results_snapshot(results, results_path, fsync=checkpoint_fsync)
            new_points_since_snapshot = 0
            logger.info("  snapshot | refreshed %s", results_path)

        end_time_per_point = time()
        elapsed_time_per_point = end_time_per_point - start_time_per_point
        logger.info("  elapsed | %s", format_elapsed_time(elapsed_time_per_point))
        total_elapsed_time = end_time_per_point - start_time
        logger.info("  total elapsed | %s", format_elapsed_time(total_elapsed_time))

save_results_snapshot(results, results_path, fsync=checkpoint_fsync)


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
