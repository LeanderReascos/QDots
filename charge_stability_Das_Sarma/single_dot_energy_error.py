'''
This example computes the charge stability diagram of a double quantum dot.

Ref: https://journals.aps.org/prb/pdf/10.1103/PhysRevB.110.205428
'''

from pathlib import Path
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pickle import dump
from time import time
import logging
import os
import faulthandler

import numpy as np

from pyscf import fci
import frayedends as fe

from potentials import Gates, SquareGate
from solvers import closed_shell_calculation, open_shell_calculation

# -----------------------------------------------------------------------------
# Paths, logging, and helpers
# -----------------------------------------------------------------------------
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
results_path = data_dir / "single_dot_error.pickle"
log_path = data_dir / "single_dot_error.log"

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

# -----------------------------------------------------------------------------
# Paths and physical constants
# -----------------------------------------------------------------------------
start_time = time()

# SI units
hbar = 1.054571817e-34  # J.s
m_e = 9.10938356e-31    # kg
m_si = 0.19 * m_e       # kg
e = 1.602176634e-19     # C
epsilon_0 = 8.854187817e-12  # F/m (vacuum permittivity)
epsilon_r = 11.7  # Relative permittivity for Si

l0 = 50e-9  # 50 nm in meters
E0 = hbar**2 / (m_si * l0**2) 

eVtoE0 = e * 1 / epsilon_r / E0  # 1 eV in effective units
C_E0 = e**2 / (4 * np.pi * epsilon_0 * epsilon_r * l0) * 1 / E0  # Coulomb energy in effective units

logger.info("1 eV in effective units: %.6f", eVtoE0)
logger.info("Coulomb energy in effective units: %.6f", C_E0)

# -----------------------------------------------------------------------------
# Device geometry and sweep configuration
# -----------------------------------------------------------------------------
L = 50 / (l0 * 1e9)
Lx_x, Ly_x = 30 / (l0 * 1e9), 50 / (l0 * 1e9)     # Barrier
z_pot = 30 / (l0 * 1e9)
d = 80 / (l0 * 1e9)  # Dot separation

JtoeV = e * 1 # J to eV conversion factor

N_electrons = [2, (2, 1)]
Vs = [0.4, 0.6]
N_orbitals_max = 36 

# -----------------------------------------------------------------------------
# Optimization parameters
# -----------------------------------------------------------------------------

eV_error = 1e-7

thresh = min(eV_error * JtoeV / E0 / 10, 1e-5)  # Set MADNESS threshold to be an order of magnitude smaller than the desired energy error
e_conv = eV_error * JtoeV / E0  # Convert eV error to effective units

opt_thresh = e_conv / 10  # Set orbital optimization threshold to be an order of magnitude smaller than the desired energy error

max_iter_orbital_optimization = 5
max_iter = 100
max_time_per_iteration = 10 * 60  # 10 minutes in seconds

# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------
results = {
    "metadata": {
        "description": "Closed-shell charge stability sweep for a double quantum dot.",
        "length_scale_nm": 50.0,
        "effective_units_reference_energy": float(E0),
        "coulomb_energy_effective_units": float(C_E0),
        "eV_error_tolerance": float(eV_error),
        "e_conv_tolerance_effective_units": float(e_conv),
        "log_file": str(log_path),
    },
    "runs": [],
}

logger.info("Starting charge stability sweep for a single quantum dot, error: %.2e eV or %.2e effective units", eV_error, e_conv)


# -----------------------------------------------------------------------------
# Main sweep loop
# -----------------------------------------------------------------------------
for electrons, v in zip(N_electrons, Vs):
    n_electrons = electrons if isinstance(electrons, int) else sum(electrons)
    run = {
        "n_electrons": int(n_electrons),
        "V": float(v),
        "points": [],
    }
    results["runs"].append(run)

    start_time_per_n = time()

    # -------------------------------------------------------------------------
    # Orbitals loop
    # -------------------------------------------------------------------------
    last_energy = None
    for n_orbitals in np.arange(n_electrons, N_orbitals_max + 1, 2):
        time_per_iteration = time()
        logger.info("")
        logger.info("Voltage sweep")
        logger.info("  electrons=%s | orbitals=%s | voltage=%.3f V, %.3f V", n_electrons, n_orbitals, v, v)

        # Gate voltages
        VL = SquareGate(x0=-L/2 - Lx_x/2, y0=0, z0=z_pot, Lx=L, Ly=L)
        VR = SquareGate(x0=L/2 + Lx_x/2, y0=0, z0=z_pot, Lx=L, Ly=L)
        VX = SquareGate(x0=0, y0=0, z0=z_pot, Lx=Lx_x, Ly=Ly_x)
        gates = Gates([VL, VX, VR])
        gates.add_voltages([v, -0.1, v])

        def potential(x, y):
                return gates.Vi(x, y) * eVtoE0 * np.exp(-max(0, (x-VL.x0-d/4))**2/(d/2)**2) # in effective units
        
        if n_electrons % 2 == 1:
            n_alpha, n_beta = electrons
            result = open_shell_calculation(potential, L, n_alpha, n_beta, n_orbitals, C_E0=C_E0, econv=e_conv, nuc_repulsion=0.0, max_iter=max_iter, logger=logger, get_potential=False,
                                                opt_thresh=opt_thresh, max_iter_orbital_optimization=max_iter_orbital_optimization, thresh=thresh)
        
        else:
            result = closed_shell_calculation(potential, L, n_electrons, n_orbitals, C_E0=C_E0, econv=e_conv, max_iter=max_iter, logger=logger, get_potential=False, opt_thresh=opt_thresh, max_iter_orbital_optimization=max_iter_orbital_optimization, thresh=thresh)

        # Clean temp logs files:
        for temp_file in Path('.').glob('mad*.log'):
            temp_file.unlink()
       
        if last_energy is None:
            last_energy = result["energy"]
        # Persist point result
        point = {
            "v": float(v),
            "v": float(v),
            "n_orbitals": int(n_orbitals),
        }
        point.update(result)
        run["points"].append(point)

        if not point["converged"]:
            logger.warning("    reached max_iter without convergence | energy=%+.10f", point["energy"])

        logger.info("  result | energy=%+.10f | iterations=%s | converged=%s", point["energy"], point["n_iterations"], point["converged"])


        with open(results_path, "wb") as f:
            dump(results, f)

        end_time_per_iteration = time()
        elapsed_time_per_iteration = end_time_per_iteration - time_per_iteration
        logger.info("  elapsed | %s", format_elapsed_time(elapsed_time_per_iteration))
        end_time_per_n = time()
        elapsed_time = end_time_per_n - start_time_per_n
        logger.info("  elapsed | %s", format_elapsed_time(elapsed_time))

        delta_energy = abs(point["energy"] - last_energy)
        last_energy = point["energy"]

        if elapsed_time_per_iteration >= max_time_per_iteration and delta_energy < e_conv:
            logger.warning("  stopping sweep due to time limit | elapsed=%s | max=%s", format_elapsed_time(elapsed_time_per_iteration), format_elapsed_time(max_time_per_iteration))
            break

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
