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

N_electrons = [(1, 0), 2, (2, 1), 4, (3, 2), 6]
M_orbitals = [2, 6, 8, 10, 12, 14]

Vs = np.linspace(0, 1, 100) # V

# -----------------------------------------------------------------------------
# Optimization parameters
# -----------------------------------------------------------------------------

eV_error = 1e-6

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