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


# -----------------------------------------------------------------------------
# Paths, logging, and helpers
# -----------------------------------------------------------------------------
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
results_path = data_dir / "open_shell_results.pickle"
log_path = data_dir / "open_shell_calculations.log"

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


@contextmanager
def suppress_external_output():
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        yield


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


N_electrons = [2, 4, 6]
N_orbitals = [4, 8, 12]
Vs = np.linspace(0, 1, 100) # V

Vx, Vy = np.meshgrid(Vs, Vs)


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------
results = {
    "metadata": {
        "description": "Open-shell charge stability sweep for a double quantum dot.",
        "length_scale_nm": 50.0,
        "effective_units_reference_energy": float(E0),
        "coulomb_energy_effective_units": float(C_E0),
        "log_file": str(log_path),
    },
    "runs": [],
}


# -----------------------------------------------------------------------------
# Main sweep loop
# -----------------------------------------------------------------------------
for n_electrons, n_orbitals, voltages_x, voltages_y in zip(N_electrons, N_orbitals, Vx, Vy):
    run = {
        "n_electrons": int(n_electrons),
        "n_orbitals": int(n_orbitals),
        "points": [],
    }
    results["runs"].append(run)

    start_time_per_n = time()

    # -------------------------------------------------------------------------
    # Voltage loop
    # -------------------------------------------------------------------------
    for v_x, v_y in zip(voltages_x, voltages_y):

        logger.info("")
        logger.info("Voltage sweep")
        logger.info("  electrons=%s | orbitals=%s | voltage=%.3f V, %.3f V", n_electrons, n_orbitals, v_x, v_y)

        # Gate voltages
        VL = SquareGate(x0=-L/2 - Lx_x/2, y0=0, z0=z_pot, Lx=L, Ly=L)
        VR = SquareGate(x0=L/2 + Lx_x/2, y0=0, z0=z_pot, Lx=L, Ly=L)
        VX = SquareGate(x0=0, y0=0, z0=z_pot, Lx=Lx_x, Ly=Ly_x)
        gates = Gates([VL, VX, VR])
        gates.add_voltages([v_x, -0.1, v_y])

        def potential(x, y):
            return gates.Vi(x, y) * eVtoE0  # in effective units

        logger.info("  > Initialising MADNESS world")
        logger.info("  > Building MRA potential")
        with suppress_external_output():
            world = fe.MadWorld(ndims=2, L=L*10) # This is required for any MADNESS calculation as it initializes the required environment

            factory = fe.MRAFunctionFactory(world, potential) # This transform a python function into a MRA function which can be read by MADNESS
            mra_pot = factory.get_function() # Potential as MRA function

        # world.line_plot(str(data_dir / f"potential_v{v:.2f}.dat"), mra_pot, axis="x", datapoints=2001)  # This plots the potential along the x-axis

        # Single-particle basis
        logger.info("  > Solving single-particle orbitals")
        eigen = fe.Eigensolver(world, mra_pot)
        orbitals = eigen.get_orbitals(n_orbitals=n_orbitals, n_guess_orbs=n_orbitals*2+1)

        integrals = fe.Integrals(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)

        nuc_repulsion = 0

        orbitals_ab = [orbitals, orbitals]

        integralsOS = fe.Integrals_open_shell(world)
 
        # Self-consistent loop
        econv = 1.e-8 # Energy convergence threshold
        max_iter = 100

        current = 0.0
        energy = 0.0
        converged = False
        n_iterations = 0

        logger.info("  > Starting SCF loop (max_iter=%s, econv=%g)", max_iter, econv)

        for iteration in range(max_iter):

            # Get initial effective Hamiltonian
            c, h1, g2 = integralsOS.compute_effective_hamiltonian(
                [], [], orbitals_ab[0], orbitals_ab[1], mra_pot, nuc_repulsion
            )
            g2[0] = g2[0].transpose(0, 2, 1, 3)  # transform g tensors to chem ordering
            g2[1] = g2[1].transpose(0, 2, 1, 3)
            g2[2] = g2[2].transpose(0, 2, 1, 3)

            # FCI calculation on active space
            e, fcivec = fci.direct_uhf.kernel(h1, g2, n_orbitals, (n_alpha, n_beta))
            rdm1, rdm2 = fci.direct_uhf.make_rdm12s(fcivec, n_orbitals, (n_alpha, n_beta))
            rdm2 = np.swapaxes(rdm2, 1, 2)
            rdm_2_phys_aa = rdm2[0].transpose(0, 2, 1, 3)  # again reordering to fit our convention
            rdm_2_phys_ab = rdm2[1].transpose(0, 2, 1, 3)
            rdm_2_phys_bb = rdm2[2].transpose(0, 2, 1, 3)

            # Orbital optimization
            opti = fe.OrbitalRefinement(world, mra_pot, nuc_repulsion=0.0)
            orbitals = opti.get_orbitals(
                orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=10*econv, occ_thresh=econv
            ) # Optimizes the orbitals and returns the new ones

            n_iterations = iteration + 1
            
            if np.isclose(energy, current, atol=econv, rtol=0.0):
                converged = True
                logger.info("    converged after %s iterations | energy=%+.10f", n_iterations, energy)
                break
            
            current = energy

        with suppress_external_output():
            del factory
            del integrals
            del opti
            del eigen
            del world
                
        # Persist point result
        point = {
            "vx": float(v_x),
            "vy": float(v_y),
            "energy": float(energy),
            "converged": converged,
            "iterations": int(n_iterations),
        }
        run["points"].append(point)

        if not converged:
            logger.warning("    reached max_iter without convergence | energy=%+.10f", energy)

        logger.info("  result | energy=%+.10f | iterations=%s | converged=%s", energy, n_iterations, converged)


        with open(results_path, "wb") as f:
            dump(results, f)

        end_time_per_n = time()
        elapsed_time = end_time_per_n - start_time_per_n
        logger.info("  elapsed | %s", format_elapsed_time(elapsed_time))


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
    