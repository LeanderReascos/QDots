from time import perf_counter

from pyscf import fci
import frayedends as fe
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import numpy as np
from profiler import Profiler


@contextmanager
def suppress_external_output():
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        yield

def prepare_kwargs(kwargs):
    kwargs.setdefault("C_E0", 1)
    kwargs.setdefault("econv", 1e-3)
    kwargs.setdefault("max_iter", 100)
    kwargs.setdefault("logger", None)
    kwargs.setdefault("opt_thresh", 1e-4)
    kwargs.setdefault("occ_thresh", 1e-5)
    kwargs.setdefault("max_iter_orbital_optimization", 3)
    kwargs.setdefault("thresh", 1e-5)
    kwargs.setdefault("get_potential", False)
    kwargs.setdefault("nuc_repulsion", 0.0)

    return kwargs


def closed_shell_calculation(potential, L, n_electrons, n_orbitals, **kwargs):

    kwargs = prepare_kwargs(kwargs)

    profiler = Profiler()

    result = {}

    if kwargs["logger"]:
        kwargs["logger"].info("  > Initialising MADNESS world")
        kwargs["logger"].info("  > Building MRA potential")
    with suppress_external_output():

        with profiler.stage("MAD world initialization"):
            world = fe.MadWorld(ndims=2, L=L*10, thresh=kwargs["thresh"]) # This is required for any MADNESS calculation as it initializes the required environment

        with profiler.stage("MRA potential creation"):
            factory = fe.MRAFunctionFactory(world, potential) # This transform a python function into a MRA function which can be read by MADNESS
            mra_pot = factory.get_function() # Potential as MRA function

    if kwargs["get_potential"]:
        world.line_plot(str("potential.dat"), mra_pot, axis="x", datapoints=2001)  # This plots the potential along the x-axis

        # Open the file with numpy and read the data
        data = np.loadtxt("potential.dat")
        result["potential"] = data.tolist()  # Convert to list for JSON serialization

        #delete the file after reading
        os.remove("potential.dat")
        
    # Single-particle basis
    if kwargs["logger"]:
        kwargs["logger"].info("  > Solving single-particle orbitals")

    with profiler.stage("Single-particle orbitals calculation"):
        eigen = fe.Eigensolver(world, mra_pot)
        orbitals = eigen.get_orbitals(n_orbitals=n_orbitals)

    # Self-consistent loop
    current = 0.0
    energy = 0.0
    converged = False
    n_iterations = 0
    story = []

    if kwargs["logger"]:
        kwargs["logger"].info("  > Starting SCF loop (max_iter=%s, econv=%g)", kwargs["max_iter"], kwargs["econv"])


    start_time = perf_counter()
    for iteration in range(kwargs["max_iter"]):
        with profiler.stage(f"Integrals computation for iteration {iteration}"):
            integrals = fe.Integrals(world)
            orbitals = integrals.orthonormalize(orbitals=orbitals)
            V = integrals.compute_potential_integrals(orbitals, V=mra_pot) 
            T = integrals.compute_kinetic_integrals(orbitals)
            G = integrals.compute_two_body_integrals(orbitals, ordering="chem")

        with profiler.stage(f"FCI calculation for iteration {iteration}"):
            energy, fcivec = fci.direct_spin0.kernel(T + V, G.elems * kwargs["C_E0"], n_orbitals, n_electrons)  # The two-body integrals are divided by the dielectric constant
            rdm1, rdm2 = fci.direct_spin0.make_rdm12(fcivec, n_orbitals, n_electrons)
            rdm2 = np.swapaxes(rdm2, 1, 2)

        with profiler.stage(f"Orbital Refinement for iteration {iteration}"):
            # Orbital optimization
            opti = fe.OrbitalRefinement(world, mra_pot, nuc_repulsion=kwargs["nuc_repulsion"])
            # opti.set_orthonormalization_method("mixed", degeneracy_tol=1e-3)
            orbitals = opti.get_orbitals(
                orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=kwargs["opt_thresh"], occ_thresh=kwargs["occ_thresh"], maxiter=kwargs["max_iter_orbital_optimization"]
            ) # Optimizes the orbitals and returns the new ones

        n_iterations = iteration + 1
        
        story.append(energy)
        if np.isclose(energy, current, atol=kwargs["econv"]):
            converged = True
            if kwargs["logger"]:
                kwargs["logger"].info("    converged after %s iterations | energy=%+.10f", n_iterations, energy)
            break
        
        current = energy
    end_time = perf_counter()
    log_time = end_time - start_time

    with suppress_external_output():
        del factory
        del integrals
        del opti
        del eigen
        del world

    result.update({
        "energy": float(energy),
        "n_iterations": int(n_iterations),
        "converged": converged,
        "story": story,
        "profiler": profiler.data,
        "log_time": log_time
    })

    return result

def open_shell_calculation(potential, L, n_alpha, n_beta, n_orbitals, **kwargs):

    kwargs = prepare_kwargs(kwargs)

    profiler = Profiler()

    result = {}

    if kwargs["logger"]:
        kwargs["logger"].info("  > Initialising MADNESS world")
        kwargs["logger"].info("  > Building MRA potential")
    with suppress_external_output():
        with profiler.stage("MAD world initialization"):
            world = fe.MadWorld(ndims=2, L=L*10, thresh=kwargs["thresh"]) # This is required for any MADNESS calculation as it initializes the required environment

        with profiler.stage("MRA function factory initialization"):
            factory = fe.MRAFunctionFactory(world, potential) # This transform a python function into a MRA function which can be read by MADNESS
            mra_pot = factory.get_function() # Potential as MRA function

    if kwargs["get_potential"]:
        world.line_plot(str("potential.dat"), mra_pot, axis="x", datapoints=2001)  # This plots the potential along the x-axis

        # Open the file with numpy and read the data
        data = np.loadtxt("potential.dat")
        result["potential"] = data.tolist()  # Convert to list for JSON serialization

        #delete the file after reading
        os.remove("potential.dat")


    # Single-particle basis
    with profiler.stage("Single-particle orbitals calculation"):
        if kwargs["logger"]:
            kwargs["logger"].info("  > Solving single-particle orbitals")
        eigen = fe.Eigensolver(world, mra_pot)
        orbitals = eigen.get_orbitals(n_orbitals=n_orbitals)

    with profiler.stage("Integral computation"):
        integrals = fe.Integrals(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)

    with profiler.stage("Two-body integrals computation"):
        orbitals_ab = [orbitals, orbitals]
        integralsOS = fe.Integrals_open_shell(world)

    current = 0.0
    converged = False
    n_iterations = 0
    story = []
    converged_orbitals = []
    if kwargs["logger"]:
        kwargs["logger"].info("  > Starting SCF loop (max_iter=%s, econv=%g)", kwargs["max_iter"], kwargs["econv"])

    start_time = perf_counter()
    
    for iteration in range(kwargs["max_iter"]):
        with profiler.stage(f"Integrals {iteration}"):
            # Get initial effective Hamiltonian
            c, h1, g2 = integralsOS.compute_effective_hamiltonian(
                [], [], orbitals_ab[0], orbitals_ab[1], mra_pot, kwargs["nuc_repulsion"]
            )
        with profiler.stage(f"Two-body integrals transformation {iteration}"):
            g2[0] = g2[0].transpose(0, 2, 1, 3) * kwargs["C_E0"] # transform g tensors to chem ordering
            g2[1] = g2[1].transpose(0, 2, 1, 3) * kwargs["C_E0"]
            g2[2] = g2[2].transpose(0, 2, 1, 3) * kwargs["C_E0"]


        # FCI calculation on active space
        with profiler.stage(f"FCI {iteration}"):
            e, fcivec = fci.direct_uhf.kernel(h1, g2, n_orbitals, (n_alpha, n_beta))
            rdm1, rdm2 = fci.direct_uhf.make_rdm12s(fcivec, n_orbitals, (n_alpha, n_beta))
        with profiler.stage(f"RDM2 transformation {iteration}"):
            rdm2 = np.swapaxes(rdm2, 1, 2)
            rdm_2_phys_aa = rdm2[0].transpose(0, 2, 1, 3)  # again reordering to fit our convention
            rdm_2_phys_ab = rdm2[1].transpose(0, 2, 1, 3)
            rdm_2_phys_bb = rdm2[2].transpose(0, 2, 1, 3)

        with profiler.stage(f"Orbital Refinement {iteration}"):
            # Orbital refinement with core orbital refinement enabled
            opti = fe.OrbitalRefinement_open_shell(world, mra_pot, kwargs["nuc_repulsion"])
            core, orbitals_ab, converged_orbital = opti.refine_orbitals(
                orbitals=[[], [], orbitals_ab[0], orbitals_ab[1]],
                rdm1=rdm1,
                rdm2=[rdm_2_phys_aa, rdm_2_phys_ab, rdm_2_phys_bb],
                opt_thresh=kwargs["opt_thresh"], occ_thresh=kwargs["occ_thresh"],
                maxiter=kwargs["max_iter_orbital_optimization"],
                redirect_filename=f"madopt{iteration}.log",
            )
            converged_orbitals.append(converged_orbital)
        n_iterations = iteration + 1

        energy = e + c
        story.append(energy)
        
        if np.isclose(energy, current, atol=kwargs["econv"]):
            converged = True
            if kwargs["logger"]:
                kwargs["logger"].info("    converged after %s iterations | energy=%+.10f", n_iterations, energy)
            break
        
        current = energy

    end_time = perf_counter()

    with suppress_external_output():
        del factory
        del integrals
        del opti
        del eigen
        del world

    result.update({
        "energy": float(energy),
        "n_iterations": int(n_iterations),
        "converged": converged,
        "converged_orbital": converged_orbital,
        "story": story,
        "profiler": profiler.data,
        "log_time": end_time - start_time
    })

    return result