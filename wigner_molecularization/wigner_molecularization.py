from time import time

import numpy as np
import tequila as tq
print("Tequila version:", tq.__version__)
from pyscf import fci

import frayedends as mad

# Save data
from pickle import dump
path = "data/"

from potentials import SquareGate, Gates


anisotropy = np.array([1.0, 1.5, 2.0, 3, 5.3, 6])

Ly = 50
Lz = 30

n_electrons = 2
n_orbitals = 2

mass_si = 0.19
epsilon_r_si = 11.7

Ground_states_energies = []

for alpha in anisotropy:
    Lx = Ly * alpha
    
    gate = SquareGate(x0=0, y0=0, z0=Lz, Lx=Lx, Ly=Ly)
    gates = Gates([gate])
    gates.add_voltages([0.6])
    gates.to_au()

    Ground_states_energies.append({
        "alpha" : alpha,
    })

    def potential(x, y):
        return -0.5 + gates.Vi(x, y)

    world = mad.MadWorld2D(L=5000, thresh=1e-6) # This is required for any MADNESS calculation as it initializes the required environment

    factory = mad.MRAFunctionFactory2D(world, potential) # This transform a python function into a MRA function which can be read by MADNESS
    mra_pot = factory.get_function() # Potential as MRA function

    world.line_plot(path + "potential.dat", mra_pot, axis="x", datapoints=2001)  # This plots the potential along the x-axis

    eigen = mad.Eigensolver2D(world, mra_pot)
    orbitals = eigen.get_orbitals(n_orbitals, n_guess_orbs=n_orbitals*2)
    del eigen        

    econv = 1.e-8 # Energy convergence threshold
    max_iter = 15

    current = 0.0
    energy = 0.0

    for iteration in range(max_iter):
        integrals = mad.Integrals2D(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)
        V = integrals.compute_potential_integrals(orbitals, V=mra_pot) / epsilon_r_si
        T = integrals.compute_kinetic_integrals(orbitals) / mass_si
        G = integrals.compute_two_body_integrals(orbitals, ordering="chem")

        del integrals


        # Try to implement the same with tequila
        geom = "H 0.0 0.0 0.0\nH 0.0 0.0 3.5"  # geometry in Angstrom
        mol = tq.Molecule(
            geom, one_body_integrals= T + V, two_body_integrals=G.elems / epsilon_r_si, nuclear_repulsion=0
        )

        energy_tq, wfn_tq = mol.compute_energy("fci", get_wfn=True)

        solver = fci.direct_spin0.FCI()
        energy, fcivec = solver.kernel(T + V, G.elems / epsilon_r_si, n_orbitals, n_electrons)
        rdm1, rdm2 = solver.make_rdm12(fcivec, n_orbitals, n_electrons)
        rdm2 = np.swapaxes(rdm2, 1, 2)


        print(f"Iteration {iteration} Energy {energy:.8f}")
        if np.isclose(energy, current, atol=econv, rtol=0.0):
            break
        
        current = energy

        # Orbital optimization
        opti = mad.Optimization2D(world, mra_pot, 0.0)
        orbitals = opti.get_orbitals(
            orbitals=orbitals, rdm1=rdm1, rdm2=rdm2, opt_thresh=10*econv, occ_thresh=econv
        ) # Optimizes the orbitals and returns the new ones

        for i in range(len(orbitals)):
            world.plane_plot(f"alpha{alpha}_orb_{i}.dat", orbitals[i]) # Plots the optimized orbitals

        del opti

    del factory
    del world

    Ground_states_energies[-1]["energies"] = energy
    Ground_states_energies[-1]["wfn"] = wfn_tq

Ground_states_energies = np.array(Ground_states_energies)
print(Ground_states_energies)


# Move the orbitals file to path
import os
for i in range(n_orbitals):
    for alpha in anisotropy:
        os.rename(f"plane_x1x2_alpha{alpha}_orb_{i}.dat", path + f"alpha{alpha}_orb_{i}.dat")
    

with open(path + "wigner_molecularization.pkl", "wb") as f:
    dump(Ground_states_energies, f)

print("Data saved in ", path + "wigner_molecularization.pkl")

