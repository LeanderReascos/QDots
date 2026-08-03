from time import time

import numpy as np
import tequila as tq
print("Tequila version:", tq.__version__)
from pyscf import fci

import madpy as mad

# Save data
from pickle import dump
path = "data/"

from potentials import SquareGate, Gates


# Si Units
hbar = 1.054571817e-34  # J.s
m_e = 9.10938356e-31    # kg
m_si = 0.19 * m_e       # kg
e = 1.602176634e-19     # C
epsilon_0 = 8.854187817e-12  # F/m (vacuum permittivity)
epsilon_r = 11.7 * epsilon_0  # Relative permittivity for Si


# Mine units
C_E0 = 5.3
x0 = 4 * np.pi * epsilon_r * hbar**2 / ( e**2 * m_si ) 

print("C_E0 = ", C_E0)
print("x0 (m) = ", x0)
print("x0 (nm) = ", x0 * 1e9)

anisotropy = np.array([0.5, 0.8, 1.0])[::-1]

n_electrons = 2
n_orbitals = 4

Ground_states_energies = []

for alpha in anisotropy:
    Lx = 1
    Ly = Lx * alpha
    
    Ground_states_energies.append({
        "alpha" : alpha,
    })

    L_box = Lx * 10

    Ground_states_energies[-1]["lambda_W"] = C_E0
    Ground_states_energies[-1]["Lx"] = Lx
    Ground_states_energies[-1]["Ly"] = Ly
    print("lambda_W = ", Ground_states_energies[-1]["lambda_W"])

    def potential(x, y):
        return  -5 * np.exp(- 1/10 * (x**2/ Lx**2 + y**2 / Ly**2))

    world = mad.MadWorld2D(L= L_box) # This is required for any MADNESS calculation as it initializes the required environment

    factory = mad.MRAFunctionFactory2D(world, potential) # This transform a python function into a MRA function which can be read by MADNESS
    mra_pot = factory.get_function() # Potential as MRA function

    world.line_plot(path + "potential.dat", mra_pot, axis="x", datapoints=2001)  # This plots the potential along the x-axis

    eigen = mad.Eigensolver2D(world, mra_pot)
    orbitals = eigen.get_orbitals(0, n_orbitals, 0, n_states=n_orbitals*2)
    del eigen        

    econv = 1.e-8 # Energy convergence threshold
    max_iter = 100

    current = 0.0
    energy = 0.0

    for iteration in range(max_iter):
        integrals = mad.Integrals2D(world)
        orbitals = integrals.orthonormalize(orbitals=orbitals)
        V = integrals.compute_potential_integrals(orbitals, V=mra_pot)
        T = integrals.compute_kinetic_integrals(orbitals)
        G = integrals.compute_two_body_integrals(orbitals, ordering="chem")

        del integrals


        # Try to implement the same with tequila
        geom = "H 0.0 0.0 0.0\nH 0.0 0.0 3.5"  # geometry in Angstrom
        mol = tq.Molecule(
            geom, one_body_integrals= T + V, two_body_integrals=G.elems * C_E0, nuclear_repulsion=0
        )

        energy_tq, wfn_tq = mol.compute_energy("fci", get_wfn=True)

        solver = fci.direct_spin0.FCI()
        energy, fcivec = solver.kernel(T + V, G.elems * C_E0, n_orbitals, n_electrons)
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

        del opti

    for i in range(len(orbitals)):
        world.plane_plot(f"alpha{alpha}_orb_{i}.dat", orbitals[i], datapoints=500, zoom=1) # Plots the optimized orbitals


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

