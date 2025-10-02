"""
sabre_3-fp_8spin_simulation_buildup_relaxation.py

This script contains a large-scale spin dynamics simulation of the SABRE
process when subjected to a static or rectangular polarisation transfer field.
The script calculates and stores spin polarisation levels of each spin while
applying fresh parahydrogen gas for 30 seconds and continues to record the
polarisation levels for another 60 seconds after switching off the parahydrogen
flow. This data is used to acquire T1 relaxation times and Tb build-up times.
The results are saved as .txt files.

The spin system consists of the following parts:
- SABRE complex: hydride protons and one equatorial 3-fluoropyridine (8 spins)
- Free ligand: 3-fluoropyridine in solution (6 spins)
- Parahydrogen: Fresh parahydrogen gas supplied to the system (2 spins)

Coherent spin dynamics along with non-linear kinetics and full Redfield-theory
relaxation with the molecular dynamics approximated by isotropic rotational
diffusion of a rigid molecule are implemented. All NMR interactions as well as
scalar relaxation of the second kind are included.
"""

# Imports
import os
import spinguin as sg
import numpy as np
from sabrefuncs import (
    sabre_buildup,
    sabre_relaxation,
    sabre_buildup_altfields,
    sabre_relaxation_altfields
)

# Change the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create spin systems
spin_system_c = sg.SpinSystem("spin_system/isotopes_complex.txt")
spin_system_s = sg.SpinSystem("spin_system/isotopes_substrate.txt")
spin_system_H2 = sg.SpinSystem("spin_system/isotopes_H2.txt")

# Set maximum spin orders
spin_system_c.basis.max_spin_order = 4
spin_system_s.basis.max_spin_order = 4
spin_system_H2.basis.max_spin_order = 2

# Build the basis sets
spin_system_c.basis.build()
spin_system_s.basis.build()
spin_system_H2.basis.build()

# Set relaxation theory
spin_system_c.relaxation.theory = "redfield"
spin_system_s.relaxation.theory = "redfield"

# Enable SR2K
spin_system_c.relaxation.sr2k = True
spin_system_s.relaxation.sr2k = True

# Set correlation times
spin_system_c.relaxation.tau_c = 50e-12
spin_system_s.relaxation.tau_c = 6e-12

# Set chemical shifts
spin_system_c.chemical_shifts = "spin_system/chemical_shifts_complex.txt"
spin_system_s.chemical_shifts = "spin_system/chemical_shifts_substrate.txt"

# Set J-couplings
spin_system_c.J_couplings = "spin_system/J_couplings_complex.txt"
spin_system_s.J_couplings = "spin_system/J_couplings_substrate.txt"

# Set coordinates
spin_system_c.xyz = "spin_system/xyz_complex.txt"
spin_system_s.xyz = "spin_system/xyz_substrate.txt"

# Set shielding tensors
spin_system_c.shielding = "spin_system/shielding_tensors_complex.txt"
spin_system_s.shielding = "spin_system/shielding_tensors_substrate.txt"

# Set EFG tensors
spin_system_c.efg = "spin_system/efg_tensors_complex.txt"
spin_system_s.efg = "spin_system/efg_tensors_substrate.txt"

# Define spin maps
spin_map_H2 = [0, 1]
spin_map_s = [2, 3, 4, 5, 6, 7]

# Define concentrations
c_c = 0.0005
c_s = 0.0135

# Define dissociation rates
kd_s = 20
kd_H2 = 2

# Set bubbling and relaxation time
bubbling_time = 30
relaxation_time = 60

# Perform the SABRE hyperpolarisation at 5 µT
time, mag_c, mag_s, rho_c, rho_s = sabre_buildup(
    spin_system_c = spin_system_c,
    spin_system_s = spin_system_s,
    spin_system_H2 = spin_system_H2,
    field = 5e-6,
    t = 4e-3,
    bubbling_time = bubbling_time,
    spin_map_s = spin_map_s,
    spin_map_H2 = spin_map_H2,
    c_c = c_c,
    c_s = c_s,
    kd_s = kd_s,
    kd_H2 = kd_H2
)

np.savetxt("buildup_5uT.txt", mag_s, delimiter=";")

# Perform the SABRE hyperpolarisation at square-modulated field
time, mag_c, mag_s, rho_c, rho_s = sabre_buildup_altfields(
    spin_system_c = spin_system_c,
    spin_system_s = spin_system_s,
    spin_system_H2 = spin_system_H2,
    field_low = -90e-6,
    field_high = 100e-6,
    t_low = 2e-3,
    t_high = 2e-3,
    bubbling_time = bubbling_time,
    spin_map_s = spin_map_s,
    spin_map_H2 = spin_map_H2,
    c_c = c_c,
    c_s = c_s,
    kd_s = kd_s,
    kd_H2 = kd_H2
)

np.savetxt("buildup_squ.txt", mag_s, delimiter=";")

# Continue relaxation from hyperpolarised state at square-modulated field
time, mag_c, mag_s = sabre_relaxation_altfields(
    spin_system_c = spin_system_c,
    spin_system_s = spin_system_s,
    spin_system_H2 = spin_system_H2,
    field_low = -90e-6,
    field_high = 100e-6,
    t_low = 2e-3,
    t_high = 2e-3,
    relaxation_time = relaxation_time,
    spin_map_s = spin_map_s,
    spin_map_H2 = spin_map_H2,
    c_c = c_c,
    c_s = c_s,
    kd_s = kd_s,
    rho_c = rho_c,
    rho_s = rho_s
)

np.savetxt("buildup_squ_relaxation_squ.txt", mag_s, delimiter=";")

# Continue relaxation from the hyperpolarised state at 5 µT
time, mag_c, mag_s = sabre_relaxation(
    spin_system_c = spin_system_c,
    spin_system_s = spin_system_s,
    spin_system_H2 = spin_system_H2,
    field = 5e-6,
    t = 4e-3,
    relaxation_time = relaxation_time,
    spin_map_s = spin_map_s,
    spin_map_H2 = spin_map_H2,
    c_c = c_c,
    c_s = c_s,
    kd_s = kd_s,
    rho_c = rho_c,
    rho_s = rho_s
)

np.savetxt("buildup_squ_relaxation_5uT.txt", mag_s, delimiter=";")

# Continue relaxation from the hyperpolarised state at 100 µT
time, mag_c, mag_s = sabre_relaxation(
    spin_system_c = spin_system_c,
    spin_system_s = spin_system_s,
    spin_system_H2 = spin_system_H2,
    field = 100e-6,
    t = 4e-3,
    relaxation_time = relaxation_time,
    spin_map_s = spin_map_s,
    spin_map_H2 = spin_map_H2,
    c_c = c_c,
    c_s = c_s,
    kd_s = kd_s,
    rho_c = rho_c,
    rho_s = rho_s
)

np.savetxt("buildup_squ_relaxation_100uT.txt", mag_s, delimiter=";")

# Perform the SABRE hyperpolarisation at 2 mT
time, mag_c, mag_s, rho_c, rho_s = sabre_buildup(
    spin_system_c = spin_system_c,
    spin_system_s = spin_system_s,
    spin_system_H2 = spin_system_H2,
    field = 2e-3,
    t = 4e-3,
    bubbling_time = bubbling_time,
    spin_map_s = spin_map_s,
    spin_map_H2 = spin_map_H2,
    c_c = c_c,
    c_s = c_s,
    kd_s = kd_s,
    kd_H2 = kd_H2
)

# Continue relaxation from hyperpolarised state at square-modulated field
time, mag_c, mag_s = sabre_relaxation_altfields(
    spin_system_c = spin_system_c,
    spin_system_s = spin_system_s,
    spin_system_H2 = spin_system_H2,
    field_low = -90e-6,
    field_high = 100e-6,
    t_low = 2e-3,
    t_high = 2e-3,
    relaxation_time = relaxation_time,
    spin_map_s = spin_map_s,
    spin_map_H2 = spin_map_H2,
    c_c = c_c,
    c_s = c_s,
    kd_s = kd_s,
    rho_c = rho_c,
    rho_s = rho_s
)

np.savetxt("buildup_2mT_relaxation_squ.txt", mag_s, delimiter=";")

# Continue relaxation from the hyperpolarised state at 5 µT
time, mag_c, mag_s = sabre_relaxation(
    spin_system_c = spin_system_c,
    spin_system_s = spin_system_s,
    spin_system_H2 = spin_system_H2,
    field = 5e-6,
    t = 4e-3,
    relaxation_time = relaxation_time,
    spin_map_s = spin_map_s,
    spin_map_H2 = spin_map_H2,
    c_c = c_c,
    c_s = c_s,
    kd_s = kd_s,
    rho_c = rho_c,
    rho_s = rho_s
)

np.savetxt("buildup_2mT_relaxation_5uT.txt", mag_s, delimiter=";")

# Continue relaxation from the hyperpolarised state at 100 µT
time, mag_c, mag_s = sabre_relaxation(
    spin_system_c = spin_system_c,
    spin_system_s = spin_system_s,
    spin_system_H2 = spin_system_H2,
    field = 100e-6,
    t = 4e-3,
    relaxation_time = relaxation_time,
    spin_map_s = spin_map_s,
    spin_map_H2 = spin_map_H2,
    c_c = c_c,
    c_s = c_s,
    kd_s = kd_s,
    rho_c = rho_c,
    rho_s = rho_s
)

np.savetxt("buildup_2mT_relaxation_100uT.txt", mag_s, delimiter=";")