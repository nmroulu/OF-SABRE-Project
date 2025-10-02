"""
sabre_3-fp_8spin_simulation_fielddep.py

This script contains a large-scale spin dynamics simulation of the SABRE
process when subjected to a rectangular polarisation transfer field. The
script calculates spin polarisation levels of each spin after 30 seconds of
SABRE hyperpolarisation using a variety of field parameters. The results are
saved as .txt files.

The spin system consists of the following parts:
- SABRE complex: hydride protons and one equatorial 3-fluoropyridine (8 spins)
- Free ligand: 3-fluoropyridine in solution (6 spins)
- Parahydrogen: Fresh parahydrogen gas supplied to the system (2 spins)

Coherent spin dynamics along with non-linear kinetics and full Redfield-theory
relaxation with the molecular dynamics approximated by isotropic rotational
diffusion of a rigid molecule are implemented. All NMR interactions as well as
scalar relaxation of the second kind are included.

NOTE: This script performs a lot of calculations. Running all calculations
serially took a few days on a laptop with 11th generation Intel i5 processor
and 16 GB of DDR4 RAM.
"""

# Imports
import os
import spinguin as sg
import numpy as np
from sabrefuncs import sabre_buildup_altfields

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

# Set bubbling time
bubbling_time = 30

# Set 1: B_low = 3 µT; B_high = 0 ... 100 µT; tau_low = tau_high = 5 ms
field_low = 3e-6
field_high_list = np.arange(101) * 1e-6
t_low = 5e-3
t_high = 5e-3

# Empty array for the results
result = np.zeros((len(field_high_list), spin_system_s.nspins))

# Perform the simulation with each B_high
for i, field_high in enumerate(field_high_list):

    # Perform the SABRE hyperpolarisation at square-modulated field
    time, mag_c, mag_s, rho_c, rho_s = sabre_buildup_altfields(
        spin_system_c = spin_system_c,
        spin_system_s = spin_system_s,
        spin_system_H2 = spin_system_H2,
        field_low = field_low,
        field_high = field_high,
        t_low = t_low,
        t_high = t_high,
        bubbling_time = bubbling_time,
        spin_map_s = spin_map_s,
        spin_map_H2 = spin_map_H2,
        c_c = c_c,
        c_s = c_s,
        kd_s = kd_s,
        kd_H2 = kd_H2
    )

    # Save results
    result[i] = mag_s[-1]

np.savetxt("fielddep_3uT_5ms_XuT_5ms.txt", result, delimiter=";")

# Set 2: B_low = 3 µT; B_high = 0 ... 100 µT; tau_low = tau_high = 10 ms
field_low = 3e-6
field_high_list = np.arange(101) * 1e-6
t_low = 10e-3
t_high = 10e-3

# Empty array for the results
result = np.zeros((len(field_high_list), spin_system_s.nspins))

# Perform the simulation with each B_high
for i, field_high in enumerate(field_high_list):

    # Perform the SABRE hyperpolarisation at square-modulated field
    time, mag_c, mag_s, rho_c, rho_s = sabre_buildup_altfields(
        spin_system_c = spin_system_c,
        spin_system_s = spin_system_s,
        spin_system_H2 = spin_system_H2,
        field_low = field_low,
        field_high = field_high,
        t_low = t_low,
        t_high = t_high,
        bubbling_time = bubbling_time,
        spin_map_s = spin_map_s,
        spin_map_H2 = spin_map_H2,
        c_c = c_c,
        c_s = c_s,
        kd_s = kd_s,
        kd_H2 = kd_H2
    )

    # Save results
    result[i] = mag_s[-1]

np.savetxt("fielddep_3uT_10ms_XuT_10ms.txt", result, delimiter=";")

# Set 3: B_low = -120 ... -80 µT; B_high = 100 µT; tau_low = tau_high = 2 ms
field_low_list = np.arange(-120, -79) * 1e-6
field_high = 100e-6
t_low = 2e-3
t_high = 2e-3

# Empty array for the results
result = np.zeros((len(field_high_list), spin_system_s.nspins))

# Perform the simulation with each B_low
for i, field_low in enumerate(field_low_list):

    # Perform the SABRE hyperpolarisation at square-modulated field
    time, mag_c, mag_s, rho_c, rho_s = sabre_buildup_altfields(
        spin_system_c = spin_system_c,
        spin_system_s = spin_system_s,
        spin_system_H2 = spin_system_H2,
        field_low = field_low,
        field_high = field_high,
        t_low = t_low,
        t_high = t_high,
        bubbling_time = bubbling_time,
        spin_map_s = spin_map_s,
        spin_map_H2 = spin_map_H2,
        c_c = c_c,
        c_s = c_s,
        kd_s = kd_s,
        kd_H2 = kd_H2
    )

    # Save results
    result[i] = mag_s[-1]

np.savetxt("fielddep_100uT_2ms_XuT_2ms.txt", result, delimiter=";")

# Set 4: B_low = 0 ... -390 µT; B_high = 10 ... 400 µT; tau_low = tau_high = 1 ms
field_low_list = np.arange(0, -395, step=-5) * 1e-6
field_high_list = np.arange(10, 405, step=5) * 1e-6
t_low = 1e-3
t_high = 1e-3

# Empty array for the results
result = np.zeros((len(field_high_list), spin_system_s.nspins))

# Perform the simulation with each B_low and B_high pair
for i, (field_low, field_high) in enumerate(zip(field_low_list, field_high_list)):

    # Perform the SABRE hyperpolarisation at square-modulated field
    time, mag_c, mag_s, rho_c, rho_s = sabre_buildup_altfields(
        spin_system_c = spin_system_c,
        spin_system_s = spin_system_s,
        spin_system_H2 = spin_system_H2,
        field_low = field_low,
        field_high = field_high,
        t_low = t_low,
        t_high = t_high,
        bubbling_time = bubbling_time,
        spin_map_s = spin_map_s,
        spin_map_H2 = spin_map_H2,
        c_c = c_c,
        c_s = c_s,
        kd_s = kd_s,
        kd_H2 = kd_H2
    )

    # Save results
    result[i] = mag_s[-1]

np.savetxt("fielddep_XuT_1ms_XuT_1ms.txt", result, delimiter=";")

# Set 5: B_low = -90 µT; B_high = 100 µT; tau_low = tau_high = 0.1 ... 8 ms
field_low = -90e-6
field_high = 100e-6
t_low_list = np.arange(1, 81, step=1)*1e-4
t_high_list = np.arange(1, 81, step=1)*1e-4

# Empty array for the results
result = np.zeros((len(t_low_list), spin_system_s.nspins))

# Perform the simulation with each tau_low and tau_high pair
for i, (t_low, t_high) in enumerate(zip(t_low_list, t_high_list)):

    # Perform the SABRE hyperpolarisation at square-modulated field
    time, mag_c, mag_s, rho_c, rho_s = sabre_buildup_altfields(
        spin_system_c = spin_system_c,
        spin_system_s = spin_system_s,
        spin_system_H2 = spin_system_H2,
        field_low = field_low,
        field_high = field_high,
        t_low = t_low,
        t_high = t_high,
        bubbling_time = bubbling_time,
        spin_map_s = spin_map_s,
        spin_map_H2 = spin_map_H2,
        c_c = c_c,
        c_s = c_s,
        kd_s = kd_s,
        kd_H2 = kd_H2
    )

    # Save results
    result[i] = mag_s[-1]

np.savetxt("fielddep_100uT_Xms_-90uT_Xms.txt", result, delimiter=";")

# Set 6: B_low = -150 ... -50 µT; B_high = 50 µT; tau_low = 1 ms, tau_high = 2 ms
field_low_list = np.arange(-150, -49) * 1e-6
field_high = 50e-6
t_low = 1e-3
t_high = 2e-3

# Empty array for the results
result = np.zeros((len(field_low_list), spin_system_s.nspins))

# Perform the simulation with each B_low
for i, field_low in enumerate(field_low_list):

    # Perform the SABRE hyperpolarisation at square-modulated field
    time, mag_c, mag_s, rho_c, rho_s = sabre_buildup_altfields(
        spin_system_c = spin_system_c,
        spin_system_s = spin_system_s,
        spin_system_H2 = spin_system_H2,
        field_low = field_low,
        field_high = field_high,
        t_low = t_low,
        t_high = t_high,
        bubbling_time = bubbling_time,
        spin_map_s = spin_map_s,
        spin_map_H2 = spin_map_H2,
        c_c = c_c,
        c_s = c_s,
        kd_s = kd_s,
        kd_H2 = kd_H2
    )

    # Save results
    result[i] = mag_s[-1]

np.savetxt("fielddep_50uT_2ms_XuT_1ms.txt", result, delimiter=";")
