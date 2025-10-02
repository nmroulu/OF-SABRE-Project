"""
sabrefuncs.py

This module contains functions that implement the SABRE hyperpolarisation at
static or rectangular polarisation transfer fields.
"""

# Imports
import numpy as np
from tqdm.auto import tqdm
import spinguin as sg
from copy import deepcopy

def sabre_buildup(
    spin_system_c : sg.SpinSystem,
    spin_system_s : sg.SpinSystem,
    spin_system_H2 : sg.SpinSystem,
    field : float,
    t : float,
    bubbling_time : float,
    spin_map_s : list,
    spin_map_H2 :list,
    c_c : float,
    c_s : float,
    kd_s : float,
    kd_H2 : float
):
    """
    Polarisation build-up in SABRE when subjected to a static magnetic field.
    """
    # Take copies of the spin systems
    spin_system_c = deepcopy(spin_system_c)
    spin_system_s = deepcopy(spin_system_s)
    spin_system_H2 = deepcopy(spin_system_H2)

    # Calculate the Liouvillian
    sg.parameters.magnetic_field=field
    H_c = sg.hamiltonian(spin_system_c)
    H_s = sg.hamiltonian(spin_system_s)

    R_c = sg.relaxation(spin_system_c)
    R_s = sg.relaxation(spin_system_s)

    L_c = sg.liouvillian(H_c, R_c)
    L_s = sg.liouvillian(H_s, R_s)

    # Truncate the basis set to ZQ
    L_c = spin_system_c.basis.truncate_by_coherence([0], L_c)
    L_s = spin_system_s.basis.truncate_by_coherence([0], L_s)

    # Construct the time propagator
    P_c = sg.propagator(L_c, t)
    P_s = sg.propagator(L_s, t)

    # Calculate number of steps
    nsteps = int(bubbling_time / t)

    # Use the correction by Lindale et al.
    kd_s = np.exp(-t*kd_s/2) * kd_s
    kd_H2 = np.exp(-t*kd_H2/2) * kd_H2

    # Construct the initial state
    rho_c = sg.unit_state(spin_system_c)
    rho_s = sg.unit_state(spin_system_s)
    rho_H2 = sg.singlet_state(spin_system_H2, 0, 1)

    # Empty arrays for storing the spin magnetizations
    mag_c = np.zeros((nsteps, spin_system_c.nspins))
    mag_s = np.zeros((nsteps, spin_system_s.nspins))

    # Evolve the spin system for the number of steps
    for step in tqdm(
        range(nsteps),
        miniters=nsteps/100,
        maxinterval=float('inf'),
        desc='Propagating...'):

        # Calculate the magnetizations
        for i in range(spin_system_c.nspins):
            mag_c[step, i] = sg.measure(spin_system_c, rho_c, f"I(z,{i})").real
        for i in range(spin_system_s.nspins):
            mag_s[step, i] = sg.measure(spin_system_s, rho_s, f"I(z,{i})").real

        # Split the complex to substrate and H2
        rho_c_H2, rho_c_s = sg.dissociate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_c, spin_map_H2, spin_map_s)

        # Calculate complexes with new substrate (and old/new pH2) 
        rho_c_oldH2 = sg.associate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_c_H2, rho_s, spin_map_H2, spin_map_s)
        rho_c_newH2 = sg.associate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_H2, rho_s, spin_map_H2, spin_map_s)

        # Exchange process of free substrate
        rho_s = rho_s + c_c/c_s * t * kd_s * (rho_c_s - rho_s)

        # Exchange process of complex
        rho_c = rho_c + t * (
            (kd_s - kd_H2) * (rho_c_oldH2 - rho_c) + \
            kd_H2 * (rho_c_newH2 - rho_c)
        )

        # Propagate the spin system
        rho_c = P_c @ rho_c
        rho_s = P_s @ rho_s

    # Construct the time array
    time = np.arange(nsteps)*t

    print("Calculations finished.")
    return time, mag_c, mag_s, rho_c, rho_s

def sabre_buildup_altfields(
        spin_system_c : sg.SpinSystem,
        spin_system_s : sg.SpinSystem,
        spin_system_H2 : sg.SpinSystem,
        field_low : float,
        field_high : float,
        t_low : float,
        t_high : float,
        bubbling_time : float,
        spin_map_s : list,
        spin_map_H2 :list,
        c_c : float,
        c_s : float,
        kd_s : float,
        kd_H2 : float
):
    """
    Polarisation build-up in SABRE when subjected to rectangular polarisation
    transfer field.
    """
    # Take copies of the spin systems
    spin_system_c = deepcopy(spin_system_c)
    spin_system_s = deepcopy(spin_system_s)
    spin_system_H2 = deepcopy(spin_system_H2)

    # Calculate the Liouvillian for LOW field
    sg.parameters.magnetic_field=field_low
    H_c_low = sg.hamiltonian(spin_system_c)
    H_s_low = sg.hamiltonian(spin_system_s)

    R_c_low = sg.relaxation(spin_system_c)
    R_s_low = sg.relaxation(spin_system_s)

    L_c_low = sg.liouvillian(H_c_low, R_c_low)
    L_s_low = sg.liouvillian(H_s_low, R_s_low)

    # Calculate the Liouvillian for HIGH field
    sg.parameters.magnetic_field=field_high
    H_c_high = sg.hamiltonian(spin_system_c)
    H_s_high = sg.hamiltonian(spin_system_s)

    R_c_high = sg.relaxation(spin_system_c)
    R_s_high = sg.relaxation(spin_system_s)

    L_c_high = sg.liouvillian(H_c_high, R_c_high)
    L_s_high = sg.liouvillian(H_s_high, R_s_high)

    # Truncate the basis set to ZQ
    L_c_low, L_c_high = \
        spin_system_c.basis.truncate_by_coherence([0], L_c_low, L_c_high)
    
    L_s_low, L_s_high = \
        spin_system_s.basis.truncate_by_coherence([0], L_s_low, L_s_high)

    # Construct the time propagator
    P_c_low = sg.propagator(L_c_low, t_low)
    P_s_low = sg.propagator(L_s_low, t_low)

    P_c_high = sg.propagator(L_c_high, t_high)
    P_s_high = sg.propagator(L_s_high, t_high)

    # Construct the overall time propagators
    P_c = P_c_low @ P_c_high
    P_s = P_s_low @ P_s_high

    # Calculate total time step and number of steps
    t = t_low + t_high
    nsteps = int(bubbling_time / t)

    # Use the correction by Lindale et al.
    kd_s = np.exp(-t*kd_s/2) * kd_s
    kd_H2 = np.exp(-t*kd_H2/2) * kd_H2

    # Construct the initial state
    rho_c = sg.unit_state(spin_system_c)
    rho_s = sg.unit_state(spin_system_s)
    rho_H2 = sg.singlet_state(spin_system_H2, 0, 1)

    # Empty arrays for storing the spin magnetizations
    mag_c = np.zeros((nsteps, spin_system_c.nspins))
    mag_s = np.zeros((nsteps, spin_system_s.nspins))

    # Evolve the spin system for the number of steps
    for step in tqdm(
        range(nsteps),
        miniters=nsteps/100,
        maxinterval=float('inf'),
        desc='Propagating...'):

        # Calculate the magnetizations
        for i in range(spin_system_c.nspins):
            mag_c[step, i] = sg.measure(spin_system_c, rho_c, f"I(z,{i})").real
        for i in range(spin_system_s.nspins):
            mag_s[step, i] = sg.measure(spin_system_s, rho_s, f"I(z,{i})").real

        # Split the complex to substrate and H2
        rho_c_H2, rho_c_s = sg.dissociate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_c, spin_map_H2, spin_map_s)

        # Calculate complexes with new substrate (and old/new pH2) 
        rho_c_oldH2 = sg.associate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_c_H2, rho_s, spin_map_H2, spin_map_s)
        rho_c_newH2 = sg.associate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_H2, rho_s, spin_map_H2, spin_map_s)

        # Exchange process of free substrate
        rho_s = rho_s + c_c/c_s * t * kd_s * (rho_c_s - rho_s)

        # Exchange process of complex
        rho_c = rho_c + t * (
            (kd_s - kd_H2) * (rho_c_oldH2 - rho_c) + \
            kd_H2 * (rho_c_newH2 - rho_c))

        # Propagate the spin system
        rho_c = P_c @ rho_c
        rho_s = P_s @ rho_s

    # Construct the time array
    time = np.arange(nsteps)*t

    print("Calculations finished.")
    return time, mag_c, mag_s, rho_c, rho_s

def sabre_relaxation(
        spin_system_c : sg.SpinSystem,
        spin_system_s : sg.SpinSystem,
        spin_system_H2 : sg.SpinSystem,
        field : float,
        t : float,
        relaxation_time : float,
        spin_map_s : list,
        spin_map_H2 :list,
        c_c : float,
        c_s : float,
        kd_s : float,
        rho_c : np.ndarray,
        rho_s : np.ndarray
):
    """
    Relaxation of a SABRE system when subjected to a static magnetic field.
    """
    # Take copies of the spin system
    spin_system_c = deepcopy(spin_system_c)
    spin_system_s = deepcopy(spin_system_s)
    spin_system_H2 = deepcopy(spin_system_H2)

    # Calculate the Liouvillian
    sg.parameters.magnetic_field=field
    H_c = sg.hamiltonian(spin_system_c)
    H_s = sg.hamiltonian(spin_system_s)

    R_c = sg.relaxation(spin_system_c)
    R_s = sg.relaxation(spin_system_s)

    L_c = sg.liouvillian(H_c, R_c)
    L_s = sg.liouvillian(H_s, R_s)

    # Truncate the basis set to ZQ
    L_c = spin_system_c.basis.truncate_by_coherence([0], L_c)
    L_s = spin_system_s.basis.truncate_by_coherence([0], L_s)

    # Construct the time propagator
    P_c= sg.propagator(L_c, t)
    P_s = sg.propagator(L_s, t)

    # Calculate number of steps
    nsteps = int(relaxation_time / t)

    # Use the correction by Lindale et al.
    kd_s = np.exp(-t*kd_s/2) * kd_s

    # Empty arrays for storing the spin magnetizations
    mag_c = np.zeros((nsteps, spin_system_c.nspins))
    mag_s = np.zeros((nsteps, spin_system_s.nspins))

    # Evolve the spin system for the number of steps
    for step in tqdm(
        range(nsteps),
        miniters=nsteps/100,
        maxinterval=float('inf'),
        desc='Propagating...'):

        # Calculate the magnetizations
        for i in range(spin_system_c.nspins):
            mag_c[step, i] = sg.measure(spin_system_c, rho_c, f"I(z,{i})").real
        for i in range(spin_system_s.nspins):
            mag_s[step, i] = sg.measure(spin_system_s, rho_s, f"I(z,{i})").real

        # Split the complex to substrate and H2
        rho_c_H2, rho_c_s = sg.dissociate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_c, spin_map_H2, spin_map_s)

        # Calculate complexes with new substrate (only old H2)
        rho_c_oldH2 = sg.associate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_c_H2, rho_s, spin_map_H2, spin_map_s)

        # Exchange process of free substrate
        rho_s = rho_s + c_c/c_s * t * kd_s * (rho_c_s - rho_s)

        # Exchange process of complex
        rho_c = rho_c + t * kd_s * (rho_c_oldH2 - rho_c)

        # Propagate the spin system
        rho_c = P_c @ rho_c
        rho_s = P_s @ rho_s

    # Construct the time array
    time = np.arange(nsteps)*t

    print("Calculations finished.")
    return time, mag_c, mag_s


def sabre_relaxation_altfields(
        spin_system_c : sg.SpinSystem,
        spin_system_s : sg.SpinSystem,
        spin_system_H2 : sg.SpinSystem,
        field_low : float,
        field_high : float,
        t_low : float,
        t_high : float,
        relaxation_time : float,
        spin_map_s : list,
        spin_map_H2 :list,
        c_c : float,
        c_s : float,
        kd_s : float,
        rho_c : np.ndarray,
        rho_s : np.ndarray
):
    """
    Relaxation of a SABRE system when subjected to a rectangular magnetic field.
    """
    # Take copies of the spin system
    spin_system_c = deepcopy(spin_system_c)
    spin_system_s = deepcopy(spin_system_s)
    spin_system_H2 = deepcopy(spin_system_H2)

    # Calculate the Liouvillian for LOW field
    sg.parameters.magnetic_field=field_low
    H_c_low = sg.hamiltonian(spin_system_c)
    H_s_low = sg.hamiltonian(spin_system_s)

    R_c_low = sg.relaxation(spin_system_c)
    R_s_low = sg.relaxation(spin_system_s)

    L_c_low = sg.liouvillian(H_c_low, R_c_low)
    L_s_low = sg.liouvillian(H_s_low, R_s_low)

    # Calculate the Liouvillian for HIGH field
    sg.parameters.magnetic_field=field_high
    H_c_high = sg.hamiltonian(spin_system_c)
    H_s_high = sg.hamiltonian(spin_system_s)

    R_c_high = sg.relaxation(spin_system_c)
    R_s_high = sg.relaxation(spin_system_s)

    L_c_high = sg.liouvillian(H_c_high, R_c_high)
    L_s_high = sg.liouvillian(H_s_high, R_s_high)

    # Truncate the basis set to ZQ
    L_c_low, L_c_high = \
        spin_system_c.basis.truncate_by_coherence([0], L_c_low, L_c_high)
    
    L_s_low, L_s_high = \
        spin_system_s.basis.truncate_by_coherence([0], L_s_low, L_s_high)

    # Construct the time propagator
    P_c_low = sg.propagator(L_c_low, t_low)
    P_s_low = sg.propagator(L_s_low, t_low)

    P_c_high = sg.propagator(L_c_high, t_high)
    P_s_high = sg.propagator(L_s_high, t_high)

    # Construct the overall time propagators
    P_c = P_c_low @ P_c_high
    P_s = P_s_low @ P_s_high

    # Calculate total time step and number of steps
    t = t_low + t_high
    nsteps = int(relaxation_time / t)

    # Use the correction by Lindale et al.
    kd_s = np.exp(-t*kd_s/2) * kd_s

    # Empty arrays for storing the spin magnetizations
    mag_c = np.zeros((nsteps, spin_system_c.nspins))
    mag_s = np.zeros((nsteps, spin_system_s.nspins))

    # Evolve the spin system for the number of steps
    for step in tqdm(
        range(nsteps),
        miniters=nsteps/100,
        maxinterval=float('inf'),
        desc='Propagating...'):

        # Calculate the magnetizations
        for i in range(spin_system_c.nspins):
            mag_c[step, i] = sg.measure(spin_system_c, rho_c, f"I(z,{i})").real
        for i in range(spin_system_s.nspins):
            mag_s[step, i] = sg.measure(spin_system_s, rho_s, f"I(z,{i})").real

        # Split the complex to substrate and H2
        rho_c_H2, rho_c_s = sg.dissociate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_c, spin_map_H2, spin_map_s)

        # Calculate complexes with new substrate (only old H2)
        rho_c_oldH2 = sg.associate(
            spin_system_H2, spin_system_s, spin_system_c,
            rho_c_H2, rho_s, spin_map_H2, spin_map_s)

        # Exchange process of free substrate
        rho_s = rho_s + c_c/c_s * t * kd_s * (rho_c_s - rho_s)

        # Exchange process of complex
        rho_c = rho_c + t * kd_s * (rho_c_oldH2 - rho_c)

        # Propagate the spin system
        rho_c = P_c @ rho_c
        rho_s = P_s @ rho_s

    # Construct the time array
    time = np.arange(nsteps)*t

    print("Calculations finished.")
    return time, mag_c, mag_s