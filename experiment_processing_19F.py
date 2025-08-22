"""
How to use?

1) Change the processing settings.
2) Run the script.
3) Choose a folder where all experiments are located.
4) The results will be saved into a "Results" folder.
"""

# Imports
import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import os
import glob
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

# Fourier transform settings
npoints = 8192              # Number of points (truncating)
lb = 1                      # Line broadening
zf = 32768                  # Zero filling

# Phase correction settings
phase_mode = "phased"       # "phased" OR "magnitude"
p0 = -43                    # Zero-order phase correction
p1 = 0                      # First order phase correction

# Baseline correction settings
baseline_correction = "yes" # "no" OR "yes"

# Integration settings
low_integral = -129.3
high_integral = -127

# Spectrum plot settings
low_spectrum = -135
high_spectrum = -120

# Show figure?
show_figure = "no"          # "no" OR "yes"

# Ask for folder location
root = tk.Tk()
root.withdraw() 
folder = filedialog.askdirectory()
root.destroy()

# Get the contents in the folder
folder_contents = glob.glob(os.path.join(folder, '*'))

# Make empty arrays for the integrals
expt_names = []
integrals = []

# Process every experiment in the folder
for experiment in folder_contents:

    # Get the experiment name
    expt_name = os.path.basename(experiment)

    # Get the integral
    try:
        
        # Read the data
        dic, data = ng.spinsolve.read(experiment)
        dic = ng.spinsolve.guess_udic(dic, data)

        # Convert the data to NMRPipe format
        C = ng.convert.converter()
        C.from_universal(dic, data)
        dic, data = C.to_pipe()

        # Truncate the data
        data = data[0:npoints]

        # Apply apodization
        data = ng.proc_base.em(data, lb / dic["FDF2SW"])

        # Apply zero-filling
        data = ng.proc_base.zf_size(data, zf)

        # Fourier transform the data
        data = ng.proc_base.fft_norm(data)

        # Apply phase correction
        if phase_mode == "phased":
            data = ng.proc_base.ps(data, p0, p1)

        elif phase_mode == "magnitude":
            data = np.abs(data)

        # Delete imaginary
        data = ng.proc_base.di(data)

        # Apply baseline correction
        if baseline_correction:
            data = ng.proc_bl.baseline_corrector(data, wd=20)

        # Get the ppm scale
        uc = ng.pipe.make_uc(dic, data)

        # Integrate the peak
        integral = ng.integration.integrate(
            data, uc, limits=[low_integral, high_integral], unit='ppm')[0]

        # Store the integral
        expt_names.append(expt_name)
        integrals.append(integral)

        print(f"Experiment {expt_name} processed.")

        # Show the spectrum
        if show_figure == "yes":
            plt.plot(uc.ppm_scale(), np.real(data))
            plt.xlim(high_spectrum, low_spectrum)
            plt.xlabel("Chemical shift (ppm)")
            plt.ylabel("Intensity (arb. units)")
            plt.show()

    except FileNotFoundError:
        print(f"Experiment {experiment} could not be processed.")

# Collect the result into single array
result = np.array(list(zip(expt_names, integrals)))

# Make output folder
result_folder = os.path.join(folder, "Results").replace('\\', '/')
os.makedirs(result_folder, exist_ok=True)

# Make output path
filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
output_path = os.path.join(result_folder, filename).replace('\\', '/')

# Save the result as CSV
np.savetxt(output_path, result, delimiter=';', fmt='%s')