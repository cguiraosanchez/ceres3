#!/usr/bin/python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

# sys.argv[0] is the script name itself.
# sys.argv[1] is the first argument passed to the script.
print("sys.argv: ", sys.argv)
if len(sys.argv) > 1:
    filename = sys.argv[1]
    print(sys.argv[1])
else:
    print("No arguments provided.")

path = Path(filename)
directory = str(path.parent) # /rute to directory 
name = path.name # archivo.txt
name_without_ext = path.stem # archivo

hdu = fits.open(filename)
data = hdu[0].data  # Shape: (10, nords, npix)

# 2. Extract key arrays
wavelengths = data[0]  
flux        = data[1]

nords, npix = wavelengths.shape
print(f"Loaded spectrum with {nords} orders and {npix} pixels per order.")
nords = nords - 3 

# 4. Plot option B: Overlay all orders across the full spectrum
plt.figure(figsize=(12, 5))
for i in range(nords):
    # Plotting flux for all orders
    plt.plot(wavelengths[i], flux[i], lw=0.7, alpha=0.8)
#i=44
#plt.plot(wavelengths[i], flux[i], lw=0.7, alpha=0.8)
#plt.plot(wavelengths[i+1], flux[i+1], lw=0.7, alpha=0.8)

plt.title(f"CERES Full ThAr Spectrum of {name}")
plt.xlabel("Wavelength [Å]")
plt.ylabel(f"Intensity")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
