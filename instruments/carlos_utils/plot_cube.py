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
wavelengths = data[0]         # Dimension: (nords, npix)
norm_flux   = data[5]         # Continuum normalized flux
blaze_flux  = data[3]         # Blaze corrected flux
data1_flux    = data[1]
data2_flux    = data[2]
data3_flux    = data[3]
data4_flux    = data[4]
data5_flux    = data[5]
data6_flux    = data[6]
data7_flux    = data[7]
data8_flux    = data[8]

nords, npix = wavelengths.shape
print(f"Loaded spectrum with {nords} orders and {npix} pixels per order.")

# 3. Plot option A: Single Order (e.g., Order 15)
#order_to_plot = 15
#order_to_plot = 4
order_to_plot = int(sys.argv[2])
data_set = int(sys.argv[2])


#plt.figure(figsize=(10, 4))
#plt.plot(wavelengths[order_to_plot], norm_flux[order_to_plot], color='black', lw=0.8)
#plt.title(f"CERES Spectrum - Order {order_to_plot}")
#plt.xlabel("Wavelength [Å]")
#plt.ylabel("Normalized Flux")
#plt.grid(True, linestyle='--', alpha=0.6)
#plt.tight_layout()
#plt.show()

# 4. Plot option B: Overlay all orders across the full spectrum
plt.figure(figsize=(12, 5))
for i in range(nords):
    # Plotting flux for all orders
    #plt.plot(wavelengths[i], blaze_flux[i], lw=0.7, alpha=0.8)
    #plt.plot(wavelengths[i], data1_flux[i], lw=0.7, alpha=0.8)
    #plt.plot(wavelengths[i], norm_flux[i], lw=0.7, alpha=0.8)
    plt.plot(wavelengths[i], data[data_set][i], lw=0.7, alpha=0.8)

if data_set == 3:
    data_set = "blazed(3)"  
if data_set == 5: 
    data_set = "normalized(5)"  
plt.title(f"CERES Full Spectrum of {name} data_set {data_set}")
plt.xlabel("Wavelength [Å]")
plt.ylabel(f"data[{data_set}]")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
