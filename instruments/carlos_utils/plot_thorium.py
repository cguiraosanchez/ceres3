#!/usr/bin/python3

import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# sys.argv[0] is the script name itself.
# sys.argv[1] is the first argument passed to the script.
print("sys.argv: ", sys.argv)
if len(sys.argv) > 1:
    fits_file = sys.argv[1]
    print(sys.argv[1])
else:
    print("No arguments provided.")

with fits.open(fits_file) as hdul:
  # El array principal contiene los datos 3D con forma (2, 49, 1756)
  data = hdul[0].data

# Verificamos los dos planos de datos
# Plano 0 y Plano 1: uno es longitud de onda y otro es flujo.
# Por lo general, los valores de longitud de onda (ej. Ångströms o nm) son mayores que los flujos.
plane_a = data[0]
plane_b = data[1]

# Identificamos automáticamente cuál es la longitud de onda comprobando el rango de valores
if np.nanmean(plane_a) > np.nanmean(plane_b):
  wavelength = plane_a
  flux = plane_b
else:
  wavelength = plane_b
  flux = plane_a

# Configuración de la figura para visualizar los 49 órdenes
plt.figure(figsize=(12, 7))

n_orders = flux.shape[0]  # Debería ser 49
for i in range(n_orders):
  wave_order = wavelength[i]
  flux_order = flux[i]

  # Filtrar valores no finitos o nulos
  mask = np.isfinite(wave_order) & np.isfinite(flux_order) & (wave_order > 0)

  plt.plot(wave_order[mask], flux_order[mask], lw=0.7, alpha=0.8)

plt.xlabel('Wavelength [Å])', fontsize=12)
plt.ylabel('Flux', fontsize=12)
plt.title(
    f"Echelle Spectra of {fits_file} - {n_orders} orders overplotted", fontsize=14, fontweight='bold'
)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
