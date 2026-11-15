#!/usr/bin/python3

import sys
import numpy as np
from astropy.io import fits

print("sys.argv: ", sys.argv)
if len(sys.argv) > 1:
    archivo = sys.argv[1]
    print(sys.argv[1])
else:
    print("No arguments provided.")

archivo_entrada = sys.argv[1]
archivo_salida = archivo_entrada + ".fits"

with fits.open(archivo_entrada) as hdul:
    datos = hdul[1].data
    wave_orig = datos["WAVELENGTH"]
    flux_orig = datos["FLUX"]

# 1. Crear una nueva rejilla de longitud de onda estrictamente lineal
npix = len(wave_orig)
start = float(wave_orig[0])
end = float(wave_orig[-1])
step = (end - start) / (npix - 1)

wave_lineal = np.linspace(start, end, npix)

# 2. Interpolar el flujo a la nueva rejilla lineal
flux_lineal = np.interp(wave_lineal, wave_orig, flux_orig)

# 3. Crear el nuevo HDU principal
hdu_nuevo = fits.PrimaryHDU(data=np.array(flux_lineal, dtype=np.float32))
hdr = hdu_nuevo.header

# 4. Asignar las cabeceras requeridas
hdr["NAXIS"] = 1
hdr["NAXIS1"] = npix
hdr["NPIX"] = npix

hdr["START"] = (start, "Longitud de onda inicial")
hdr["STEP"] = (step, "Paso constante de longitud de onda")
hdr["CUNIT1"] = ("Angstrom", "Unidades")

hdr["LHCUTS"] = (
    f"{float(np.nanmin(flux_lineal))} {float(np.nanmax(flux_lineal))}",
    "Limites de flujo",
)

# WCS Estándar
hdr["CRPIX1"] = 1.0
hdr["CRVAL1"] = start
hdr["CDELT1"] = step
hdr["CTYPE1"] = "WAVE"

hdu_nuevo.writeto(archivo_salida, overwrite=True)
print(f"Espectro reinterpolado guardado en {archivo_salida}")
