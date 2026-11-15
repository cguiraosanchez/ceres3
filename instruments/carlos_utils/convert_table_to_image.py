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
    wave = datos["WAVELENGTH"]
    flux = datos["FLUX"]
    header_orig = hdul[0].header.copy()

# 1. Calcular parámetros de la escala lineal de longitud de onda
npix = len(wave)
start = float(wave[0])
# Paso medio entre puntos
step = float((wave[-1] - wave[0]) / (npix - 1)) if npix > 1 else 1.0

# 2. Límites de contraste/corte para visualización (LHCUTS: min y max de flujo)
flux_min = float(np.nanmin(flux))
flux_max = float(np.nanmax(flux))

# 3. Crear el nuevo HDU principal con los datos de flujo
# Aseguramos un tipo de dato estándar (float32 o float64)
hdu_nuevo = fits.PrimaryHDU(data=np.array(flux, dtype=np.float32))

# 4. Asignar las palabras clave requeridas por el visualizador
hdr = hdu_nuevo.header

# Dimensiones de la imagen
hdr["NAXIS"] = 1
hdr["NAXIS1"] = npix
hdr["NPIX"] = npix  # Copia directa requerida por software legacy

# Malla de longitud de onda (formato personalizado/legacy)
hdr["START"] = (start, "Longitud de onda inicial")
hdr["STEP"] = (step, "Paso en longitud de onda")
hdr["CUNIT1"] = ("Angstrom", "Unidades del eje de longitud de onda")  # O "nm"

# Límites para el escalado del visualizador
hdr["LHCUTS"] = (
    f"{flux_min} {flux_max}",
    "Limites inferior y superior de flujo",
)

# Estándar oficial WCS (recomendado mantenerlo para compatibilidad universal)
hdr["CRPIX1"] = 1.0
hdr["CRVAL1"] = start
hdr["CDELT1"] = step
hdr["CTYPE1"] = "WAVE"

# Guardar el archivo modificado
hdu_nuevo.writeto(archivo_salida, overwrite=True)
print(f"Archivo guardado correctamente como: {archivo_salida}")
