#!/usr/bin/python3
import sys
from astropy.io import fits
import matplotlib.pyplot as plt

# Ruta de tu archivo FITS
# sys.argv[0] is the script name itself.
# sys.argv[1] is the first argument passed to the script.
print("sys.argv: ", sys.argv)
if len(sys.argv) > 1:
    archivo = sys.argv[1]
    print(sys.argv[1])
else:
    print("No arguments provided.")


# Abrir el archivo FITS
with fits.open(archivo) as hdul:
    # Las tablas en FITS suelen estar en la extensión 1 (HDU 1)
    datos = hdul[1].data
    
    longitud_onda = datos["WAVELENGTH"]
    flujo = datos["FLUX"]
    error = datos["ERROR"]

# Configurar y generar la gráfica
plt.figure(figsize=(10, 5))
plt.plot(longitud_onda, flujo, color="tab:blue", linewidth=1, label="Flujo")

plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title("Espectro: Wavelength vs Flux")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
