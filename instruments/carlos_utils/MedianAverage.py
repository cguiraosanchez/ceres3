#!/usr/bin/python3
import sys
import argparse
import numpy as np
from astropy.io import fits

def calcular_median_average(lista_archivos, archivo_salida="MedianAverage.fits"):
    """
    Lee una lista de archivos FITS 2D, calcula la mediana píxel a píxel
    y guarda el resultado en un nuevo archivo FITS.
    """
    if not lista_archivos:
        print("Error: No se han proporcionado archivos FITS de entrada.")
        sys.exit(1)

    print(f"Cargando {len(lista_archivos)} archivo(s) FITS...")

    imagenes = []
    header_referencia = None
    shape_referencia = None

    for ruta in lista_archivos:
        try:
            with fits.open(ruta) as hdul:
                data = hdul[0].data
                
                # Validar que contenga datos
                if data is None:
                    print(f"[OMITIDO] {ruta}: El HDU primario no contiene datos.")
                    continue

                # Guardar el header del primer archivo para usarlo de plantilla
                if header_referencia is None:
                    header_referencia = hdul[0].header.copy()
                    shape_referencia = data.shape

                # Verificar compatibilidad de dimensiones
                if data.shape != shape_referencia:
                    print(f"[OMITIDO] {ruta}: Dimensiones {data.shape} no coinciden con la referencia {shape_referencia}.")
                    continue

                imagenes.append(data)
                print(f"  [OK] Cargar: {ruta}")

        except Exception as e:
            print(f"[ERROR] No se pudo leer '{ruta}': {e}")

    if not imagenes:
        print("Error: No hay imágenes válidas para calcular la mediana.")
        sys.exit(1)

    # Convertir la lista de imágenes a un stack 3D de NumPy (N_imagenes, Y, X)
    stack = np.array(imagenes)

    print(f"\nCalculando la mediana de {stack.shape[0]} imagen(es)...")
    # Calcular la mediana a lo largo del eje 0 (a través de la pila de imágenes). Manteniendo float32 original
    mediana_data = np.median(stack, axis=0).astype(np.float32)

    # Añadir información al header de referencia
    header_referencia['COMMENT'] = f"Combinacion por mediana de {stack.shape[0]} imagenes."
    header_referencia['NCOMBINE'] = stack.shape[0]

    # Guardar la nueva imagen FITS
    nuevo_hdu = fits.PrimaryHDU(data=mediana_data, header=header_referencia)
    nuevo_hdu.writeto(archivo_salida, overwrite=True)

    print(f"Proceso completado. Resultado guardado en: {archivo_salida}")

def main():
    parser = argparse.ArgumentParser(
        description="Calcula la mediana (median average) de una lista de archivos FITS."
    )
    parser.add_argument(
        "archivos",
        nargs="+",
        help="Rutas a los archivos FITS de entrada (ej. img1.fits img2.fits ... o *.fits)"
    )
    parser.add_argument(
        "-o", "--output",
        default="MedianAverage.fits",
        help="Nombre del archivo FITS de salida (por defecto: MedianAverage.fits)"
    )

    args = parser.parse_args()
    calcular_median_average(args.archivos, args.output)

if __name__ == "__main__":
    main()
