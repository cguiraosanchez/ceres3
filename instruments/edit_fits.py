#!/usr/bin/python3
import os
import sys
import glob
from astropy.io import fits

def procesar_fits_directorio(directorio, start_x, start_y, size_x, size_y):
    """
    Procesa todos los archivos .fits en 'directorio':
      1. Realiza un flip en el eje X (horizontal).
      2. Extrae una subventana (crop) según los parámetros indicados.
      3. Guarda el resultado con el sufijo 'sub_'.

    :param directorio: Ruta del directorio con las imágenes FITS.
    :param start_x: Índice inicial de corte en el eje X (base 0, eje horizontal).
    :param start_y: Índice inicial de corte en el eje Y (base 0, eje vertical).
    :param size_x: Número de píxeles a recortar en X.
    :param size_y: Número de píxeles a recortar en Y.
    """
    # Buscar todos los archivos .fits en el directorio especificado
    patron = os.path.join(directorio, "*.fit*")
    archivos = glob.glob(patron)

    if not archivos:
        print(f"No se encontraron archivos .fits en '{directorio}'.")
        return

    print(f"Se encontraron {len(archivos)} archivo(s) FITS. Iniciando procesamiento...\n")

    for ruta_archivo in archivos:
        nombre_base = os.path.basename(ruta_archivo)

        # Omitir archivos que ya hayan sido procesados previamente
        if nombre_base.startswith("sub_"):
            continue

        try:
            with fits.open(ruta_archivo) as hdul:
                # Trabajar sobre el HDU primario
                header = hdul[0].header.copy()
                data = hdul[0].data

                # Verificar que el HDU contiene datos y que es bidimensional (NAXIS == 2)
                if data is None or data.ndim != 2:
                    print(f"[OMITIDO] {nombre_base}: No es una imagen bidimensional (dimensiones: {data.ndim if data is not None else 0}).")
                    continue

                # 1. Flip en el eje X (horizontal)
                # En NumPy arrays (Y, X), la segunda dimensión representa el eje X.
                data_flipped = data[:, ::-1]

                # 2. Recorte / Subventana (crop)
                end_x = start_x + size_x
                end_y = start_y + size_y

                # Comprobar límites de la imagen
                max_y, max_x = data_flipped.shape
                if end_x > max_x or end_y > max_y or start_x < 0 or start_y < 0:
                    print(f"[ERROR] {nombre_base}: La subventana ({start_x}:{end_x}, {start_y}:{end_y}) excede las dimensiones de la imagen ({max_x}x{max_y}).")
                    continue

                # Slicing en NumPy: data_sub[filas(Y), columnas(X)]
                data_sub = data_flipped[start_y:end_y, start_x:end_x]

                # Actualizar el header con las nuevas dimensiones de la matriz
                header['NAXIS1'] = size_x
                header['NAXIS2'] = size_y
                header['COMMENT'] = "Imagen volteada en X y recortada (subventana)."

                # Crear la nueva estructura FITS y guardarla
                nuevo_hdu = fits.PrimaryHDU(data=data_sub, header=header)
                
                directorio_origen = os.path.dirname(ruta_archivo)
                nombre_salida = f"sub_{nombre_base}"
                ruta_salida = os.path.join(directorio_origen, nombre_salida)

                nuevo_hdu.writeto(ruta_salida, overwrite=True)
                print(f"[OK] {nombre_base} -> {nombre_salida} (Dimensión final: {size_x}x{size_y})")

        except Exception as e:
            print(f"[ERROR] No se pudo procesar {nombre_base}: {e}")

# ==========================================
# EJEMPLO DE USO
# ==========================================
if __name__ == "__main__":
    # Configuración de parámetros
    START_X = 780    # Píxel inicial en X (0-indexed)
    #START_Y = 255    # Píxel inicial en Y (0-indexed)
    START_Y = 290    # Píxel inicial en Y (0-indexed)
    SIZE_X  = 1756   # Ancho en píxeles
    SIZE_Y  = 1636   # Alto en píxeles

    # sys.argv[0] is the script name itself.
    # sys.argv[1] is the first argument passed to the script.
    print("sys.argv: ", sys.argv)
    if len(sys.argv) > 1:
        DIRECTORIO_IMAGENES = sys.argv[1]
    else:
        print("No arguments provided.")

    if len(sys.argv) > 2:
        if sys.argv[2] != '?':
          START_X = sys.argv[2]
    if len(sys.argv) > 3:
        if sys.argv[3] != '?':
          START_Y = sys.argv[3]
    if len(sys.argv) > 4:
        SIZE_X  = sys.argv[4]
    if len(sys.argv) > 5:
       SIZE_Y  = sys.argv[5]

    print("start_x:", START_X)
    print("start_y:", START_Y)
    print("size_x:", SIZE_X)
    print("size_y:", SIZE_Y)

    procesar_fits_directorio(
        directorio=DIRECTORIO_IMAGENES,
        start_x=START_X,
        start_y=START_Y,
        size_x=SIZE_X,
        size_y=SIZE_Y
    )
