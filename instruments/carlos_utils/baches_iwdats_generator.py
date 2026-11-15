#!/usr/bin/python3

from collections import defaultdict


def procesar_fichero_ascii(fichero_entrada):
    # Diccionario para agrupar las filas procesadas según el valor de NN (columna 7)
    datos_por_nn = defaultdict(list)

    with open(fichero_entrada, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue  # Ignorar líneas vacías

            # Separar por cualquier cantidad de espacios en blanco
            partes = linea.split()

            if len(partes) < 7:
                continue

            try:
                # Extraer columnas requeridas (índices 0-based: col 2 es índice 1, col 5 es índice 4, col 7 es índice 6)
                col2_float = float(partes[1])
                col5_float = float(partes[4])
                nn_val = int(partes[6])

                # Transformaciones
                col2_redondeada = round(col2_float)
                col5_formateada = f"{col5_float:.4f}"

                # Guardar tupla: (valor float original de col2 para ordenar, col2 redondeada, col5 a 4 dec)
                datos_por_nn[nn_val].append(
                    (col2_float, col2_redondeada, col5_formateada)
                )

            except ValueError:
                # Omitir cabeceras o líneas con formato no numérico
                continue

    # Generar ficheros del 31 al 53
    for nn in range(31, 53):
        filas = datos_por_nn.get((nn), [])
       

        # Ordenar por el valor decimal original de la segunda columna
        filas.sort(key=lambda x: x[0])

        nombre_fichero = f"baches_{nn}.iwdat"

        with open(nombre_fichero, "w") as f_out:
            for _, col2_red, col5_fmt in filas:
                # Formato alineado en columnas separadas por espacios en blanco:
                #   '1': constante
                #   'col2_red': entero alineado en ancho de 8 caracteres
                #   'col5_fmt': flotante formateado a 4 dec alineado en ancho de 14 caracteres
                #   'NN': string constante
                f_out.write(f"1   {col2_red:>8d}   {col5_fmt:>14s}   NN\n")


if __name__ == "__main__":
    # Reemplaza 'datos.txt' por el nombre o ruta de tu fichero de entrada
    procesar_fichero_ascii("bachesLINE.txt")
