#!/usr/bin/python3
import os
import re

def renombrar_archivos(directorio='.'):
    # Patrón para identificar whapos_{NN}.py
    patron = re.compile(r'^whapos_(\d{2})\.iwdat$')
    
    archivos_a_renombrar = []
    
    # 1. Buscar coincidencias en el directorio
    for nombre in os.listdir(directorio):
        coincidencia = patron.match(nombre)
        if coincidencia:
            num_original = int(coincidencia.group(1))
            archivos_a_renombrar.append((num_original, nombre))
    
    # 2. Ordenar en orden descendente para evitar colisiones
    archivos_a_renombrar.sort(reverse=True, key=lambda x: x[0])
    
    # 3. Renombrar cada archivo
    for num_original, nombre_viejo in archivos_a_renombrar:
        nuevo_num = num_original + 49
        
        # Mantiene el formato de al menos 2 cifras con ceros a la izquierda (zfill)
        nuevo_num_str = str(nuevo_num).zfill(2)
        nombre_nuevo = f"whapos_{nuevo_num_str}.iwdat"
        
        ruta_vieja = os.path.join(directorio, nombre_viejo)
        ruta_nueva = os.path.join(directorio, nombre_nuevo)
        
        os.rename(ruta_vieja, ruta_nueva)
        print(f"Renombrado: {nombre_viejo} -> {nombre_nuevo}")

if __name__ == '__main__':
    # Modifica '.' por la ruta de la carpeta si es diferente
    renombrar_archivos('.')
