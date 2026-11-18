from astroquery.simbad import Simbad
import astropy.units as u
from astropy.coordinates import SkyCoord

def consultar_simbad_tap(nombre_objeto):
    """
    Realiza una consulta TAP/ADQL a SIMBAD para obtener RA y DEC de un objeto.
    """
    # Construir la consulta ADQL
    # Se cruza la tabla 'basic' (datos principales) con 'ident' (identificadores/nombres alternativos)
    query = f"""
    SELECT TOP 1 
        basic.main_id, 
        basic.ra, 
        basic.dec 
    FROM basic 
    JOIN ident ON basic.oid = ident.oid 
    WHERE id = '{nombre_objeto}'
    """
    
    print(f"Ejecutando consulta ADQL para '{nombre_objeto}'...")
    
    try:
        # Ejecutar la consulta TAP a través de Astroquery.simbad
        resultado = Simbad.query_tap(query)
        
        if resultado is not None and len(resultado) > 0:
            fila = resultado[0]
            main_id = fila['main_id'].decode('utf-8') if isinstance(fila['main_id'], bytes) else fila['main_id']
            ra_str = fila['ra']
            dec_str = fila['dec']
            
            print(f"\n[Éxito] Objeto principal en SIMBAD: {main_id}")
            print(f"  - RA (sexagesimal): {ra_str}")
            print(f"  - DEC (sexagesimal): {dec_str}")
            
            # Convertir las coordenadas de texto (formato sexagesimal o grados según devuelva TAP)
            # SIMBAD TAP por defecto suele devolver RA/DEC en formato string o float en grados.
            # Si vienen en formato texto o grados, SkyCoord lo maneja robustamente:
            coord = SkyCoord(f"{ra_str} {dec_str}", unit=(u.deg, u.deg) if isinstance(ra_str, (float, int)) else (u.hourangle, u.deg))
            
            print(f"  - RA (grados): {coord.ra.deg:.6f}°")
            print(f"  - DEC (grados): {coord.dec.deg:.6f}°")
            
            return coord
        else:
            print(f"No se encontraron resultados para el objeto '{nombre_objeto}'.")
            return None
            
    except Exception as e:
        print(f"Error al conectar con el servicio TAP de SIMBAD: {e}")
        return None

if __name__ == "__main__":
    objeto_objetivo = "HD 73773"
    coordenadas = consultar_simbad_tap(objeto_objetivo)
