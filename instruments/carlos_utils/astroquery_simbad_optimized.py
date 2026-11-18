from astroquery.simbad import Simbad
import astropy.units as u
from astropy.coordinates import SkyCoord

def obtener_coordenadas_simbad(nombre_objeto):
    """
    Busca un objeto en SIMBAD intentando primero simbad.harvard.edu 
    y recurriendo a simbad.cds.unistra.fr si falla.
    """
    servidores = ['simbad.harvard.edu', 'simbad.cds.unistra.fr']
    resultado = None
    
    print(f"Buscando '{nombre_objeto}' en SIMBAD...")
    
    for servidor in servidores:
        print(f" -> Intentando conexión con el servidor: {servidor}")
        try:
            simbad = Simbad()
            simbad.server = servidor
            resultado = simbad.query_object(nombre_objeto)
            
            if resultado is not None and len(resultado) > 0:
                print(f"    [Éxito] Conexión establecida con {servidor}\n")
                break
        except Exception as e:
            print(f"    [Aviso] Falló la conexión con {servidor}: {e}")
            continue
            
    # Procesar el resultado si se obtuvo alguno
    if resultado is not None and len(resultado) > 0:
        main_id = resultado['main_id'][0]
        ra_str = resultado['ra'][0]
        dec_str = resultado['dec'][0]
        
        print(f"Objeto principal: {main_id}")
        print(f"  - RA (sexagesimal): {ra_str}")
        print(f"  - DEC (sexagesimal): {dec_str}")
        
        # Convertir a SkyCoord
        coord = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))
        
        print(f"  - RA (grados): {coord.ra.deg:.6f}°")
        print(f"  - DEC (grados): {coord.dec.deg:.6f}°")
        
        return coord
    else:
        print(f"\n[Error] No se pudo recuperar información para '{nombre_objeto}' en ningún servidor disponible.")
        return None

if __name__ == "__main__":
    #objeto = "HD72673"
    objeto = "HD 72737"
    coordenadas = obtener_coordenadas_simbad(objeto)
