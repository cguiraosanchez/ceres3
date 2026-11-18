from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

def obtener_coordenadas_simbad(nombre_objeto, mjd):
    """
    Busca un objeto en Gaia DR3 a través de VizieR probando múltiples 
    espejos internacionales para evitar errores 404 del servidor principal.
    """
    # Lista de espejos oficiales de VizieR
    servidores_vizier = [
        'vizier.cfa.harvard.edu', 
        'vizier.nao.ac.jp', 
        'vizier.cds.unistra.fr'
    ]
    resultado = None
    
    print(f"Buscando '{nombre_objeto}' en Gaia DR3 a través de VizieR...")
    
    for servidor in servidores_vizier:
        print(f" -> Intentando conexión con el servidor VizieR: {servidor}")
        try:
            # Configurar el servidor espejo actual
            Vizier.VIZIER_SERVER = servidor
            
            # Consultar Gaia DR3
            v = Vizier(columns=['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE'], row_limit=1)
            resultado = v.query_object(nombre_objeto, catalog="I/355/gaiadr3")
            
            if resultado is not None and len(resultado) > 0 and len(resultado[0]) > 0:
                print(f"    [Éxito] Conexión establecida con {servidor}\n")
                break
        except Exception as e:
            print(f"    [Aviso] Falló la conexión con {servidor}: {e}")
            continue
            
    # Procesar el resultado si se obtuvo con éxito en algún espejo
    if resultado is not None and len(resultado) > 0 and len(resultado[0]) > 0:
        tabla = resultado[0]
        fila = tabla[0]
        
        ra_deg = float(fila['RAJ2000'])
        dec_deg = float(fila['DEJ2000'])
        
        pmra = float(fila['pmRA']) if not np.ma.is_masked(fila['pmRA']) else 0.0
        pmdec = float(fila['pmDE']) if not np.ma.is_masked(fila['pmDE']) else 0.0
        
        print(f"[Éxito] Objeto localizado")
        print(f"  - RA (base J2000): {ra_deg:.6f}°")
        print(f"  - DEC (base J2000): {dec_deg:.6f}°")
        print(f"  - Movimiento propio (pmRA*cosDEC): {pmra} mas/yr")
        print(f"  - Movimiento propio (pmDEC): {pmdec} mas/yr")
        
        # Crear SkyCoord base en J2000.0 (MJD 51544.5) con su movimiento propio
        coord_base = SkyCoord(
            ra=ra_deg * u.deg, 
            dec=dec_deg * u.deg,
            pm_ra_cosdec=pmra * u.mas / u.yr,
            pm_dec=pmdec * u.mas / u.yr,
            epoch=2000.0,
            frame='icrs'
        )
        
        # Calcular intervalo de tiempo en años desde J2000.0 hasta el MJD indicado
        mjd_j2000 = 51544.5
        dt_anos = (mjd - mjd_j2000) / 365.25
        
        # Aplicar corrección por movimiento propio usando Astropy
        coord_corregida = coord_base.apply_space_motion(dt=dt_anos * u.yr)
        
        print(f"\n[Coordenadas corregidas para MJD {mjd}]")
        print(f"  - RA (grados): {coord_corregida.ra.deg:.6f}°")
        print(f"  - DEC (grados): {coord_corregida.dec.deg:.6f}°")
        print(f"  - RA (sexagesimal): {coord_corregida.ra.to_string(unit=u.hourangle, sep=':', precision=3)}")
        print(f"  - DEC (sexagesimal): {coord_corregida.dec.to_string(unit=u.deg, sep=':', precision=2)}")
        
        return coord_corregida
    else:
        print(f"\n[Error] No se pudo recuperar información para '{nombre_objeto}' en ningún espejo de VizieR.")
        return None

if __name__ == "__main__":
    objeto = "HD 73773"
    mjd_ejemplo = 61000.0  
    coordenadas = obtener_coordenadas_simbad(objeto, mjd_ejemplo)