import tempfile
import requests
import numpy as np

def simbad_coords_test(obname, mjd=51544.5):
    """
    Versión robusta y corregida para pruebas de conexión con SIMBAD.
    """
    # Normalización de nombres (mantenida del original)
    name_mapping = {
        'alphacent': 'alpha cent', 'alphaboo': 'alpha boo',
        'hadar': 'beta cen', 'betacen': 'beta cen', 'diphda': 'bet cet',
        'betacar': 'beta car', 'betscl': 'bet scl', 'bvel': 'b vel',
        'deltasco': 'del sco', 'delcen': 'del cen', 'epsilonaqr': 'eps aqr',
        'epspsa': 'eps psa', 'etahya': 'eta Hya', 'ethahydra': 'eta Hya',
        'etapsa': 'eta psa', 'etacen': 'eta cen', 'opup': 'o Pup',
        'etacar': 'eta Car', 'agcar': 'ag Car', 'hrcar': 'hr Car',
        'sslep': 'ss lep', 'thetavir': 'theta vir', 'mucen': 'mu cen',
        'lesath': 'ups sco', 'mulup': 'mu lup', 'chioph': 'chi oph',
        'dlup': 'd lup', '48lib': '48 lib', 'iotara': 'iot ara',
        'qvtel': 'qv tel', 'taucet': 'tau cet', 'pi2ori': 'pi2 ori',
        'zetapeg': 'zet peg', 'tpyx': 't pyx', 'omicronpup': 'omi pup'
    }
    
    obname_lower = obname.lower().replace(" ", "")
    if obname_lower in name_mapping:
        obname = name_mapping[obname_lower]

    # Usar HTTPS y el endpoint oficial actual de Estrasburgo
    paths = ['https://simbad.u-strasbg.fr/simbad/sim-script']

    cond = True
    i = 0
    result = ""
    
    while cond:
        try:
            (th, tfile) = tempfile.mkstemp(prefix='CP', text=True)
            with open(tfile, 'w', encoding='utf-8') as tf:
                tf.write("output console=off\n")
                tf.write("output script=off\n")
                tf.write("output error=merge\n")
                tf.write("set limit 1\n")
                tf.write("format object fmt1 \"%IDLIST(1) | %OTYPELIST(S) | %SP(S) | %COO(A) | %COO(D) | %PM(A) | %PM(D)\"\n")
                tf.write("result full\n")
                tf.write(f"query id {obname}\n")

            with open(tfile, 'rb') as script_file:
                # Corregido: 'scriptFile' en minúsculas y timeout añadido para evitar bloqueos
                resp = requests.post(
                    paths[i],
                    files={"scriptFile": script_file},
                    timeout=10
                )
            
            # Lanzar excepción si hay error HTTP (4xx, 5xx)
            resp.raise_for_status()
            result = resp.text
            cond = False
        except Exception as e:
            print(f"Error conectando a {paths[i]}: {e}")
            cond = False # Salimos en el test para evitar bucles infinitos si la URL falla

    if not result:
        print("No se obtuvo respuesta de la API.")
        return None

    # Procesamiento de líneas de forma segura buscando la línea con los separadores '|'
    lines = result.split('\n')
    info = None
    for line in lines:
        if '|' in line and not line.startswith('::'):
            info = [x.strip() for x in line.split('|')]
            break

    if not info or len(info) < 7:
        print("La respuesta de SIMBAD no tiene el formato esperado:", result[:200])
        return 0, 0, 0, False

    print("Datos parseados correctamente:", info)
    
    if 'unrecognized' in info[0].lower() or 'not' in info[0].lower():
        know = False
        sp, ra, dec = 0, 0, 0
    else:
        know = True
        sp, ra, dec, pmra, pmdec = info[2], info[3], info[4], info[5], info[6]

        if '~' in pmra:
            pmra = '0.'
        if '~' in pmdec:
            pmdec = '0.'

        rad = ra.split()
        decd = dec.split()
        ra_val = float(rad[0])*360./24. + float(rad[1])*6./24. + float(rad[2])/240. + (float(pmra)/(3600*1000.))*((mjd-51544.5)/365.)
        
        if float(decd[0]) < 0:
            dec_val = -(np.absolute(float(decd[0])) + float(decd[1])/60. + float(decd[2])/3600.) + (float(pmdec)/(3600*1000.))*((mjd-51544.5)/365.)
        else:
            dec_val = float(decd[0]) + float(decd[1])/60. + float(decd[2])/3600. + (float(pmdec)/(3600*1000.))*((mjd-51544.5)/365.)
            
        sp, ra, dec = sp, ra_val, dec_val

    return sp, ra, dec, know

# --- Ejecución del Test ---
if __name__ == "__main__":
    print("Iniciando test de conexión con la API de SIMBAD...")
    
    # Probamos con un objeto astronómico de referencia conocido (ej. Sirius o Alpha Centauri)
    test_object = "Sirius"
    mjd_test = 60000.0 # Ejemplo de MJD
    
    sp, ra, dec, know = simbad_coords_test(test_object, mjd_test)
    
    print("\n--- Resultados del Test ---")
    print(f"Objeto: {test_object}")
    print(f"¿Conocido?: {know}")
    print(f"Tipo espectral (sp): {sp}")
    print(f"Ascensión Recta (ra): {ra}")
    print(f"Declinación (dec): {dec}")
    
    if know:
        print("\n ¡Test superado con éxito! La conexión y el parsing funcionan correctamente.")
    else:
        print("\n La conexión se estableció, pero el objeto no fue reconocido por SIMBAD.")