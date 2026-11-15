#!/usr/bin/env python3
import argparse
import sys
from astropy.io import fits


def parse_value(val_str):
  """Intenta convertir el valor ingresado al tipo de dato adecuado (int, float o bool).

  Si no es ninguno de ellos, lo mantiene como string.
  """
  # Caso Booleano
  if val_str.upper() == "TRUE":
    return True
  if val_str.upper() == "FALSE":
    return False

  # Caso Entero
  try:
    return int(val_str)
  except ValueError:
    pass

  # Caso Flotante
  try:
    return float(val_str)
  except ValueError:
    pass

  # Si no es numérico ni booleano, se queda como texto
  return val_str


def main():
  parser = argparse.ArgumentParser(
      description="Modifica o crea un KEY en el header de un archivo FITS."
  )
  parser.add_argument("fits_file", help="Ruta al archivo FITS")
  parser.add_argument(
      "key", help="Nombre del KEY/Clave a modificar o añadir"
  )
  parser.add_argument("value", help="Nuevo valor a escribir en el KEY")
  parser.add_argument(
      "--hdu",
      type=int,
      default=0,
      help="Índice del HDU a modificar (por defecto 0)",
  )

  args = parser.parse_args()

  # Convertir el valor a su tipo correspondiente (int, float, bool o str)
  value = parse_value(args.value)

  try:
    # Abrir en modo 'update' para poder modificar el archivo directamente
    with fits.open(args.fits_file, mode="update") as hdul:
      header = hdul[args.hdu].header

      # Guardar el valor previo si existe para informar al usuario
      old_value = header.get(args.key.upper(), None)

      # Asignar el nuevo valor (astropy crea la clave si no existe)
      header[args.key.upper()] = value

      # Guardar los cambios
      hdul.flush()

      if old_value is not None:
        print(
            f"✓ KEY '{args.key.upper()}' actualizado: '{old_value}' ->"
            f" '{value}' ({type(value).__name__})"
        )
      else:
        print(
            f"✓ KEY '{args.key.upper()}' creado con valor: '{value}'"
            f" ({type(value).__name__})"
        )

  except FileNotFoundError:
    print(
        f"Error: No se encontró el archivo '{args.fits_file}'.", file=sys.stderr
    )
    sys.exit(1)
  except Exception as e:
    print(f"Error procesando el archivo FITS: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
