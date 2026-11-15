#!/usr/bin/python3

import sys
from pathlib import Path
from astropy.io import fits
from astropy.time import Time


def get_date_obs(filepath: Path) -> Time:
    """Extracts the 'DATE-OBS' keyword from a FITS header and converts it to an astropy Time object.

    Only loads the header metadata to optimize performance and reduce memory usage.
    """
    # Open header only to avoid loading heavy image/data arrays into memory
    with fits.open(filepath, mode="readonly") as hdul:
        # Search for DATE-OBS in the primary HDU (HDU 0)
        date_str = hdul[0].header.get("DATE-OBS")
        print("DATE-OBS:",date_str)

        # Fallback for multi-extension FITS (MEF) where DATE-OBS might be in HDU 1
        if date_str is None and len(hdul) > 1:
            date_str = hdul[1].header.get("DATE-OBS")

        if date_str is None:
            raise KeyError(
                f"Header keyword 'DATE-OBS' not found in {filepath.name}"
            )

        return Time(date_str)


def find_closest_fits(science_filepath: Path, search_directory: Path) -> tuple:
    """Finds the FITS file in search_directory whose DATE-OBS is closest to the reference science FITS file.

    Returns a tuple of (closest_filepath, time_difference_in_seconds).
    """
    science_time = get_date_obs(science_filepath)

    closest_file = None
    min_time_diff = None

    # Iterate over all .fits files in the directory
    for filepath in search_directory.glob("ThAr*.fits"):
        # Skip the reference science file if it is in the same directory
        if filepath.resolve() == science_filepath.resolve():
            continue

        try:
            candidate_time = get_date_obs(filepath)

            # Calculate absolute time difference in seconds
            time_diff = abs((candidate_time - science_time).sec)

            if min_time_diff is None or time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_file = filepath

        except KeyError:
            print(f"Skipping {filepath.name}: 'DATE-OBS' missing")
        except Exception as err:
            print(f"Error processing {filepath.name}: {err}")

    return closest_file, min_time_diff


if __name__ == "__main__":
    # Define file and directory paths
    #science_file = Path("/path/to/your/science_file.fits")
    science_file = Path(sys.argv[1])
    fits_directory = Path(sys.argv[2])

    # Locate nearest FITS file
    best_match, diff_seconds = find_closest_fits(science_file, fits_directory)

    # Print results
    if best_match:
        print(f"Closest FITS file: {best_match.name}")
        print(
            f"Time difference: {diff_seconds:.2f} seconds ({diff_seconds / 60.0:.2f} minutes)"
        )
    else:
        print("No valid FITS files with 'DATE-OBS' found.")
