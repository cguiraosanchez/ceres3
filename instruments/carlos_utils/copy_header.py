from astropy.io import fits


def copy_fits_header(source_file, target_file, output_file=None, hdu_index=0):
  """Copies the header from source_file to target_file while keeping

  the target file's data intact.
  """
  # 1. Read header from the source file
  with fits.open(source_file) as hdul_src:
    src_header = hdul_src[hdu_index].header.copy()

  # 2. Open the target file in update mode (or write to a new file)
  with fits.open(target_file) as hdul_tgt:
    target_data = hdul_tgt[hdu_index].data

    # Create a new HDU using the source header and target data.
    # Astropy automatically preserves target data structure (NAXIS, BITPIX, etc.)
    new_hdu = fits.PrimaryHDU(data=target_data, header=src_header)

    # 3. Save to output path or overwrite target file
    save_path = output_file if output_file else target_file
    new_hdu.writeto(save_path, overwrite=True)
    print(f"Header successfully copied from {source_file} to {save_path}")


# Example usage:
copy_fits_header(
    #source_file="2022-12-20T08_25_25.461507-HD72673.fits",
    #target_file="2022-12-19T23_22_30.431132-ThAr_60s_2F.fits",
    #output_file="2022-12-20T08_25_25.999999-HD72673.fits",  # Omit to overwrite target.fits
    source_file="source.fits",
    target_file="target.fits",
    output_file="output.fits",  # Omit to overwrite target.fits

)
