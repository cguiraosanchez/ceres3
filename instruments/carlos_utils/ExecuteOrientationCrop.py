#!/usr/bin/python3

"""
Date: 20.09.2026
Author: Carlos Guirao

The WHAPOS detector (Sony IMX571) generate images of 3162x2088 pixels, while the spectrum is limited to a
sub-image of (aprox) 1756x1636 pixels somewhere centered in the larger image.

Also, the image might need to be rotated, tilted or tipped to be used by the pipeline system.

Thss script provides two funcions:
   1- First, it compares the image of a Thorium-Argon lamp with a reference full image of a Th-Ar file 
      to determine if files in the same directory requires to be manipulated (rotated, tilted or tipped). 
      If the files requires manipulation, the script overwrites the original files.
   2- Second, it compares the image of the same Thorium-Argon lamp as in step 1, with a reference image of a 
      ThAr already cropped. It identifies the size (height and width) and X,Y initial pixel for the crop operation
      The cropped files are uploade under the subdirectory "<initial_dir>/cropped
"""

import sys
import os
import glob
import numpy as np
from pathlib import Path
from astropy.io import fits
from skimage.feature import match_template

TRANSFORMATIONS = [
    {"rot": 0, "flip_lr": False, "flip_ud": False, "desc": "No changes"},
    {"rot": 1, "flip_lr": False, "flip_ud": False, "desc": "Rotate 90° CCW"},
    {"rot": 2, "flip_lr": False, "flip_ud": False, "desc": "Rotate 180° CW"},
    {"rot": 3, "flip_lr": False, "flip_ud": False, "desc": "Rotate 270° CCW (90° CW)"},
    {"rot": 0, "flip_lr": True,  "flip_ud": False, "desc": "Flip mirror horizontal"},
    {"rot": 0, "flip_lr": False, "flip_ud": True,  "desc": "Flip mirror vertical"},
    {"rot": 1, "flip_lr": True,  "flip_ud": False, "desc": "Rotate 90° CCW + Flip horizontal"},
    {"rot": 1, "flip_lr": False, "flip_ud": True,  "desc": "Rotate 90° CCW + Flip vertical"},
]

def aply_transformation(img, trans):
    """Aplica rotación y volteos a una matriz 2D."""
    res = np.copy(img)
    if trans["rot"] > 0:
        res = np.rot90(res, trans["rot"])
    if trans["flip_lr"]:
        res = np.fliplr(res)
    if trans["flip_ud"]:
        res = np.flipud(res)
    return res

def evaluate_similarity(ref, test):
    """Calcula la correlación de Pearson entre dos imágenes."""
    if ref.shape != test.shape:
        return -1.0
    
    ref_centered = ref - np.mean(ref)
    test_centered = test - np.mean(test)
    
    norm = np.sqrt(np.sum(ref_centered**2) * np.sum(test_centered**2))
    if norm == 0:
        return 0.0
    
    return np.sum(ref_centered * test_centered) / norm

def find_orientation(fits_ref_path, fits_test_path, ext=0):
    data_ref = fits.getdata(fits_ref_path, ext=ext).astype(float)
    data_test = fits.getdata(fits_test_path, ext=ext).astype(float)
    
    data_ref = np.nan_to_num(data_ref)
    data_test = np.nan_to_num(data_test)
    
    best_score = -np.inf
    best_trans = None
    
    print("\nEvaluating the 8 possible orientations...\n")
    
    for idx, trans in enumerate(TRANSFORMATIONS):
        img_transformed = aply_transformation(data_test, trans)
        
        if img_transformed.shape != data_ref.shape:
            print(f"[{idx+1}/8] Transformation: {trans['desc']} --> Descartada por dimensiones ({img_transformed.shape} != {data_ref.shape})")
            continue
            
        score = evaluate_similarity(data_ref, img_transformed)
        print(f"[{idx+1}/8] Transformation: {trans['desc']} --> Score Similitud: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_trans = trans

    print("\n" + "="*50)
    if best_trans is not None:
        print(f"FINAL RESULTS (Score máximo: {best_score:.4f}):")
        print(f"To align the test image with reference image you must do:")
        print(f" -> {best_trans['desc']}")
    else:
        print("ERROR: No orientation found compatible in dimensions.")
    print("="*50)
    
    return best_trans

def transform_and_save_fits(filepath, trans, output_prefix="corr_"):
    """Load a file FITS, apply de transformation and save it."""
    with fits.open(filepath) as hdul:
        # Assuming data is in HDU primary (extensión 0)
        data = hdul[0].data
        if data is not None:
            # Detect the type IEEE original (p. ej. float32 o float64)
            original_dtype = data.dtype
            
            # If data were not IEEE, force them to float32 (IEEE single precision)
            if not np.issubdtype(original_dtype, np.floating):
                original_dtype = np.float32

            #transformed_data = aply_transformation(data.astype(float), trans)
            transformed_data = aply_transformation(data, trans).astype(original_dtype, copy=False)
            
            # Create new header/HDU preserving original header
            new_hdu = fits.PrimaryHDU(data=transformed_data, header=hdul[0].header)
            
            dirname, filename = os.path.split(filepath)
            out_path = os.path.join(dirname, f"{output_prefix}{filename}")
            
            new_hdu.writeto(out_path, overwrite=True)
            print(f"  [OK] Proccessed and saved: {out_path}")

            # Overwrite the original file
            #new_hdu.writeto(filepath, overwrite=True)
            #print(f"  [OK] Proccessed and saved: {filepath}")
        else:
            print(f"  [SKIP] El file {filepath} does not contain an bi-demensional image.")

def crop_and_save_fits(fpath, y_offset, x_offset, height, width, dirout):

    path = Path(fpath)
    name = path.name
    # Rename .fit/.FIT  to .fits/.FITs
    if name.lower().endswith(".fit"): 
        name = name +  "s"

    data_a = fits.getdata(fpath).astype(np.float32)

    # 5. Execute the exact crop over the file
    data_a_cropped = data_a[y_offset : y_offset + height_b, x_offset : x_offset + width_b]

    # 6. Save the cropped image preserving the original header (without suffix "corr_")
    with fits.open(fpath) as hdul_b:
        header_b = hdul_b[0].header

    name = name.removeprefix("corr_")
    path_cropped_file = dirout + '/' + name
    fits.writeto(path_cropped_file, data_a_cropped, header=header_b, overwrite=True)
    print(f"  [OK] Cropped and saved {path_cropped_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: please provide the path to a ThAr_image FITS file to test its orientation.")
        print(f"Use: {sys.argv[0]} <path_to_file>")
        sys.exit(1)
        
    filename = sys.argv[1]
    ref_file = './whapos_ThAr_full_reference.fits'

    if os.path.isdir(filename):
        print(f"Error: '{filename}' is a directory.")
        print("Error: please provide the path to a ThAr_image FITS file to test its orientation.")
        sys.exit(1)
    
    if not os.access(ref_file, os.F_OK):
        print(f"\tNo reference file: {ref_file}")
        sys.exit(1)
        
    # Validate if filename is a FITS file by reading its header
    try:
        with fits.open(filename) as hdul:
            print(f"GOOD: '{filename}' is a valid FITS file.")
            print(f"Number of HDUs: {len(hdul)}")
    except Exception as e:
        print(f"Error: '{filename}' is not a valid FITS file or is damaged.")
        print(f"Details: {e}")
        sys.exit(1)

    print("Evaluating the orientation of the ThAr image by comparing it with the reference file:")
    print(f"\t{ref_file}")
    best_trans = find_orientation(ref_file, filename)
    
    # Check if the best transformation requires changes
    is_identity = (best_trans is not None and 
                   best_trans["rot"] == 0 and 
                   not best_trans["flip_lr"] and 
                   not best_trans["flip_ud"])
    
    if best_trans is not None and not is_identity:
        print(f"\nThe image requires manipulation ({best_trans['desc']}).")
        
        # Ask operator
        answer = input("\nDo you want to apply the same transformation to ALL images FITS in the same directory? [y/N]: ")
        
        if answer.lower() in ['y', 'yes']:
            target_dir = os.path.dirname(os.path.abspath(filename))
            if not target_dir:
                target_dir = '.'
                
            print(f"\nSearching FITS files in: {target_dir}")
            
            # Find extensiones .fits y .FITS
            fits_files = glob.glob(os.path.join(target_dir, "*.fit")) + \
                         glob.glob(os.path.join(target_dir, "*.FIT")) 
            
            # Rename .fit to .fits and .FIT to .fits (required by the pipeline)
            for file in fits_files:
                 # change the last 4 characters ('.fit') por '.fits'
                 new_name = file[:-4] + ".fits"
                 os.rename(file, new_name)
                 print(f"   [OK] Renamed: {new_name}")

            # Rename the argument filename to filename+s
            if filename.endswith(".fit") and not filename.endswith(".fits"):
                filename = filename.removesuffix(".fit") + ".fits"

            # Find extensiones .fits y .FITS
            fits_files = glob.glob(os.path.join(target_dir, "*.fits")) + \
                         glob.glob(os.path.join(target_dir, "*.FITS"))

            # Create a subdirectory with the new oriented files (corr) 
            # Do not proccess files already corrected (prefix corr_)
            ref_abs_path = os.path.abspath(ref_file)
            fits_files = [f for f in fits_files 
                          if not os.path.basename(f).startswith("corr_") 
                          and os.path.abspath(f) != ref_abs_path]
            
            print(f"\nI've found {len(fits_files)} files to be processed:\n")
            
            for fpath in fits_files:
                transform_and_save_fits(fpath, best_trans)
                
            print("\nTransformation in directory finished!")
        else:
            print("Operation canceled by user.")
            sys.exit()
    else:
        if is_identity:
            print("\nThe image already has the correct orientation (No changes). No transformation required.")
        else:
            print("\nNo transformation executed due to no matching dimensions.")


    # Request to continue with the crop process

    if is_identity:
        answer = input("\nDo you want to apply the crop transformation to ALL images FITS in the same directory? [y/N]: ")
    else: 
        answer = input("\nDo you want to apply the crop transformation to the corrected FITS images in the same directory? [y/N]: ")

    if answer.lower() in ['y', 'yes']:
        target_dir = os.path.dirname(os.path.abspath(filename))
        if not target_dir:
           target_dir = '.'

        dirout = os.path.join(target_dir, 'cropped')
        if not os.access(dirout,os.F_OK):
            os.makedirs(dirout, exist_ok=True)
    else:
        print("Operaction canceled by user.")
        sys.exit()
         
    
    # Detect Crop area and request to continue

    # Cropped reference file 
    cropped_ref_file = './whapos_ThAr_reference.fits'


    # 1. Load the images to be compared. data_a target, data_b reference
    data_a = fits.getdata(filename).astype(np.float32)
    data_b = fits.getdata(cropped_ref_file).astype(np.float32)

    # 2. Find the position of B in A using match_template
    #    It returns a matrix of correlation values between -1.0 and 1.0
    result = match_template(data_a, data_b)

    # 3. Find the point of maximal match
    ij = np.unravel_index(np.argmax(result), result.shape)
    y_offset, x_offset = ij  # Coordinates of the upper left corner [Y, X]

    print(f"Spectrum {cropped_ref_file} identified in \n\t{filename} in: \n\tY = {y_offset}, X = {x_offset}")

    # 4. Obtain the dimensions of reference file
    height_b, width_b = data_b.shape
    print(f"\theight = {height_b}, width = {width_b}")

    # 5. Find extensiones .fits y .FITS
    # Use the original files if they were already corrected
    if is_identity:
        fits_files = glob.glob(os.path.join(target_dir, "*.fits")) + \
            glob.glob(os.path.join(target_dir, "*.FITS")) + \
            glob.glob(os.path.join(target_dir, "*.fit")) + \
            glob.glob(os.path.join(target_dir, "*.FIT")) 
    else:
        # Only those already corrected
        fits_files = glob.glob(os.path.join(target_dir, "corr_*.fits"))
        
    print(f"\nI've found {len(fits_files)} files to be processed:\n")
            
    for fpath in fits_files:
        crop_and_save_fits(fpath, y_offset, x_offset, height_b, width_b, dirout)
