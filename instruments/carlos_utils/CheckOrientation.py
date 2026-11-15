#!/usr/bin/python3

import sys
import os
import numpy as np
from astropy.io import fits as pyfits
from scipy.signal import fftconvolve

# Definition of the 8 transformations of the orientation in 2D 
TRANSFORMATIONS = [
    {"rot": 0, "flip_lr": False, "flip_ud": False, "desc": "No changes"},
    {"rot": 1, "flip_lr": False, "flip_ud": False, "desc": "Rotate 90° CCW"},
    {"rot": 2, "flip_lr": False, "flip_ud": False, "desc": "Rotate 180° CW"},
    {"rot": 3, "flip_lr": False, "flip_ud": False, "desc": "Rotate 270° CCW (90° CW)"},
    {"rot": 0, "flip_lr": True,  "flip_ud": False, "desc": "Flix mirror horizontal"},
    {"rot": 0, "flip_lr": False, "flip_ud": True,  "desc": "Flip mirror vertical"},
    {"rot": 1, "flip_lr": True,  "flip_ud": False, "desc": "Rotate 90° CCW + Flip horizontal"},
    {"rot": 1, "flip_lr": False, "flip_ud": True,  "desc": "Rotate 90° CCW + Flip vertical"},
]

def aply_transformation(img, trans):
    """Aply rotate and flips in a matriz 2D."""
    res = np.copy(img)
    if trans["rot"] > 0:
        res = np.rot90(res, trans["rot"])
    if trans["flip_lr"]:
        res = np.fliplr(res)
    if trans["flip_ud"]:
        res = np.flipud(res)
    return res

def evaluate_similarity(ref, test):
    """Calculate the coeficient of the Pearson correlation normalized between two images."""
    if ref.shape != test.shape:
        return -1.0  # dimensions do not match after la transformation
    
    ref_centered = ref - np.mean(ref)
    test_centered = test - np.mean(test)
    
    norm = np.sqrt(np.sum(ref_centered**2) * np.sum(test_centered**2))
    if norm == 0:
        return 0.0
    
    return np.sum(ref_centered * test_centered) / norm

def find_orientation(fits_ref_path, fits_test_path, ext=0):
    """
    Compare fits_test con fits_ref and point to the  required operations in fits to 
    align fits_test with fits_ref 
    """
    # Load data FITS
    data_ref = pyfits.getdata(fits_ref_path, ext=ext).astype(float)
    data_test = pyfits.getdata(fits_test_path, ext=ext).astype(float)
    
    # Normalize basic cleaning (remoción de NaNs o Inf)
    data_ref = np.nan_to_num(data_ref)
    data_test = np.nan_to_num(data_test)

    print(f"Dimensiones Referencia : {data_ref.shape}")
    print(f"Dimensiones Imagen Test: {data_test.shape}\n")
    
    best_score = -np.inf
    best_trans = None
    
    print("Evaluating the 8 posible orientations...\n")
    
    for idx, trans in enumerate(TRANSFORMATIONS):
        img_transformed = aply_transformation(data_test, trans)
        
        # If dimensions after rotation 90/270 do not match with the reference, skip operation
        if img_transformed.shape != data_ref.shape:
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
        print("ERROR: Ninguna transformación produjo dimensiones compatibles con la imagen de referencia.")
    print("="*50)

    return best_trans

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No arguments provided.")
        sys.exit(1)
        
    filename = sys.argv[1]
    ref_file = './whapos_ThAr_full_reference.fits'
    
    if not os.access(ref_file, os.F_OK):
        print(f"\tNo reference file: {ref_file}")
        sys.exit(1)
        
    print("Evaluating the orientation of the ThAr image")
    result = find_orientation(ref_file, filename)
    print("result:",result)

