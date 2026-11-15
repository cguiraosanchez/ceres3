# bachespipe V1 Date: 2026-09-10
# bachespipe V11 Date: 2026-09-11 Indent to 4 chars
# bachespipe V2 Date: 2026-09-20 
# bachespipe V3 Date: 2026-09-28  Cleaned. Choosing the closest in time ThAr file
# bachespipe V4 Date: 2026-10-09  with wavcals with official order numbers
# bachespipe V5 Date: 2026-10-09  Further developments
import sys
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import traceback

# ceres modules
import os
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_data_dir = os.path.join(_pkg_dir, 'instruments', 'data')
_xc_masks_dir = os.path.join(_pkg_dir, 'utils', 'data', 'xc_masks') + '/'

baryc_dir = ''
ephemeris = 'DEc403'

from ceres3.instruments import bachesutils
from ceres3.utils import correlation
from ceres3.ext import Marsh
from ceres3.utils import globalutils as GLOBALutils

# other useful modules
import argparse
import ephem
import glob
from ceres3.utils import jplephem
from math import radians as rad
from astropy.io import fits as pyfits
from astropy.time import Time
import pickle
import scipy
import scipy.interpolate
from scipy import interpolate, ndimage
import datetime
from pathlib import Path
import statsmodels.api as sm

lowess = sm.nonparametric.lowess

# Recive input parameters
parser = argparse.ArgumentParser()
parser.add_argument('directorio')
parser.add_argument('-avoid_plot', action="store_true", default=False)
parser.add_argument('-dirout',default='default')
parser.add_argument('-do_class', action="store_true", default=False)
parser.add_argument('-just_extract', action="store_true", default=False)
parser.add_argument('-npools', default=1)
parser.add_argument('-o2do',default='all')
parser.add_argument('-reffile',default='default')
parser.add_argument('-absref',default='none')
parser.add_argument('-just_bias', action="store_true", default=False)
parser.add_argument('-just_flats', action="store_true", default=False)
parser.add_argument('-just_thar', action="store_true", default=False)

args = parser.parse_args()
dirin            = args.directorio
avoid_plot       = args.avoid_plot
dirout           = args.dirout
DoClass          = args.do_class
JustExtract      = args.just_extract
npools           = int(args.npools)
object2do        = args.o2do
reffile          = args.reffile
absref           = args.absref

directory = Path(dirin)
if not directory.exists() or not directory.is_dir():
    print(f"\n\tError: the path {dirin} does not exist or is not a valid directory.")
    sys.exit()

if dirin[-1] != '/':
    dirin = dirin + '/'

dirin= dirin.replace('//','/')

if dirout == 'default':
    dirout = dirin[:-1]+'_red/'

if not os.access(dirout,os.F_OK):
    os.makedirs(dirout, exist_ok=True)

os.makedirs(dirout+'proc', exist_ok=True)

f_res = open(dirout+'proc/'+'results.txt','w')

if reffile == 'default':
    reffile = dirin+'reffile.txt'

####### GLOBAL VARIABLES #####
## perhaps put into options ##
force_pre_process  = False
force_sci_extract  = False
force_thar_extract = False
force_flat_extract = False
force_tharxc       = False
force_thar_wavcal  = False
force_spectral_file_build = True
force_stellar_pars = False
just_comp          = False
dumpargon          = False
minlines_glob      = 500

Inverse_m          = True
use_cheby          = True
MRMS               = 120   # max rms in m/s, global wav solution

order0             = 0
ncoef_x            = 4
ncoef_m            = 8
npar_wsol          = (min(ncoef_x,ncoef_m) + 1) * (2*max(ncoef_x,ncoef_m) - min(ncoef_x,ncoef_m) + 2) // 2
#trace_degree       = 4
trace_degree       = 3
Marsh_alg          = 0
ext_aperture       = 10
NSigma_Marsh       = 10
NCosmic_Marsh      = 10
S_Marsh            = 0.4
N_Marsh            = 4     # grado polinomio 
min_extract        = 0
max_extract        = 1550
mid_extract        = 775

order_dir   = "baches_wavcals/"
models_path = os.path.join(_pkg_dir, 'data', 'COELHO_MODELS', 'R_40000b/')    # path to the synthetic models 

#ronoise, gain = 6.82, 1.9
#ronoise and gain(ADU/e-) for the Moravian C3 26000 is  XXX, YYY 
ronoise, gain = 1.5, 1

log = dirout+'night.log'

just_bias      = args.just_bias
just_flats     = args.just_flats
just_thar      = args.just_thar

if just_bias or just_flats:
    force_pre_process = True

print("\n\n\tWHAPOS Baader-Planetarium PIPELINE\n")
print("\tRAW data is in ",dirin)
print("\tProducts of reduction will be in",dirout)
print('\n')

biases, darks, flats, flats_old, thars, thars_old, sim_sci, dar_sci = bachesutils.FileClassify(dirin,log)
print("\tfound biases:\t", len(biases))  
print("\tfound darks:\t", len(darks))  
print("\tfound flats:\t", len(flats))  
print("\tfound thars:\t", len(thars))  
print("\tfound science:\t", len(sim_sci))  
print('\n')

if len(flats) == 0 or len(thars) == 0: 
    print("No flats or ThAr files found to work with")
    sys.exit()

if ( (os.access(dirout+'MasterFlat.fits',os.F_OK) == False) or \
    (os.access(dirout+'trace.pkl',os.F_OK) == False)  or \
    (os.access(dirout+'MasterBias.fits',os.F_OK) == False)  or \
    (force_pre_process) == True ):

    print("\tNo previous pre-processing files found, or pre-process forced...")
    pre_process = 1
else:
    print("\tPre-processing files found, going straight to extraction")
    pre_process = 0

trace_file = dirout+'trace.pkl'

if (pre_process == 1):
    if just_flats == False: 

        # If there biases:
        if len(biases)>1: 
            # Open the first biases[0] as hdu1 to retrieve the EGAIN value from header
            with pyfits.open(biases[0]) as hdu1:
                header = hdu1[0].header
                # Check if the keyword 'EGAIN' exist in header
                if 'EGAIN' in header:
                    # Retrieve the value of 'EGAIN' in gain (as float32 value)
                    gain = float(header['EGAIN'])
            MasterBias = GLOBALutils.MedianCombine_simple(biases,ZF=0.)
            ronoise = np.around(bachesutils.compute_RON(MasterBias,biases,gain=gain),1)

        # If there are no bias, fake one with the flats and with data values set to 0
        if len(biases) == 0 and len(flats)>1: 
            # Open the first flats[0] as hdu
            with pyfits.open(flats[0]) as hdu1:
                header = hdu1[0].header
                # Check if the keyword 'EGAIN' exist in header
                if 'EGAIN' in header:
                    # Retrieve the value of 'EGAIN' in gain (as float value)
                    gain = float(header['EGAIN'])
            MasterBias = GLOBALutils.MedianCombine_simple(flats,ZF=0.)

        hdu = pyfits.PrimaryHDU( MasterBias )
        mbdate = str(datetime.datetime.now()).replace(' ','T')
        print('\n\t\tRON:', ronoise,'e')
        print('\t\tGAIN:', gain)
        hdu = GLOBALutils.update_header(hdu,'RON',ronoise)
        hdu = GLOBALutils.update_header(hdu,'EGAIN',gain)
        hdu = GLOBALutils.update_header(hdu,'DATE',mbdate)
        hdu.data = hdu.data.astype(np.float32)
        hdu.writeto(dirout+'MasterBias.fits',overwrite=True)
        print("\t\t-> Masterbias: done!")

    # Read the just/previous created MasterBias
    MasterBias = pyfits.open(dirout+'MasterBias.fits')
    ronoise = MasterBias[0].header['RON']

    if just_bias == False:
        MasterFlat = GLOBALutils.MedianCombine_simple(flats,ZF=MasterBias[0].data)
        mfdate = str(datetime.datetime.now()).replace(' ','T')
        MasterFlat = np.fliplr(MasterFlat.T)
        MasterFlat = np.ascontiguousarray(MasterFlat) # Improve RAM speed
        hdu = pyfits.PrimaryHDU( MasterFlat )
        hdu = GLOBALutils.update_header(hdu,'BDATE',MasterBias[0].header['DATE'])
        hdu = GLOBALutils.update_header(hdu,'DATE',mfdate)
        hdu = bachesutils.fill_list_header(hdu,flats)
        if (os.access(dirout+'MasterFlat.fits',os.F_OK)):
            os.remove(dirout+'MasterFlat.fits')
        hdu.data = hdu.data.astype(np.float32)
        hdu.writeto(dirout+'MasterFlat.fits')

        print("\t\t-> Masterflats: done!")

        print("\t\t-> Tracing orders...")
        #c, nord = GLOBALutils.get_them(MasterFlat.T, 18.,trace_degree,maxords=-1,nsigmas=5,mode=1)
        c, nord = GLOBALutils.get_them(MasterFlat.T, ext_aperture,trace_degree,maxords=-1,nsigmas=5,mode=1)
        print("\t\t-> Orders found in flat:", nord)
        c, nord = bachesutils.good_orders(c,nord,MasterFlat.shape[1],MasterFlat.shape[0],ext_aperture)
        print("\t\t-> Good orders found:", nord)

        trace_dict = {'c':c,'nord':nord,'bdate':hdu.header['BDATE'],'fdate':hdu.header['DATE']}
        pickle.dump(trace_dict, open(trace_file, 'wb'))

        trace_dict = pickle.load(open(trace_file, 'rb'), encoding='latin1')
        c    = trace_dict['c']
        nord = len(c)
        MasterBias = pyfits.open(dirout+'MasterBias.fits')
        ronoise = MasterBias[0].header['RON']
        MasterFlat = pyfits.open(dirout+'MasterFlat.fits')
else:
    #trace_dict = {'c':c,'nord':nord,'bdate':hdu.header['BDATE'],'fdate':hdu.header['DATE']}
    trace_dict = pickle.load(open(trace_file, 'rb'), encoding='latin1')
    c    = trace_dict['c']
    nord = len(c) 
    MasterBias = pyfits.open(dirout+'MasterBias.fits')
    ronoise = MasterBias[0].header['RON']
    MasterFlat = pyfits.open(dirout+'MasterFlat.fits')


if just_bias:
    sys.exit()

if len(darks)>1:
    dtimes = GLOBALutils.get_dark_times(darks,key='EXPTIME')
    for dtime in dtimes:
        tdarks = GLOBALutils.get_tdarks(darks,dtime,key='EXPTIME')
        dark   = GLOBALutils.MedianCombine_simple(tdarks,ZF=MasterBias[0].data)
        hdu = pyfits.PrimaryHDU( dark )
        if (os.access(dirout+'Dark_'+str(dtime)+'.fits',os.F_OK)):
            os.remove(dirout+'Dark_'+str(dtime)+'.fits')
        hdu.data = hdu.data.astype(np.float32)
        hdu.writeto(dirout+'Dark_'+str(dtime)+'.fits')
    print("\t\t-> MasterDarks: done!")


P_fits = dirout + 'P.fits'
S_flat_fits = dirout +'S_flat.fits'
S_flat_simple_fits = dirout +'S_flat_simple.fits'
#force_flat_extract=True

if ( os.access(P_fits,os.F_OK) == False ) or \
    ( os.access(S_flat_fits,os.F_OK) == False ) or \
    ( os.access(S_flat_simple_fits,os.F_OK) == False ) or \
    (force_flat_extract) or just_flats:

    print('\n\tExtracting Flats...')
    ejx = np.arange(MasterFlat[0].data.shape[0])

    Centers = np.zeros((nord,MasterFlat[0].data.shape[0]))
    for i in range(nord):
        Centers[i,:]=np.polyval(c[i],ejx)
    bac = GLOBALutils.get_scat(MasterFlat[0].data.T,Centers,span=8)

    hdu = pyfits.PrimaryHDU(bac)
    if os.access(dirout +'BKG_flat.fits',os.F_OK):
        os.remove(dirout +'BKG_flat.fits')
    hdu.data = hdu.data.astype(np.float32)
    hdu.writeto(dirout +'BKG_flat.fits')

    Flat = MasterFlat[0].data.T - bac

    P = GLOBALutils.obtain_P(Flat, c, ext_aperture, ronoise,\
        gain,NSigma_Marsh, S_Marsh, \
        N_Marsh, Marsh_alg, min_extract,\
        max_extract, npools)

    S_flat_simple = GLOBALutils.simple_extraction(Flat,c,ext_aperture,min_extract,\
        max_extract,npools)
    S_flat  = GLOBALutils.optimal_extraction(Flat,P,c,ext_aperture,ronoise,gain,S_Marsh,\
        NCosmic_Marsh,min_extract,max_extract,npools)
    sfdate = str(datetime.datetime.now()).replace(' ','T')

    S_flat_simple = GLOBALutils.invert(S_flat_simple)
    S_flat = GLOBALutils.invert(S_flat)

    if (os.access(P_fits,os.F_OK)):
        os.remove( P_fits )

    hdu = pyfits.PrimaryHDU( P )
    hdu.data = hdu.data.astype(np.float32)
    hdu.writeto( P_fits )

    if (os.access(S_flat_fits,os.F_OK)):
        os.remove( S_flat_fits )
    if (os.access(S_flat_simple_fits,os.F_OK)):
        os.remove( S_flat_simple_fits )

    hdu = pyfits.PrimaryHDU( S_flat )
    hdu = GLOBALutils.update_header(hdu,'FDATE',MasterFlat[0].header['DATE'])
    hdu = GLOBALutils.update_header(hdu,'DATE',sfdate)
    hdu.data = hdu.data.astype(np.float32)
    hdu.writeto( S_flat_fits )
    hdu = pyfits.PrimaryHDU( S_flat_simple )
    hdu.data = hdu.data.astype(np.float32)
    hdu.writeto( S_flat_simple_fits )

print("\tExtracted flat comparison spectra found, loading...\n")
P       = pyfits.getdata( P_fits )
S_flat  = pyfits.open( S_flat_fits )
S_flat_simple = pyfits.getdata( S_flat_simple_fits )

S_flat_n, S_flat_simple_n = GLOBALutils.FlatNormalize( S_flat[0].data, S_flat_simple)

if just_flats:
   sys.exit()

reffiles = []

thar_times = []
for fsim in thars:
    print("\tthars: ",fsim)
    hthar = pyfits.open( fsim )
    dthar = pyfits.getdata( fsim ) - MasterBias[0].data
    dthar = np.fliplr(dthar.T)
    dthar = np.ascontiguousarray(dthar)  # Improve RAM speed
    mjd, mjd0 = bachesutils.mjd_fromheader( hthar )
    thar_times.append(mjd)
    thar_fits_simple = dirout + Path(fsim).stem + '.spec.simple.fits.S'
    thar_fits = dirout + Path(fsim).stem + '.spec.fits.S'
    bac_fits_thar = dirout + 'BKG_'+ Path(fsim).stem + '.fits'
    if ( os.access(thar_fits_simple,os.F_OK) == False ) or (force_thar_extract) or (just_thar):
        print("\tNo previous extraction or extraction forced for ThAr file", fsim, "extracting...")
        bac = bachesutils.get_scat(dthar.astype('float32'), P.astype('float32'),1)
        dthar -= bac
        thar_Ss_simple  = GLOBALutils.simple_extraction(dthar.T,c,ext_aperture,min_extract,\
             max_extract,npools)
        thar_Ss  = GLOBALutils.optimal_extraction(dthar.T,P,c,ext_aperture,ronoise,gain,S_Marsh,\
             NCosmic_Marsh,min_extract,max_extract,npools)
        thdate = str(datetime.datetime.now()).replace(' ','T')
    
        thar_Ss_simple = GLOBALutils.invert(thar_Ss_simple)
        thar_Ss = GLOBALutils.invert(thar_Ss)

        if (os.access(thar_fits,os.F_OK)):
            os.remove( thar_fits )
        hdu = pyfits.PrimaryHDU( thar_Ss )
        hdu = GLOBALutils.update_header(hdu,'BDATE',MasterBias[0].header['DATE'])
        hdu = GLOBALutils.update_header(hdu,'DATE',thdate)
        hdu.data = hdu.data.astype(np.float32)
        hdu.writeto( thar_fits )

        if (os.access(thar_fits_simple,os.F_OK)):
            os.remove( thar_fits_simple )
        hdu = pyfits.PrimaryHDU( thar_Ss_simple )
        hdu.data = hdu.data.astype(np.float32)
        hdu.writeto( thar_fits_simple )

thar_times = np.array(thar_times)
thars      = thars[np.argsort(thar_times)]
thar_times = thar_times[np.argsort(thar_times)]

print('\tSelection of echelle orders...')
force_thar_id = False
if os.access(dirout+'order_id.pkl', os.F_OK) == False or force_thar_id:
    print('\tUsing image:', thars[0])
    thar_fits = dirout + Path(thars[0]).stem + '.spec.fits.S'
    thar = pyfits.getdata(thar_fits)[:,1,:]
    # identify the order with the lines defined as in baches_64.iwdat
    or34, offset = GLOBALutils.identify_order(thar,order_dir+'baches_34.iwdat',window=100,di=0.5)

    start = or34 - 3
    end   = start + 19
    idorder0 = 34 - 3  # first start order "baches_{idorder0}.iwdat" or 34 - 3 

    print('\t\tWill start on order', start, 'for comparison fibre and finishes on order', end)
    print('\t\tOffset for comparison fibre is ', offset, 'pixels')
    thar_dict = { 'start':start, 'end':end, 'offset':offset, 'idorder0':idorder0 }
    pickle.dump(thar_dict, open(dirout+'order_id.pkl', 'wb'))
else:
    print('\t\tValues loaded from file...')
    thar_dict   = pickle.load(open(dirout+'order_id.pkl', 'rb'), encoding='latin1')
    start       = thar_dict['start']
    end         = thar_dict['end']
    offset      = thar_dict['offset']
    idorder0    = thar_dict['idorder0']
    
    print('\t\tWill start on order', start, 'for comparison fibre...')
    print('\t\tOffset for comparison fibre is ', offset, 'pixels')

ntot = end - start +1

# Calibrating all the Thorium-Argon files:
#force_thar_wavcal = True
for fsim in thars:
    thar_fits_simple = dirout + Path(fsim).stem + '.spec.simple.fits.S'
    thar_spec    = dirout + Path(fsim).stem + '.sp.fits'
    wavsol_pkl = dirout + Path(fsim).stem + '.wavsolpars.pkl'
    if ( os.access(wavsol_pkl,os.F_OK) == False ) or (force_thar_wavcal):
        print("\tWorking on initial ThAr file", fsim)
        hthar = pyfits.open( fsim )
        mjd, mjd0 = bachesutils.mjd_fromheader( hthar )
        thar_fits = dirout + Path(fsim).stem + '.spec.fits.S'
        thar_S = pyfits.open( thar_fits )

        lines_thar  = thar_S[0].data[:,1,:]
        iv_thar     = thar_S[0].data[:,2,:]

        All_Pixel_Centers = np.array([])
        All_Wavelengths   = np.array([])
        All_Orders        = np.array([])
        All_Centroids     = np.array([])
        All_Sigmas        = np.array([])
        All_Intensities   = np.array([])
    
        order = start
        idorder = idorder0
        print('\t\tWill start on order', start, 'and finishes on order', end)
        while order <= end:
            order_s = str(idorder)
            if (idorder < 10):
                order_s = '0'+str(idorder)
            
            thar_order_orig = lines_thar[order,:]/scipy.signal.medfilt(S_flat[0].data[order,1],31)
            IV              = iv_thar[order,:]
            wei             = np.ones(len(thar_order_orig))#np.sqrt( IV )
            bkg             = scipy.signal.medfilt(thar_order_orig,101)        
            thar_order      = thar_order_orig - bkg
            #IO = np.where(G_ord == idorder)[0]
            #print idorder
            #print pixel_centers
            #print wavelengths
            #print order_s


            # Next block performs the initial 1D wavelength calibration for a single echelle order using Thorium-Argon (ThAr) emission lines, 
            # and subsequently calculates a reference Zero Point wavelength for a specific echelle order.
            #
            # 1. 1D Wavelength Calibration (Initial_Wav_Calibration)
            #     Purpose: Identifies known ThAr emission lines in a 1D extracted arc spectrum (thar_order) 
            #         for the current order and fits a pixel-to-wavelength mapping polynomial.
            # Input Parameters:
            #     order_dir + 'baches_' + order_s + '.iwdat': Path to the reference catalog containing identified ThAr line 
            #         wavelengths and expected pixel positions for this order.
            #     thar_order: The 1D extracted flux array of the ThAr arc lamp for the current order.
            #     order: Echelle order index. 
            #     wei: Variance/inverse-variance weights array associated with the ThAr spectrum pixels.
            #     rmsmax=400: Maximum acceptable root-mean-square (RMS) scatter threshold (in m/s or relative units) for line position residuals.
            #     minlines=10: Minimum required number of identified emission lines to consider the 1D fit valid.
            #     FixEnds=False: Allows the endpoints of the polynomial to remain unconstrained by fixed boundary conditions.
            #     Dump_Argon=False: Disables dumping/filtering of specific Argon lines.
            #     Cheby=True: Specifies the use of Chebyshev polynomials (rather than standard power-series polynomials) for numeric stability during fitting.
            #     porder=ncoef_x: The degree of the polynomial along the dispersion (pixel) direction (N = ncoef_x).
            #     rough_shift=offset: Applies an initial cross-correlation or pixel offset to align observed ThAr lines with the reference catalog.
            #     do_xc=False: Disables internal cross-correlation re-alignment during line matching.
            #     line_width=6: Expected full width of emission lines in pixels for Gaussian fitting.
            #     pixelization=True: Accounts for pixel integration effects when fitting line profiles.
            # Returned Outputs:
            #     coeffs_pix2wav: Coefficients for the pixel-to-wavelength mapping polynomial (x -> lambda).
            #     coeffs_pix2sigma: Coefficients mapping pixel positions to spectral resolution/line widths (sigma).
            #     pixel_centers, wavelengths: Fitted pixel locations and matched catalog wavelengths (lambda) for detected ThAr lines.
            #     rms_ms, residuals: Fit quality metrics (RMS error and individual line residuals).
            #     centroids, sigmas, intensities: Gaussian profile parameters measured for each line.
            # 2. Zero Point Evaluation for Reference Order (Global_ZP)
            #    Purpose: Sets a global reference wavelength zero point when processing the 11th echelle order (idorder == 11 + 30).
            #    0.5 * (len(thar_order)) - 1: Evaluates the polynomial at the exact central pixel of the detector along the dispersion axis.
            #    GLOBALutils.Cheby_eval(...): Evaluates the Chebyshev polynomial coeffs_pix2wav at this central pixel index, 
            #        scaling the pixel coordinate appropriately across the domain [0, len(thar_order)]. 
            #    Global_ZP: Stores the central wavelength (lambda0) of order 11th, which serves as a global anchor for global 2D wavelength 
            #        solutions and velocity shift determinations across orders.
            coeffs_pix2wav, coeffs_pix2sigma, pixel_centers, wavelengths, rms_ms, residuals, centroids, sigmas, intensities \
                = bachesutils.Initial_Wav_Calibration( order_dir+'baches_'+order_s+'.iwdat', thar_order, order, wei, \
                rmsmax=400, minlines=10,FixEnds=False,Dump_Argon=False,Cheby=True,porder=ncoef_x,rough_shift = offset,do_xc=False,line_width=6,pixelization=True)
            if (idorder == 11 + 30): 
                Global_ZP = GLOBALutils.Cheby_eval( coeffs_pix2wav,  0.5*(len(thar_order))-1, len(thar_order) )

            All_Pixel_Centers = np.append( All_Pixel_Centers, pixel_centers )
            All_Wavelengths   = np.append( All_Wavelengths, wavelengths )
            All_Orders        = np.append( All_Orders, np.zeros( len(pixel_centers) ) + idorder )
            All_Centroids     = np.append( All_Centroids, centroids)
            All_Sigmas        = np.append( All_Sigmas, sigmas)
            All_Intensities   = np.append( All_Intensities, intensities )
            order += 1
            idorder += 1

        # Next code block performs the global 2D wavelength calibration fit across all echelle orders of the spectrum
        # using the previously detected Thorium-Argon (ThAr) emission lines.
        #
        # 1. Initialization of the Parameter Vector 
        #    p0 = np.zeros(npar_wsol): Creates an initial zero vector of size npar_wsol (the total number of 
        #         coefficients in the 2D surface polynomial).
        #    p0[0] = (11 + 30) * Global_ZP: Sets the initial seed value for the first fit coefficient ($C_{0,0}$):
        #         11: Represents the internal order index analyzed previously (idorder == 41).
        #         30: Represents order0 (the base physical echelle order number of the spectrograph). 
        #         Thus, (11 + 30) = 44 is the absolute physical order number.
        #    Global_ZP: The central wavelength of that reference order.
        #
        # Underlying Physics: In an echelle spectrograph, the fundamental grating equation dictates that the product 
        # of the physical order 'm' and the wavelength 'lambda' is roughly constant (m x lambda =~ is-a-constant). 
        # Multiplying the absolute physical order number by its central wavelength yields an accurate starting seed 
        # for the 2D surface fitting algorithm.
        #
        # 2. Fitting the 2D Global Wavelength Solution (GLOBALutils.Fit_Global_Wav_Solution)
        #    Purpose: Fits a 2D polynomial surface—typically mapped over pixel position 'x' and order 'm' (or 1/m)— 
        #    that maps detector positions and spectral orders to precise wavelengths lambda(x, m)$ across the entire CCD.
        #
        #    All_Pixel_Centers, All_Wavelengths, All_Orders: Combined arrays containing pixel centroids, catalog wavelengths, 
        #       and order numbers for all identified ThAr lines across all orders.
        #    np.ones(All_Intensities.shape): Applies uniform unit weighting to all ThAr lines.
        #    p0: The initial guess vector calculated in the previous step.
        #    Cheby=True: Employs 2D Chebyshev polynomials instead of standard power series polynomials 
        #        to maintain numerical stability during inversion.
        #    maxrms=MRMS: Outlier rejection threshold (clipping). Any emission lines whose residuals exceed this 
        #        maximum RMS value are iteratively removed.
        #    Inv=Inverse_m: When set to True, the cross-dispersion fit is performed against the inverse of the physical order number (1/m), 
        #        accurately modeling the physical dispersion of the echelle grating.
        #    minlines_glob=minlines_glob: Minimum required number of valid ThAr lines across the detector to accept the global solution.
        #    order0: Offset to convert relative order indices into absolute physical order numbers m (m = relative_order + 49).
        #    ntotal=ntot, npix=len(thar_order): Total number of processed orders and total number of pixels along the dispersion axis per order.
        #    nx=ncoef_x, nm=ncoef_m: Degrees of the polynomial fit along the dispersion/pixel axis (nx) and cross-dispersion/order axis (nm).
        #
        # Returned Values:
        #    p1: Optimized 2D polynomial coefficients vector.
        #    G_pix, G_ord, G_wav: Pixel positions, order numbers, and catalog wavelengths of the ThAr lines retained after outlier rejection.
        #    II: Indices of the accepted lines in the final calibration solution.
        #    rms_ms: Final global RMS calibration error (expressed in m/s or relative units).
        #    G_res: Individual line residuals relative to the fitted 2D global solution.
        p0 = np.zeros( npar_wsol )
        p0[0] =  (11+30) * Global_ZP 
        p1, G_pix, G_ord, G_wav, II, rms_ms, G_res = \
            GLOBALutils.Fit_Global_Wav_Solution(All_Pixel_Centers, All_Wavelengths, All_Orders,\
            np.ones(All_Intensities.shape), p0, Cheby=True,\
            maxrms=MRMS, Inv=Inverse_m,minlines=minlines_glob,order0=order0, \
            ntotal=ntot,npix=len(thar_order),nx=ncoef_x,nm=ncoef_m)

        # Purpose: This code block evaluates the fitted 2D wavelength solution across all extracted echelle orders to 
        #          produce the final calibrated ThAr reference array (thar_out).
        #
        # 1. Pre-allocation and Variable Initialization
        #    thar_out: Allocates a 3D output array with dimensions (2, nord, npix) to store the wavelength values and flux profiles for all orders:
        #       Index 0: Evaluated wavelengths (lambda) for each pixel.
        #       Index 1: Raw/extracted ThAr flux counts.
        #    equis: Generates a 1D array of pixel indices x = [0, 1, 2, ... npix-1] along the dispersion direction.
        # Tracking Indices:
        #    order: Current index in the input spectra matrix lines_thar.
        #    idorder: Internal order counter (starts at 11 for this specific spectral sub-range).
        #    out_order: Sequential index (0 to nord - 1) for populating the output array thar_out.
        # 2. Evaluating the Wavelength Surface per Order (while order <= end:)
        #    For each echelle order:
        #    Physical Order Number (m = idorder + 49): Calculates the absolute physical grating order m by adding the base offset order0 
        #        to the relative order index idorder.
        #    Chebyshev Basis Polynomials (Calculate_chebs): Evaluates the 2D Chebyshev basis functions for every pixel x (equis) 
        #        at physical order m (or 1/m if Inverse_m=True).
        #    2D Surface Evaluation (Joint_Polynomial_Cheby): Computes the joint 2D Chebyshev polynomial using the global coefficient vector p1 fitted previously.
        #    Wavelength Conversion: Multiplies by 1/m to convert the grating equation output (m x lambda) into true wavelength values lambda(x):
        # 3. Output Array Population and Loop Increment
        #    Stores the evaluated wavelength array WavSol into thar_out[0, out_order, :].
        #    Copies the corresponding extracted ThAr flux spectrum into thar_out[1, out_order, :].
        #    Increments all iteration counters to advance to the next echelle order.
        thar_out = np.zeros((2,nord,lines_thar.shape[1]))
        equis = np.arange( lines_thar.shape[1] )        
        order = start
        idorder = idorder0
        out_order = 0
        #print("Starting in order: ", order, " and finish at end: ", end)
        #print("equis:", equis)
        #print("Inverse_m:", Inverse_m)
        #print("ntot:", ntot)
        #print("len(thar_order):", len(thar_order))
        #print("ncoef_x:",ncoef_x)
        #print("ncoef_m:",ncoef__m)
        #print("chebs:",chebs)
        #print("p1:",p1)

        while order <= end:
            m   = idorder
            chebs = GLOBALutils.Calculate_chebs(equis, m, Inverse=Inverse_m,order0=order0,ntotal=ntot,npix=len(thar_order),nx=ncoef_x,nm=ncoef_m)
            WavSol = (1.0/m) * GLOBALutils.Joint_Polynomial_Cheby(p1,chebs,ncoef_x,ncoef_m)
            #print("WavSol:",WavSol)
            thar_out[0,out_order,:] = WavSol
            thar_out[1,out_order,:] = lines_thar[order]
            order+=1
            idorder += 1
            out_order += 1

        wsdate = str(datetime.datetime.now()).replace(' ','T')
        if os.access(thar_spec,os.F_OK):
            os.remove(thar_spec)
        hdu = pyfits.PrimaryHDU(thar_out)
        hdu = GLOBALutils.update_header(hdu,'THDATE',thar_S[0].header['DATE'])
        hdu = GLOBALutils.update_header(hdu,'DATE',wsdate)
        hdu.data = hdu.data.astype(np.float32)
        hdu.writeto(thar_spec)

        # if fsim == thars[0]:
        #    Logic: Checks whether the current science frame or calibration frame being processed (fsim) matches the 
        #       very first Thorium-Argon (ThAr) reference image in the dataset (thars[0]).
        #    Purpose: Ensures that the reference wavelength is set only once—using the baseline arc 
        #       lamp exposure—to establish a fixed epoch for drift calculations.
        # ref_pix = thar_out[0, 11, len(thar_order) // 2]
        #    Logic: Retrieves a specific wavelength value from the 3D calibrated ThAr array thar_out (which has dimensions [2, nord, npix]):
        #       0: Index for the wavelength array (λ).
        #       11: Selects the 11th echelle order (often used as the central anchor or reference order for the spectrograph).
        #       len(thar_order) // 2: Calculates the integer midpoint along the dispersion axis (the central pixel of the detector).
        #    Purpose: Stores the exact wavelength at the center of order 11 in the initial ThAr calibration frame into ref_pix. 
        #         This value serves as the global reference wavelength anchor (λref) against which subsequent instrumental drifts 
        #        or velocity shifts are measured throughout the observing night.
        if fsim == thars[0]:
            ref_pix = thar_out[0, 11, len(thar_order) // 2]
            print("here, ref_pix:",ref_pix)
 
        pdict = {'p1':p1,'mjd':mjd, 'G_pix':G_pix, 'G_ord':G_ord, 'G_wav':G_wav, 'II':II, 'rms_ms':rms_ms,\
            'G_res':G_res, 'All_Centroids':All_Centroids, 'All_Orders':All_Orders, 'All_Sigmas':All_Sigmas,\
            'All_Pixel_Centers': All_Pixel_Centers, 'All_Wavelengths':All_Wavelengths, \
            'thar_order':thar_order, 'All_Intensities': All_Intensities}
        pickle.dump(pdict, open(wavsol_pkl, 'wb'))
        print("\t--> Dumping in wavsol_pkl:", wavsol_pkl)

    else:
        print("\t--> Using previously computed wavelength solution in file",wavsol_pkl)
        pdict = pickle.load(open(wavsol_pkl, 'rb'), encoding='latin1')
        All_Pixel_Centers = pdict['All_Pixel_Centers']
        All_Wavelengths   = pdict['All_Wavelengths']
        All_Orders        = pdict['All_Orders']
        All_Intensities   = pdict['All_Intensities']
        p1                = pdict['p1']
        thar_order           = pdict['thar_order']
        thar_out = pyfits.getdata(thar_spec)
        if fsim == thars[0]:
            ref_pix = thar_out[0, 11, mid_extract]

drift_file = dirout + '/drifts.txt'
drift_names_file = dirout + '/drifts_names.txt'
rednames = []

force_drifts = False

if os.path.isfile(drift_names_file):
    orednames = np.loadtxt(drift_names_file,dtype='str')[:-1]
    for thar in thars:
        if not thar.split('/')[-1] in orednames:
            force_drifts = True
            break

drift_out = []
best_id=0

pkl_wsol = dirout + Path(thars[best_id]).stem + '.wavsolpars.pkl'
print("\t--> Unpickling wavelength solution from", pkl_wsol, " ...")
wsol_dict = pickle.load(open(pkl_wsol, 'rb'), encoding='latin1')

print('\n\t--> Computing instrumental drifts for ThAr + Dark images...')
sci_all = np.hstack((sim_sci, dar_sci))

if absref != 'none' and os.path.isfile(absref):
    pkl0 = pickle.load(open(absref, 'rb'), encoding='latin1')
    pkl  = pickle.load(open(pkl_wsol, 'rb'), encoding='latin1')
    thar_fits2 = dirout + Path(thars[int(.5*len(thars))]).stem + '.spec.fits.S'
    thar_S2 = pyfits.getdata( thar_fits2 )
    p_shift, pix_centers, orders, wavelengths, I, rms_ms, residuals  = \
        GLOBALutils.Global_Wav_Solution_vel_shift(pkl['G_pix'], pkl['G_wav'],\
        pkl['G_ord'], np.ones(len(pkl['G_ord'])), pkl0['p1'], Cheby=True, \
        Inv=True,maxrms=MRMS,minlines=minlines_glob, order0=order0,ntotal=ntot,npix=thar_S2.shape[2],nx=ncoef_x,nm=ncoef_m)
    werr = rms_ms/np.sqrt(float(len(residuals)))
    drift0 = (1e-6*p_shift)*299792458.0
    print("drift0:",drift0,"werr:",werr)
    print("residuals:",residuals)
    print("rms_ms:",rms_ms)
    thar_dict = pickle.load(open(dirout+'order_id.pkl', 'rb'), encoding='latin1')
    thar_dict['absref'] = absref
    thar_dict['absp_shift'] = drift0
    thar_dict['werr'] = werr
    pickle.dump(thar_dict, open(dirout+'order_id.pkl', 'wb'))

if just_thar:
    sys.exit()

### start of science frame reductions ###
justdo = np.array([])
if os.path.isfile(dirin+'/justdo.txt'):
    justdo = open(dirin+'/justdo.txt').readlines()

#print(justdo)
if len(justdo) != 0:
    njustdo = []
    for jso in justdo:
        njsm = dirin+'/'+jso[:-1]
        njsm = njsm.replace('//','/')
        njustdo.append(njsm)
    njustdo = np.array(njustdo)
    sci_all = njustdo.copy()

#print(sci_all)
new_list = []
new_list_obnames = []
new_list_texp = []
for i in range(len(sci_all)):
    fsim = sci_all[i]
    hd = pyfits.getheader(fsim)
    texp = hd['EXPTIME']
    obname = bachesutils.get_name(fsim)

    if (object2do == 'all'):
        new_list.append(fsim)
        new_list_obnames.append( obname )
        new_list_texp.append( texp )

    else:
        if (obname == object2do):
            new_list.append(fsim)
            new_list_obnames.append( obname )
            new_list_texp.append( texp )

print('\n\t--> The following targets will be processed:')
for nlisti in range(len(new_list)):
    print('\t\t'+new_list_obnames[nlisti]+'\t'+new_list[nlisti])


# Does any image have a special requirement for dealing with the moonlight?
if os.access(dirin + 'moon_corr.txt', os.F_OK):
    fmoon = open(dirin + 'moon_corr.txt','r')
    moon_lns = fmoon.readlines()
    spec_moon = []
    use_moon = []
    for line in moon_lns:
        spec_moon.append(line.split()[0])
        if line.split()[1] == '0':
            use_moon.append(False)
        else:
            use_moon.append(True)
else:
    spec_moon = []
    use_moon = []

spec_moon = np.array(spec_moon)
use_moon  = np.array(use_moon)
for fsim in new_list:
    try:
            print('\n')
            print("\t--> Working on image: ", fsim)
            hd = pyfits.getheader(fsim)

            hsim = pyfits.open(fsim)
            mjd, mjd0 = bachesutils.mjd_fromheader( hsim )

            exptime = hd['EXPTIME']
            #obname = fsim.split('/')[-1]
            #obname = obname.split('-')[-1][:-5]
            obname = bachesutils.get_name(fsim)

            print("\t\tObject name:",obname)

            know_moon = False
            if fsim.split('/')[-1] in spec_moon:
                I = np.where(fsim.split('/')[-1] == spec_moon)[0]
                know_moon = True
                here_moon = use_moon[I]

            # Sierra Nevada Observatory
            altitude    = 2896.
            latitude    = 37.0642
            longitude   = 3.3847
            epoch       = 2000.0

            known_coords = False
            try:  
                sp,ra,dec,known_coords = GLOBALutils.simbad_coords(obname,mjd)
            except:
                ra,dec = -999,-999
            ras,decs = ra,dec
            ra2,dec2 = GLOBALutils.getcoords(obname,mjd,filen=reffile)
            if ra2 !=0 and dec2 != 0:
                ra = ra2
                dec = dec2
                print('\t\tUsing the coordinates found in the image header.')
            else:
                print('\t\tUsing the coordinates found in SIMBAD.')

            airmass = bachesutils.get_airmass(ra,dec,latitude,longitude,altitude,hd['DATE-OBS'].replace('T',' '))

            iers                    = GLOBALutils.JPLiers( baryc_dir, mjd-999.0, mjd+999.0 )
            obsradius, R0           = GLOBALutils.JPLR0( latitude, altitude)
            obpos                   = GLOBALutils.obspos( longitude, obsradius, R0 )

            jplephem.set_ephemeris_dir( baryc_dir , ephemeris )
            jplephem.set_observer_coordinates( float(obpos[0]), float(obpos[1]), float(obpos[2]) )

            res         = jplephem.doppler_fraction(ra/15.0, dec, int(mjd), mjd%1, 1, 0.0)
            lbary_ltopo = 1.0 + res['frac'][0]
            bcvel_baryc = ( lbary_ltopo - 1.0 ) * 2.99792458E5

            print("\t\tBarycentric velocity:", bcvel_baryc)

            res = jplephem.pulse_delay(ra/15.0, dec, int(mjd), mjd%1, 1, 0.0)   
            mbjd = mjd + res['delay'][0] / (3600.0 * 24.0)

            # Moon Phase Calculations
            gobs      = ephem.Observer()  
            gobs.name = 'OSN1.5'  
            gobs.lat  = rad(latitude)  # lat/long in decimal degrees  
            gobs.long = rad(longitude)
            gobs.date = hd['DATE-OBS'][:10] + ' ' + hd['DATE-OBS'][11:]
            mephem = ephem.Moon()
            mephem.compute(gobs)
            Mcoo = jplephem.object_track("Moon", int(mjd), float(mjd%1), 1, 0.0)
            Mp   = jplephem.barycentric_object_track("Moon", int(mjd), float(mjd%1), 1, 0.0)
            Sp   = jplephem.barycentric_object_track("Sun", int(mjd), float(mjd%1), 1, 0.0)
            res  = jplephem.object_doppler("Moon", int(mjd), mjd%1, 1, 0.0)
            lunation,moon_state,moonsep,moonvel = GLOBALutils.get_lunar_props(ephem,gobs,Mcoo,Mp,Sp,res,ra,dec)
            refvel = bcvel_baryc + moonvel
            print('\t\tRadial Velocity of scattered moonlight:',refvel) # CG typo

            if os.access(dirout+'Dark_'+str(exptime)+'.fits',os.F_OK):
                DARK = pyfits.getdata(dirout+'Dark_'+str(exptime)+'.fits')
            else:
                DARK = np.zeros(MasterBias[0].data.shape)

            hdat  = pyfits.open( fsim )
            data  = pyfits.getdata( fsim )
            #data = data - MasterBias[0].data #- DARK
            data = data - MasterBias[0].data - DARK

            data  = np.fliplr(data.T)
            data  = np.ascontiguousarray(data)  # To improve speed in RAM
            #plt.imshow(data.T, vmin=np.median(data), vmax=np.median(data) + 500)
            #ejx = np.arange(1756)
            #for i in range(nord):
            #   y = np.polyval(c[i],ejx)
            #   plt.plot(ejx,y)
            #plt.savefig("AAAAA.png", bbox_inches="tight")
            #plt.close()

            #plt.show()

            c_new = c.copy()
            P_new = P.copy()

            #plt.figure(figsize=(10, 4))
            #plt.plot(P_new[:,1000])
            #plt.plot(P[:,1000])
            #plt.show()
            bac_fits    = dirout + 'BKG_'+ Path(fsim).stem + '.fits'  # CG '.fits' to 'fits'

            if os.access(bac_fits,os.F_OK) == False:
                bac = bachesutils.get_scat(data, P_new,1)
                hdu = pyfits.PrimaryHDU(bac)
                hdu.data = hdu.data.astype(np.float32)
                hdu.writeto(bac_fits)
            else:
                bac = pyfits.getdata(bac_fits)

            #plt.imshow(bac)    
            #plt.show()
            #plt.plot(data[1000])
            #plt.plot(bac[1000])
            #plt.show()
            data -= bac
            #plt.plot(data[1000])
            #plt.show()

            sci_fits_simple = dirout + Path(fsim).stem + '.spec.simple.fits.S'
            bac_fits_simple = dirout + Path(fsim).stem + '.spec.bac.fits.S'
            sci_fits = dirout + Path(fsim).stem + '.spec.fits.S'
            thar_spec = dirout + Path(fsim).stem + '.sp.fits'  

            if ( os.access(sci_fits,os.F_OK) == False ) or \
                ( os.access(sci_fits_simple,os.F_OK) == False ) or \
                ( force_sci_extract ):

                print("No previous extraction or extraction forced for SCIENCE file", fsim, "extracting...")
                sci_Ss = GLOBALutils.simple_extraction(data.T,c_new,ext_aperture,min_extract,max_extract,npools)
                bac    = GLOBALutils.simple_extraction(bac.T,c_new,ext_aperture,min_extract,max_extract,npools)
                sci_S  = GLOBALutils.optimal_extraction(data.T,P_new,c_new,ext_aperture,ronoise,gain,S_Marsh,NCosmic_Marsh,min_extract,max_extract,npools)
                sci_Ss = GLOBALutils.invert(sci_Ss)
                sci_S  = GLOBALutils.invert(sci_S)
                if (os.access(sci_fits,os.F_OK)):
                    os.remove( sci_fits )
                if (os.access(sci_fits_simple,os.F_OK)):
                    os.remove( sci_fits_simple )
                if (os.access(bac_fits_simple,os.F_OK)):
                    os.remove( bac_fits_simple )

                hdu = pyfits.PrimaryHDU( sci_S )
                hdu.data = hdu.data.astype(np.float32)
                hdu.writeto( sci_fits )
                hdu = pyfits.PrimaryHDU( sci_Ss )
                hdu.data = hdu.data.astype(np.float32)
                hdu.writeto( sci_fits_simple )
                hdu = pyfits.PrimaryHDU( bac )
                hdu.data = hdu.data.astype(np.float32)
                hdu.writeto( bac_fits_simple )

            else:
                print("\t--> Image already extracted, reading in product fits files...")
                sci_S  = pyfits.getdata( sci_fits )
                sci_Ss = pyfits.getdata( sci_fits_simple )
                bac = pyfits.getdata( bac_fits_simple )

            p_shift = 0.

            if fsim in sim_sci:
                print("\t--> Wavelength calibration for image: ", fsim)
                best_match_index, diff_seconds = bachesutils.find_closest_index(mjd,thar_times)
                print(f"\t\tTarget Science MJD  : {mjd}")
                print(f"\t\tClosest ThAr MJD    : {thar_times[best_match_index]}")
                print(f"\t\tBest match index    : {best_match_index}")
                print(f"\t\tTime difference     : {diff_seconds:.2f} seconds")
                print(f"\t\tBest ThAr file      : {thars[best_match_index]}")

                # Retrieve the calibration of the closest ThAr file
                wavsol_pkl = dirout + Path(thars[best_match_index]).stem + '.wavsolpars.pkl'
                pdict = pickle.load(open(wavsol_pkl, 'rb'), encoding='latin1')
                All_Pixel_Centers = pdict['All_Pixel_Centers']
                All_Wavelengths   = pdict['All_Wavelengths']
                All_Orders        = pdict['All_Orders']
                All_Intensities   = pdict['All_Intensities']
                p1                = pdict['p1']
                thar_order           = pdict['thar_order']

                #sci_S  = pyfits.getdata( thar_fits )		# HERE HERE, already done?
                #sci_Ss = pyfits.getdata( thar_fits_simple )	# HERE HERE, already done?
                #print("thar_fits:",thar_fits)
                #print("All_Pixel_Centers:",All_Pixel_Centers)
                #print("All_Wavelengths:",All_Wavelengths)
                #print("All_Orders:",All_Orders)
                #print("All_Intensities.shape:",All_Intensities.shape)
                #print("p1:",p1)
                #print("wsol_dict:",wsol_dict['p1'])
                #print("MRMS:",MRMS)
                #print("minlines_glob:",minlines_glob)
                #print("ntot:",ntot)
                #print("len(thar_order):",len(thar_order))
                #print("ncoef_x:",ncoef_x)
                #print("ncoef_m:",ncoef_m)
                p_shift, pix_centers, orders, wavelengths, I, rms_ms, residuals  = \
                    GLOBALutils.Global_Wav_Solution_vel_shift(All_Pixel_Centers, All_Wavelengths, All_Orders,\
                    np.ones(All_Intensities.shape), wsol_dict['p1'],\
                    Cheby=True,Inv=True,maxrms=MRMS,minlines=minlines_glob,\
                    order0=order0,ntotal=ntot,npix=len(thar_order),nx=ncoef_x,nm=ncoef_m)

            werr = rms_ms/np.sqrt(float(len(residuals)))
            rv_drift = (1e-6*p_shift)*299792458.0
            mat = np.array([mjd, rv_drift[0], werr,exptime,1]) # CG
                
            shift_pix = ref_pix*(1.+rv_drift/299792458.0)

            if werr<6:
                if len(drift_out)==0:
                    drift_out = mat.copy()
                else:
                    drift_out = np.vstack((drift_out,mat))
            else:
                p_shift = 0.
                print('\t 2-Warning!! RMS of wavelength solution is to high, setting Instrumental drift to 0 m/s...')

            #drift_out = []
            #print("p_shift:",p_shift)
            #p_shift = 0.
            np.savetxt(drift_file,drift_out)

            fout = 'proc/'+ Path(fsim).stem + '.sp.fits'
            spec = np.zeros((11, ntot, data.shape[0]))
            hdu = pyfits.PrimaryHDU( spec )

            hdu = GLOBALutils.update_header(hdu,'HIERARCH MJD', mjd)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH MBJD', mbjd)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH SHUTTER START DATE', hd['DATE-OBS'][:10] )
            hdu = GLOBALutils.update_header(hdu,'HIERARCH SHUTTER START UT',  hd['DATE-OBS'][11:])
            hdu = GLOBALutils.update_header(hdu,'HIERARCH TEXP (S)',hd['EXPTIME'])
            hdu = GLOBALutils.update_header(hdu,'HIERARCH BARYCENTRIC CORRECTION (KM/S)', bcvel_baryc)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH (LAMBDA_BARY / LAMBDA_TOPO)', lbary_ltopo)    
            hdu = GLOBALutils.update_header(hdu,'HIERARCH TARGET NAME', obname)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH RA',ras)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH DEC',decs)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH RA DEG',ras)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH DEC DEG',decs)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH RA BARY',ra)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH DEC BARY',dec)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH EQUINOX',2000)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH OBS LATITUDE',latitude)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH OBS LONGITUDE',longitude)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH OBS ALTITUDE',altitude)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH DRIFT_CO',np.around(rv_drift[0],1))
            hdu = GLOBALutils.update_header(hdu,'HIERARCH DRIFT_CO_E',np.around(werr,1))
            hdu = GLOBALutils.update_header(hdu,'HIERARCH AIRMASS',np.around(airmass.value,3))
            hdu = GLOBALutils.update_header(hdu,'BDATE',MasterBias[0].header['DATE'])
            hdu = GLOBALutils.update_header(hdu,'HIERARCH REFWAV',thars[best_id].split('/')[-1])

            equis = np.arange( data.shape[0] )
            out_order = 0
            order = start
            idorder = 31

            print("Finally starting in order: ", order, " and finish at end: ", end)
            while order <= end:
                m = idorder
                chebs = GLOBALutils.Calculate_chebs(equis, m, Inverse=Inverse_m,order0=order0,ntotal=ntot,npix=len(equis),nx=ncoef_x,nm=ncoef_m)
                WavSol = lbary_ltopo * (1.0 + 1.0e-6*p_shift) * (1.0/m) * GLOBALutils.Joint_Polynomial_Cheby(wsol_dict['p1'],chebs,ncoef_x,ncoef_m)
                spec[0,out_order,:] = GLOBALutils.ToVacuum(WavSol)
                #spec[0,out_order,:] = WavSol
                spec[1,out_order,:] = sci_S[order,1, :]
                spec[2,out_order,:] = sci_S[order,2, :]
                # Flat-fielded spectrum
                fn = S_flat_n[order,1,:]
                L  = np.where( fn > 0 )
                spec[3,out_order,:][L] = sci_S[order,1,:][L] / S_flat_n[order,1,:][L]
                spec[4,out_order,:][L] = sci_S[order,2,:][L] * ( S_flat_n[order,1,:][L] ** 2 )
                ccoef = GLOBALutils.get_cont_single(spec[0,out_order],spec[3,out_order],spec[4,out_order],ll=1.5,lu=5,nc=3)
                L  = np.where( spec[1,out_order] != 0 )
                spec[5,out_order,:][L] = spec[3,out_order][L] / np.polyval(ccoef,spec[0,out_order][L])

                nJ = np.where(np.isnan(spec[5,out_order])==True)[0]
                nJ2 = np.where(np.isinf(spec[5,out_order])==True)[0]
                spec[5,out_order,nJ] = 1.0
                spec[5,out_order,nJ2] = 1.0
                ratio            = spec[3,out_order,:][L] / spec[5,out_order,:][L]
                spec[6,out_order,:][L] = spec[4,out_order,:][L] * (ratio ** 2 )
                spec[7,out_order,:][L] = ratio
                spec[8,out_order,:][L] = ratio * S_flat_n[order,1,:][L] / np.sqrt( ratio * S_flat_n[order,1,:][L] / gain + bac[order,:][L] / gain + 2*ext_aperture*(ronoise/gain)**2 )
                II = np.where(spec[8,out_order]<0)[0]
                if len(II)>0:
                        spec[8,out_order,II] = 0.

                rI = np.where(spec[5,out_order] > 1. + 8./spec[8,out_order])
                spec[5,out_order,rI] = 1.
                rI1 = np.where(spec[5,out_order] > 10)
                rI2 = np.where(spec[5,out_order] < -10)
                spec[5,out_order,rI1] = 1.
                spec[5,out_order,rI2] = 1.
                spl           = scipy.interpolate.splrep(np.arange(WavSol.shape[0]), WavSol,k=3)
                dlambda_dx    = scipy.interpolate.splev(np.arange(WavSol.shape[0]), spl, der=1)
                NN            = np.average(dlambda_dx)
                dlambda_dx    /= NN

                spec[9,out_order,:][L]  = spec[5,out_order,:][L] * (dlambda_dx[L] ** 1) 
                spec[10,out_order,:][L] = spec[6,out_order,:][L] / (dlambda_dx[L] ** 2)

                order += 1
                out_order += 1
                idorder += 1

            hdu = GLOBALutils.update_header(hdu,'SFDATE',MasterBias[0].header['DATE'])
            if (os.access( dirout + fout,os.F_OK)):
                os.remove( dirout + fout)
            hdu.data = hdu.data.astype(np.float32)
            hdu.writeto( dirout + fout )

            if (not JustExtract):

                if DoClass:
                    print('\t\tSpectral Analysis:')
                    # spectral analysis
                    # First, query SIMBAD with the object name
                    query_success = False
                    sp_type_query = 'None'
                    #query_success,sp_type_query = GLOBALutils.simbad_query_obname(obname)
                    # Now, query SIMBAD by coordinates if above not successful
                    #if (not query_success):
                    #    query_success,sp_type_query = GLOBALutils.simbad_query_coords('12:00:00','00:00:00')
                    #print "\t\t\tSpectral type returned by SIMBAD query:",sp_type_query

                    hdu = GLOBALutils.update_header(hdu,'HIERARCH SIMBAD SPTYP', sp_type_query)
                    pars_file = dirout + fsim.split('/')[-1][:-8]+'_stellar_pars.txt'

                    if os.access(pars_file,os.F_OK) == False or force_stellar_pars:
                        print("\t\t\tEstimating atmospheric parameters:")
                        spec2 = spec.copy()
                        T_eff, logg, Z, vsini, vel0, ccf = correlation.CCF(spec2,model_path=models_path,npools=npools)
                        line = "%6d %4.1f %4.1f %8.1f %8.1f\n" % (T_eff,logg, Z, vsini, vel0)
                        f = open(pars_file,'w')
                        f.write(line)
                        f.close()
                    else:
                        print("\t\t\tAtmospheric parameters loaded from file:")
                        T_eff, logg, Z, vsini, vel0 = np.loadtxt(pars_file,unpack=True)

                    print("\t\t\t\tT_eff=",T_eff,"np.log(g)=",logg,"Z=",Z,"vsin(i)=",vsini,"vel0",vel0)

                else:
                    T_eff, logg, Z, vsini, vel0 = -999,-999,-999,-999,-999

                T_eff_epoch = T_eff
                logg_epoch  = logg
                Z_epoch     = Z
                vsini_epoch = vsini
                vel0_epoch  = vel0
                hdu = GLOBALutils.update_header(hdu,'HIERARCH TEFF', float(T_eff))
                hdu = GLOBALutils.update_header(hdu,'HIERARCH LOGG', float(logg))
                hdu = GLOBALutils.update_header(hdu,'HIERARCH Z', Z)
                hdu = GLOBALutils.update_header(hdu,'HIERARCH VSINI', vsini)
                hdu = GLOBALutils.update_header(hdu,'HIERARCH VEL0', vel0)

                print("\t\tRadial Velocity analysis:")
                # assign mask
                sp_type, mask = GLOBALutils.get_mask_reffile(obname,reffile=reffile,base=_xc_masks_dir)
                print("\t\t\tWill use",sp_type,"mask for CCF.")

                # Read in mask
                ml, mh, weight = np.loadtxt(mask,unpack=True)
                ml_v = GLOBALutils.ToVacuum( ml )
                mh_v = GLOBALutils.ToVacuum( mh )
               
                # make mask larger accounting for factor ~2 lower res in CORALIE w/r to HARPS
                av_m = 0.5*( ml_v + mh_v )
                ml_v -= 1.5*(av_m - ml_v)
                mh_v += 1.5*(mh_v - av_m)
                mask_hw_kms = (GLOBALutils.Constants.c/1e3) * 0.5*(mh_v - ml_v) / av_m

                #sigma_fout = stellar_pars_dir + obname + '_' +'sigma.txt'

                disp = GLOBALutils.get_disp(obname, reffile=reffile)
                if disp == 0:
                    known_sigma = False
                    if vsini != -999 and vsini > 4.:
                        disp = vsini
                    else:
                        disp = 4.
                else:
                    known_sigma = True

                mask_hw_wide = av_m * disp / (GLOBALutils.Constants.c/1.0e3)
                ml_v = av_m - mask_hw_wide
                mh_v = av_m + mask_hw_wide 

                print('\t\t\tComputing the CCF...')
                cond = True
                while (cond):
                    # first rough correlation to find the minimum
                    vels, xc_full, sn, nlines_ccf, W_ccf = \
                        GLOBALutils.XCor(spec, ml_v, mh_v, weight, 0, lbary_ltopo, vel_width=200,vel_step=3,\
                        spec_order=9,iv_order=10,sn_order=8,max_vel_rough=200)
                    v1,x1 = vels.copy(), xc_full.copy()

                    for i in range(spec.shape[1]):
                        I = np.where((spec[0,i]>7030) & (spec[0,i] < 7050))[0]
                        if len(I)>0:
                                W_ccf[i] = 0
                        I = np.where((spec[0,i]>6920) & (spec[0,i] < 6940))[0]
                        if len(I)>0:
                                W_ccf[i] = 0
                        I = np.where((spec[0,i]>4300) & (spec[0,i] < 4340))[0]
                        if len(I)>0:
                                W_ccf[i] = 0
                                
                    # Next code block handles averaging the cross-correlation functions (CCFs) from 
                    # the different spectral orders and normalizing the continuum of the combined 
                    # profile using a robust non-parametric fit.
                    # 
                    # 1. Averaging CCFs by Order: GLOBALutils.Average()
                    #    sn: Signal-to-noise ratio per order, used to weight orders with better signal. 
                    #    sn_min=0.0: Discards orders with a signal-to-noise ratio below this threshold. 
                    #    W=W_ccf: Matrix of weights assigned to each order or CCF point. 
                    #    start_order=2: Skips the first two echelle orders (which are often noisy or defective).
                    # 2. Median filter: scipy.signal.medfilt().
                    #    Ease the median with a window of 11 points. It eliminates cosmis rays and fluctuations 
                    #    of high frequency 
                    # 3. Non-parametric Continuum Fit (LOWESS): pred = lowess()
                    #    frac=0.4: Uses 40% of neighboring data points to construct each local fit 
                    #           (captures smooth, low-frequency undulations). 
                    #    it=10: Performs 10 iterations of robust reweighting to exclude outliers, 
                    #           such as the CCF dip itself (the main absorption/emission line). 
                    #    vels: Vector of velocities associated with the CCF data points.
                    # 4. Linear Spline Interpolation `tck1 = scipy.interpolate.splrep(vels, pred, k=1)`
                    #    Purpose: Generates the spline representation ($k=1$, linear order) of the predicted continuum 
                    #    curve `pred` over the velocity grid `vels`. Allows for the subsequent evaluation or 
                    #    interpolation of the continuum trend at any arbitrary velocity point, if necessary. 
                    # 5. Preservation and Normalization 
                    #    Purpose: `xc_av_orig`: Saves a copy of the unnormalized averaged CCF for potential 
                    #    diagnostics or plotting. 
                    #    Divides the averaged CCF point-by-point by the estimated continuum curve (`pred`).
                    xc_av = GLOBALutils.Average_CCF(xc_full, sn, sn_min=0.0, Simple=True, W=W_ccf, start_order=2)
                    yy = scipy.signal.medfilt(xc_av,11)
                    pred = lowess(yy, vels,frac=0.4,it=10,return_sorted=False)
                    tck1 = scipy.interpolate.splrep(vels,pred,k=1)
                    xc_av_orig = xc_av.copy()
                    xc_av /= pred

                    vel0_xc = vels[ np.argmin( xc_av ) ]
                    rvels, rxc_av, rpred, rxc_av_orig, rvel0_xc = vels.copy(), \
                        xc_av.copy(), pred.copy(), xc_av_orig.copy(), vel0_xc

                    xc_av_rough = xc_av
                    vels_rough  = vels
                    if disp > 30:
                        disp = 30.
                    vel_width = np.maximum( 20.0, 6*disp )

                    vels, xc_full, sn, nlines_ccf, W_ccf =\
                        GLOBALutils.XCor(spec, ml_v, mh_v, weight, vel0_xc, lbary_ltopo, vel_width=vel_width,vel_step=0.1,\
                        spec_order=9,iv_order=10,sn_order=8,max_vel_rough=300)
                    for i in range(spec.shape[1]):
                        I = np.where((spec[0,i]>7030) & (spec[0,i] < 7050))[0]
                        if len(I)>0:
                                W_ccf[i] = 0
                        I = np.where((spec[0,i]>6920) & (spec[0,i] < 6940))[0]
                        if len(I)>0:
                                W_ccf[i] = 0
                        I = np.where((spec[0,i]>4300) & (spec[0,i] < 4340))[0]
                        if len(I)>0:
                                W_ccf[i] = 0

                    #I = np.where(W_ccf!=0)[0]
                    #W_ccf[I] = 1.

                    xc_av = GLOBALutils.Average_CCF(xc_full, sn, sn_min=0.0, Simple=True, W=W_ccf, start_order=2)
                    pred = scipy.interpolate.splev(vels,tck1)
                    xc_av /= pred
                
                    if sp_type == 'M5':
                        moon_sig = 2.5
                    elif sp_type == 'K5':
                        moon_sig = 3.3
                    else:
                        moon_sig = 4.5

                    p1,XCmodel,p1gau,XCmodelgau,Ls2 = GLOBALutils.XC_Final_Fit( vels, xc_av ,\
                        sigma_res = 4, horder=8, moonv = refvel, moons = moon_sig, moon = False)

                    moonmatters = False
                    if (know_moon and here_moon):
                        moonmatters = True
                        ismoon = True
                        confused = False
                        p1_m,XCmodel_m,p1gau_m,XCmodelgau_m,Ls2_m = GLOBALutils.XC_Final_Fit( vels, xc_av , sigma_res = 4, horder=8, moonv = refvel, moons = moon_sig, moon = True)
                        moon_flag = 1
                    else:
                        confused = False
                        ismoon = False
                        p1_m,XCmodel_m,p1gau_m,XCmodelgau_m,Ls2_m = p1,XCmodel,p1gau,XCmodelgau,Ls2
                        moon_flag = 0

                    bspan = GLOBALutils.calc_bss(vels,xc_av)
                    SP = bspan[0]

                    #print 'Bisector span:', SP
                    if (not known_sigma):
                        disp = np.floor(p1gau[2])
                        if (disp < 4.0): 
                            disp = 4.0
                        mask_hw_wide = av_m * disp / (GLOBALutils.Constants.c/1.0e3)
                        ml_v = av_m - mask_hw_wide
                        mh_v = av_m + mask_hw_wide            
                        known_sigma = True
                    else:
                        cond = False

                xc_dict = {'vels':vels,'xc_av':xc_av,'XCmodelgau':XCmodelgau,'Ls2':Ls2,'refvel':refvel,\
                    'rvels':rvels,'rxc_av':rxc_av,'rpred':rpred,'rxc_av_orig':rxc_av_orig,\
                    'rvel0_xc':rvel0_xc,'xc_full':xc_full, 'p1':p1, 'sn':sn, 'p1gau':p1gau,\
                    'p1_m':p1_m,'XCmodel_m':XCmodel_m,'p1gau_m':p1gau_m,'Ls2_m':Ls2_m,\
                    'XCmodelgau_m':XCmodelgau_m}

                moon_dict = {'moonmatters':moonmatters,'moon_state':moon_state,'moonsep':moonsep,\
                    'lunation':lunation,'mephem':mephem,'texp':hd['EXPTIME']}

                pkl_xc = dirout + fsim.split('/')[-1][:-8]+obname+'_XC_'+sp_type+'.pkl'
                pickle.dump(xc_dict, open(pkl_xc, 'wb'))

                #ccf_pdf = dirout + 'proc/' + Path(fsim).stem + obname + '_XCs_' + sp_type + '.pdf'
                ccf_pdf = dirout + 'proc/' + Path(fsim).stem + '_XCs_' + sp_type + '.pdf'

                if not avoid_plot:
                    GLOBALutils.plot_CCF(xc_dict,moon_dict,path=ccf_pdf)

                #SNR_5130 = np.median(spec[8,21,1000:1101] )
                SNR_5130 = np.median(spec[8,11,1000:1101] )
                if SNR_5130 < 1:
                        SNR_5130 = 1.
                seeing   = -999
                
                if sp_type == 'G2':
                    if T_eff < 6000:
                        A = 0.06544
                        B = 0.00146
                        D = 0.24416
                        C = 0.00181
                    else:
                        A = 0.09821
                        B = 0.00014
                        D = 0.33491
                        C = 0.00113
                elif  sp_type == 'K5':
                    A = 0.05348
                    B = 0.00147 
                    D = 0.20695
                    C = 0.00321
                else:
                    A = 0.05348
                    B = 0.00147 
                    D = 0.20695
                    C = 0.00321

                RVerr =  B + ( 1.6 + 0.2 * p1gau[2] ) * A / np.round(SNR_5130)
                depth_fact = 1. + p1gau[0]/(p1gau[2]*np.sqrt(2*np.pi))

                RVerr2_ori = -999.0
                if depth_fact >= 1.:
                    RVerr2 = -999.000
                else:
                    if sp_type == 'G2':
                        depth_fact = (1 - 0.62) / (1 - depth_fact)
                    else:
                        depth_fact = (1 - 0.59) / (1 - depth_fact)
                    RVerr2 = RVerr * depth_fact
                    RVerr2 = np.sqrt(RVerr2**2+0.004**2)
                    #if (RVerr2 <= 0.004):
                    #    RVerr2 = 0.004
                
                #if not good_quality:
                #    RVerr2 = np.sqrt(0.03**2 + RVerr2**2)

                BSerr = D / float(np.round(SNR_5130)) + C

                RV     = np.around(p1gau_m[1],4)  
                BS     = np.around(SP,4)   
                RVerr2 = np.around(RVerr2,4)
                BSerr = np.around(BSerr,4)

                print('\t\t\tRV = '+str(RV)+' +- '+str(RVerr2))
                print('\t\t\tBS = '+str(BS)+' +- '+str(BSerr))

                bjd_out = 2400000.5 + mbjd
                T_eff_err = 100
                logg_err = 0.5
                Z_err = 0.5
                vsini_err = 2
                XC_min = np.abs(np.around(np.min(XCmodel),2))

                SNR_5130 = np.around(SNR_5130)
                SNR_5130_R = np.around(SNR_5130*np.sqrt(2.9))

                disp_epoch = np.around(p1gau_m[2],1)
                hdu = GLOBALutils.update_header(hdu,'RV', RV)
                hdu = GLOBALutils.update_header(hdu,'RV_E', RVerr2)
                hdu = GLOBALutils.update_header(hdu,'BS', BS)
                hdu = GLOBALutils.update_header(hdu,'BS_E', BSerr)
                hdu = GLOBALutils.update_header(hdu,'DISP', disp_epoch)
                hdu = GLOBALutils.update_header(hdu,'SNR', SNR_5130)
                hdu = GLOBALutils.update_header(hdu,'SNR_R', SNR_5130_R)
                hdu = GLOBALutils.update_header(hdu,'INST', 'WHAPOS')
                hdu = GLOBALutils.update_header(hdu,'RESOL', '24000')
                hdu = GLOBALutils.update_header(hdu,'PIPELINE', 'CERES')
                hdu = GLOBALutils.update_header(hdu,'XC_MIN', XC_min)
                hdu = GLOBALutils.update_header(hdu,'BJD_OUT', bjd_out)
                hdu = GLOBALutils.update_header(hdu,'DATE', str(datetime.datetime.now()).replace(' ','T'))
                
                line_out = "%-15s %18.8f %9.4f %7.4f %9.3f %5.3f   bachesutils   ceres   24000 %6d %5.2f %5.2f %5.1f %4.2f %5.2f %6.1f %4d %7.4f %7.4f %7.4f %s\n"%\
                    (obname, bjd_out, RV, RVerr2, BS, BSerr, T_eff_epoch, logg_epoch, Z_epoch, vsini_epoch, XC_min, disp_epoch,\
                    hd['EXPTIME'], SNR_5130_R, airmass, shift_pix[0], rv_drift[0], ccf_pdf) # CG rv_drift[0]
                f_res.write(line_out)
                if (os.access( dirout + fout,os.F_OK)):
                    os.remove( dirout + fout)
                hdu.data = hdu.data.astype(np.float32)
                hdu.writeto( dirout + fout )


                # Activity indicators + merged 1D rest-frame spectrum
                spectrum_1d_path = os.path.join(dirout, fout.replace('.sp.fits', '.1d.fits'))
                print("\tspectrum_1d_path:",spectrum_1d_path)
                try:
                    from ceres3.utils.activity import compute_activity
                    activity = compute_activity(spec, instrument='fideos', output_1d_path=spectrum_1d_path,
                                        teff=T_eff if T_eff != -999 else None)
                except Exception:
                    print("Exception")
                    print("setting activity to None")

            #else:
            #    print "\t\tReading spectral file from", fout
            #    spec = pyfits.getdata( fout )
    except (FileNotFoundError, KeyError) as e:
        print(f"[FILE ERROR] Falta un archivo de calibración o una cabecera FITS en {fsim}: {e}")
    except ValueError as e:
        print(f"[MATH/VALUE ERROR] Error en dimensiones de matriz o ajustes en {fsim}: {e}")
    except Exception as e:
        print(f"[UNEXPECTED ERROR] Error no previsto en {fsim}: {e}")
        traceback.print_exc()

#os.remove(dirin+'/justdo.txt') # CG. Do not return an error if it does not exist
f_res.close()
np.savetxt(drift_file,drift_out)
