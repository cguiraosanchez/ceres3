"""Structured Whapos CCD reduction and radial-velocity pipeline.

This entrypoint preserves the products and numerical routines used by
``whapospipe.py`` while separating argument parsing, calibration, wavelength
solution, and per-science-frame work. Importing the module has no side effects;
run it with ``python -m ceres3.instruments.whaposAIpipe``.

The extraction, wavelength-fitting, barycentric, lunar, and CCF algorithms are
kept in CERES' established utility modules. This driver owns orchestration,
validation, caching, output paths, and recovery from individual bad frames.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import pickle
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import ephem
import numpy as np
import scipy
import scipy.interpolate
import scipy.signal
import statsmodels.api as sm
from astropy.io import fits
from math import radians

from ceres3.ext import Marsh
from ceres3.instruments import whaposAIutils
from ceres3.utils import correlation, globalutils as GLOBALutils, jplephem

LOWESS = sm.nonparametric.lowess
C_KMS = 299792.458
C_MS = 299792458.0


@dataclass(frozen=True)
class PipelineConfig:
    """User-selected paths and modes, independent of pipeline state."""

    input_dir: Path
    output_dir: Path
    avoid_plot: bool = False
    do_class: bool = False
    just_extract: bool = False
    npools: int = 1
    object_to_do: str = "all"
    reference_file: Path | None = None
    absolute_reference: Path | None = None
    just_bias: bool = False
    just_flats: bool = False
    just_thar: bool = False


@dataclass(frozen=True)
class InstrumentConfig:
    """Whapos detector and fit parameters; aperture is resolved from the flat."""

    gain: float = 1.0
    read_noise: float = 1.5
    aperture: int = 3
    trace_degree: int = 4
    marsh_algorithm: int = 0
    marsh_spacing: float = 0.4
    marsh_degree: int = 3
    marsh_sigma: float = 10.0
    cosmic_sigma: float = 10.0
    min_column: int = 0
    max_column: int = 1756
    order_zero: int = 0
    wavelength_degree_x: int = 4
    wavelength_degree_order: int = 8
    min_global_lines: int = 500
    max_wavelength_rms: float = 120.0
    inverse_order: bool = True
    force_preprocess: bool = False
    force_flat_extract: bool = False
    force_thar_extract: bool = False
    force_science_extract: bool = False
    force_wavelength_fit: bool = False
    force_stellar_parameters: bool = False


class WhaposPipeline:
    """Coordinate one night's calibration and science reduction."""

    def __init__(self, config: PipelineConfig):
        if config.npools < 1:
            raise ValueError("npools must be at least 1")
        if not config.input_dir.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {config.input_dir}")

        self.config = config
        self.instrument = InstrumentConfig(
            force_preprocess=config.just_bias or config.just_flats
        )
        self.gain = self.instrument.gain
        self.read_noise = self.instrument.read_noise
        self.package_root = Path(__file__).resolve().parents[1]
        self.order_dir = self.package_root / "instruments" / "data" / "whapos_wavcals"
        self.model_path = self.package_root / "data" / "COELHO_MODELS" / "R_40000b"
        self.mask_dir = self.package_root / "utils" / "data" / "xc_masks"
        self.reffile = config.reference_file or config.input_dir / "reffile.txt"
        self.log_path = config.output_dir / "night.log"
        self.results_path = config.output_dir / "proc" / "results.txt"
        self.drift_path = config.output_dir / "drifts.txt"
        self.barycenter_dir = ""
        self.ephemeris = "DEc403"
        self.drift_rows: list[np.ndarray] = []

    def run(self) -> None:
        """Run calibration stages, then reduce each selected science frame."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "proc").mkdir(parents=True, exist_ok=True)
        print("\n\tWHAPOS Baader-Planetarium PIPELINE\n")
        print(f"\tRAW data is in {self.config.input_dir}")
        print(f"\tProducts of reduction will be in {self.config.output_dir}\n")

        with self.results_path.open("w", encoding="utf-8") as results:
            self.results = results
            self._classify_frames()
            self._prepare_calibrations()
            if self.config.just_bias:
                return
            self._extract_flat()
            if self.config.just_flats:
                return
            self._calibrate_thars()
            self._estimate_absolute_reference()
            if self.config.just_thar:
                return
            self._reduce_science_frames()

    def _classify_frames(self) -> None:
        """Classify FITS files once and reject incomplete observing nights."""
        (self.biases, self.darks, self.flats, self.flats_old,
         self.thars, self.thars_old, self.science, self.dark_science) = \
            whaposAIutils.FileClassify(str(self.config.input_dir) + os.sep, str(self.log_path))
        for label, files in (("biases", self.biases), ("darks", self.darks),
                             ("flats", self.flats), ("ThAr", self.thars),
                             ("science", self.science)):
            print(f"\tfound {label}:\t{len(files)}")
        if len(self.flats) == 0 or len(self.thars) == 0:
            raise RuntimeError("No flats or ThAr files found to work with")

    def _prepare_calibrations(self) -> None:
        """Create missing master frames and trace, or load cached products."""
        out = self.config.output_dir
        bias_path = out / "MasterBias.fits"
        flat_path = out / "MasterFlat.fits"
        trace_path = out / "trace.pkl"
        flat_rebuilt = False
        preprocess = (self.instrument.force_preprocess or not bias_path.exists()
                      or not flat_path.exists() or not trace_path.exists())

        if preprocess:
            self._make_master_bias(bias_path)
            if not self.config.just_bias:
                self._make_master_flat(flat_path, trace_path)
                flat_rebuilt = True
        elif self.config.just_flats:
            self._make_master_flat(flat_path, trace_path)
            flat_rebuilt = True

        if not bias_path.exists():
            raise FileNotFoundError(f"Master bias is missing: {bias_path}")
        with fits.open(bias_path, memmap=False) as hdul:
            self.master_bias = np.asarray(hdul[0].data, dtype=np.float32)
            self.bias_header = hdul[0].header.copy()
            self.read_noise = float(hdul[0].header.get("RON", self.instrument.read_noise))
            self.gain = float(hdul[0].header.get("EGAIN", self.instrument.gain))

        if not self.config.just_bias:
            if not flat_path.exists() or not trace_path.exists():
                raise FileNotFoundError("Master flat or order trace was not created")
            with fits.open(flat_path, memmap=False) as hdul:
                self.master_flat = np.asarray(hdul[0].data, dtype=np.float32)
                self.flat_header = hdul[0].header.copy()
            with trace_path.open("rb") as stream:
                trace = pickle.load(stream, encoding="latin1")
            self.trace_coefficients = np.asarray(trace["c"])
            if not flat_rebuilt:
                aperture = self._set_aperture_from_flat(
                    self.master_flat, self.trace_coefficients
                )
                self.trace_coefficients, count = whaposAIutils.good_orders(
                    self.trace_coefficients, len(self.trace_coefficients),
                    self.master_flat.shape[1], self.master_flat.shape[0], aperture,
                )
                if count == 0:
                    raise RuntimeError("No usable echelle orders found in the master flat")
            self.order_count = len(self.trace_coefficients)

        self._make_master_darks()

    def _make_master_bias(self, destination: Path) -> None:
        """Combine bias frames, handling single-frame and no-bias nights."""
        if len(self.biases):
            with fits.open(self.biases[0], memmap=False) as hdul:
                self.gain = float(hdul[0].header.get("EGAIN", self.gain))
            master = GLOBALutils.MedianCombine_simple(self.biases, ZF=0.0)
            if len(self.biases) > 1:
                self.read_noise = float(np.around(
                    whaposAIutils.compute_RON(master, self.biases, gain=self.gain), 1
                ))
            else:
                self.read_noise = self.instrument.read_noise
        else:
            with fits.open(self.flats[0], memmap=False) as hdul:
                master = np.zeros_like(hdul[0].data, dtype=np.float32)
                self.gain = float(hdul[0].header.get("EGAIN", self.gain))
            self.read_noise = self.instrument.read_noise

        hdu = fits.PrimaryHDU(np.asarray(master, dtype=np.float32))
        hdu.header["RON"] = self.read_noise
        hdu.header["EGAIN"] = self.gain
        hdu.header["DATE"] = self._now()
        hdu.writeto(destination, overwrite=True)
        print(f"\tMaster bias written: {destination}")

    def _make_master_flat(self, destination: Path, trace_path: Path) -> None:
        """Build the master flat, copy detector metadata, and trace orders."""
        with fits.open(self.config.output_dir / "MasterBias.fits", memmap=False) as hdul:
            bias_data = np.asarray(hdul[0].data)
            bias_header = hdul[0].header.copy()
        master = GLOBALutils.MedianCombine_simple(self.flats, ZF=bias_data)
        master = np.ascontiguousarray(np.fliplr(master.T), dtype=np.float32)
        hdu = fits.PrimaryHDU(master)
        hdu.header["BDATE"] = bias_header["DATE"]
        hdu.header["DATE"] = self._now()
        hdu = whaposAIutils.fill_list_header(hdu, self.flats)
        hdu.writeto(destination, overwrite=True)
        self._print_detector_metadata(self.flats[0])

        coefficients, _ = GLOBALutils.get_them(
            master.T, 4.0, self.instrument.trace_degree,
            maxords=-1, nsigmas=5, mode=1,
        )
        aperture = self._set_aperture_from_flat(master, coefficients)
        coefficients, count = whaposAIutils.good_orders(
            coefficients, len(coefficients), master.shape[1], master.shape[0],
            aperture,
        )
        if count == 0:
            raise RuntimeError("No usable echelle orders found in the master flat")
        trace = {"c": coefficients, "nord": count,
                 "bdate": hdu.header["BDATE"], "fdate": hdu.header["DATE"]}
        with trace_path.open("wb") as stream:
            pickle.dump(trace, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\tMaster flat and trace written: {destination}")

    def _set_aperture_from_flat(
        self, master_flat: np.ndarray, trace_coefficients: np.ndarray
    ) -> int:
        """Set aperture to half the median spatial FWHM measured on the flat."""
        flat = np.asarray(master_flat, dtype=float)
        coefficients = np.asarray(trace_coefficients, dtype=float)
        if flat.ndim != 2 or coefficients.ndim != 2 or len(coefficients) == 0:
            raise ValueError("master flat and order traces must be non-empty 2-D arrays")

        ncolumns, nrows = flat.shape
        columns = np.unique(np.linspace(
            0, ncolumns - 1, min(ncolumns, 64), dtype=int
        ))
        centers = np.asarray([
            np.polyval(coefficient, columns) for coefficient in coefficients
        ])
        widths: list[float] = []
        for column_index, column in enumerate(columns):
            column_centers = centers[:, column_index]
            for order_index, center in enumerate(column_centers):
                other_centers = np.delete(column_centers, order_index)
                if not other_centers.size or not np.isfinite(center):
                    continue
                spacing = float(np.min(np.abs(other_centers - center)))
                radius = min(12, max(4, int(spacing // 3)))
                pixel = int(np.rint(center))
                lo, hi = pixel - radius, pixel + radius
                if lo < 0 or hi >= nrows:
                    continue

                profile = flat[column, lo:hi + 1]
                if not np.all(np.isfinite(profile)):
                    continue
                baseline = 0.5 * (np.median(profile[:2]) + np.median(profile[-2:]))
                peak_index = int(np.argmax(profile))
                peak = profile[peak_index]
                if peak <= baseline:
                    continue
                half_max = baseline + 0.5 * (peak - baseline)

                left = peak_index
                while left > 0 and profile[left - 1] >= half_max:
                    left -= 1
                right = peak_index
                while right < len(profile) - 1 and profile[right + 1] >= half_max:
                    right += 1
                if left == 0 or right == len(profile) - 1:
                    continue

                left_denominator = profile[left] - profile[left - 1]
                right_denominator = profile[right] - profile[right + 1]
                if left_denominator <= 0 or right_denominator <= 0:
                    continue
                left_crossing = left - 1 + (half_max - profile[left - 1]) / left_denominator
                right_crossing = right + (profile[right] - half_max) / right_denominator
                width = float(right_crossing - left_crossing)
                if np.isfinite(width) and width > 0:
                    widths.append(width)

        if len(widths) < 10:
            raise RuntimeError(
                "Could not measure enough order profiles to determine the aperture"
            )
        median_fwhm = float(np.median(widths))
        aperture = max(1, int(np.ceil(median_fwhm / 2.0))) + 1
        self.instrument = replace(self.instrument, aperture=aperture)
        print(
            f"\tEstimated extraction aperture: {aperture} px "
            f"(median order FWHM {median_fwhm:.1f} px from {len(widths)} profiles)"
        )
        return aperture

    @staticmethod
    def _print_detector_metadata(flat_path: str) -> None:
        """Print camera metadata when recorded in the source flat header."""
        header = fits.getheader(flat_path)
        size = header.get("CCD-SIZE", header.get("CCDSIZE"))
        if size is None:
            size = f"{header.get('NAXIS1', '?')} x {header.get('NAXIS2', '?')} pixels"
        pixel_size = header.get("PIXEL-SIZE", header.get("PIXSIZE"))
        if pixel_size is None:
            xpix = header.get("XPIXSZ", header.get("XPIXELSZ"))
            ypix = header.get("YPIXSZ", header.get("YPIXELSZ"))
            pixel_size = f"{xpix} x {ypix}" if xpix is not None and ypix is not None else "not recorded"
        binning = header.get("BINNING")
        if binning is None:
            xbin, ybin = header.get("XBINNING"), header.get("YBINNING")
            binning = f"{xbin} x {ybin}" if xbin is not None and ybin is not None else "not recorded"
        print(f"\tCCD size: {size}")
        print(f"\tPixel size: {pixel_size}")
        print(f"\tBinning: {binning}")
        print(f"\tCCD temperature: {header.get('CCD-TEMP', 'not recorded')}")

    def _make_master_darks(self) -> None:
        """Write dark masters grouped by exposure time for science subtraction."""
        if len(self.darks) == 0:
            return
        times = GLOBALutils.get_dark_times(self.darks, key="EXPTIME")
        for exposure in times:
            group = GLOBALutils.get_tdarks(self.darks, exposure, key="EXPTIME")
            dark = GLOBALutils.MedianCombine_simple(group, ZF=self.master_bias)
            fits.PrimaryHDU(np.asarray(dark, dtype=np.float32)).writeto(
                self.config.output_dir / f"Dark_{exposure}.fits", overwrite=True
            )

    def _extract_flat(self) -> None:
        """Create or load the flat extraction profile and normalized spectra."""
        out = self.config.output_dir
        profile_path = out / "P.fits"
        optimal_path = out / "S_flat.fits"
        simple_path = out / "S_flat_simple.fits"
        force = self.instrument.force_flat_extract or self.config.just_flats
        if force or not all(path.exists() for path in (profile_path, optimal_path, simple_path)):
            x = np.arange(self.master_flat.shape[0])
            centers = np.asarray([np.polyval(row, x) for row in self.trace_coefficients])
            background = GLOBALutils.get_scat(self.master_flat.T, centers, span=8)
            fits.PrimaryHDU(np.asarray(background, dtype=np.float32)).writeto(
                out / "BKG_flat.fits", overwrite=True
            )
            flat = self.master_flat.T - background
            profile = GLOBALutils.obtain_P(
                flat, self.trace_coefficients, self.instrument.aperture,
                self.read_noise, self.gain, self.instrument.marsh_sigma,
                self.instrument.marsh_spacing, self.instrument.marsh_degree,
                self.instrument.marsh_algorithm, self.instrument.min_column,
                self.instrument.max_column, self.config.npools,
            )
            simple = GLOBALutils.simple_extraction(
                flat, self.trace_coefficients, self.instrument.aperture,
                self.instrument.min_column, self.instrument.max_column, self.config.npools,
            )
            optimal = GLOBALutils.optimal_extraction(
                flat, profile, self.trace_coefficients, self.instrument.aperture,
                self.read_noise, self.gain, self.instrument.marsh_spacing,
                self.instrument.cosmic_sigma, self.instrument.min_column,
                self.instrument.max_column, self.config.npools,
            )
            simple = GLOBALutils.invert(simple)
            optimal = GLOBALutils.invert(optimal)
            fits.PrimaryHDU(np.asarray(profile, dtype=np.float32)).writeto(profile_path, overwrite=True)
            fits.PrimaryHDU(np.asarray(simple, dtype=np.float32)).writeto(simple_path, overwrite=True)
            hdu = fits.PrimaryHDU(np.asarray(optimal, dtype=np.float32))
            hdu.header["FDATE"] = self.flat_header.get("DATE", "")
            hdu.header["DATE"] = self._now()
            hdu.writeto(optimal_path, overwrite=True)

        self.profile = fits.getdata(profile_path)
        with fits.open(optimal_path, memmap=False) as hdul:
            self.flat_spectrum = np.asarray(hdul[0].data)
        self.flat_simple = fits.getdata(simple_path)
        self.flat_normalized, self.flat_simple_normalized = GLOBALutils.FlatNormalize(
            self.flat_spectrum, self.flat_simple
        )

    def _calibrate_thars(self) -> None:
        """Extract arc spectra and fit or reload each global wavelength solution."""
        mjds = []
        for path in self.thars:
            with fits.open(path, memmap=False) as hdul:
                mjd, _ = whaposAIutils.mjd_fromheader(hdul)
                raw = np.asarray(hdul[0].data, dtype=np.float32) - self.master_bias
            mjds.append(mjd)
            simple_path = self.config.output_dir / f"{Path(path).stem}.spec.simple.fits.S"
            optimal_path = self.config.output_dir / f"{Path(path).stem}.spec.fits.S"
            if self.instrument.force_thar_extract or self.config.just_thar or not simple_path.exists() or not optimal_path.exists():
                image = np.ascontiguousarray(np.fliplr(raw.T))
                background = whaposAIutils.get_scat(image.astype(np.float32), self.profile.astype(np.float32), 1)
                image -= background
                simple = GLOBALutils.simple_extraction(
                    image.T, self.trace_coefficients, self.instrument.aperture,
                    self.instrument.min_column, self.instrument.max_column, self.config.npools,
                )
                optimal = GLOBALutils.optimal_extraction(
                    image.T, self.profile, self.trace_coefficients, self.instrument.aperture,
                    self.read_noise, self.gain, self.instrument.marsh_spacing,
                    self.instrument.cosmic_sigma, self.instrument.min_column,
                    self.instrument.max_column, self.config.npools,
                )
                fits.PrimaryHDU(np.asarray(GLOBALutils.invert(simple), dtype=np.float32)).writeto(simple_path, overwrite=True)
                hdu = fits.PrimaryHDU(np.asarray(GLOBALutils.invert(optimal), dtype=np.float32))
                hdu.header["BDATE"] = fits.getheader(self.config.output_dir / "MasterBias.fits").get("DATE", "")
                hdu.header["DATE"] = self._now()
                hdu.writeto(optimal_path, overwrite=True)

        order_file = self.config.output_dir / "order_id.pkl"
        if order_file.exists():
            with order_file.open("rb") as stream:
                order_info = pickle.load(stream, encoding="latin1")
            self.start = int(order_info["start"])
            self.end = int(order_info["end"])
            self.offset = float(order_info["offset"])
            self.first_physical_order = int(order_info["idorder0"])
        else:
            first_arc = self.config.output_dir / f"{Path(self.thars[0]).stem}.spec.fits.S"
            spectrum = fits.getdata(first_arc)[:, 1, :]
            order_at_69, self.offset = GLOBALutils.identify_order(
                spectrum, str(self.order_dir / "whapos_69.iwdat"), window=100, di=0.5
            )
            self.start = order_at_69 - 9
            self.end = self.start + 48
            self.first_physical_order = 60
            order_info = {"start": self.start, "end": self.end,
                          "offset": self.offset, "idorder0": self.first_physical_order}
            with order_file.open("wb") as stream:
                pickle.dump(order_info, stream, protocol=pickle.HIGHEST_PROTOCOL)
        if self.end - self.start + 1 > self.order_count:
            raise ValueError("Wavelength order range exceeds the traced-order count")

        order = np.argsort(mjds)
        self.thar_times = np.asarray(mjds)[order]
        self.thars = np.asarray(self.thars)[order]
        self.ntotal = self.end - self.start + 1
        self.wav_solutions = {}
        self.thar_products = {}
        for index, path in enumerate(self.thars):
            solution_path = self.config.output_dir / f"{Path(path).stem}.wavsolpars.pkl"
            calibrated_path = self.config.output_dir / f"{Path(path).stem}.sp.fits"
            if self.instrument.force_wavelength_fit or not solution_path.exists() or not calibrated_path.exists():
                solution, calibrated = self._fit_one_thar(path, index)
                with solution_path.open("wb") as stream:
                    pickle.dump(solution, stream, protocol=pickle.HIGHEST_PROTOCOL)
                fits.PrimaryHDU(np.asarray(calibrated, dtype=np.float32)).writeto(calibrated_path, overwrite=True)
            else:
                with solution_path.open("rb") as stream:
                    solution = pickle.load(stream, encoding="latin1")
                calibrated = fits.getdata(calibrated_path)
            self.wav_solutions[str(path)] = solution
            self.thar_products[str(path)] = calibrated
        self.reference_solution = self.wav_solutions[str(self.thars[0])]
        first_product = self.thar_products[str(self.thars[0])]
        self.reference_pixel = first_product[0, 25, len(self.reference_solution["thar_order"]) // 2]

    def _fit_one_thar(self, path: str, arc_index: int) -> tuple[dict[str, Any], np.ndarray]:
        """Fit per-order line lists, then one robust 2-D wavelength surface."""
        with fits.open(path, memmap=False) as hdul:
            mjd, _ = whaposAIutils.mjd_fromheader(hdul)
        extracted_path = self.config.output_dir / f"{Path(path).stem}.spec.fits.S"
        with fits.open(extracted_path, memmap=False) as hdul:
            extracted = np.asarray(hdul[0].data)
        lines = extracted[:, 1, :]
        flux_variance = extracted[:, 2, :]
        pixels: list[np.ndarray] = []
        wavelengths: list[np.ndarray] = []
        orders: list[np.ndarray] = []
        centroids: list[np.ndarray] = []
        centroid_uncertainties: list[np.ndarray] = []
        wavelength_uncertainties: list[np.ndarray] = []
        sigmas: list[np.ndarray] = []
        intensities: list[np.ndarray] = []
        global_zero = None
        last_order = None

        for row, physical_order in zip(
            range(self.start, self.end + 1),
            range(self.first_physical_order, self.first_physical_order + self.ntotal),
        ):
            order_text = f"{physical_order:02d}" if physical_order < 10 else str(physical_order)
            normalization = scipy.signal.medfilt(self.flat_spectrum[row, 1], 31)
            raw = lines[row] / normalization
            arc_order = raw - scipy.signal.medfilt(raw, 101)
            pixel_variance = np.asarray(flux_variance[row], dtype=float)
            weights = np.zeros_like(pixel_variance)
            valid_variance = (np.isfinite(pixel_variance) & (pixel_variance > 0)
                              & np.isfinite(normalization) & (normalization > 0))
            weights[valid_variance] = (
                normalization[valid_variance] / np.sqrt(pixel_variance[valid_variance])
            )
            fit = whaposAIutils.Initial_Wav_Calibration(
                str(self.order_dir / f"whapos_{order_text}.iwdat"), arc_order,
                row, weights, rmsmax=400, minlines=10, FixEnds=False,
                Dump_Argon=False, Cheby=True,
                porder=self.instrument.wavelength_degree_x,
                rough_shift=self.offset, do_xc=False, line_width=6,
                pixelization=True, return_centroid_uncertainty=True,
            )
            (coeff_wav, _, pix, wav, _, _, line_centers, line_sigmas,
             line_intensity, line_centroid_uncertainties) = fit
            if physical_order == 74:
                global_zero = GLOBALutils.Cheby_eval(coeff_wav, 0.5 * (len(arc_order) - 1), len(arc_order))
            wavelength_derivative = (
                GLOBALutils.Cheby_eval(coeff_wav, pix + 0.5, len(arc_order))
                - GLOBALutils.Cheby_eval(coeff_wav, pix - 0.5, len(arc_order))
            )
            line_wavelength_uncertainties = (
                np.abs(wavelength_derivative) * line_centroid_uncertainties
            )
            valid_lines = (np.isfinite(line_wavelength_uncertainties)
                           & (line_wavelength_uncertainties > 0))
            pixels.append(pix[valid_lines])
            wavelengths.append(wav[valid_lines])
            orders.append(np.full(np.count_nonzero(valid_lines), physical_order))
            centroids.append(line_centers[valid_lines])
            centroid_uncertainties.append(line_centroid_uncertainties[valid_lines])
            wavelength_uncertainties.append(line_wavelength_uncertainties[valid_lines])
            sigmas.append(line_sigmas[valid_lines])
            intensities.append(line_intensity[valid_lines])
            last_order = arc_order

        if global_zero is None or last_order is None:
            raise RuntimeError("Reference order 74 was not successfully wavelength-calibrated")
        all_pixels = np.concatenate(pixels)
        all_wavelengths = np.concatenate(wavelengths)
        all_orders = np.concatenate(orders)
        all_centroids = np.concatenate(centroids)
        all_centroid_uncertainties = np.concatenate(centroid_uncertainties)
        all_wavelength_uncertainties = np.concatenate(wavelength_uncertainties)
        all_sigmas = np.concatenate(sigmas)
        all_intensities = np.concatenate(intensities)
        global_weights = 1.0 / all_wavelength_uncertainties
        global_weights /= np.median(global_weights)
        initial = np.zeros((min(self.instrument.wavelength_degree_x,
                                self.instrument.wavelength_degree_order) + 1)
                           * (2 * max(self.instrument.wavelength_degree_x,
                                      self.instrument.wavelength_degree_order)
                              - min(self.instrument.wavelength_degree_x,
                                    self.instrument.wavelength_degree_order) + 2) // 2)
        initial[0] = 74 * global_zero
        fit = GLOBALutils.Fit_Global_Wav_Solution(
            all_pixels, all_wavelengths, all_orders, global_weights,
            initial, Cheby=True, maxrms=self.instrument.max_wavelength_rms,
            Inv=self.instrument.inverse_order,
            minlines=self.instrument.min_global_lines,
            order0=self.instrument.order_zero, ntotal=self.ntotal,
            npix=len(last_order), nx=self.instrument.wavelength_degree_x,
            nm=self.instrument.wavelength_degree_order,
        )
        coefficients, gpix, gord, gwav, accepted, rms, residuals = fit
        output = np.zeros((2, self.ntotal, lines.shape[1]))
        x = np.arange(lines.shape[1])
        for out_order, physical_order in enumerate(
            range(self.first_physical_order, self.first_physical_order + self.ntotal)
        ):
            chebyshev = GLOBALutils.Calculate_chebs(
                x, physical_order, Inverse=self.instrument.inverse_order,
                order0=self.instrument.order_zero, ntotal=self.ntotal,
                npix=len(last_order), nx=self.instrument.wavelength_degree_x,
                nm=self.instrument.wavelength_degree_order,
            )
            output[0, out_order] = (1.0 / physical_order) * GLOBALutils.Joint_Polynomial_Cheby(
                coefficients, chebyshev, self.instrument.wavelength_degree_x,
                self.instrument.wavelength_degree_order,
            )
            output[1, out_order] = lines[self.start + out_order]
        solution = {
            "p1": coefficients, "mjd": mjd, "G_pix": gpix, "G_ord": gord,
            "G_wav": gwav, "II": accepted, "rms_ms": rms,
            "G_res": residuals, "All_Centroids": all_centroids,
            "All_Orders": all_orders, "All_Sigmas": all_sigmas,
            "All_Centroid_Uncertainties": all_centroid_uncertainties,
            "All_Wavelength_Uncertainties": all_wavelength_uncertainties,
            "All_Pixel_Centers": all_pixels, "All_Wavelengths": all_wavelengths,
            "thar_order": last_order, "All_Intensities": all_intensities,
        }
        return solution, output

    def _estimate_absolute_reference(self) -> None:
        """Optionally compare the night solution against an absolute pickle."""
        reference = self.config.absolute_reference
        if reference is None or not reference.is_file():
            return
        with reference.open("rb") as stream:
            absolute = pickle.load(stream, encoding="latin1")
        solution = self.reference_solution
        product = self.thar_products[str(self.thars[len(self.thars) // 2])]
        shift = GLOBALutils.Global_Wav_Solution_vel_shift(
            solution["G_pix"], solution["G_wav"], solution["G_ord"],
            np.ones(len(solution["G_ord"])), absolute["p1"], Cheby=True,
            Inv=True, maxrms=self.instrument.max_wavelength_rms,
            minlines=self.instrument.min_global_lines,
            order0=self.instrument.order_zero, ntotal=self.ntotal,
            npix=product.shape[2], nx=self.instrument.wavelength_degree_x,
            nm=self.instrument.wavelength_degree_order,
        )
        shift_ppm, _, _, _, _, rms, residuals = shift
        info_path = self.config.output_dir / "order_id.pkl"
        with info_path.open("rb") as stream:
            info = pickle.load(stream, encoding="latin1")
        info.update({"absref": str(reference),
                     "absp_shift": (1e-6 * shift_ppm) * C_MS,
                     "werr": rms / np.sqrt(max(len(residuals), 1))})
        with info_path.open("wb") as stream:
            pickle.dump(info, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def _reduce_science_frames(self) -> None:
        """Filter science targets and isolate errors to the failing exposure."""
        all_science = np.hstack((self.science, self.dark_science))
        justdo_path = self.config.input_dir / "justdo.txt"
        if justdo_path.is_file():
            selected = [self.config.input_dir / item.strip()
                        for item in justdo_path.read_text(encoding="utf-8").splitlines()
                        if item.strip()]
            all_science = np.asarray([str(path) for path in selected])
        files = []
        for path in all_science:
            name = whaposAIutils.get_name(path)
            if self.config.object_to_do == "all" or name == self.config.object_to_do:
                files.append((str(path), name))
        print("\n\t--> The following targets will be processed:")
        for path, name in files:
            print(f"\t\t{name}\t{path}")
        moon_settings = self._read_moon_settings()
        for path, name in files:
            try:
                self._reduce_one_science(path, name, moon_settings)
            except (FileNotFoundError, KeyError) as error:
                print(f"[FILE ERROR] Missing calibration or FITS header for {path}: {error}")
            except (ValueError, IndexError, FloatingPointError) as error:
                print(f"[DATA ERROR] Invalid dimensions or fit for {path}: {error}")
            except Exception as error:
                print(f"[UNEXPECTED ERROR] Failed to reduce {path}: {error}")
                traceback.print_exc()
        np.savetxt(self.drift_path, np.vstack(self.drift_rows) if self.drift_rows else np.empty((0, 5)))

    def _read_moon_settings(self) -> dict[str, bool]:
        """Read optional per-file lunar contamination decisions."""
        settings_path = self.config.input_dir / "moon_corr.txt"
        if not settings_path.is_file():
            return {}
        settings = {}
        for line in settings_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2:
                settings[fields[0]] = fields[1] != "0"
        return settings

    def _reduce_one_science(self, path: str, object_name: str,
                            moon_settings: dict[str, bool]) -> None:
        """Correct, extract, calibrate, and analyze a single science exposure."""
        header = fits.getheader(path)
        with fits.open(path, memmap=False) as hdul:
            mjd, _ = whaposAIutils.mjd_fromheader(hdul)
            raw = np.asarray(hdul[0].data, dtype=np.float32)
        exposure = float(header["EXPTIME"])
        date_obs = str(header["DATE-OBS"])
        altitude, latitude, longitude = 2896.0, 37.0642, 3.3847
        try:
            _, ra, dec, _ = GLOBALutils.simbad_coords(object_name, mjd)
        except Exception:
            ra, dec = -999.0, -999.0
        simbad_ra, simbad_dec = ra, dec
        header_ra, header_dec = GLOBALutils.getcoords(object_name, mjd, filen=str(self.reffile))
        if header_ra != 0 and header_dec != 0:
            ra, dec = header_ra, header_dec
        if ra == -999 or dec == -999:
            raise ValueError(f"No usable coordinates for {object_name}")

        airmass = whaposAIutils.get_airmass(ra, dec, latitude, longitude, altitude,
                                          date_obs.replace("T", " "))
        self._set_observer(latitude, longitude, altitude)
        bary_result = jplephem.doppler_fraction(ra / 15.0, dec, int(mjd), mjd % 1, 1, 0.0)
        bary_factor = 1.0 + bary_result["frac"][0]
        bary_velocity = (bary_factor - 1.0) * C_KMS
        delay = jplephem.pulse_delay(ra / 15.0, dec, int(mjd), mjd % 1, 1, 0.0)["delay"][0]
        bary_mjd = mjd + delay / 86400.0
        moon_info = self._moon_velocity(ra, dec, mjd, date_obs, latitude, longitude)
        reference_velocity = bary_velocity + moon_info["moon_velocity"]
        moon_enabled = moon_settings.get(Path(path).name, False)

        dark_path = self.config.output_dir / f"Dark_{exposure}.fits"
        dark = fits.getdata(dark_path) if dark_path.exists() else np.zeros_like(self.master_bias)
        image = np.ascontiguousarray(np.fliplr((raw - self.master_bias - dark).T))
        profile = self.profile.copy()
        stem = Path(path).stem
        background_path = self.config.output_dir / f"BKG_{stem}.fits"
        if background_path.exists():
            background = fits.getdata(background_path)
        else:
            background = whaposAIutils.get_scat(image, profile, 1)
            fits.PrimaryHDU(np.asarray(background, dtype=np.float32)).writeto(background_path)
        image -= background

        simple_path = self.config.output_dir / f"{stem}.spec.simple.fits.S"
        optimal_path = self.config.output_dir / f"{stem}.spec.fits.S"
        background_spec_path = self.config.output_dir / f"{stem}.spec.bac.fits.S"
        force = self.instrument.force_science_extract
        if force or not all(item.exists() for item in (simple_path, optimal_path, background_spec_path)):
            simple = GLOBALutils.simple_extraction(
                image.T, self.trace_coefficients, self.instrument.aperture,
                self.instrument.min_column, self.instrument.max_column, self.config.npools,
            )
            background_spec = GLOBALutils.simple_extraction(
                background.T, self.trace_coefficients, self.instrument.aperture,
                self.instrument.min_column, self.instrument.max_column, self.config.npools,
            )
            optimal = GLOBALutils.optimal_extraction(
                image.T, profile, self.trace_coefficients, self.instrument.aperture,
                self.read_noise, self.gain, self.instrument.marsh_spacing,
                self.instrument.cosmic_sigma, self.instrument.min_column,
                self.instrument.max_column, self.config.npools,
            )
            for destination, array in ((simple_path, GLOBALutils.invert(simple)),
                                       (optimal_path, GLOBALutils.invert(optimal)),
                                       (background_spec_path, GLOBALutils.invert(background_spec))):
                fits.PrimaryHDU(np.asarray(array, dtype=np.float32)).writeto(destination, overwrite=True)
        simple_spectrum = fits.getdata(simple_path)
        optimal_spectrum = fits.getdata(optimal_path)
        background_spectrum = fits.getdata(background_spec_path)

        arc_index, seconds = whaposAIutils.find_closest_index(mjd, self.thar_times)
        arc_path = str(self.thars[arc_index])
        arc_solution = self.wav_solutions[arc_path]
        reference = self.reference_solution
        shift_result = GLOBALutils.Global_Wav_Solution_vel_shift(
            arc_solution["All_Pixel_Centers"], arc_solution["All_Wavelengths"],
            arc_solution["All_Orders"], np.ones(arc_solution["All_Intensities"].shape),
            reference["p1"], Cheby=True, Inv=True,
            maxrms=self.instrument.max_wavelength_rms,
            minlines=self.instrument.min_global_lines,
            order0=self.instrument.order_zero, ntotal=self.ntotal,
            npix=len(arc_solution["thar_order"]),
            nx=self.instrument.wavelength_degree_x,
            nm=self.instrument.wavelength_degree_order,
        )
        shift_ppm, _, _, _, _, rms, residuals = shift_result
        error = rms / np.sqrt(max(float(len(residuals)), 1.0))
        drift = float(np.asarray(shift_ppm).flat[0]) * 1e-6 * C_MS
        if error < 6.0:
            self.drift_rows.append(np.array([mjd, drift, error, exposure, 1.0]))
        else:
            print("\tWarning: wavelength RMS too high; setting instrumental drift to zero")
            drift = 0.0
        np.savetxt(self.drift_path, np.vstack(self.drift_rows) if self.drift_rows else np.empty((0, 5)))
        ref_pixel = float(self.reference_pixel)
        shifted_reference_pixel = ref_pixel * (1.0 + drift / C_MS)

        spec = self._build_spectrum(
            image, optimal_spectrum, background_spectrum, bary_factor, drift,
        )
        product = self.config.output_dir / "proc" / f"{stem}.sp.fits"
        hdu = fits.PrimaryHDU(np.asarray(spec, dtype=np.float32))
        self._write_product_header(hdu.header, header, object_name, mjd,
                                   bary_mjd, bary_factor, bary_velocity,
                                   airmass, drift, error, arc_path,
                                   simbad_ra, simbad_dec, ra, dec)
        if not self.config.just_extract:
            analysis = self._analyze_star(spec, object_name, path, header,
                                          moon_info, reference_velocity,
                                          bary_factor, moon_enabled)
            self._write_science_metadata(hdu.header, analysis, bary_mjd)
            ccf_path = self.config.output_dir / "proc" / f"{stem}_XCs_{analysis['spectral_type']}.pdf"
            legacy_prefix = Path(path).name[:-8]
            ccf_pickle = self.config.output_dir / f"{legacy_prefix}{object_name}_XC_{analysis['spectral_type']}.pkl"
            with ccf_pickle.open("wb") as stream:
                pickle.dump(analysis["ccf"], stream, protocol=pickle.HIGHEST_PROTOCOL)
            if not self.config.avoid_plot:
                GLOBALutils.plot_CCF(analysis["ccf"], analysis["moon"], path=str(ccf_path))
            self.results.write(self._format_result(path, object_name, analysis,
                                                   bary_mjd, airmass, shifted_reference_pixel,
                                                   drift, header["EXPTIME"]))
        hdu.writeto(product, overwrite=True)
        if not self.config.just_extract:
            self._write_activity(spec, product, analysis.get("teff"))
        print(f"\tCompleted {path}; closest ThAr is {arc_path} ({seconds:.1f} s)")

    def _set_observer(self, latitude: float, longitude: float, altitude: float) -> None:
        """Configure ephemeris routines for the Sierra Nevada Observatory."""
        radius, r0 = GLOBALutils.JPLR0(latitude, altitude)
        position = GLOBALutils.obspos(longitude, radius, r0)
        jplephem.set_ephemeris_dir(self.barycenter_dir, self.ephemeris)
        jplephem.set_observer_coordinates(*map(float, position))

    def _moon_velocity(self, ra: float, dec: float, mjd: float, date_obs: str,
                       latitude: float, longitude: float) -> dict[str, Any]:
        """Compute moonlight velocity context for the current target exposure."""
        observer = ephem.Observer()
        observer.name = "OSN1.5"
        observer.lat = radians(latitude)
        observer.long = radians(longitude)
        observer.date = date_obs.replace("T", " ")
        moon = ephem.Moon()
        moon.compute(observer)
        moon_track = jplephem.object_track("Moon", int(mjd), float(mjd % 1), 1, 0.0)
        moon_bary = jplephem.barycentric_object_track("Moon", int(mjd), float(mjd % 1), 1, 0.0)
        sun_bary = jplephem.barycentric_object_track("Sun", int(mjd), float(mjd % 1), 1, 0.0)
        doppler = jplephem.object_doppler("Moon", int(mjd), mjd % 1, 1, 0.0)
        lunation, state, separation, velocity = GLOBALutils.get_lunar_props(
            ephem, observer, moon_track, moon_bary, sun_bary, doppler, ra, dec
        )
        return {"observer": observer, "moon": moon, "lunation": lunation,
                "state": state, "separation": separation, "moon_velocity": velocity}

    def _build_spectrum(self, image: np.ndarray, extracted: np.ndarray,
                        background: np.ndarray, bary_factor: float,
                        drift: float) -> np.ndarray:
        """Assemble the legacy 11-plane, order-by-pixel spectral product."""
        spec = np.zeros((11, self.ntotal, image.shape[0]), dtype=np.float64)
        x = np.arange(image.shape[0])
        for output_order, order in enumerate(range(self.start, self.end + 1)):
            physical_order = self.first_physical_order + output_order
            basis = GLOBALutils.Calculate_chebs(
                x, physical_order, Inverse=self.instrument.inverse_order,
                order0=self.instrument.order_zero, ntotal=self.ntotal,
                npix=len(x), nx=self.instrument.wavelength_degree_x,
                nm=self.instrument.wavelength_degree_order,
            )
            wavelengths = bary_factor * (1.0 + drift / C_MS) / physical_order * \
                GLOBALutils.Joint_Polynomial_Cheby(
                    self.reference_solution["p1"], basis,
                    self.instrument.wavelength_degree_x,
                    self.instrument.wavelength_degree_order,
                )
            spec[0, output_order] = GLOBALutils.ToVacuum(wavelengths)
            spec[1, output_order] = extracted[order, 1]
            spec[2, output_order] = extracted[order, 2]
            flat_flux = self.flat_normalized[order, 1]
            valid_flat = flat_flux > 0
            spec[3, output_order, valid_flat] = extracted[order, 1, valid_flat] / flat_flux[valid_flat]
            spec[4, output_order, valid_flat] = extracted[order, 2, valid_flat] * flat_flux[valid_flat] ** 2
            continuum = GLOBALutils.get_cont_single(
                spec[0, output_order], spec[3, output_order], spec[4, output_order],
                ll=1.5, lu=5, nc=3,
            )
            valid = spec[1, output_order] != 0
            spec[5, output_order, valid] = spec[3, output_order, valid] / np.polyval(
                continuum, spec[0, output_order, valid]
            )
            flux = spec[5, output_order]
            flux[~np.isfinite(flux) | (np.abs(flux) <= np.finfo(float).eps)] = 1.0
            ratio = np.ones(image.shape[0])
            normalized_flux = np.nan_to_num(spec[3, output_order], nan=0.0,
                                            posinf=0.0, neginf=0.0)
            ratio[valid] = normalized_flux[valid] / flux[valid]
            spec[6, output_order, valid] = spec[4, output_order, valid] * ratio[valid] ** 2
            spec[7, output_order, valid] = ratio[valid]
            variance = ratio[valid] * flat_flux[valid] / self.gain + \
                background[order, valid] / self.gain + \
                2 * self.instrument.aperture * (self.read_noise / self.gain) ** 2
            spec[8, output_order, valid] = ratio[valid] * flat_flux[valid] / np.sqrt(np.maximum(variance, 1e-12))
            spec[8, output_order] = np.maximum(spec[8, output_order], 0.0)
            bad = (flux > 1.0 + 8.0 / np.maximum(spec[8, output_order], 1e-12)) | (flux > 10) | (flux < -10)
            flux[bad] = 1.0
            derivative = scipy.interpolate.splev(
                np.arange(len(wavelengths)),
                scipy.interpolate.splrep(np.arange(len(wavelengths)), wavelengths, k=3),
                der=1,
            )
            derivative /= np.mean(derivative)
            spec[9, output_order, valid] = flux[valid] * derivative[valid]
            spec[10, output_order, valid] = spec[6, output_order, valid] / derivative[valid] ** 2
        return np.nan_to_num(spec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _analyze_star(self, spec: np.ndarray, object_name: str, path: str,
                      header: fits.Header, moon_info: dict[str, Any],
                      reference_velocity: float,
                      bary_factor: float, moon_enabled: bool) -> dict[str, Any]:
        """Run optional stellar classification and the CCF radial-velocity fit."""
        teff = logg = metallicity = vsini = velocity0 = -999.0
        if self.config.do_class:
            parameters = self.config.output_dir / f"{Path(path).name[:-8]}_stellar_pars.txt"
            if parameters.exists() and not self.instrument.force_stellar_parameters:
                teff, logg, metallicity, vsini, velocity0 = np.loadtxt(parameters, unpack=True)
            else:
                teff, logg, metallicity, vsini, velocity0, _ = correlation.CCF(
                    spec.copy(), model_path=str(self.model_path), npools=self.config.npools
                )
                np.savetxt(parameters, np.asarray([[teff, logg, metallicity, vsini, velocity0]]))
        spectral_type, mask_path = GLOBALutils.get_mask_reffile(
            object_name, reffile=str(self.reffile), base=str(self.mask_dir) + os.sep
        )
        mask_low, mask_high, weights = np.loadtxt(mask_path, unpack=True)
        mask_low, mask_high = GLOBALutils.ToVacuum(mask_low), GLOBALutils.ToVacuum(mask_high)
        centers = 0.5 * (mask_low + mask_high)
        mask_low -= 1.5 * (centers - mask_low)
        mask_high += 1.5 * (mask_high - centers)
        dispersion = GLOBALutils.get_disp(object_name, reffile=str(self.reffile))
        known_dispersion = dispersion != 0
        if not known_dispersion:
            dispersion = vsini if vsini > 4 else 4.0
        for _ in range(2):
            half_width = centers * dispersion / (GLOBALutils.Constants.c / 1e3)
            low, high = centers - half_width, centers + half_width
            velocities, xc, sn, _, order_weights = GLOBALutils.XCor(
                spec, low, high, weights, 0, bary_factor, vel_width=200, vel_step=3,
                spec_order=9, iv_order=10, sn_order=8, max_vel_rough=200,
            )
            self._mask_problem_orders(spec, order_weights)
            averaged = GLOBALutils.Average_CCF(xc, sn, sn_min=0.0, Simple=True,
                                               W=order_weights, start_order=2)
            continuum = LOWESS(scipy.signal.medfilt(averaged, 11), velocities,
                               frac=0.4, it=10, return_sorted=False)
            rough_velocity = velocities[np.argmin(averaged / continuum)]
            rough_state = {
                "rvels": velocities.copy(),
                "rxc_av": (averaged / continuum).copy(),
                "rpred": continuum.copy(),
                "rxc_av_orig": averaged.copy(),
                "rvel0_xc": rough_velocity,
            }
            spline = scipy.interpolate.splrep(velocities, continuum, k=1)
            width = max(20.0, 6.0 * min(dispersion, 30.0))
            velocities, xc, sn, _, order_weights = GLOBALutils.XCor(
                spec, low, high, weights, rough_velocity, bary_factor,
                vel_width=width, vel_step=0.1, spec_order=9,
                iv_order=10, sn_order=8, max_vel_rough=300,
            )
            self._mask_problem_orders(spec, order_weights)
            averaged = GLOBALutils.Average_CCF(xc, sn, sn_min=0.0, Simple=True,
                                               W=order_weights, start_order=2)
            averaged /= scipy.interpolate.splev(velocities, spline)
            moon_sigma = {"M5": 2.5, "K5": 3.3}.get(spectral_type, 4.5)
            fit = GLOBALutils.XC_Final_Fit(
                velocities, averaged, sigma_res=4, horder=8,
                moonv=reference_velocity, moons=moon_sigma, moon=False,
            )
            moon_fit = fit
            if moon_enabled:
                moon_fit = GLOBALutils.XC_Final_Fit(
                    velocities, averaged, sigma_res=4, horder=8,
                    moonv=reference_velocity, moons=moon_sigma, moon=True,
                )
            if known_dispersion:
                break
            dispersion = max(4.0, float(np.floor(fit[2][2])))
            known_dispersion = True
        p1, model, gaussian, gaussian_model, levels = fit
        p1_m, model_m, gaussian_m, gaussian_model_m, levels_m = moon_fit
        bisector = GLOBALutils.calc_bss(velocities, averaged)[0]
        snr_order = min(21, spec.shape[1] - 1)
        snr_values = spec[8, snr_order, 1000:min(1101, spec.shape[2])]
        snr_values = snr_values[np.isfinite(snr_values)]
        snr = max(float(np.median(snr_values)) if snr_values.size else 1.0, 1.0)
        coeffs = ({"G2": (0.06544, 0.00146, 0.24416, 0.00181) if teff < 6000
                   else (0.09821, 0.00014, 0.33491, 0.00113)}
                  .get(spectral_type, (0.05348, 0.00147, 0.20695, 0.00321)))
        a, b, d, c = coeffs
        rv_error = b + (1.6 + 0.2 * gaussian[2]) * a / round(snr)
        depth = 1.0 + gaussian[0] / (gaussian[2] * np.sqrt(2 * np.pi))
        if depth >= 1.0:
            rv_error = -999.0
        else:
            depth = (1.0 - (0.62 if spectral_type == "G2" else 0.59)) / (1.0 - depth)
            rv_error = np.sqrt((rv_error * depth) ** 2 + 0.004 ** 2)
        return {"teff": teff, "logg": logg, "metallicity": metallicity,
                "vsini": vsini, "velocity0": velocity0, "spectral_type": spectral_type,
            "rv": float(gaussian_m[1]), "rv_error": float(rv_error),
                "bisector": float(bisector), "bisector_error": d / round(snr) + c,
                "dispersion": float(gaussian[2]), "snr": snr,
                "xc_min": abs(float(np.around(np.min(model), 2))),
            "ccf": {"vels": velocities, "xc_av": averaged,
                "XCmodelgau": gaussian_model, "Ls2": levels,
                "refvel": reference_velocity, **rough_state,
                "xc_full": xc, "p1": p1, "sn": sn,
                "p1gau": gaussian, "p1_m": p1_m,
                "XCmodel_m": model_m, "p1gau_m": gaussian_m,
                "Ls2_m": levels_m, "XCmodelgau_m": gaussian_model_m},
            "moon": {"moonmatters": moon_enabled,
                 "moon_state": moon_info["state"],
                 "moonsep": moon_info["separation"],
                 "lunation": moon_info["lunation"],
                 "mephem": moon_info["moon"],
                 "texp": float(header["EXPTIME"])} }

    @staticmethod
    def _mask_problem_orders(spec: np.ndarray, weights: np.ndarray) -> None:
        """Disable orders containing strong telluric or contaminating bands."""
        for order in range(spec.shape[1]):
            wave = spec[0, order]
            if any(np.any((wave > low) & (wave < high))
                   for low, high in ((7030, 7050), (6920, 6940), (4300, 4340))):
                weights[order] = 0

    def _write_product_header(self, header_out: fits.Header, source: fits.Header,
                              object_name: str, mjd: float, bary_mjd: float,
                              bary_factor: float, bary_velocity: float,
                              airmass: Any, drift: float, error: float,
                              reference_arc: str, simbad_ra: float,
                              simbad_dec: float, bary_ra: float,
                              bary_dec: float) -> None:
        """Write stable observation and calibration metadata to the spectrum."""
        date_obs = str(source["DATE-OBS"])
        values = {
            "HIERARCH MJD": mjd, "HIERARCH MBJD": bary_mjd,
            "HIERARCH SHUTTER START DATE": date_obs[:10],
            "HIERARCH SHUTTER START UT": date_obs[11:],
            "HIERARCH TEXP (S)": source["EXPTIME"],
            "HIERARCH BARYCENTRIC CORRECTION (KM/S)": bary_velocity,
            "HIERARCH (LAMBDA_BARY / LAMBDA_TOPO)": bary_factor,
            "HIERARCH TARGET NAME": object_name,
            "HIERARCH RA": simbad_ra,
            "HIERARCH DEC": simbad_dec,
            "HIERARCH RA DEG": simbad_ra,
            "HIERARCH DEC DEG": simbad_dec,
            "HIERARCH RA BARY": bary_ra,
            "HIERARCH DEC BARY": bary_dec,
            "HIERARCH EQUINOX": 2000,
            "HIERARCH OBS LATITUDE": 37.0642,
            "HIERARCH OBS LONGITUDE": 3.3847,
            "HIERARCH OBS ALTITUDE": 2896.0,
            "HIERARCH DRIFT_CO": np.around(drift, 1),
            "HIERARCH DRIFT_CO_E": np.around(error, 1),
            "HIERARCH AIRMASS": np.around(float(np.asarray(airmass.value).flat[0]), 3),
            "BDATE": fits.getheader(self.config.output_dir / "MasterBias.fits").get("DATE", ""),
            "SFDATE": fits.getheader(self.config.output_dir / "MasterBias.fits").get("DATE", ""),
            "HIERARCH REFWAV": Path(reference_arc).name,
            "INST": "WHAPOS", "RESOL": "24000", "PIPELINE": "CERES",
            "DATE": self._now(),
        }
        for key, value in values.items():
            header_out[key] = value

    def _write_science_metadata(self, header: fits.Header,
                                analysis: dict[str, Any], bary_mjd: float) -> None:
        """Append RV, bisector, SNR, and classification results to the FITS header."""
        snr = np.around(analysis["snr"])
        values = {
            "RV": np.around(analysis["rv"], 4),
            "RV_E": np.around(analysis["rv_error"], 4),
            "BS": np.around(analysis["bisector"], 4),
            "BS_E": np.around(analysis["bisector_error"], 4),
            "DISP": np.around(analysis["dispersion"], 1),
            "SNR": snr, "SNR_R": np.around(snr * np.sqrt(2.9)),
            "XC_MIN": analysis["xc_min"], "BJD_OUT": 2400000.5 + bary_mjd,
            "HIERARCH TEFF": float(analysis["teff"]),
            "HIERARCH LOGG": float(analysis["logg"]),
            "HIERARCH Z": analysis["metallicity"],
            "HIERARCH VSINI": analysis["vsini"],
            "HIERARCH VEL0": analysis["velocity0"],
            "HIERARCH SIMBAD SPTYP": "None",
        }
        for key, value in values.items():
            header[key] = value

    def _format_result(self, path: str, object_name: str, result: dict[str, Any],
                       bary_mjd: float, airmass: Any, shifted_reference_pixel: float,
                       drift: float, exposure: float) -> str:
        """Format the tabular nightly summary used by existing consumers."""
        return ("%-15s %18.8f %9.4f %7.4f %9.3f %5.3f   whaposutils   ceres   24000 "
                "%6d %5.2f %5.2f %5.1f %4.2f %5.2f %6.1f %4d %7.4f %7.4f %7.4f %s\n" %
                (object_name, 2400000.5 + bary_mjd, result["rv"], result["rv_error"],
                 result["bisector"], result["bisector_error"], result["teff"],
                 result["logg"], result["metallicity"], result["vsini"], result["xc_min"],
                 result["dispersion"], exposure, result["snr"] * np.sqrt(2.9),
                 float(np.asarray(airmass.value).flat[0]), shifted_reference_pixel,
                 drift, self.config.output_dir / "proc" / f"{Path(path).stem}_XCs_{result['spectral_type']}.pdf"))

    def _write_activity(self, spec: np.ndarray, product: Path, teff: float | None) -> None:
        """Attempt activity indicators without invalidating the spectrum product."""
        activity_path = product.with_name(product.name.replace(".sp.fits", ".1d.fits"))
        try:
            from ceres3.utils.activity import compute_activity
            compute_activity(spec, instrument="whapos", output_1d_path=str(activity_path),
                             teff=teff if teff != -999 else None)
        except Exception as error:
            print(f"\tActivity products unavailable: {error}")

    @staticmethod
    def _now() -> str:
        """Return a FITS-compatible local timestamp."""
        return dt.datetime.now().isoformat(timespec="seconds")


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    """Parse the legacy command-line options without import-time execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing raw FITS frames")
    parser.add_argument("-avoid_plot", action="store_true", default=False)
    parser.add_argument("-dirout", default="default")
    parser.add_argument("-do_class", action="store_true", default=False)
    parser.add_argument("-just_extract", action="store_true", default=False)
    parser.add_argument("-npools", type=int, default=1)
    parser.add_argument("-o2do", default="all")
    parser.add_argument("-reffile", default="default")
    parser.add_argument("-absref", default="none")
    parser.add_argument("-just_bias", action="store_true", default=False)
    parser.add_argument("-just_flats", action="store_true", default=False)
    parser.add_argument("-just_thar", action="store_true", default=False)
    args = parser.parse_args(argv)
    input_dir = args.directory.expanduser().resolve()
    output_dir = (input_dir.parent / f"{input_dir.name}_red" if args.dirout == "default"
                  else Path(args.dirout).expanduser().resolve())
    reffile = None if args.reffile == "default" else Path(args.reffile).expanduser().resolve()
    absref = None if args.absref == "none" else Path(args.absref).expanduser().resolve()
    return PipelineConfig(
        input_dir=input_dir, output_dir=output_dir, avoid_plot=args.avoid_plot,
        do_class=args.do_class, just_extract=args.just_extract, npools=args.npools,
        object_to_do=args.o2do, reference_file=reffile, absolute_reference=absref,
        just_bias=args.just_bias, just_flats=args.just_flats, just_thar=args.just_thar,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; return a process status instead of calling ``sys.exit``."""
    try:
        WhaposPipeline(parse_args(argv)).run()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Whapos pipeline failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
