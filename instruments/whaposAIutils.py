"""Documented, defensive utilities for the Whapos reduction pipeline.

This module keeps the public function names, positional arguments, and return
shapes of :mod:`whaposutils` while making input validation and FITS handling
more explicit. Numerical work still uses the same Astropy, SciPy, and CERES
primitives as the legacy implementation. Functions that fit detector profiles
or wavelength solutions raise ``ValueError`` with context when their inputs
cannot support a fit, rather than returning silently malformed arrays.
"""

import glob
import os
from pathlib import Path
from typing import Any

import emcee
import numpy as np
import scipy
from scipy import interpolate, ndimage, optimize, signal
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits as pyfits
from astropy.time import Time

from ceres3.utils import globalutils as GLOBALutils


_C_MS = 299792458.0
_SECONDS_PER_DAY = 86400.0
_SUBPIXEL_FACTOR = 100


def _image2d(value: Any, name: str) -> np.ndarray:
    """Return a numeric 2-D view or raise a consistent validation error."""
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array; got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    return array


def _header_from(value: Any):
    """Accept an Astropy HDUList, HDU, Header, path, or file-like FITS source."""
    if hasattr(value, "header"):
        return value.header
    if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes, os.PathLike)):
        try:
            return value[0].header
        except (IndexError, TypeError, KeyError):
            pass
    if isinstance(value, (str, bytes, os.PathLike)):
        return pyfits.getheader(value)
    raise TypeError("expected a FITS path, HDU, HDUList, or Header")


def FileClassify(diri, log):
    """Classify raw FITS images and return the legacy eight-array tuple.

    Classification uses case-insensitive filename tokens. ``bad_files.txt``
    may contain basenames or paths relative to the input directory. ThAr and
    science frames must provide ``DATE-OBS`` and ``EXPTIME`` because sorting
    them chronologically is part of the return contract.
    """
    directory = Path(diri).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"Raw-data directory does not exist: {directory}")
    log_path = Path(log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    bad_path = directory / "bad_files.txt"
    excluded = set()
    if bad_path.is_file():
        excluded = {line.strip() for line in bad_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")}

    groups: dict[str, list[str]] = {
        "bias": [], "dark": [], "flat": [], "thar": [], "science": []
    }
    timed: dict[str, list[float]] = {"thar": [], "science": []}
    files = sorted({*directory.glob("*.fit"), *directory.glob("*.fits"),
                    *directory.glob("*.FIT"), *directory.glob("*.FITS")})
    with log_path.open("w", encoding="utf-8"):
        for path in files:
            relative = path.name
            if relative in excluded or str(path) in excluded:
                continue
            name = path.name.lower()
            if "bias" in name:
                groups["bias"].append(str(path))
                continue
            if "dark" in name:
                groups["dark"].append(str(path))
                continue
            if "flat" in name:
                groups["flat"].append(str(path))
                continue
            category = "thar" if "thar" in name else "science"
            with pyfits.open(path, memmap=False) as hdul:
                try:
                    mjd, _ = mjd_fromheader(hdul)
                except KeyError as error:
                    raise KeyError(f"{error} in {path}") from error
            groups[category].append(str(path))
            timed[category].append(mjd)

    def strings(values: list[str]) -> np.ndarray:
        return np.asarray(values, dtype=str)

    thars = strings(groups["thar"])
    science = strings(groups["science"])
    if timed["thar"]:
        thars = thars[np.argsort(timed["thar"])]
    if timed["science"]:
        science = science[np.argsort(timed["science"])]
    empty = np.asarray([], dtype=str)
    return (strings(groups["bias"]), strings(groups["dark"]), strings(groups["flat"]),
            empty.copy(), thars, empty.copy(), science, empty.copy())


def make_flatOB(MasterFlat, c_co, exap=5):
    """Replace order regions with local minima to form a comparison-fibre flat."""
    flat = _image2d(MasterFlat, "MasterFlat").T
    coefficients = np.asarray(c_co, dtype=float)
    if coefficients.ndim != 2:
        raise ValueError("c_co must be a 2-D array of polynomial coefficients")
    if exap < 0:
        raise ValueError("exap must be non-negative")
    output = flat.copy()
    x = np.arange(flat.shape[1])
    centers = np.asarray([np.polyval(row, x) for row in coefficients])
    for column in range(flat.shape[1]):
        baseline = np.min(flat[:, column])
        for center in centers[:, column]:
            pixel = int(np.rint(center))
            lo = max(0, int(np.rint(pixel - 3 * exap)))
            hi = min(flat.shape[0], int(np.rint(pixel + 3 * exap)))
            region = flat[lo:hi, column]
            level = float(np.min(region)) if region.size else baseline
            lo = max(0, int(np.rint(pixel - exap)))
            hi = min(flat.shape[0], int(np.rint(pixel + exap + 1)))
            output[lo:hi, column] = level
    return output


def get_flatOB(MasterFlat, MasterFlat_co):
    """Subtract the comparison-fibre flat after estimating its subpixel shift."""
    master = _image2d(MasterFlat, "MasterFlat")
    comparison = _image2d(MasterFlat_co, "MasterFlat_co")
    if master.shape != comparison.shape:
        raise ValueError("MasterFlat arrays must have matching shapes")
    row = min(1000, master.shape[0] - 1)
    reference = np.arange(master.shape[1], dtype=float)
    spline = interpolate.splrep(reference, comparison[row], k=min(3, master.shape[1] - 1))
    shifts = np.arange(-6.0, 6.0, 0.001)
    correlations = np.asarray([
        np.sum(interpolate.splev(reference + shift, spline) * master[row])
        for shift in shifts
    ])
    shift = float(shifts[int(np.argmax(correlations))])
    result = np.empty_like(comparison, dtype=np.result_type(master, comparison, float))
    for row_index in range(comparison.shape[0]):
        order = comparison[row_index]
        spline = interpolate.splrep(reference, order, k=1)
        result[row_index] = master[row_index] - interpolate.splev(reference + shift, spline)
    return result


def get_data(path):
    """Load a FITS image and return the orientation used by Whapos routines."""
    data = pyfits.getdata(path)
    return np.fliplr(_image2d(data, "FITS image").T)


def good_orders(coef, nord, ny, nx, ext_aperture):
    """Keep the contiguous leading orders whose aperture fits on the detector."""
    coefficients = np.asarray(coef)
    if coefficients.ndim != 2:
        raise ValueError("coef must be a 2-D array")
    count = min(int(nord), len(coefficients))
    if count < 0 or ny <= 0 or nx <= 0 or ext_aperture < 0:
        raise ValueError("order count, image dimensions, and aperture must be valid")
    x = np.arange(int(nx))
    for index in range(count):
        center = np.polyval(coefficients[index], x)
        if np.any(center - ext_aperture < 0) or np.any(center + ext_aperture >= ny):
            return coefficients[:index], index
    return coefficients[:count], count


def clean_orders(c_all, data, exap=5):
    """Separate alternating order traces into object and comparison sets."""
    coefficients = np.asarray(c_all)
    image = _image2d(data, "data")
    if coefficients.ndim != 2 or len(coefficients) < 3:
        raise ValueError("at least three order traces are required")
    if exap < 0 or 2 * exap + 1 > image.shape[1]:
        raise ValueError("exap is incompatible with the image width")
    mid = image.shape[1] // 2
    centers = np.polyval(coefficients, mid)
    use_last = centers[-1] - centers[-2] < centers[-2] - centers[-3]
    last_index = len(coefficients) - 1 if use_last else len(coefficients) - 2
    comparison, object_orders = [], []
    for index in range(last_index, 0, -2):
        comparison.append(coefficients[index])
        object_orders.append(coefficients[index - 1])
    object_orders = np.asarray(object_orders[::-1])
    comparison = np.asarray(comparison[::-1])
    return object_orders, comparison, len(object_orders), len(comparison)


def mjd_fromheader(h):
    """Return exposure-midpoint MJD and the MJD epoch origin from a FITS header."""
    header = _header_from(h)
    date_obs = header.get("DATE-OBS")
    if not date_obs:
        raise KeyError("DATE-OBS")
    try:
        observation_time = Time(str(date_obs), scale="utc")
    except Exception as error:
        raise ValueError(f"Invalid DATE-OBS value: {date_obs!r}") from error
    exposure = float(header.get("EXPTIME", 0.0))
    if not np.isfinite(exposure) or exposure < 0:
        raise ValueError(f"EXPTIME must be finite and non-negative; got {exposure}")
    return float(observation_time.mjd + exposure / (2 * _SECONDS_PER_DAY)), 2400000.5


def get_scat(sc, ps, binning):
    """Estimate scattered-light background between spatial-profile traces.

    Each detector row is sampled in the zero-valued gaps of ``ps``. A linear
    interpolation through gap medians is edge-extended, then median-filtered
    across adjacent detector rows. Missing gap samples safely yield a zero row.
    """
    image = _image2d(sc, "sc")
    profile = _image2d(ps, "ps")
    if profile.shape == image.shape:
        profile_rows = profile
    elif profile.shape == image.T.shape:
        profile_rows = profile.T
    else:
        raise ValueError(
            f"ps must match sc or its transpose: {image.shape} vs {profile.shape}"
        )
    if float(binning) <= 0:
        raise ValueError("binning must be positive")
    x = np.arange(image.shape[1], dtype=float)
    background = np.zeros(image.shape, dtype=np.result_type(image.dtype, np.float32))
    for row in range(image.shape[0]):
        in_gap = profile_rows[row] == 0
        edges = np.diff(np.r_[False, in_gap, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        points, values = [], []
        for start, end in zip(starts, ends):
            if end <= start:
                continue
            sample = np.asarray(image[row, start:end], dtype=float)
            sample = sample[np.isfinite(sample)]
            if sample.size:
                points.append(0.5 * (start + end - 1))
                values.append(float(np.median(sample)))
        if not values:
            continue
        points = np.asarray(points)
        values = np.asarray(values)
        valid = np.isfinite(points) & np.isfinite(values) & (values < 10000)
        points, values = points[valid], values[valid]
        if not values.size:
            continue
        if len(values) == 1:
            background[row] = values[0]
        else:
            background[row] = np.interp(x, points, values)
    kernel = min(51, image.shape[0] if image.shape[0] % 2 else max(1, image.shape[0] - 1))
    if kernel > 1:
        background = signal.medfilt2d(background, kernel_size=(kernel, 1))
    return background


def get_name(arch):
    """Read the target name from a FITS header, normalizing HD 10700."""
    name = pyfits.getheader(arch).get("OBJECT")
    if name is None:
        raise KeyError(f"OBJECT not found in FITS header: {arch}")
    if "HD10700" in str(arch):
        return "HD10700"
    return name


def Initial_Wav_Calibration(filename, spec, order, wei, porder=3, rmsmax=75,
                            minlines=10, FixEnds=True, Dump_Argon=False,
                            Dump_AllLines=False, Cheby=False, rough_shift=0.0,
                            del_width=5.0, binning=1, line_width=4, fact=1,
                            do_xc=True, sigmai=0.7, pixelization=False,
                            fibre_ap=2.3):
    """Fit line-list wavelengths to measured ThAr line centers for one order.

    Returns the legacy nine-tuple of wavelength/width coefficients, accepted
    pixel and wavelength vectors, fit RMS/residuals, centroids, widths, and
    line strengths. Outlying lines are removed iteratively while preserving
    the requested minimum number of calibration lines.
    """
    spectrum = np.asarray(spec, dtype=float)
    weights = np.asarray(wei, dtype=float)
    if spectrum.ndim != 1 or weights.shape != spectrum.shape:
        raise ValueError("spec and wei must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("wei must contain finite, non-negative inverse-standard-deviation weights")
    if binning == 0 or fact == 0 or porder < 0 or minlines < 1:
        raise ValueError("binning, fact, polynomial degree, and minimum line count must be valid")
    records = []
    with open(filename, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 3:
                continue
            try:
                pixel = float(fields[1]) * fact / float(binning) + rough_shift
                wavelength = float(fields[2])
            except ValueError as error:
                raise ValueError(f"Malformed line list at {filename}:{line_number}") from error
            if 20 < pixel < len(spectrum) - 20:
                x = np.arange(int(np.rint(pixel)) - 5, int(np.rint(pixel)) + 6)
                guess = np.array([pixel - 0.5 * fibre_ap, np.max(spectrum[x]), sigmai])
                fitted = FitFideosCompProf(x, spectrum[x], guess, fibre_ap, weights[x])
                records.append((fitted[0], wavelength, fitted[0], fitted[2], fitted[1]))
    if len(records) < max(minlines, porder + 1):
        raise ValueError(f"Only {len(records)} usable lines found; need at least {max(minlines, porder + 1)}")
    data = np.asarray(records, dtype=float)
    pixels, wavelengths = data[:, 0], data[:, 1]
    centroids, sigmas, intensities = data[:, 2], data[:, 3], data[:, 4]
    accepted = np.flatnonzero((pixels >= 50) & (pixels <= len(spectrum) - 50))
    if len(accepted) < max(minlines, porder + 1):
        accepted = np.arange(len(data))
    fit = Cheby_Fit if Cheby else None

    def fit_values(indices):
        if Cheby:
            return (Cheby_Fit(pixels[indices], wavelengths[indices], porder, len(spectrum)),
                    Cheby_Fit(pixels[indices], sigmas[indices], porder, len(spectrum)))
        return (np.polyfit(pixels[indices], wavelengths[indices], porder),
                np.polyfit(pixels[indices], sigmas[indices], porder))

    while True:
        coeff_wavelength, coeff_sigma = fit_values(accepted)
        rms, residuals = rms_ms(coeff_wavelength, pixels[accepted], wavelengths[accepted],
                                len(spectrum), Cheby=Cheby)
        if rms <= rmsmax or len(accepted) <= max(minlines, porder + 1):
            break
        worst = int(np.argmax(np.abs(residuals)))
        accepted = np.delete(accepted, worst)
    return (coeff_wavelength, coeff_sigma, pixels[accepted], wavelengths[accepted],
            rms, residuals, centroids[accepted], sigmas[accepted], intensities[accepted])


def rms_ms(coeffs_pix2wav, pixel_centers, wavelengths, npix, Cheby=False):
    """Return wavelength-fit RMS in m/s and per-line residuals."""
    pixels = np.asarray(pixel_centers, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if pixels.shape != wavelengths.shape or pixels.ndim != 1 or not pixels.size:
        raise ValueError("pixel_centers and wavelengths must be non-empty 1-D arrays of equal length")
    predicted = Cheby_eval(coeffs_pix2wav, pixels, npix) if Cheby else np.polyval(coeffs_pix2wav, pixels)
    residuals = predicted - wavelengths
    if npix <= 0:
        raise ValueError("npix must be positive")
    if Cheby:
        edge_a = Cheby_eval(coeffs_pix2wav, 50.0, npix)
        edge_b = Cheby_eval(coeffs_pix2wav, npix - 50.0, npix)
    else:
        edge_a = np.polyval(coeffs_pix2wav, 50.0)
        edge_b = np.polyval(coeffs_pix2wav, npix - 50.0)
    central_wavelength = 0.5 * (edge_a + edge_b)
    if not np.isfinite(central_wavelength) or central_wavelength == 0:
        raise ValueError("wavelength solution has an invalid central wavelength")
    return float(np.sqrt(np.var(residuals)) * _C_MS / central_wavelength), residuals


def fit2d(sc, trace, filename, spec, order, wei, porder=3, rmsmax=75,
          minlines=10, FixEnds=True, Dump_Argon=False, Dump_AllLines=False,
          Cheby=False, rough_shift=0.0, del_width=5.0, binning=1,
          line_width=4, fact=1, do_xc=True, sigmai=0.8, pixelization=False):
    """Fit the experimental 2-D fibre profile using the historical MCMC model."""
    image = _image2d(sc, "sc")
    spectrum = np.asarray(spec, dtype=float)
    output = []
    with open(filename, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#") or len(fields) < 3:
                continue
            pixel = float(fields[1]) * fact / float(binning) + rough_shift
            if not (20 < pixel < len(spectrum) - 20):
                continue
            y_center = 2064 - np.polyval(trace, 2048.0 - pixel)
            x_slice = np.arange(int(np.rint(2048 - pixel)) - 5, int(np.rint(2048 - pixel)) + 5)
            y_slice = np.arange(int(np.rint(y_center)) - 5, int(np.rint(y_center)) + 6)
            x_slice = x_slice[(x_slice >= 0) & (x_slice < image.shape[1])]
            y_slice = y_slice[(y_slice >= 0) & (y_slice < image.shape[0])]
            if not x_slice.size or not y_slice.size:
                continue
            patch = image[np.ix_(y_slice, x_slice)].T.astype(float)
            background = np.median(np.r_[patch[0], patch[-1], patch[:, 0], patch[:, -1]])
            patch -= background
            error = np.sqrt(np.abs(patch) + 9.0)
            guess = np.array([2048 - pixel, y_center, patch.max(), 2.0, 10.0])
            ndim, nwalkers = len(guess), 100
            positions = [guess + np.random.randn(ndim) * np.array([2, 2, 500, 2, 4])
                         for _ in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                             args=(x_slice, y_slice, patch, error))
            sampler.run_mcmc(positions, 500, progress=False)
            samples = sampler.get_chain(discard=50, flat=True)
            output.append(np.median(samples, axis=0))
    return np.asarray(output) if output else np.empty((0, 5))


def Cheby_Fit(x, y, order, npix):
    """Fit a Chebyshev polynomial using the legacy descending coefficient order."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not x.size:
        raise ValueError("x and y must be non-empty 1-D arrays of equal length")
    if order < 0 or npix <= 0 or x.size < order + 1:
        raise ValueError("insufficient samples or invalid polynomial dimensions")
    normalized = (x - 0.5 * npix) / (0.5 * npix)
    basis = [scipy.special.chebyt(index)(normalized) for index in range(order + 1)]

    def residual(parameters):
        fitted = sum(parameters[order - index] * basis[index]
                     for index in range(order + 1))
        return fitted - y

    initial = np.zeros(order + 1)
    initial[order] = np.mean(y)
    coefficients, _ = optimize.leastsq(residual, initial)
    return coefficients


def Cheby_eval(p, x, npix):
    """Evaluate coefficients returned by :func:`Cheby_Fit`."""
    coefficients = np.asarray(p, dtype=float)
    if coefficients.ndim != 1 or not coefficients.size or npix <= 0:
        raise ValueError("p must be non-empty 1-D and npix must be positive")
    normalized = (np.asarray(x, dtype=float) - 0.5 * npix) / (0.5 * npix)
    return np.polynomial.chebyshev.chebval(normalized, coefficients[::-1])


def LineFit_SingleSigma(X, Y, B, mu, sigma, weight, pixelization=False):
    """Fit several Gaussian components sharing one width; return parameters and status."""
    x, y, baseline = np.asarray(X, float), np.asarray(Y, float), np.asarray(B, float)
    centers, widths, weights = np.asarray(mu, float), np.asarray(sigma, float), np.asarray(weight, float)
    if x.ndim != 1 or y.shape != x.shape or baseline.shape != x.shape or weights.shape != x.shape:
        raise ValueError("X, Y, B, and weight must have the same one-dimensional shape")
    if centers.ndim != 1 or widths.shape != centers.shape or not centers.size:
        raise ValueError("mu and sigma must be non-empty arrays of equal length")
    if np.any(widths <= 0):
        raise ValueError("all Gaussian widths must be positive")
    n = len(centers)

    def model(parameters):
        shared_width = parameters[0]
        if shared_width <= 0:
            return np.full_like(x, np.inf)
        if pixelization:
            sample_x = x[:, None] + np.linspace(-0.495, 0.495, 100)[None, :]
            values = np.zeros_like(sample_x)
        else:
            sample_x = x
            values = np.zeros_like(x)
        for component in range(n):
            amplitude = parameters[1 + 2 * component]
            center = parameters[2 + 2 * component]
            values += amplitude * GLOBALutils.IntGaussian(sample_x, center, shared_width)
        return values.mean(axis=1) if pixelization else values

    initial = np.zeros(2 * n + 1)
    initial[0] = widths[0]
    for i, center in enumerate(centers):
        idx = int(np.clip(np.rint(center) - x[0], 0, len(y) - 1))
        initial[1 + 2 * i] = y[idx]
        initial[2 + 2 * i] = center
    result = optimize.least_squares(lambda p: (model(p) - (y - baseline)) * weights, initial)
    packed = np.zeros(3 * n)
    for i in range(n):
        packed[3 * i:3 * i + 3] = (result.x[1 + 2 * i], result.x[2 + 2 * i], result.x[0])
    return packed, int(result.status)


def TruncGaussian(x, y, x0, y0, dev):
    """Evaluate a circular Gaussian truncated behind its center and at 3 sigma."""
    if dev <= 0 or x > x0 or x < x0 - 3 * dev:
        return 0.0
    return float(np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * dev ** 2)))


def HalfFiber(x, y, x0, y0, lim):
    """Indicator for the left half of a circular fibre aperture."""
    return float(x <= x0 and (x - x0) ** 2 + (y - y0) ** 2 < lim ** 2)


def HalfFiberPos(x, y, x0, y0, lim):
    """Indicator for the right half of a circular fibre aperture."""
    return float(x >= x0 and (x - x0) ** 2 + (y - y0) ** 2 < lim ** 2)


def AnaliticProfile1D(p, x, R=2.3):
    """Integrate a blurred half-circle fibre profile over input pixel bins."""
    parameters = np.asarray(p, dtype=float)
    x = np.asarray(x, dtype=float)
    if parameters.size < 3 or x.ndim != 1:
        raise ValueError("p needs center/amplitude/width and x must be one-dimensional")
    if not x.size:
        return np.empty(0)
    if R <= 0 or parameters[2] <= 0:
        raise ValueError("fibre radius and profile width must be positive")
    high_x = np.arange(x[0], x[-1] + 1.0, 1.0 / _SUBPIXEL_FACTOR)
    expected_count = len(x) * _SUBPIXEL_FACTOR
    if len(high_x) != expected_count:
        high_x = x[0] + np.arange(expected_count) / _SUBPIXEL_FACTOR
    distance = high_x - parameters[0]
    profile = np.zeros(len(high_x))
    inside = (distance >= 0) & (distance <= R)
    profile[inside] = 2.0 * np.sqrt(np.maximum(R * R - distance[inside] ** 2, 0.0))
    profile = ndimage.gaussian_filter1d(profile, parameters[2] * _SUBPIXEL_FACTOR, mode="reflect")
    maximum = float(profile.max())
    if maximum > 0:
        profile *= parameters[1] / maximum
    return profile.reshape(len(x), _SUBPIXEL_FACTOR).mean(axis=1)


def FitFideosCompProf(X, Y, p0, fibre_ap, weights=None):
    """Fit a 1-D fibre profile, optionally weighting samples by inverse noise."""
    x, y, initial = np.asarray(X, float), np.asarray(Y, float), np.asarray(p0, float)
    if x.ndim != 1 or y.shape != x.shape or initial.size != 3:
        raise ValueError("X/Y must match and p0 must contain center, amplitude, width")
    if fibre_ap <= 0:
        raise ValueError("fibre_ap must be positive")
    if initial[2] <= 0:
        raise ValueError("initial profile width must be positive")
    if weights is None:
        fit_weights = np.ones_like(y)
    else:
        fit_weights = np.asarray(weights, dtype=float)
        if fit_weights.shape != y.shape:
            raise ValueError("weights must have the same shape as Y")
        if not np.all(np.isfinite(fit_weights)) or np.any(fit_weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if np.count_nonzero(fit_weights) < 3:
            raise ValueError("at least three samples must have positive weights")
    fitted = optimize.least_squares(
        lambda p: (AnaliticProfile1D(p, x, fibre_ap) - y) * fit_weights,
        initial,
        bounds=([-np.inf, -np.inf, np.finfo(float).eps],
                [np.inf, np.inf, np.inf]),
    )
    return fitted.x


def FideosCompProf(p, X, Y):
    """Render an oversampled, pixel-integrated half-fibre image profile."""
    parameters = np.asarray(p, dtype=float)
    x, y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if parameters.size < 5 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("p must contain five values and X/Y must be one-dimensional")
    if parameters[3] <= 0 or parameters[4] <= 0:
        raise ValueError("fibre radius and blur width must be positive")
    scale = 10
    xs = np.arange(x[0] - 0.5, x[-1] + 0.6, 1.0 / scale)
    ys = np.arange(y[0] - 0.5, y[-1] + 0.6, 1.0 / scale)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    disk = ((xx <= parameters[0]) &
            ((xx - parameters[0]) ** 2 + (yy - parameters[1]) ** 2 < parameters[3] ** 2))
    blurred = ndimage.gaussian_filter(disk.astype(float), parameters[4])
    maximum = float(blurred.max())
    if maximum:
        blurred *= parameters[2] / maximum
    output = np.zeros((len(x), len(y)))
    for ix, pixel_x in enumerate(x):
        ix_mask = (xs >= pixel_x - 0.5) & (xs < pixel_x + 0.4999)
        for iy, pixel_y in enumerate(y):
            iy_mask = (ys >= pixel_y - 0.5) & (ys < pixel_y + 0.4999)
            patch = blurred[np.ix_(ix_mask, iy_mask)]
            output[ix, iy] = patch.mean() if patch.size else 0.0
    return output


def FitFideos2DProf(X, Y, F, x1, y1):
    """Fit center and amplitude to a 2-D fibre patch without plotting side effects."""
    x, y, image = np.asarray(X), np.asarray(Y), _image2d(F, "F")
    if image.shape != (len(x), len(y)):
        raise ValueError("F shape must be (len(X), len(Y))")
    initial = np.array([x1, y1, np.max(image)], dtype=float)

    def model(parameters):
        full = np.array([parameters[0], parameters[1], parameters[2], 2.0, 5.0])
        return FideosCompProf(full, x, y)

    result = optimize.least_squares(lambda p: (model(p) - image).ravel(), initial)
    return result.x


def lnlike(theta, X, Y, F, Ferr):
    """Return the Gaussian log likelihood for a 2-D fibre-profile model."""
    data, error = _image2d(F, "F"), _image2d(Ferr, "Ferr")
    if data.shape != error.shape or np.any(error <= 0):
        return -np.inf
    model = FideosCompProf(theta, X, Y)
    if model.shape != data.shape:
        return -np.inf
    inv_variance = 1.0 / error ** 2
    value = -0.5 * np.sum(inv_variance * (data - model) ** 2 - np.log(inv_variance))
    return float(value) if np.isfinite(value) else -np.inf


def lnprior(theta, X, Y):
    """Uniform prior enforcing the legacy profile parameter bounds."""
    theta = np.asarray(theta, dtype=float)
    if theta.size < 5:
        return -np.inf
    if (X[0] < theta[0] < X[-1] and Y[0] < theta[1] < Y[-1]
            and 0 < theta[2] < 70000 and 0.1 < theta[3] < 5 and 0.1 < theta[4] < 20):
        return 0.0
    return -np.inf


def lnprob(theta, X, Y, F, Ferr):
    """Return prior plus likelihood, or negative infinity outside the prior."""
    prior = lnprior(theta, X, Y)
    return prior + lnlike(theta, X, Y, F, Ferr) if np.isfinite(prior) else -np.inf


def get_airmass(ra, dec, lat, lon, alt, dateobs):
    """Return secant zenith distance for a target and observatory location."""
    coordinates = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
    location = EarthLocation(lat=float(lat) * u.deg, lon=float(lon) * u.deg,
                             height=float(alt) * u.m)
    time = Time(dateobs, scale="utc")
    return coordinates.transform_to(AltAz(obstime=time, location=location)).secz


def compute_RON(mbias, biases, gain=1.0):
    """Estimate read noise from the median per-frame bias residual standard deviation."""
    master = np.asarray(mbias, dtype=float)
    if len(biases) == 0:
        raise ValueError("at least one bias frame is required")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")
    deviations = []
    for path in biases:
        image = np.asarray(pyfits.getdata(path), dtype=float)
        if image.shape != master.shape:
            raise ValueError(f"bias shape mismatch in {path}: {image.shape} != {master.shape}")
        residual = image - master
        finite = residual[np.isfinite(residual)]
        if finite.size:
            deviations.append(float(np.std(finite)))
    if not deviations:
        raise ValueError("bias frames contain no finite pixels")
    return float(np.median(deviations) * gain)


def fill_list_header(hdu, ofiles):
    """Record source basenames in sequential RAWn FITS cards."""
    for index, filename in enumerate(ofiles):
        hdu.header[f"RAW{index}"] = Path(filename).name
    return hdu


def get_date_obs(filepath: Path) -> Time:
    """Read DATE-OBS from the primary HDU or the first extension."""
    with pyfits.open(filepath, mode="readonly", memmap=False) as hdul:
        value = hdul[0].header.get("DATE-OBS")
        if value is None:
            for hdu in hdul[1:]:
                value = hdu.header.get("DATE-OBS")
                if value is not None:
                    break
    if value is None:
        raise KeyError(f"Header keyword 'DATE-OBS' not found in {Path(filepath).name}")
    try:
        return Time(value, scale="utc")
    except Exception as error:
        raise ValueError(f"Invalid DATE-OBS value {value!r} in {filepath}") from error


def find_closest_fits(science_filepath: Path, search_directory: Path) -> tuple:
    """Return the closest ThAr FITS path and absolute time difference in seconds."""
    science_time = get_date_obs(science_filepath)
    closest_path, closest_seconds = None, None
    for candidate in sorted(Path(search_directory).glob("ThAr*.fits")):
        if candidate.resolve() == Path(science_filepath).resolve():
            continue
        try:
            difference = abs(float((get_date_obs(candidate) - science_time).sec))
        except (KeyError, ValueError, OSError):
            continue
        if closest_seconds is None or difference < closest_seconds:
            closest_path, closest_seconds = candidate, difference
    return closest_path, closest_seconds


def find_closest_index(mjd: float, thar_times: np.ndarray) -> tuple[int, float]:
    """Return the nearest arc index and absolute time separation in seconds."""
    times = np.asarray(thar_times, dtype=float)
    if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)):
        raise ValueError("thar_times must be a non-empty finite one-dimensional array")
    if not np.isfinite(mjd):
        raise ValueError("mjd must be finite")
    if np.any(np.diff(times) < 0):
        index = int(np.argmin(np.abs(times - mjd)))
    else:
        insertion = int(np.searchsorted(times, mjd))
        if insertion == 0:
            index = 0
        elif insertion == len(times):
            index = len(times) - 1
        else:
            index = insertion - 1 if mjd - times[insertion - 1] <= times[insertion] - mjd else insertion
    return index, float(abs(times[index] - mjd) * _SECONDS_PER_DAY)
