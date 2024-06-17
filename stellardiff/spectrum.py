#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" An object for dealing with one-dimensional spectra. """

__all__ = ["Spectrum1D"]

import logging
import numpy as np
import os
import re
from collections import OrderedDict
from hashlib import md5

from astropy.io import fits

logger = logging.getLogger(__name__)

class Spectrum1D(object):
    """ A one-dimensional spectrum. """

    def __init__(self, dispersion, flux, ivar, metadata=None):
        """
        Initialie a `Spectrum1D` object with the given dispersion, flux and
        inverse variance arrays.

        :param dispersion:
            An array containing the dispersion values for every pixel.

        :param flux:
            An array containing flux values for all pixels.

        :param ivar:
            An array containing the inverse variances for the flux values in all
            pixels.

        :param metadata: [optional]
            A dictionary containing metadata for this spectrum.
        """

        dispersion = np.array(dispersion)
        flux = np.array(flux)
        ivar = np.array(ivar)

        if max([len(_.shape) for _ in (dispersion, flux, ivar)]) > 1:
            raise ValueError(
                "dispersion, flux and ivar must be one dimensional arrays")
        
        if flux.size != dispersion.size:
            raise ValueError(
                "dispersion and flux arrays must have the same "
                "size ({0} != {1})".format(dispersion.size, flux.size, ))

        if ivar.size != dispersion.size:
            raise ValueError(
                "dispersion and ivar arrays must have the same "
                "size ({0} != {1})".format(dispersion.size, ivar.size, ))

        self.metadata = metadata or {}

        # Don't allow orders to be back-to-front.
        if np.all(np.diff(dispersion) < 0):
            dispersion = dispersion[::-1]
            flux = flux[::-1]
            ivar = ivar[::-1]

        # HACK so that *something* can be done with spectra when there is no
        # inverse variance array.
        if not np.any(np.isfinite(ivar)) or np.nanmax(ivar) == 0:
            ivar = np.ones_like(flux) * 1.0/np.nanmean(flux)

        self._dispersion = dispersion
        self._flux = flux
        self._ivar = ivar

        return None
    

    @property
    def dispersion(self):
        """ Return the dispersion points for all pixels. """
        return self._dispersion


    @property
    def flux(self):
        """ Return the flux for all pixels. """
        return self._flux


    @property
    def ivar(self):
        """ Return the inverse variance of the flux for all pixels. """
        return self._ivar


    @classmethod
    def read(cls, path, **kwargs):
        """
        Create a Spectrum1D class from a path on disk.

        :param path:
            The path of the filename to load.
        """

        if not os.path.exists(path):
            raise IOError("filename '{}' does not exist".format(path))

        # Try multi-spec first since this is currently the most common use case.
        methods = (
            cls.read_fits_multispec,
            cls.read_fits_spectrum1d,
            cls.read_ascii_spectrum1d,
            cls.read_fits_spectrum1d_table
        )

        for method in methods:
            try:
                dispersion, flux, ivar, metadata = method(path, **kwargs)

            except:
                continue

            else:
                orders = [cls(dispersion=d, flux=f, ivar=i, metadata=metadata) \
                    for d, f, i in zip(dispersion, flux, ivar)]
                break
        else:
            raise ValueError("cannot read spectrum from path {}".format(path))

        # If it's a single order, just return that instead of a 1-length list.
        orders = orders if len(orders) > 1 else orders[0]
        return orders


    @classmethod
    def read_fits_spectrum1d_table(cls, path, **kwargs):

        image = fits.open(path)
        dispersion = np.atleast_2d(image[1].data)
        flux = np.atleast_2d(image[2].data)
        ivar = np.atleast_2d(image[3].data**-2)

        # Merge headers into a metadata dictionary.
        metadata = OrderedDict()
        for key, value in image[0].header.items():

            # NOTE: In the old SMH we did a try-except block to string-ify and
            #       JSON-dump the header values, and if they could not be
            #       forced to a string we didn't keep that header.

            #       I can't remember what types caused that problem, but it was
            #       to prevent SMH being unable to save a session.
            
            #       Since we are pickling now, that shouldn't be a problem
            #       anymore, but this note is here to speed up debugging in case
            #       that issue returns.

            if key in metadata:
                metadata[key] += value
            else:
                metadata[key] = value

        metadata["smh_read_path"] = path

        return (dispersion, flux, ivar, metadata)


    @classmethod
    def read_fits_multispec(cls, path, flux_ext=None, ivar_ext=None, **kwargs):
        """
        Create multiple Spectrum1D classes from a multi-spec file on disk.

        :param path:
            The path of the multi-spec filename to load.

        :param flux_ext: [optional]
            The zero-indexed extension number containing flux values.

        :param ivar_ext: [optional]
            The zero-indexed extension number containing the inverse variance of
            the flux values.
        """

        image = fits.open(path)

        # Merge headers into a metadata dictionary.
        metadata = OrderedDict()
        for key, value in image[0].header.items():

            # NOTE: In the old SMH we did a try-except block to string-ify and
            #       JSON-dump the header values, and if they could not be
            #       forced to a string we didn't keep that header.

            #       I can't remember what types caused that problem, but it was
            #       to prevent SMH being unable to save a session.
            
            #       Since we are pickling now, that shouldn't be a problem
            #       anymore, but this note is here to speed up debugging in case
            #       that issue returns.

            if key in metadata:
                metadata[key] += value
            else:
                metadata[key] = value

        metadata["smh_read_path"] = path

        flux = image[0].data

        assert metadata["CTYPE1"].upper().startswith("MULTISPE") \
            or metadata["WAT0_001"].lower() == "system=multispec"

        # Join the WAT keywords for dispersion mapping.
        i, concatenated_wat, key_fmt = (1, str(""), "WAT2_{0:03d}")
        while key_fmt.format(i) in metadata:
            # .ljust(68, " ") had str/unicode issues across Python 2/3
            value = metadata[key_fmt.format(i)]
            concatenated_wat += value + (" "  * (68 - len(value)))
            i += 1

        # Split the concatenated header into individual orders.
        order_mapping = np.array([map(float, each.rstrip('" ').split()) \
                for each in re.split('spec[0-9]+ ?= ?"', concatenated_wat)[1:]])

        # Parse the order mapping into dispersion values.
        dispersion = np.array(
            [compute_dispersion(*mapping) for mapping in order_mapping])

        # Get the flux and inverse variance arrays.
        # NOTE: Most multi-spec data previously used with SMH have been from
        #       Magellan/MIKE, and reduced with CarPy.
        md5_hash = md5(";".join([v for k, v in metadata.items() \
            if k.startswith("BANDID")])).hexdigest()
        is_carpy_mike_product = (md5_hash == "0da149208a3c8ba608226544605ed600")
        is_carpy_mage_product = (md5_hash == "6b2c2ec1c4e1b122ccab15eb9bd305bc")
        is_apo_product = (image[0].header.get("OBSERVAT", None) == "APO")

        if is_carpy_mike_product or is_carpy_mage_product:
            # CarPy gives a 'noise' spectrum, which we must convert to an
            # inverse variance array
            flux_ext = flux_ext or 1
            noise_ext = ivar_ext or 2

            logger.info(
                "Recognized CarPy product. Using zero-indexed flux/noise "
                "extensions (bands) {}/{}".format(flux_ext, noise_ext))

            flux = image[0].data[flux_ext]
            ivar = image[0].data[noise_ext]**(-2)

        elif is_apo_product:
            flux_ext = flux_ext or 0
            noise_ext = ivar_ext or -1

            logger.info(
                "Recognized APO product. Using zero-indexed flux/noise "
                "extensions (bands) {}/{}".format(flux_ext, noise_ext))

            flux = image[0].data[flux_ext]
            ivar = image[0].data[noise_ext]**(-2)

        else:
            ivar = np.nan * np.ones_like(flux)
            #raise ValueError("could not identify flux and ivar extensions")

        dispersion = np.atleast_2d(dispersion)
        flux = np.atleast_2d(flux)
        ivar = np.atleast_2d(ivar)

        # Ensure dispersion maps from blue to red direction.
        if np.min(dispersion[0]) > np.min(dispersion[-1]):

            dispersion = dispersion[::-1]
            if len(flux.shape) > 2:
                flux = flux[:, ::-1]
                ivar = ivar[:, ::-1]
            else:
                flux = flux[::-1]
                ivar = ivar[::-1]

        # Do something sensible regarding zero or negative fluxes.
        ivar[0 >= flux] = 0
        flux[0 >= flux] = np.nan

        return (dispersion, flux, ivar, metadata)


    @classmethod
    def read_fits_spectrum1d(cls, path, **kwargs):
        """
        Read Spectrum1D data from a binary FITS file.

        :param path:
            The path of the FITS filename to read.
        """

        image = fits.open(path)

        # Merge headers into a metadata dictionary.
        metadata = OrderedDict()
        for key, value in image[0].header.items():
            if key in metadata:
                metadata[key] += value
            else:
                metadata[key] = value
        metadata["smh_read_path"] = path

        # Find the first HDU with data in it.
        for hdu_index, hdu in enumerate(image):
            if hdu.data is not None: break

        ctype1 = image[0].header.get("CTYPE1", None)

        if len(image) == 2 and hdu_index == 1:

            dispersion_keys = ("dispersion", "disp", "WAVELENGTH[COORD]", "wave")
            for key in dispersion_keys:
                try:
                    dispersion = image[hdu_index].data[key]

                except KeyError:
                    continue

                else:
                    break

            else:
                raise KeyError("could not find any dispersion key: {}".format(
                    ", ".join(dispersion_keys)))

            flux_keys = ("flux", "SPECTRUM[FLUX]")
            for key in flux_keys:
                try:
                    flux = image[hdu_index].data[key]
                except KeyError:
                    continue
                else:
                    break
            else:
                raise KeyError("could not find any flux key: {}".format(
                    ", ".join(flux_keys)))

            # Try ivar, then error, then variance.
            unc_keys = ("ivar", "err", "SPECTRUM[SIGMA]", "variance")
            for key in unc_keys:
                try:
                    unc = image[hdu_index].data[key]
                    working_key = np.copy(key)
                except KeyError:
                    continue
                else:
                    break
            else:
                raise KeyError("could not find any uncertainty key: {}".format(
                    ", ".join(unc_keys))) 
                
            # Transform uncertainty into ivar
            if working_key == "ivar":
                ivar = unc
            elif np.isin(working_key, ("err", "SPECTRUM[SIGMA]")):
                ivar = 1.0/unc**2.
            elif working_key == "variance":
                ivar = 1.0/unc
            else:
                raise KeyError("transformation from {0} to ivar failed".format(working_key))

        else:
            # Build a simple linear dispersion map from the headers.
            # See http://iraf.net/irafdocs/specwcs.php
            crval = image[0].header["CRVAL1"]
            naxis = image[0].header["NAXIS1"]
            crpix = image[0].header.get("CRPIX1", 0)
            cdelt = image[0].header["CDELT1"]
            ltv = image[0].header.get("LTV1", 0)

            dispersion = \
                crval + (np.arange(naxis) - crpix) * cdelt - ltv * cdelt

            flux = image[0].data
            if len(image) == 1:
                ivar = np.ones_like(flux)*1e+5 # HACK S/N ~300 just for training/verification purposes
            else:
                ivar = image[1].data

        dispersion = np.atleast_2d(dispersion)
        flux = np.atleast_2d(flux)
        ivar = np.atleast_2d(ivar)

        return (dispersion, flux, ivar, metadata)


    @classmethod
    def read_ascii_spectrum1d(cls, path, **kwargs):
        """
        Read Spectrum1D data from an ASCII-formatted file on disk.

        :param path:
            The path of the ASCII filename to read.
        """

        kwds = kwargs.copy()
        kwds.update({
            "unpack": True
        })
        kwds.setdefault("usecols", (0, 1, 2))

        try:
            dispersion, flux, ivar = np.loadtxt(path, **kwds)
        except:
            # Try by ignoring the first row.
            kwds.setdefault("skiprows", 1)
            dispersion, flux, ivar = np.loadtxt(path, **kwds)

        dispersion = np.atleast_2d(dispersion)
        flux = np.atleast_2d(flux)
        ivar = np.atleast_2d(ivar)
        metadata = { "smh_read_path": path }
        
        return (dispersion, flux, ivar, metadata)


    def write(self, filename, clobber=True, output_verify="warn"):
        """ Write spectrum to disk. """

        if os.path.exists(filename) and not clobber:
            raise IOError("Filename '%s' already exists and we have been asked not to clobber it." % (filename, ))
        
        if not filename.endswith('fits'):
            a = np.array([self.dispersion, self.flux, self.ivar]).T
            np.savetxt(filename, a, fmt="%.4f".encode('ascii'))
            return
        
        else:

            crpix1, crval1 = 0, self.dispersion.min()
            cdelt1 = np.mean(np.diff(self.dispersion))
            naxis1 = len(self.dispersion)
            
            linear_dispersion = crval1 + (np.arange(naxis1) - crpix1) * cdelt1

            ## Check for linear dispersion map
            maxdiff = np.max(np.abs(linear_dispersion - self.dispersion))
            if maxdiff > 1e-3:
                ## TODO Come up with something better...
                ## Frustratingly, it seems like there's no easy way to make an IRAF splot-compatible
                ## way of storing the dispersion data for an arbitrary dispersion sampling.
                ## The standard way of doing it requires putting every dispersion point in the
                ## WAT2_xxx header.
                
                ## So just implemented it as a binary table with names according to 
                ## http://iraf.noao.edu/projects/spectroscopy/formats/sptable.html
                ## It doesn't work because I have not put in WCS headers, but I do not know
                ## how to tell it to look for those.
                
                ## I think some of these links below may provide a better solution though
                ## http://iraf.noao.edu/projects/spectroscopy/formats/onedspec.html
                ## http://www.cv.nrao.edu/fits/
                ## http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?specwcs
                
                ## python 2(?) hack needs the b prefix
                ## https://github.com/numpy/numpy/issues/2407
                
                headers = {}

                dispcol = fits.Column(name=b"WAVELENGTH[COORD]",
                                      format="D",
                                      array=self.dispersion)
                fluxcol = fits.Column(name=b"SPECTRUM[FLUX]",
                                      format="D",
                                      array=self.flux)
                errscol = fits.Column(name=b"SPECTRUM[SIGMA]",
                                      format="D",
                                      array=(self.ivar)**-0.5)
                
                coldefs = fits.ColDefs([dispcol, fluxcol, errscol])
                hdu = fits.BinTableHDU.from_columns(coldefs)
                hdu.header.update(headers)
                hdu.writeto(filename, output_verify=output_verify, clobber=clobber)

                return

            else:
                # We have a linear dispersion!
                hdu = fits.PrimaryHDU(np.array(self.flux))
                hdu2 = fits.ImageHDU(np.array(self.ivar))
    
                #headers = self.headers.copy()
                headers = {}
                headers.update({
                    'CTYPE1': 'LINEAR  ',
                    'CRVAL1': crval1,
                    'CRPIX1': crpix1,
                    'CDELT1': cdelt1
                })
                
                for key, value in headers.iteritems():
                    try:
                        hdu.header[key] = value
                    except ValueError:
                        #logger.warn("Could not save header key/value combination: %s = %s" % (key, value, ))
                        print("Could not save header key/value combination: %s = %s".format(key, value))
                hdulist = fits.HDUList([hdu,hdu2])
                hdulist.writeto(filename, output_verify=output_verify, clobber=clobber)
                return



    # State functionality for serialization.
    def __getstate__(self):
        """ Return the spectrum state. """
        return (self.dispersion, self.flux, self.ivar, self.metadata)


    def __setstate__(self, state):
        """
        Set the state of the spectrum.

        :param state:
            A four-length tuple containing the dispersion array, flux array, the
            inverse variance of the fluxes, and a metadata dictionary.
        """
        
        dispersion, flux, ivar, metadata = state
        self._dispersion = dispersion
        self._flux = flux
        self._ivar = ivar
        self.metadata = metadata
        return None


    def copy(self):
        """
        Create a copy of the spectrum.
        """
        return self.__class__(
            dispersion=self.dispersion.copy(),
            flux=self.flux.copy(), ivar=self.ivar.copy(),
            metadata=self.metadata.copy())


def compute_dispersion(aperture, beam, dispersion_type, dispersion_start,
    mean_dispersion_delta, num_pixels, redshift, aperture_low, aperture_high,
    weight=1, offset=0, function_type=None, order=None, Pmin=None, Pmax=None,
    *coefficients):
    """
    Compute a dispersion mapping from a IRAF multi-spec description.

    :param aperture:
        The aperture number.

    :param beam:
        The beam number.

    :param dispersion_type:
        An integer representing the dispersion type:

        0: linear dispersion
        1: log-linear dispersion
        2: non-linear dispersion

    :param dispersion_start:
        The value of the dispersion at the first physical pixel.

    :param mean_dispersion_delta:
        The mean difference between dispersion pixels.

    :param num_pixels:
        The number of pixels.

    :param redshift:
        The redshift of the object. This is accounted for by adjusting the
        dispersion scale without rebinning:

        >> dispersion_adjusted = dispersion / (1 + redshift)

    :param aperture_low:
        The lower limit of the spatial axis used to compute the dispersion.

    :param aperture_high:
        The upper limit of the spatial axis used to compute the dispersion.

    :param weight: [optional]
        A multiplier to apply to all dispersion values.

    :param offset: [optional]
        A zero-point offset to be applied to all the dispersion values.

    :param function_type: [optional]
        An integer representing the function type to use when a non-linear 
        dispersion mapping (i.e. `dispersion_type = 2`) has been specified:

        1: Chebyshev polynomial
        2: Legendre polynomial
        3: Cubic spline
        4: Linear spline
        5: Pixel coordinate array
        6: Sampled coordinate array

    :param order: [optional]
        The order of the Legendre or Chebyshev function supplied.

    :param Pmin: [optional]
        The minimum pixel value, or lower limit of the range of physical pixel
        coordinates.

    :param Pmax: [optional]
        The maximum pixel value, or upper limit of the range of physical pixel
        coordinates.

    :param coefficients: [optional]
        The `order` number of coefficients that define the Legendre or Chebyshev
        polynomial functions.

    :returns:
        An array containing the computed dispersion values.
    """

    if dispersion_type in (0, 1):
        # Simple linear or logarithmic spacing
        dispersion = \
            dispersion_start + np.arange(num_pixels) * mean_dispersion_delta

        if dispersion_start == 1:
            dispersion = 10.**dispersion

    elif dispersion_type == 2:
        # Non-linear mapping.
        if function_type is None:
            raise ValueError("function type required for non-linear mapping")
        elif function_type not in range(1, 7):
            raise ValueError(
                "function type {0} not recognised".format(function_type))

        if function_type == 1:
            order = int(order)
            n = np.linspace(-1, 1, Pmax - Pmin + 1)
            temp = np.zeros((Pmax - Pmin + 1, order), dtype=float)
            temp[:, 0] = 1
            temp[:, 1] = n
            for i in range(2, order):
                temp[:, i] = 2 * n * temp[:, i-1] - temp[:, i-2]
            
            for i in range(0, order):
                temp[:, i] *= coefficients[i]

            dispersion = temp.sum(axis=1)


        elif function_type == 2:
            # Legendre polynomial.
            if None in (order, Pmin, Pmax, coefficients):
                raise TypeError("order, Pmin, Pmax and coefficients required "
                                "for a Chebyshev or Legendre polynomial")

            Pmean = (Pmax + Pmin)/2
            Pptp = Pmax - Pmin
            x = (np.arange(num_pixels) + 1 - Pmean)/(Pptp/2)
            p0 = np.ones(num_pixels)
            p1 = mean_dispersion_delta

            dispersion = coefficients[0] * p0 + coefficients[1] * p1
            for i in range(2, int(order)):
                if function_type == 1:
                    # Chebyshev
                    p2 = 2 * x * p1 - p0
                else:
                    # Legendre
                    p2 = ((2*i - 1)*x*p1 - (i - 1)*p0) / i

                dispersion += p2 * coefficients[i]
                p0, p1 = (p1, p2)

        elif function_type == 3:
            # Cubic spline.
            if None in (order, Pmin, Pmax, coefficients):
                raise TypeError("order, Pmin, Pmax and coefficients required "
                                "for a cubic spline mapping")
            s = (np.arange(num_pixels, dtype=float) + 1 - Pmin)/(Pmax - Pmin) \
              * order
            j = s.astype(int).clip(0, order - 1)
            a, b = (j + 1 - s, s - j)
            x = np.array([
                a**3,
                1 + 3*a*(1 + a*b),
                1 + 3*b*(1 + a*b),
                b**3])
            dispersion = np.dot(np.array(coefficients), x.T)

        else:
            raise NotImplementedError("function type not implemented yet")

    else:
        raise ValueError(
            "dispersion type {0} not recognised".format(dispersion_type))

    # Apply redshift correction.
    dispersion = weight * (dispersion + offset) / (1 + redshift)
    return dispersion


