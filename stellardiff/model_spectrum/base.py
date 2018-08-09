#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Models for fitting spectral data. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["BaseSpectralModel"]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .quality_constraints import constraints

class BaseSpectralModel(object):

    def __init__(self, transitions, **kwargs):
        """
        Initialize a base class for modelling spectra.
        """

        self._transitions = transitions
        
        self.metadata = {
            "is_upper_limit": False,
            "use_for_stellar_composition_inference": True,
            "use_for_stellar_parameter_inference": (
                "Fe I" in self.transitions["element"] or
                "Fe II" in self.transitions["element"])
        }

        # Create a _repr_wavelength property.
        if len(self.transitions) == 1:
            self._repr_wavelength \
                = "{0:.1f}".format(self.transitions["wavelength"][0])
        else:
            self._repr_wavelength \
                = "~{0:.0f}".format(np.mean(self.transitions["wavelength"]))
        
        return None



    @property
    def wavelength(self):
        """
        Return a (sometimes approximate) wavelength for where this spectral line
        occurs.
        """
        wavelength = np.mean(self.transitions["wavelength"])
        return int(wavelength) if len(self.transitions) > 1 else wavelength


    @property
    def is_acceptable(self):
        """ Return whether this spectral model is acceptable. """
        return self.metadata.get("is_acceptable", False)


    @is_acceptable.setter
    def is_acceptable(self, decision):
        """
        Mark the spectral model as acceptable or unacceptable.

        :param decision:
            A boolean flag.
        """
        decision = bool(decision)
        if not decision or (decision and "fitted_result" in self.metadata):
            self.metadata["is_acceptable"] = bool(decision)
        return None


    @property
    def is_upper_limit(self):
        """ Return whether this spectral model is acceptable. """
        return self.metadata.get("is_upper_limit", False)


    @is_upper_limit.setter
    def is_upper_limit(self, decision):
        """
        Mark the spectral model as acceptable or unacceptable.

        :param decision:
            A boolean flag.
        """
        self.metadata["is_upper_limit"] = bool(decision)
        return None


    @property
    def use_for_stellar_parameter_inference(self):
        """
        Return whether this spectral model should be used during the
        determination of stellar parameters.
        """
        return self.metadata["use_for_stellar_parameter_inference"]


    @use_for_stellar_parameter_inference.setter
    def use_for_stellar_parameter_inference(self, decision):
        """
        Mark whether this spectral model should be used when inferring stellar
        parameters.

        :param decision:
            A boolean flag.
        """
        self.metadata["use_for_stellar_parameter_inference"] = bool(decision)
        return None


    @property
    def use_for_stellar_composition_inference(self):
        """
        Return whether this spectral model should be used for the determination
        of stellar composition.
        """
        return self.metadata["use_for_stellar_composition_inference"]



    @use_for_stellar_composition_inference.setter
    def use_for_stellar_composition_inference(self, decision):
        """
        Mark whether this spectral model should be used when inferring the 
        stellar composition.

        :param decision:
            A boolean flag.
        """
        self.metadata["use_for_stellar_composition_inference"] = bool(decision)
        return None


    def apply_quality_constraints(self, quality_constraints):
        """
        Apply quality constraints to the this spectral model. If the model does
        not meet the quality constraints it will be marked as unacceptable.

        :param quality_constraints:
            A dictionary containing constraint names as keys and a 2-length
            tuple with the (lower, upper) bounds specified as values.

        :returns:
            Whether this model met the specified quality constraints.
        """

        is_ok = constraints(self, quality_constraints)
        self.is_acceptable = is_ok
        return is_ok


    def meets_quality_constraints(self, quality_constraints):
        """
        Check whether this spectral model meets specified quality constraints.

        :param quality_constraints:
            A dictionary containing constraint names as keys and a 2-length
            tuple with the (lower, upper) bounds specified as values.

        :returns:
            Whether this model met the specified quality constraints.
        """
        return constraints(self, quality_constraints)



    @property
    def transitions(self):
        """ Return the transitions associateed with this class. """

        # This is left as a property to prevent users from arbitrarily setting
        # the .transitions attribute.
        return self._transitions


    @property
    def elements(self):
        """ Return the elements to be measured from this class. """
        return self.metadata["elements"]


    @property
    def species(self):
        """ Return the species to be measured from this class. """
        return self.metadata["species"]


    @property
    def abundances(self):
        """ Return abundances if fit, else None """
        try:
            return self.metadata["fitted_result"][2]["abundances"]
        except KeyError:
            return None
        

    @property
    def parameters(self):
        """
        Return the model parameters.
        This must be implemented by the sub-classes.
        """
        raise NotImplementedError(
            "parameters property must be implemented by the sub-classes")


    @property
    def parameter_bounds(self):
        """ Return the fitting limits on the parameters. """
        return self._parameter_bounds


    @property
    def parameter_names(self):
        """ Return the model parameter names. """
        return self._parameter_names


    def __call__(self, dispersion, *args, **kwargs):
        """ The data-generating function. """
        raise NotImplementedError(
            "the data-generating function must be implemented by sub-classes")


    def __getstate__(self):
        """ Return a serializable state of this spectral model. """
        raise NotImplementedError


    def __setstate__(self, state):
        """ Disallow the state to be instantiated from a serialised object. """
        return None


    def _verify_transitions(self):
        """
        Verify that the transitions provided are valid.
        """
        # TODO
        return True


    def _verify_spectrum(self, spectrum):
        """
        Check that the spectrum provided is valid and has data in the wavelength
        range that we are interested.

        :param spectrum:
            The observed rest-frame normalized spectrum.
        """

        # Check the transition is in the spectrum range.
        wavelength = self.transitions["wavelength"]
        try:
            wavelength = wavelength[0]
        except IndexError:
            None
        if wavelength + 1 > spectrum.dispersion[-1] \
        or wavelength - 1 < spectrum.dispersion[0]:
            raise ValueError(
                "the observed spectrum contains no data over the wavelength "
                "range we require")

        return spectrum


    def mask(self, spectrum):
        """
        Return a pixel mask based on the metadata and existing mask information
        available.

        :param spectrum:
            A spectrum to generate a mask for.
        """

        # HACK
        if "antimask_flag" not in self.metadata:
            self.metadata["antimask_flag"] = False
        if self.metadata["antimask_flag"]:
            antimask = np.ones_like(spectrum.dispersion,dtype=bool)
            for start, end in self.metadata["mask"]:
                antimask *= ~((spectrum.dispersion >= start) \
                            * (spectrum.dispersion <= end))
            return ~antimask

        mask = self.window_mask(spectrum)

        # Any masked ranges specified in the metadata?
        for start, end in self.metadata["mask"]:
            mask *= ~((spectrum.dispersion >= start) \
                     * (spectrum.dispersion <= end))

        return mask


    def window_mask(self, spectrum):
        window = abs(self.metadata["window"])
        wavelengths = self.transitions["wavelength"]
        try:
            lower_wavelength = wavelengths[0]
            upper_wavelength = wavelengths[-1]
        except IndexError:
            # Single row.
            lower_wavelength, upper_wavelength = (wavelengths, wavelengths)

        mask = (spectrum.dispersion >= lower_wavelength - window) \
             * (spectrum.dispersion <= upper_wavelength + window)

        return mask

    def fitting_function(self, dispersion, *parameters):
        """
        Generate data at the dispersion points, given the parameters, but
        respect the boundaries specified on model parameters.

        :param dispersion:
            An array of dispersion points to calculate the data for.

        :param parameters:
            Keyword arguments of the model parameters and their values.
        """

        for parameter_name, (lower, upper) in self.parameter_bounds.items():
            value = parameters[self.parameter_names.index(parameter_name)]
            if not (upper >= value and value >= lower):
                return np.nan * np.ones_like(dispersion)

        return self.__call__(dispersion, *parameters)


    def _fill_masked_arrays(self, spectrum, x, *y):
        """
        Detect masked regions and fill masked regions in y-axis arrays with
        NaNs.

        :param spectrum:
            The spectrum used in the fit.

        :param x:
            The x values that were used in the fit.

        :param *y:
            The y-axis arrays to fill.
        """

        indices = spectrum.dispersion.searchsorted(x)
        x_actual = spectrum.dispersion[indices[0]:1 + indices[-1]]

        filled_arrays = [x_actual]
        for yi in y:
            yi_actual = np.nan * np.ones_like(x_actual)
            if len(yi_actual.shape) == 2:
                yi_actual[:, indices - indices[0]] = yi
            else:
                yi_actual[indices - indices[0]] = yi
            filled_arrays.append(yi_actual)

        return tuple(filled_arrays)



    def plot(self, spectrum, draws=100, percentiles=(2.5, 97.5), **kwargs):


        model_color = kwargs.get("model_color", "tab:blue")
        mask_color = kwargs.get("mask_color", "tab:red")
        data_color = kwargs.get("data_color", "#000000")


        window_mask = self.window_mask(spectrum)

        x = spectrum.dispersion[window_mask]
        y = spectrum.flux[window_mask]
        yerr = spectrum.ivar[window_mask]**-0.5

        xx = np.array(x).repeat(2)[1:]
        xstep = np.repeat((x[1:] - x[:-1]), 2)
        xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
        xx = np.append(xx, xx.max() + xstep[-1]) - xstep/2.0
        yy = np.array(y).repeat(2)
        yyerr = np.array(yerr).repeat(2)

        fig, axes = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[1, 4]))

        ax_residual, ax_spectrum = axes

        ax_spectrum.plot(xx, yy, "-", c=data_color)
        ax_spectrum.fill_between(xx, yy - yyerr, yy + yyerr, 
                                 facecolor=data_color, alpha=0.3,
                                 edgecolor="none", zorder=-1)


        theta = self.metadata["fitted_result"][0].values()
        model = self(x, *theta)
        ax_spectrum.plot(x, model, "-", c=model_color)

        diff = yy - model.repeat(2)
        ax_residual.plot(xx, diff, "-", c=data_color)
        ax_residual.fill_between(xx, diff - yyerr, diff + yyerr, 
                                 facecolor=data_color, alpha=0.3, 
                                 edgecolor="none", zorder=-1)


        mu, cov = np.array(list(theta)), self.metadata["fitted_result"][1]
        theta_draws = np.random.multivariate_normal(mu, cov, draws)

        ys = np.array([self(x, *theta_draw) for theta_draw in theta_draws])
        y_lower, y_upper = np.percentile(ys, percentiles, axis=0)
        ax_spectrum.fill_between(x, y_lower, y_upper,
                                facecolor=model_color, alpha=0.3,
                                edgecolor="none", zorder=-1)
        ax_residual.fill_between(x, y_lower - model, y_upper - model,
                                 facecolor=model_color, alpha=0.3,
                                 edgecolor="none", zorder=-1)

        # show masks.
        for start, end in self.metadata["mask"]:
            for ax in axes:
                ax.axvspan(start, end, 
                           facecolor=mask_color, alpha=0.3, 
                           edgecolor="none", zorder=-1)

        ax_residual.axhline(0, c=model_color)
        ax_spectrum.axhline(1, c="#666666", linestyle=":", linewidth=1)
        ax_residual.set_ylim(-0.05, 0.05)
        ax_residual.set_ylabel(r"$\Delta$")

        ax_spectrum.set_ylim(0, 1.15)
        for ax in axes:
            ax.set_xlim(x[0], x[-1])

        ax_residual.set_xticks([])
        ax_spectrum.set_xlabel(r"Wavelength")
        ax_spectrum.set_ylabel(r"Normalized flux")
        ax_spectrum.xaxis.set_major_locator(MaxNLocator(6))
        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        return fig
