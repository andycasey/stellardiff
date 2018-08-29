
import numpy as np
import os
import yaml
import astropy.units as u
from glob import glob

import stellardiff as sd

sd.mpl_utils.use_style()


MAKE_FIGURES = True

# Load in the transitions and settings.
transitions = sd.linelist.LineList.read("sun_linelist.moog")
with open("sun_settings.yaml", "r") as fp:
    profile_settings = yaml.load(fp)

transitions["equivalent_width"] = np.nan * np.ones(len(transitions))
transitions["equivalent_width_err_pos"] = np.nan * np.ones(len(transitions))
transitions["equivalent_width_err_neg"] = np.nan * np.ones(len(transitions))



quality_constraints = dict(
  abundance=[-10, 10],
  abundance_uncertainty=[0, 1],
  equivalent_width=[1, 1000],
  equivalent_width_percentage_uncertainty=[0, 25],
  equivalent_width_uncertainty=[0, 1000],
  reduced_equivalent_width=[-10, -3],
)

spectrum_glob_mask = "twin_binaries/*/*.fits"

get_output_path = lambda star, basename: "twin_binaries_voight_output/{0}/{1}"\
                                         .format(star, basename)
overwrite_kwds = dict()#profile="gaussian")


input_spectrum_path = glob(spectrum_glob_mask)
for input_spectrum in input_spectrum_path:

    star = os.path.basename(input_spectrum).split(".fits")[0]
    spectrum = sd.spectrum.Spectrum1D.read(input_spectrum)

    print("Running on star {} from {}".format(star, input_spectrum))

    os.makedirs(os.path.dirname(get_output_path(star, "")), exist_ok=True)

    models = []
    indices = []

    for i, settings in enumerate(profile_settings):
        # Skip bad lines.
        if not settings["metadata"]["is_acceptable"]:
            continue

        # Create a transitions mask.
        tm = np.in1d(transitions["hash"], settings["transition_hashes"])

        kwds = settings["metadata"].copy()
        kwds.update(overwrite_kwds)
        kwds.pop("is_acceptable", None)
        
        model = sd.model_spectrum.ProfileFittingModel(transitions[tm], **kwds)

        try:
            result = model.fit(spectrum)

        except:
            continue

        # Only use the line if it meets quality constraints.
        if "fitted_result" not in model.metadata \
        or not model.meets_quality_constraints(quality_constraints):
            continue

        # Save the equivalent width.
        ew, ew_err_neg, ew_err_pos = model.equivalent_width

        transitions["equivalent_width"][tm] = ew.to(10**-3 * u.Angstrom).value
        transitions["equivalent_width_err_neg"][tm] = ew_err_neg.to(10**-3 * u.Angstrom).value
        transitions["equivalent_width_err_pos"][tm] = ew_err_pos.to(10**-3 * u.Angstrom).value

        models.append(model)
        indices.append(np.where(tm)[0][0])

        if MAKE_FIGURES:
            fig = model.plot(spectrum)
            basename = "{element}-{wavelength:.0f}.png".format(
                element=model.transitions["element"][0].replace(" ", "-"),
                wavelength=model.transitions["wavelength"][0])
            output_path = get_output_path(star, basename)
            fig.savefig(output_path)
            print("Created figure {}".format(output_path))

            plt.close("all")

    output_path = get_output_path(star, "linelist.moog")
    transitions.write(output_path, format="moog")
    print("Saved line list to {}".format(output_path))


