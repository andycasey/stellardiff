
import numpy as np
import matplotlib.pyplot as plt
import yaml
import astropy.units as u

import stellardiff as sd

sd.mpl_utils.use_style()


spectrum = sd.spectrum.Spectrum1D.read("twin_binaries/standards/0_sun_n.fits")

# Load in the transitions and settings.
transitions = sd.linelist.LineList.read("sun_linelist.moog")
with open("sun_settings.yaml", "r") as fp:
    profile_settings = yaml.load(fp)

transitions["equivalent_width"] = np.nan * np.ones(len(transitions))
transitions["equivalent_width_err_pos"] = np.nan * np.ones(len(transitions))
transitions["equivalent_width_err_neg"] = np.nan * np.ones(len(transitions))


overwrite_kwds = dict(profile="gaussian")

quality_constraints = dict(
  abundance=[-10, 10],
  abundance_uncertainty=[0, 1],
  equivalent_width=[1, 1000],
  equivalent_width_percentage_uncertainty=[0, 25],
  equivalent_width_uncertainty=[0, 1000],
  reduced_equivalent_width=[-10, -3],
)


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
    
    model = sd.model_spectrum.ProfileFittingModel(transitions[tm], **kwds)

    result = model.fit(spectrum)

    # Save the equivalent width.
    ew, ew_err_neg, ew_err_pos = model.equivalent_width

    transitions["equivalent_width"][tm] = ew.to(10**-3 * u.Angstrom).value
    transitions["equivalent_width_err_neg"][tm] = ew_err_neg.to(10**-3 * u.Angstrom).value
    transitions["equivalent_width_err_pos"][tm] = ew_err_pos.to(10**-3 * u.Angstrom).value

    models.append(model)
    indices.append(np.where(tm)[0][0])



# Load Spina results.
spina_result_path = "twin_binaries/standards/sun_2015_05_18-2.moog.edited"
spina = sd.linelist.LineList.read(spina_result_path)
spina["equivalent_width"] = np.loadtxt(spina_result_path, 
                                       usecols=(-1, ), skiprows=1)


# Plot comparison.
x = np.array(spina["equivalent_width"])
y = np.array(transitions["equivalent_width"])
y_err_pos = transitions["equivalent_width_err_pos"]
y_err_neg = transitions["equivalent_width_err_neg"]

fig, ax = plt.subplots()
ax.scatter(x, y, facecolor="tab:blue", s=5)
ax.errorbar(x, y, yerr=(y_err_pos, -y_err_neg), fmt=None, ecolor="tab:blue",
            linewidth=1)

limits = ax.get_xlim()
ax.plot(limits, limits, c="#666666", linestyle=":", lw=1)
ax.set_xlim(limits)
ax.set_ylim(limits)

ax.set_xlabel(r"$\textrm{EW}_\textrm{Spina}$ \textrm{/ m\AA}")
ax.set_ylabel(r"$\textrm{EW}_\textrm{Auto}$ \textrm{/ m\AA}")

fig.tight_layout()

raise a


# Show the K most discrepant measurements.
K = 10
diffs = np.abs(y - x)
diffs[~np.isfinite(diffs)] = 0.0

for index in np.argsort(diffs)[::-1][:K]:


    model = models[indices.index(index)]
    print(index, model.transitions["element"][0], model.transitions["wavelength"][0], spina["equivalent_width"][index], transitions["equivalent_width"][index], 
        model.meets_quality_constraints(quality_constraints), model.transitions["hash"])

    fig = model.plot(spectrum)
    fig.axes[0].set_title(
        r"\textrm{{{0} {1:.0f}}} $\textrm{{EW}}_\textrm{{Spina}}$ = {2:.1f}; $\textrm{{EW}}_\textrm{{Auto}}$ = {3:.1f}".format(
                          model.transitions["element"][0],
                          model.transitions["wavelength"][0],
                          spina["equivalent_width"][index],
                          transitions["equivalent_width"][index]))

