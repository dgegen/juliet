import copy

import astropy.constants as const
import numpy as np
import radvel

from .utils import (
    correct_light_travel_time,
    get_quantiles,
    init_batman,
    init_catwoman,
    init_radvel,
    reverse_ld_coeffs,
)

__all__ = ["model"]

LOG_2_PI = np.log(2.0 * np.pi)  # ln(2*pi)


class model(object):
    """
    Given a juliet data object, this kernel generates either a lightcurve or a radial-velocity object. Example usage:

               >>> model = juliet.model(data, modeltype = 'lc')

    :param data: (juliet.load object)
        An object containing all the information about the current dataset.

    :param modeltype: (optional, string)
        String indicating whether the model to generate should be a lightcurve ('lc') or a radial-velocity ('rv') model.

    :param pl: (optional, float)
        If the ``(r1,r2)`` parametrization for ``(b,p)`` is used, this defines the lower limit of the planet-to-star radius ratio to be sampled.
        Default is ``0``.

    :param pu: (optional, float)
        Same as ``pl``, but for the upper limit. Default is ``1``.

    :param ecclim: (optional, float)
        This parameter sets the maximum eccentricity allowed such that a model is actually evaluated. Default is ``1``.

    :param light_travel_delay: (optinal, bool)
        Boolean indicating if light travel time delay wants to be included on eclipse time calculations.

    :param stellar_radius: (optional, float)
        Stellar radius in units of solar-radii to use for the light travel time corrections.

    :param log_like_calc: (optional, boolean)
        If True, it is assumed the model is generated to generate likelihoods values, and thus this skips the saving/calculation of the individual
        models per planet (i.e., ``self.model['p1']``, ``self.model['p2']``, etc. will not exist). Default is False.

    """

    def __init__(
        self,
        data,
        modeltype,
        pl=0.0,
        pu=1.0,
        ecclim=1.0,
        ta=2458460.0,
        light_travel_delay=False,
        stellar_radius=None,
        log_like_calc=False,
    ):
        # Inhert the priors dictionary from data:
        self.priors = data.priors
        # Define the ecclim value:
        self.ecclim = ecclim
        # Define ta:
        self.ta = ta
        # Define light travel time option:
        self.light_travel_delay = light_travel_delay
        if self.light_travel_delay and (stellar_radius is None):
            raise Exception(
                "Error: if light_travel_delay is activated, a stellar radius needs to be given as well via stellar_radius = yourvalue; e.g., dataset.fit(..., light_travel_delay = True, stellar_radius = 1.1234)."
            )

        self.stellar_radius = stellar_radius

        # Save the log_like_calc boolean:
        self.log_like_calc = log_like_calc
        # Define variable that at each iteration defines if the model is OK or not (not OK means something failed in terms of the
        # parameter space being explored):
        self.modelOK = True
        # Define a variable that will save the posterior samples:
        self.posteriors = None
        self.median_posterior_samples = None
        # Set nlm:
        self.non_linear_functions = data.non_linear_functions
        # Check if multiplicative of additive functions for each instrument:
        if self.non_linear_functions is not None:
            self.multiplicative_non_linear_function = {}

            for k in list(self.non_linear_functions.keys()):
                if "multiplicative" in self.non_linear_functions[k].keys():
                    if self.non_linear_functions[k]["multiplicative"]:
                        self.multiplicative_non_linear_function[k] = True

                    else:
                        self.multiplicative_non_linear_function[k] = False

                else:
                    # For back-compatibility:
                    self.multiplicative_non_linear_function[k] = False

        # Number of datapoints per instrument variable:
        self.ndatapoints_per_instrument = {}
        if modeltype == "lc":
            self.modeltype = "lc"
            # Inhert times, fluxes, errors, indexes, etc. from data.
            # FYI, in case this seems confusing: self.t, self.y and self.yerr save what we internally call
            # "global" data-arrays. These have the data from all the instruments stacked into an array; to recover
            # the data for a given instrument, one uses the self.instrument_indexes dictionary. On the other hand,
            # self.times, self.data and self.errors are dictionaries that on each key have the data of a given instrument.
            # Calling dictionaries is faster than calling indexes of arrays, so we use the latter in general to evaluate models.
            self.t = data.t_lc
            self.y = data.y_lc
            self.yerr = data.yerr_lc
            self.times = data.times_lc
            self.data = data.data_lc
            self.errors = data.errors_lc
            self.instruments = data.instruments_lc
            self.ninstruments = data.ninstruments_lc
            self.inames = data.inames_lc
            self.instrument_indexes = data.instrument_indexes_lc
            self.lm_boolean = data.lm_lc_boolean
            self.nlm_boolean = data.nlm_lc_boolean
            self.lm_arguments = data.lm_lc_arguments
            self.lm_n = {}
            self.pl = pl
            self.pu = pu
            self.Ar = (self.pu - self.pl) / (2.0 + self.pl + self.pu)
            self.global_model = data.global_lc_model
            self.dictionary = data.lc_options
            self.numbering = data.numbering_transiting_planets
            self.numbering.sort()
            self.nplanets = len(self.numbering)
            self.model = {}
            # First, if a global model, generate array that will save this:
            if self.global_model:
                self.model["global"] = np.zeros(len(self.t))
                self.model["global_variances"] = np.zeros(len(self.t))
                self.model["deterministic"] = np.zeros(len(self.t))
            # If limb-darkening, dilution factors or eclipse depth will be shared by different instruments, set the correct variable name for each:
            self.ld_iname = {}
            self.sigmaw_iname = {}
            self.mdilution_iname = {}
            self.mflux_iname = {}
            self.theta_iname = {}
            self.fp_iname = {}
            self.phaseoffset_iname = {}
            # To make transit depth (for batman and catwoman models) will be shared by different instruments, set the correct variable name for each:
            self.p_iname = {}
            self.p1_iname = {}
            # Since p, p1 (p2) and fp are all planetary and instrumental parameters,
            # we want to make sure that we have correct variable name for each instruments (when the instruments are shared) for _every_ planets.
            for i in self.numbering:
                self.p_iname["p" + str(i)] = {}
                self.p1_iname["p" + str(i)] = {}
                self.fp_iname["p" + str(i)] = {}
                self.phaseoffset_iname["p" + str(i)] = {}
            self.ndatapoints_all_instruments = 0
            # Variable that turns to false only if there are no TTVs. Otherwise, always positive:
            self.Tflag = False
            # Variable that sets the total number of transit times in the whole dataset:
            self.N_TTVs = {}
            # Variable that sets if the T-parametrization will be True:
            self.Tparametrization = {}
            for pi in self.numbering:
                self.N_TTVs[pi] = 0.0

            for instrument in self.inames:
                for pi in self.numbering:
                    if self.dictionary[instrument]["TTVs"][pi]["status"]:
                        if (
                            self.dictionary[instrument]["TTVs"][pi]["parametrization"]
                            == "T"
                        ):
                            self.Tparametrization[pi] = True
                            self.Tflag = True
                        self.N_TTVs[pi] += self.dictionary[instrument]["TTVs"][pi][
                            "totalTTVtransits"
                        ]
                self.model[instrument] = {}
                # Extract number of datapoints per instrument:
                self.ndatapoints_per_instrument[instrument] = len(
                    self.instrument_indexes[instrument]
                )
                self.ndatapoints_all_instruments += self.ndatapoints_per_instrument[
                    instrument
                ]
                # Extract number of linear model terms per instrument:
                if self.lm_boolean[instrument]:
                    self.lm_n[instrument] = self.lm_arguments[instrument].shape[1]
                # An array of ones to copy around:
                self.model[instrument]["ones"] = np.ones(
                    len(self.instrument_indexes[instrument])
                )

                # Generate internal model variables of interest to the user. First, the lightcurve model in the notation of juliet (Mi)
                # (full lightcurve plus dilution factors and mflux):
                self.model[instrument]["M"] = np.ones(
                    len(self.instrument_indexes[instrument])
                )
                # Linear model (in the notation of juliet, LM):
                self.model[instrument]["LM"] = np.zeros(
                    len(self.instrument_indexes[instrument])
                )
                # Now, generate dictionary that will save the final full, deterministic model (M + LM):
                self.model[instrument]["deterministic"] = np.zeros(
                    len(self.instrument_indexes[instrument])
                )
                # Same for the errors:

                self.model[instrument]["deterministic_errors"] = np.zeros(
                    len(self.instrument_indexes[instrument])
                )
                if (
                    self.dictionary[instrument]["TransitFit"]
                    or self.dictionary[instrument]["EclipseFit"]
                    or self.dictionary[instrument]["TranEclFit"]
                ):
                    # First, take the opportunity to initialize transit lightcurves for each instrument:
                    if self.dictionary[instrument]["resampling"]:
                        if not self.dictionary[instrument]["TransitFitCatwoman"]:
                            if self.dictionary[instrument]["TransitFit"]:
                                (
                                    self.model[instrument]["params"],
                                    [self.model[instrument]["m"], _],
                                ) = init_batman(
                                    self.times[instrument],
                                    self.dictionary[instrument]["ldlaw"],
                                    nresampling=self.dictionary[instrument][
                                        "nresampling"
                                    ],
                                    etresampling=self.dictionary[instrument][
                                        "exptimeresampling"
                                    ],
                                )
                            elif self.dictionary[instrument]["EclipseFit"]:
                                (
                                    self.model[instrument]["params"],
                                    [_, self.model[instrument]["m"]],
                                ) = init_batman(
                                    self.times[instrument],
                                    self.dictionary[instrument]["ldlaw"],
                                    nresampling=self.dictionary[instrument][
                                        "nresampling"
                                    ],
                                    etresampling=self.dictionary[instrument][
                                        "exptimeresampling"
                                    ],
                                )
                            elif self.dictionary[instrument]["TranEclFit"]:
                                (
                                    self.model[instrument]["params"],
                                    self.model[instrument]["m"],
                                ) = init_batman(
                                    self.times[instrument],
                                    self.dictionary[instrument]["ldlaw"],
                                    nresampling=self.dictionary[instrument][
                                        "nresampling"
                                    ],
                                    etresampling=self.dictionary[instrument][
                                        "exptimeresampling"
                                    ],
                                )
                        else:
                            (
                                self.model[instrument]["params"],
                                self.model[instrument]["m"],
                            ) = init_catwoman(
                                self.times[instrument],
                                self.dictionary[instrument]["ldlaw"],
                                nresampling=self.dictionary[instrument]["nresampling"],
                                etresampling=self.dictionary[instrument][
                                    "exptimeresampling"
                                ],
                            )
                    else:
                        if not self.dictionary[instrument]["TransitFitCatwoman"]:
                            if self.dictionary[instrument]["TransitFit"]:
                                (
                                    self.model[instrument]["params"],
                                    [self.model[instrument]["m"], _],
                                ) = init_batman(
                                    self.times[instrument],
                                    self.dictionary[instrument]["ldlaw"],
                                )
                            elif self.dictionary[instrument]["EclipseFit"]:
                                (
                                    self.model[instrument]["params"],
                                    [_, self.model[instrument]["m"]],
                                ) = init_batman(
                                    self.times[instrument],
                                    self.dictionary[instrument]["ldlaw"],
                                )
                            elif self.dictionary[instrument]["TranEclFit"]:
                                (
                                    self.model[instrument]["params"],
                                    self.model[instrument]["m"],
                                ) = init_batman(
                                    self.times[instrument],
                                    self.dictionary[instrument]["ldlaw"],
                                )
                        else:
                            (
                                self.model[instrument]["params"],
                                self.model[instrument]["m"],
                            ) = init_catwoman(
                                self.times[instrument],
                                self.dictionary[instrument]["ldlaw"],
                            )
                    # Individual transit lightcurves for each planet:
                    for i in self.numbering:
                        self.model[instrument]["p" + str(i)] = np.ones(
                            len(self.instrument_indexes[instrument])
                        )

                # First, check some edge cases of user input error. First, if user decided to use a_p1 and rho, raise an error:
                if ("a_p1" in self.priors.keys()) and (
                    ("rho" in self.priors.keys())
                    or (
                        ("r_star" in self.priors.keys())
                        and ("m_star" in self.priors.keys())
                    )
                ):
                    raise Exception(
                        "Priors currently define a_p1 (a/Rstar) and rho (stellar density) --- these are redundant. Please choose to fit either a_p1 or rho (or r_star+m_star) in your fit."
                    )

                # Now proceed with instrument namings:
                for pname in self.priors.keys():
                    # Check if variable name is a limb-darkening coefficient:
                    if pname[0:2] == "q1" or pname[0:2] == "u1" or pname[0:2] == "c1":
                        vec = pname.split("_")
                        if len(vec) > 2:
                            if instrument in vec:
                                self.ld_iname[instrument] = "_".join(vec[1:])

                        else:
                            if instrument in vec:
                                self.ld_iname[instrument] = vec[1]

                    # Check if it is a theta LM:
                    if pname[0:5] == "theta":
                        vec = pname.split("_")
                        theta_number = vec[0][5:]
                        if len(vec) > 2:
                            if instrument in vec:
                                self.theta_iname[theta_number + instrument] = "_".join(
                                    vec[1:]
                                )
                        else:
                            if instrument in vec:
                                self.theta_iname[theta_number + instrument] = vec[1]
                    # Check if sigma_w:
                    if pname[0:7] == "sigma_w":
                        vec = pname.split("_")
                        if len(vec) > 3:
                            if instrument in vec:
                                self.sigmaw_iname[instrument] = "_".join(vec[2:])
                        else:
                            if instrument in vec:
                                self.sigmaw_iname[instrument] = vec[2]
                    # Check if it is a dilution factor:
                    if pname[0:9] == "mdilution":
                        vec = pname.split("_")
                        if len(vec) > 2:
                            if instrument in vec:
                                self.mdilution_iname[instrument] = "_".join(vec[1:])
                        else:
                            if instrument in vec:
                                self.mdilution_iname[instrument] = vec[1]
                    if pname[0:5] == "mflux":
                        vec = pname.split("_")
                        if len(vec) > 2:
                            if instrument in vec:
                                self.mflux_iname[instrument] = "_".join(vec[1:])
                        else:
                            if instrument in vec:
                                self.mflux_iname[instrument] = vec[1]

                    if pname[0:2] == "fp":
                        # Note that eclipse and transit depths can be a planetary and instrumental parameter
                        vec = pname.split("_")
                        if len(vec) > 3:
                            # This is the case in which multiple instruments share an eclipse depth, e.g., fp_p1_TESS1_TESS2
                            if instrument in vec:
                                self.fp_iname[vec[1]][instrument] = "_" + "_".join(
                                    vec[2:]
                                )

                        elif len(vec) == 3:
                            # This is the case of a single instrument with fp, e.g., fp_p1_TESS
                            if instrument in vec:
                                self.fp_iname[vec[1]][instrument] = "_" + vec[2]

                        elif len(vec) == 2:
                            # This adds back-compatibility so users can define a common fp for all instruments (e.g., fp_p1):
                            self.fp_iname[vec[1]][instrument] = ""

                        else:
                            raise Exception(
                                "Prior for fp is not properly defined: must be, e.g., fp_p1, fp_p1_inst or fp_p1_inst1_inst2. Currently is "
                                + pname
                            )

                    if pname[0:11] == "phaseoffset":
                        # Note that amplitude can be a planetary and instrumental parameter
                        vec = pname.split("_")
                        if len(vec) > 3:
                            # This is the case in which multiple instruments share the parameter:
                            if instrument in vec:
                                self.phaseoffset_iname[vec[1]][instrument] = (
                                    "_" + "_".join(vec[2:])
                                )

                        elif len(vec) == 3:
                            # This is the case of a single instrument:
                            if instrument in vec:
                                self.phaseoffset_iname[vec[1]][instrument] = (
                                    "_" + vec[2]
                                )

                        elif len(vec) == 2:
                            # This adds back-compatibility so users can define a common for all instruments:
                            self.phaseoffset_iname[vec[1]][instrument] = ""

                        else:
                            raise Exception(
                                "Prior for phaseoffset is not properly defined: must be, e.g., phaseoffset_p1, phaseoffset_p1_inst or phaseoffset_p1_inst1_inst2. Currently is "
                                + pname
                            )

                    if pname[0:2] == "p_":
                        vec = pname.split("_")
                        if len(vec) > 3:
                            # This is the case in which multiple instruments share a planet-to-star ratio, e.g., p_p1_TESS1_TESS2
                            if instrument in vec:
                                self.p_iname[vec[1]][instrument] = "_" + "_".join(
                                    vec[2:]
                                )

                        elif len(vec) == 3:
                            # This is the case of a single instrument with p, e.g., p_p1_TESS:
                            if instrument in vec:
                                self.p_iname[vec[1]][instrument] = "_" + vec[2]

                        elif len(vec) == 2:
                            # This adds back-compatibility so users can define a common p for all instruments (e.g., p_p1):
                            self.p_iname[vec[1]][instrument] = ""

                        else:
                            raise Exception(
                                "Prior for p is not properly defined: must be, e.g., p_p1, p_p1_inst or p_p1_inst1_inst2. Currently is "
                                + pname
                            )

                    if pname[0:2] == "p1":
                        vec = pname.split("_")
                        if len(vec) > 3:
                            # This is the case in which multiple instruments share a CW semi-planet-to-star ratio, e.g., p1_p1_TESS1_TESS2
                            if instrument in vec:
                                self.p1_iname[vec[1]][instrument] = "_" + "_".join(
                                    vec[2:]
                                )

                        elif len(vec) == 3:
                            # This is the case of a single instrument with p1, e.g., p1_p1_TESS:
                            if instrument in vec:
                                self.p1_iname[vec[1]][instrument] = "_" + vec[2]

                        elif len(vec) == 2:
                            # This adds back-compatibility so users can define a common p for all instruments (e.g., p_p1):
                            self.p1_iname[vec[1]][instrument] = ""

                        else:
                            raise Exception(
                                "Prior for p1/p2 is not properly defined: must be, e.g., p1_p1, p1_p1_inst or p1_p1_inst1_inst2. Currently is "
                                + pname
                            )

            # Set the model-type to M(t):
            self.evaluate = self.evaluate_model
            self.generate = self.generate_lc_model

        elif modeltype == "rv":
            self.modeltype = "rv"
            # Inhert times, RVs, errors, indexes, etc. from data:
            self.t = data.t_rv
            self.y = data.y_rv
            self.yerr = data.yerr_rv
            self.times = data.times_rv
            self.data = data.data_rv
            self.errors = data.errors_rv
            self.instruments = data.instruments_rv
            self.ninstruments = data.ninstruments_rv
            self.inames = data.inames_rv
            self.instrument_indexes = data.instrument_indexes_rv
            self.nlm_boolean = data.nlm_rv_boolean
            self.lm_boolean = data.lm_rv_boolean
            self.lm_arguments = data.lm_rv_arguments
            self.lm_n = {}
            self.global_model = data.global_rv_model
            self.dictionary = data.rv_options
            self.numbering = data.numbering_rv_planets
            self.numbering.sort()
            self.nplanets = len(self.numbering)
            self.model = {}
            self.ndatapoints_all_instruments = 0
            # First, if a global model, generate array that will save this:
            if self.global_model:
                self.model["global"] = np.zeros(len(self.t))
                self.model["global_variances"] = np.zeros(len(self.t))
            # Initialize radvel:
            self.model["radvel"] = init_radvel(nplanets=self.nplanets)
            # First go around all planets to compute the full RV models:
            for i in self.numbering:
                self.model["p" + str(i)] = np.ones(len(self.t))
            # Now variable to save full RV Keplerian model:
            self.model["Keplerian"] = np.ones(len(self.t))
            # Same for Keplerian + trends:
            self.model["Keplerian+Trend"] = np.ones(len(self.t))
            # Go around each instrument:
            for instrument in self.inames:
                self.model[instrument] = {}
                # Extract number of datapoints per instrument:
                self.ndatapoints_per_instrument[instrument] = len(
                    self.instrument_indexes[instrument]
                )
                self.ndatapoints_all_instruments += self.ndatapoints_per_instrument[
                    instrument
                ]
                # Extract number of linear model terms per instrument:
                if self.lm_boolean[instrument]:
                    self.lm_n[instrument] = self.lm_arguments[instrument].shape[1]

                # Generate internal model variables of interest to the user. First, the RV model in the notation of juliet (Mi)
                # (full RV model plus offset velocity, plus trend):
                self.model[instrument]["M"] = np.ones(
                    len(self.instrument_indexes[instrument])
                )
                # Linear model (in the notation of juliet, LM):
                self.model[instrument]["LM"] = np.zeros(
                    len(self.instrument_indexes[instrument])
                )
                # Now, generate dictionary that will save the final full model (M + LM):
                self.model[instrument]["deterministic"] = np.zeros(
                    len(self.instrument_indexes[instrument])
                )
                # Same for the errors:
                self.model[instrument]["deterministic_errors"] = np.zeros(
                    len(self.instrument_indexes[instrument])
                )
                # Individual keplerians for each planet:
                for i in self.numbering:
                    self.model[instrument]["p" + str(i)] = np.ones(
                        len(self.instrument_indexes[instrument])
                    )
                # An array of ones to copy around:
                self.model[instrument]["ones"] = np.ones(
                    len(self.t[self.instrument_indexes[instrument]])
                )
            # Set the model-type to M(t):
            self.evaluate = self.evaluate_model
            self.generate = self.generate_rv_model
        else:
            raise Exception(
                'Model type "'
                + lc
                + '" not recognized. Currently it can only be "lc" for a light-curve model or "rv" for radial-velocity model.'
            )

    def generate_rv_model(self, parameter_values, evaluate_global_errors=True):
        self.modelOK = True
        # Before anything continues, check the periods are chronologically ordered (this is to avoid multiple modes due to
        # periods "jumping" between planet numbering):
        first_time = True
        for i in self.numbering:
            if first_time:
                cP = parameter_values["P_p" + str(i)]
                first_time = False
            else:
                if cP < parameter_values["P_p" + str(i)]:
                    cP = parameter_values["P_p" + str(i)]
                else:
                    self.modelOK = False
                    return False

        # First, extract orbital parameters and save them, which will be common to all instruments:
        for n in range(self.nplanets):
            i = self.numbering[n]

            # Semi-amplitudes, t0 and P:
            K, t0, P = (
                parameter_values["K_p" + str(i)],
                parameter_values["t0_p" + str(i)],
                parameter_values["P_p" + str(i)],
            )

            # Extract eccentricity and omega depending on the used parametrization for each planet:
            if self.dictionary["ecc_parametrization"][i] == 0:
                ecc, omega = (
                    parameter_values["ecc_p" + str(i)],
                    parameter_values["omega_p" + str(i)] * np.pi / 180.0,
                )
            elif self.dictionary["ecc_parametrization"][i] == 1:
                ecc = np.sqrt(
                    parameter_values["ecosomega_p" + str(i)] ** 2
                    + parameter_values["esinomega_p" + str(i)] ** 2
                )
                omega = np.arctan2(
                    parameter_values["esinomega_p" + str(i)],
                    parameter_values["ecosomega_p" + str(i)],
                )
            else:
                ecc = (
                    parameter_values["secosomega_p" + str(i)] ** 2
                    + parameter_values["sesinomega_p" + str(i)] ** 2
                )
                omega = np.arctan2(
                    parameter_values["sesinomega_p" + str(i)],
                    parameter_values["secosomega_p" + str(i)],
                )

            # Generate lightcurve for the current planet if ecc is OK:
            if ecc > self.ecclim:
                self.modelOK = False
                return False

            # Save them to radvel:
            self.model["radvel"]["per" + str(n + 1)] = radvel.Parameter(value=P)
            self.model["radvel"]["tc" + str(n + 1)] = radvel.Parameter(value=t0)
            self.model["radvel"]["w" + str(n + 1)] = radvel.Parameter(
                value=omega
            )  # note given in radians
            self.model["radvel"]["e" + str(n + 1)] = radvel.Parameter(value=ecc)
            self.model["radvel"]["k" + str(n + 1)] = radvel.Parameter(value=K)

        # If log_like_calc is True (by default during juliet.fit), don't bother saving the RVs of planet p_i:
        if self.log_like_calc:
            self.model["Keplerian"] = radvel.model.RVModel(
                self.model["radvel"]
            ).__call__(self.t)
        else:
            self.model["Keplerian"] = radvel.model.RVModel(
                self.model["radvel"]
            ).__call__(self.t)
            for n in range(self.nplanets):
                i = self.numbering[n]
                self.model["p" + str(i)] = radvel.model.RVModel(
                    self.model["radvel"]
                ).__call__(self.t, planet_num=n + 1)

        # If trends are being fitted, add them to the Keplerian+Trend model:
        if self.dictionary["fitrvline"]:
            self.model["Keplerian+Trend"] = (
                self.model["Keplerian"]
                + parameter_values["rv_intercept"]
                + (self.t - self.ta) * parameter_values["rv_slope"]
            )

        elif self.dictionary["fitrvquad"]:
            self.model["Keplerian+Trend"] = (
                self.model["Keplerian"]
                + parameter_values["rv_intercept"]
                + (self.t - self.ta) * parameter_values["rv_slope"]
                + ((self.t - self.ta) ** 2) * parameter_values["rv_quad"]
            )
        else:
            self.model["Keplerian+Trend"] = self.model["Keplerian"]

        # Populate the self.model[instrument]['deterministic'] array. This hosts the full (deterministic) model for each RV instrument.
        for instrument in self.inames:
            assert isinstance(instrument, str), "Instrument must be of type string"
            self.model[instrument]["deterministic"] = (
                self.model["Keplerian+Trend"][self.instrument_indexes[instrument]]
                + parameter_values["mu_" + instrument]
            )

            self.model[instrument]["deterministic_variances"] = (
                self.errors[instrument] ** 2
                + parameter_values["sigma_w_" + instrument] ** 2
            )

            if self.lm_boolean[instrument]:
                self.model[instrument]["LM"] = np.zeros(
                    self.ndatapoints_per_instrument[instrument]
                )
                for i in range(self.lm_n[instrument]):
                    self.model[instrument]["LM"] += (
                        parameter_values[
                            "theta"
                            + str(i)
                            + "_"
                            + self.theta_iname[str(i) + instrument]
                        ]
                        * self.lm_arguments[instrument][:, i]
                    )
                self.model[instrument]["deterministic"] += self.model[instrument]["LM"]
            # If the model under consideration is a global model, populate the global model dictionary:
            if self.global_model:
                self.model["global"][self.instrument_indexes[instrument]] = self.model[
                    instrument
                ]["deterministic"]
                if evaluate_global_errors:
                    self.model["global_variances"][
                        self.instrument_indexes[instrument]
                    ] = (
                        self.yerr[self.instrument_indexes[instrument]] ** 2
                        + parameter_values["sigma_w_" + instrument] ** 2
                    )

    def get_GP_plus_deterministic_model(self, parameter_values, instrument=None):
        if self.global_model:
            if self.dictionary["global_model"]["GPDetrend"]:
                # residuals = self.residuals #self.y - self.model['global']
                self.dictionary["global_model"]["noise_model"].set_parameter_vector(
                    parameter_values
                )
                self.dictionary["global_model"]["noise_model"].yerr = np.sqrt(
                    self.variances
                )
                self.dictionary["global_model"]["noise_model"].compute_GP(
                    X=self.original_GPregressors
                )
                # Return mean signal plus GP model:
                self.model["GP"] = self.dictionary["global_model"][
                    "noise_model"
                ].GP.predict(
                    self.residuals,
                    self.dictionary["global_model"]["noise_model"].X,
                    return_var=False,
                    return_cov=False,
                )
                return (
                    self.model["global"],
                    self.model["GP"],
                    self.model["global"] + self.model["GP"],
                )
            else:
                return self.model["global"]
        else:
            if self.dictionary[instrument]["GPDetrend"]:
                # residuals = self.residuals#self.data[instrument] - self.model[instrument]['deterministic']
                self.dictionary[instrument]["noise_model"].set_parameter_vector(
                    parameter_values
                )
                self.model[instrument]["GP"] = self.dictionary[instrument][
                    "noise_model"
                ].GP.predict(
                    self.residuals,
                    self.dictionary[instrument]["noise_model"].X,
                    return_var=False,
                    return_cov=False,
                )
                return (
                    self.model[instrument]["deterministic"],
                    self.model[instrument]["GP"],
                    self.model[instrument]["deterministic"]
                    + self.model[instrument]["GP"],
                )
            else:
                return self.model[instrument]["deterministic"]

    def evaluate_model(
        self,
        instrument=None,
        parameter_values=None,
        all_samples=False,
        nsamples=1000,
        return_samples=False,
        t=None,
        GPregressors=None,
        LMregressors=None,
        return_err=False,
        alpha=0.68,
        return_components=False,
        evaluate_transit=False,
    ):
        """
        This function evaluates the current lc or rv model given a set of posterior distribution samples and/or parameter values. Example usage:

                             >>> dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error)
                             >>> results = dataset.fit()
                             >>> transit_model, error68_up, error68_down = results.lc.evaluate('TESS', return_err=True)

        Or:

                             >>> dataset = juliet.load(priors=priors, t_rv = times, y_rv = fluxes, yerr_rv = fluxes_error)
                             >>> results = dataset.fit()
                             >>> rv_model, error68_up, error68_down = results.rv.evaluate('FEROS', return_err=True)

        :param instrument: (optional, string)
        Instrument the user wants to evaluate the model on. It is expected to be given for non-global models, not necessary for global models.

        :param parameter_values: (optional, dict)
        Dictionary containing samples of the posterior distribution or, more generally, parameter valuesin it. Each key is a parameter name (e.g. 'p_p1',
        'q1_TESS', etc.), and inside each of those keys an array of N samples is expected (i.e., parameter_values['p_p1'] is an array of length N). The
        indexes have to be consistent between different parameters.

        :param all_samples: (optional, boolean)
        If True, all posterior samples will be used to evaluate the model. Default is False.

        :param nsamples: (optional, int)
        Number of posterior samples to be used to evaluate the model. Default is 1000 (note each call to this function will sample `nsamples` different samples
        from the posterior, so no two calls are exactly the same).

        :param return_samples: (optional, boolean)
        Boolean indicating whether the user wants the posterior model samples (i.e., the models evaluated in each of the posterior sample draws) to be returned. Default
        is False.

        :param t: (optional, numpy array)
        Array with the times at which the model wants to be evaluated.

        :param GPRegressors: (optional, numpy array)
        Array containing the GP Regressors onto which to evaluate the models. Dimensions must be consistent with input `t`. If model is global, this needs to be a dictionary.

        :param LMRegressors: (optional, numpy array or dictionary)
        If the model is not global, this is an array containing the Linear Regressors onto which to evaluate the model for the input instrument.
        Dimensions must be consistent with input `t`. If model is global, this needs to be a dictionary.

        :param return_err: (optional, boolean)
        If True, this returns the credibility interval on the evaluated model. Default credibility interval is 68%.

        :param alpha: (optional, double)
        Credibility interval for return_err. Default is 0.68, i.e., the 68% credibility interval.

        :param return_components: (optional, boolean)
        If True, each component of the model is returned (i.e., the Gaussian Process component, the Linear Model component, etc.).

        :param evaluate_transit: (optional, boolean)
        If True, the function evaluates only the transit model and not the Gaussian Process or Linear Model components.

        :returns: By default, the function returns the median model as evaluated with the posterior samples. Depending on the options chosen by the user, this can return up to 5 elements (in that order): `model_samples`, `median_model`, `upper_CI`, `lower_CI` and `components`. The first is an array with all the model samples as evaluated from the posterior. The second is the median model. The third and fourth are the uppper and lower Credibility Intervals, and the latter is a dictionary with the model components.

        """
        if evaluate_transit:
            if self.modeltype != "lc":
                raise Exception(
                    "Trying to evaluate a transit (evaluate_transit = True) in a non-lightcurve model is not allowed."
                )

            # Save LM and GP booleans, turn them off:
            true_lm_boolean = self.lm_boolean[instrument]
            self.lm_boolean[instrument] = False
            if self.global_model:
                true_gp_boolean = self.dictionary["global_model"]["GPDetrend"]
                self.dictionary["global_model"]["GPDetrend"] = False
            else:
                true_gp_boolean = self.dictionary[instrument]["GPDetrend"]
                self.dictionary[instrument]["GPDetrend"] = False

        # If no instrument is given, assume user wants a global model evaluation:
        if instrument is None:
            if not self.global_model:
                raise Exception(
                    "Input error: an instrument has to be defined for non-global models in order to evaluate the model."
                )

        if self.modeltype == "lc":
            nresampling = self.dictionary[instrument].get("nresampling")
            etresampling = self.dictionary[instrument].get("exptimeresampling")

            if not self.dictionary[instrument]["TransitFitCatwoman"]:
                if self.dictionary[instrument]["TransitFit"]:
                    (
                        self.model[instrument]["params"],
                        [self.model[instrument]["m"], _],
                    ) = init_batman(
                        self.times[instrument],
                        self.dictionary[instrument]["ldlaw"],
                        nresampling=nresampling,
                        etresampling=etresampling,
                    )
                elif self.dictionary[instrument]["EclipseFit"]:
                    (
                        self.model[instrument]["params"],
                        [_, self.model[instrument]["m"]],
                    ) = init_batman(
                        self.times[instrument],
                        self.dictionary[instrument]["ldlaw"],
                        nresampling=nresampling,
                        etresampling=etresampling,
                    )

                elif self.dictionary[instrument]["TranEclFit"]:
                    self.model[instrument]["params"], self.model[instrument]["m"] = (
                        init_batman(
                            self.times[instrument],
                            self.dictionary[instrument]["ldlaw"],
                            nresampling=nresampling,
                            etresampling=etresampling,
                        )
                    )
            else:
                self.model[instrument]["params"], self.model[instrument]["m"] = (
                    init_catwoman(
                        self.times[instrument],
                        self.dictionary[instrument]["ldlaw"],
                        nresampling=nresampling,
                        etresampling=etresampling,
                    )
                )

        # Save the original inames in the case of non-global models, and set self.inames to the input model. This is because if the model
        # is not global, in general we don't care about generating the models for the other instruments (and in the lightcurve and RV evaluation part,
        # self.inames is used to iterate through the instruments one wants to evaluate the model):

        if not self.global_model:
            original_inames = copy.deepcopy(self.inames)
            self.inames = [instrument]
            instruments = self.dictionary.keys()
        else:
            instruments = self.inames
        # Check if user gave input parameter_values dictionary. If that's the case, generate again the
        # full lightcurve/rv model:
        if parameter_values is not None:
            # If return_components, generate the components dictionary:
            if return_components:
                self.log_like_calc = False
                components = {}
            # Now, consider two possible cases. If the user is giving a parameter_values where the dictionary contains *arrays* of values
            # in it, then iterate through all the values in order to calculate the median model. If the dictionary contains only individual
            # values, evaluate the model only at those values:
            parameters = list(self.priors.keys())
            input_parameters = list(parameter_values.keys())
            if type(parameter_values[input_parameters[0]]) is np.ndarray:
                # To generate a median model first generate an output_model_samples array that will save the model at each evaluation. This will
                # save nsamples samples of the posterior model. If all_samples = True, all samples from the posterior are used for the evaluated model
                # (this is slower, but user might not care). First create idx_samples, which will save the indexes of the samples:
                nsampled = len(parameter_values[input_parameters[0]])
                if all_samples:
                    nsamples = nsampled
                    idx_samples = np.arange(nsamples)
                else:
                    idx_samples = np.random.choice(
                        np.arange(nsampled), np.min([nsamples, nsampled]), replace=False
                    )
                    idx_samples = idx_samples[np.argsort(idx_samples)]

                # Create the output_model arrays: these will save on each iteration the full model (lc/rv + GP, output_model_samples),
                # the GP-only model (GP, output_modelGP_samples) and the lc/rv-only model (lc/rv, output_modelDET_samples) --- the latter ones
                # will make sense only if there is a GP model. If not, it will be a zero-array throughout the evaluation process:
                if t is None:
                    # If user did not give input times, then output samples follow the times on which the model was fitted:
                    if self.global_model:
                        output_model_samples = np.zeros(
                            [nsamples, self.ndatapoints_all_instruments]
                        )
                    else:
                        output_model_samples = np.zeros(
                            [nsamples, self.ndatapoints_per_instrument[instrument]]
                        )
                else:
                    # If user gave input times (usually extrapolating from the times the model was fitted on), then
                    # save the number of points in this input array:
                    nt = len(t)
                    # And modify the length of the output samples, which will now be a matrix with dimensions (number of samples, input times):
                    output_model_samples = np.zeros([nsamples, nt])
                    if self.global_model:
                        # If model is global, it means there is an underlying global noise model, so we have to evaluate the model in *all* the instruments
                        # because the GP component is only extractable once we have the full residuals. Because of this, we generate dictionaries that save
                        # the original number of datapoints for each instrument and the original times of each instrument. This is useful because later we
                        # will switch back and forth from the original times (to evaluate the model and get the residuals) to the input times (to generate
                        # predictions):
                        nt_original, original_instrument_times = {}, {}
                        for ginstrument in instruments:
                            nt_original[ginstrument] = len(self.times[ginstrument])
                            original_instrument_times[ginstrument] = copy.deepcopy(
                                self.times[ginstrument]
                            )
                    else:
                        # If model is not global, we don't care about generating the model for all the instruments --- we do it only for the instrument
                        # of interest. In this case, the nt_original and original_instrument_times are not dictionaries but "simple" arrays saving the
                        # number of datapoints for that instrument and the times for that instrument.
                        nt_original = len(self.times[instrument])
                        original_instrument_times = copy.deepcopy(
                            self.times[instrument]
                        )
                    if self.modeltype == "lc":
                        # If we are trying to evaluate a lightcurve mode then, again what we do will depend depending if this is a global model or not. In both,
                        # the idea is to save the lightcurve generating objects both using the input times and the original times:
                        if self.global_model:
                            # If global model, then iterate through all the instruments of the fit. If the TransitFit or TransitFitCatwoman is true,
                            # then generate the model-generating objects for those instruments using both the input times and the model-fit times. Save
                            # those in dictionaries:
                            for ginstrument in instruments:
                                if (
                                    self.dictionary[ginstrument]["TransitFit"]
                                    or self.dictionary[ginstrument][
                                        "TransitFitCatwoman"
                                    ]
                                    or self.dictionary[ginstrument]["EclipseFit"]
                                    or self.dictionary[ginstrument]["TranEclFit"]
                                ):
                                    nresampling = self.dictionary[ginstrument].get(
                                        "nresampling"
                                    )
                                    etresampling = self.dictionary[ginstrument].get(
                                        "exptimeresampling"
                                    )
                                    supersample_params, supersample_m = {}, {}
                                    sample_params, sample_m = {}, {}

                                    if not self.dictionary[ginstrument][
                                        "TransitFitCatwoman"
                                    ]:
                                        if self.dictionary[ginstrument]["TransitFit"]:
                                            (
                                                supersample_params[ginstrument],
                                                [supersample_m[ginstrument], _],
                                            ) = init_batman(
                                                t,
                                                self.dictionary[ginstrument]["ldlaw"],
                                                nresampling=nresampling,
                                                etresampling=etresampling,
                                            )
                                            (
                                                sample_params[ginstrument],
                                                [sample_m[ginstrument], _],
                                            ) = init_batman(
                                                self.times[ginstrument],
                                                self.dictionary[ginstrument]["ldlaw"],
                                                nresampling=nresampling,
                                                etresampling=etresampling,
                                            )

                                        elif self.dictionary[ginstrument]["EclipseFit"]:
                                            (
                                                supersample_params[ginstrument],
                                                [_, supersample_m[ginstrument]],
                                            ) = init_batman(
                                                t,
                                                self.dictionary[ginstrument]["ldlaw"],
                                                nresampling=nresampling,
                                                etresampling=etresampling,
                                            )
                                            (
                                                sample_params[ginstrument],
                                                [_, sample_m[ginstrument]],
                                            ) = init_batman(
                                                self.times[ginstrument],
                                                self.dictionary[ginstrument]["ldlaw"],
                                                nresampling=nresampling,
                                                etresampling=etresampling,
                                            )

                                        elif self.dictionary[ginstrument]["TranEclFit"]:
                                            (
                                                supersample_params[ginstrument],
                                                supersample_m[ginstrument],
                                            ) = init_batman(
                                                t,
                                                self.dictionary[ginstrument]["ldlaw"],
                                                nresampling=nresampling,
                                                etresampling=etresampling,
                                            )
                                            (
                                                sample_params[ginstrument],
                                                sample_m[ginstrument],
                                            ) = init_batman(
                                                self.times[ginstrument],
                                                self.dictionary[ginstrument]["ldlaw"],
                                                nresampling=nresampling,
                                                etresampling=etresampling,
                                            )

                                    else:
                                        (
                                            supersample_params[ginstrument],
                                            supersample_m[ginstrument],
                                        ) = init_catwoman(
                                            t,
                                            self.dictionary[ginstrument]["ldlaw"],
                                            nresampling=nresampling,
                                            etresampling=etresampling,
                                        )
                                        (
                                            sample_params[ginstrument],
                                            sample_m[ginstrument],
                                        ) = init_catwoman(
                                            self.times[ginstrument],
                                            self.dictionary[ginstrument]["ldlaw"],
                                            nresampling=nresampling,
                                            etresampling=etresampling,
                                        )
                        else:
                            # If model is not global, the variables saved are not dictionaries but simply the objects, as we are just going to evaluate the
                            # model for one dataset (the one of the input instrument):

                            if (
                                self.dictionary[instrument]["TransitFit"]
                                or self.dictionary[instrument]["TransitFitCatwoman"]
                                or self.dictionary[instrument]["EclipseFit"]
                                or self.dictionary[instrument]["TranEclFit"]
                            ):
                                nresampling = self.dictionary[instrument].get(
                                    "nresampling"
                                )
                                etresampling = self.dictionary[instrument].get(
                                    "exptimeresampling"
                                )

                                if not self.dictionary[instrument][
                                    "TransitFitCatwoman"
                                ]:
                                    if self.dictionary[instrument]["TransitFit"]:
                                        supersample_params, [supersample_m, _] = (
                                            init_batman(
                                                t,
                                                self.dictionary[instrument]["ldlaw"],
                                                nresampling=nresampling,
                                                etresampling=etresampling,
                                            )
                                        )

                                        sample_params, [sample_m, _] = init_batman(
                                            self.times[instrument],
                                            self.dictionary[instrument]["ldlaw"],
                                            nresampling=nresampling,
                                            etresampling=etresampling,
                                        )

                                    elif self.dictionary[instrument]["EclipseFit"]:
                                        supersample_params, [_, supersample_m] = (
                                            init_batman(
                                                t,
                                                self.dictionary[instrument]["ldlaw"],
                                                nresampling=nresampling,
                                                etresampling=etresampling,
                                            )
                                        )

                                        sample_params, [_, sample_m] = init_batman(
                                            self.times[instrument],
                                            self.dictionary[instrument]["ldlaw"],
                                            nresampling=nresampling,
                                            etresampling=etresampling,
                                        )

                                    elif self.dictionary[instrument]["TranEclFit"]:
                                        supersample_params, supersample_m = init_batman(
                                            t,
                                            self.dictionary[instrument]["ldlaw"],
                                            nresampling=nresampling,
                                            etresampling=etresampling,
                                        )

                                        sample_params, sample_m = init_batman(
                                            self.times[instrument],
                                            self.dictionary[instrument]["ldlaw"],
                                            nresampling=nresampling,
                                            etresampling=etresampling,
                                        )

                                else:
                                    supersample_params, supersample_m = init_catwoman(
                                        t,
                                        self.dictionary[instrument]["ldlaw"],
                                        nresampling=nresampling,
                                        etresampling=etresampling,
                                    )
                                    sample_params, sample_m = init_catwoman(
                                        self.times[instrument],
                                        self.dictionary[instrument]["ldlaw"],
                                        nresampling=nresampling,
                                        etresampling=etresampling,
                                    )

                    else:
                        # If we are trying to evaluate radial-velocities, we don't need to generate objects because radvel receives the times as inputs
                        # on each call. In this case then we save the original times (self.t has *all* the times of all the instruments) and instrument
                        # indexes (remember self.t[self.instrument_indexes[yourinstrument]] returns the times of yourinstrument):
                        original_t = copy.deepcopy(self.t)
                        if self.global_model:
                            # If global model, copy all the possible instrument indexes to the original_instrument_indexes:
                            original_instrument_indexes = copy.deepcopy(
                                self.instrument_indexes
                            )
                        else:
                            # If not global, assume indexes for selected instrument are all the user-inputted t's. Also, save only the instrument
                            # indexes corresponding to the instrument of interest. The others don't matter so we don't save them:
                            original_instrument_index = self.instrument_indexes[
                                instrument
                            ]
                        dummy_indexes = np.arange(len(t))
                # Fill the components dictionary in case return_components is true; use the output_model_samples for the size of each component array.
                # If global model, and the model being evaluated is a lightcurve, remember to give back one planet component per instrument because
                # each instrument might have different limb-darkening laws. To this, end, in that case, the components['p'+str(i)] dictionary is, itself,
                # a dictionary. Same thing for the components['transit'] dictionary:
                if return_components:
                    for i in self.numbering:
                        if self.global_model and self.modeltype == "lc":
                            components["p" + str(i)] = {}
                            for ginstrument in instruments:
                                components["p" + str(i)][ginstrument] = np.zeros(
                                    output_model_samples.shape
                                )
                        else:
                            components["p" + str(i)] = np.zeros(
                                output_model_samples.shape
                            )
                    if self.global_model:
                        components["lm"] = {}
                        for ginstrument in instruments:
                            components["lm"][ginstrument] = np.zeros(
                                output_model_samples.shape
                            )
                    else:
                        components["lm"] = np.zeros(output_model_samples.shape)
                    if self.modeltype == "lc":
                        if self.global_model:
                            components["transit"] = {}
                            for ginstrument in instruments:
                                components["transit"][ginstrument] = np.zeros(
                                    output_model_samples.shape
                                )
                        else:
                            components["transit"] = np.zeros(output_model_samples.shape)
                    else:
                        components["keplerian"] = np.zeros(output_model_samples.shape)
                        components["trend"] = np.zeros(output_model_samples.shape)

                        if self.global_model:
                            components["mu"] = {}
                            for ginstrument in instruments:
                                components["mu"][ginstrument] = np.zeros(
                                    output_model_samples.shape[0]
                                )
                        else:
                            components["mu"] = np.zeros(output_model_samples.shape[0])

                # IF GP detrend, there is an underlying GP being applied. Generate arrays that will save the GP and deterministic component:
                if self.global_model:
                    if self.dictionary["global_model"]["GPDetrend"]:
                        output_modelGP_samples = copy.deepcopy(output_model_samples)
                        output_modelDET_samples = copy.deepcopy(output_model_samples)
                else:
                    if self.dictionary[instrument]["GPDetrend"]:
                        output_modelGP_samples = copy.deepcopy(output_model_samples)
                        output_modelDET_samples = copy.deepcopy(output_model_samples)

                # Create dictionary that saves the current parameter_values to evaluate:
                current_parameter_values = dict.fromkeys(parameters)

                # Having defined everything, we now finally start evaluation the model. First go through all parameters in the prior; fix the ones
                # which are fixed:
                for parameter in parameters:
                    if self.priors[parameter]["distribution"] == "fixed":
                        current_parameter_values[parameter] = self.priors[parameter][
                            "hyperparameters"
                        ]

                # If extrapolating the model, save the current GPregressors and current linear
                # regressors. Save the input GPRegressors to the self.dictionary. Note this is done because
                # we won't be evaluating the likelihood on each iteration, so we don't need the original GP Regressors,
                # but only the input ones ad the residuals are generated deterministically. These residuals are passed
                # to the GP objet to generate samples from the GP. This latter is not true for the linear model, because it
                # is a determinisitc model an needs to be evaluated on each iteration on both the input regressors of the
                # fit (to generate the residuals) and on the input regressors to this function (to generate predictions):
                if t is not None:
                    if self.global_model:
                        original_lm_arguments = copy.deepcopy(self.lm_arguments)
                        if self.dictionary["global_model"]["GPDetrend"]:
                            self.original_GPregressors = copy.deepcopy(
                                self.dictionary["global_model"]["noise_model"].X
                            )
                            self.dictionary["global_model"][
                                "noise_model"
                            ].X = GPregressors
                            if GPregressors is None:
                                raise Exception(
                                    "\t Gobal model has a GP, and requires a GPregressors to be inputted to be evaluated."
                                )
                    else:
                        if self.dictionary[instrument]["GPDetrend"]:
                            self.dictionary[instrument]["noise_model"].X = GPregressors
                            if GPregressors is None:
                                raise Exception(
                                    "\t Model for instrument "
                                    + instrument
                                    + " has a GP, and requires a GPregressors to be inputted to be evaluated."
                                )
                        if self.lm_boolean[instrument]:
                            original_lm_arguments = copy.deepcopy(
                                self.lm_arguments[instrument]
                            )

                # Now iterate through all samples:
                counter = 0
                for i in idx_samples:
                    # Get parameters for the i-th sample:
                    for parameter in input_parameters:
                        # Populate the current parameter_values
                        current_parameter_values[parameter] = parameter_values[
                            parameter
                        ][i]

                    # Evaluate rv/lightcurve at the current parameter values, calculate residuals, save them:
                    if self.modeltype == "lc":
                        self.generate_lc_model(
                            current_parameter_values, evaluate_lc=True
                        )
                    else:
                        self.generate_rv_model(
                            current_parameter_values, evaluate_global_errors=True
                        )

                    # Save residuals (and global errors, in the case of global models):
                    if self.global_model:
                        self.residuals = self.y - self.model["global"]
                        self.variances = self.model["global_variances"]
                    else:
                        self.residuals = (
                            self.data[instrument]
                            - self.model[instrument]["deterministic"]
                        )

                    # If extrapolating (t is not None), evaluate the extrapolated model with a lightcurve/rv model
                    # considering the input times and not the current dataset times:
                    if t is not None:
                        if self.modeltype == "lc":
                            if self.global_model:
                                # If global model, set all super-sample objects to evaluate at the input times:
                                for ginstrument in instruments:
                                    if (
                                        self.dictionary[ginstrument]["TransitFit"]
                                        or self.dictionary[ginstrument][
                                            "TransitFitCatwoman"
                                        ]
                                        or self.dictionary[ginstrument]["EclipseFit"]
                                        or self.dictionary[ginstrument]["TranEclFit"]
                                    ):
                                        (
                                            self.model[ginstrument]["params"],
                                            self.model[ginstrument]["m"],
                                        ) = (
                                            supersample_params[ginstrument],
                                            supersample_m[ginstrument],
                                        )

                                    if self.lm_boolean[ginstrument]:
                                        self.lm_arguments[ginstrument] = LMregressors[
                                            ginstrument
                                        ]
                                    self.model[ginstrument]["ones"] = np.ones(nt)
                                    self.ndatapoints_per_instrument[ginstrument] = nt
                                    self.instrument_indexes[ginstrument] = dummy_indexes
                                original_inames = copy.deepcopy(self.inames)
                                self.inames = [instrument]
                                self.generate_lc_model(
                                    current_parameter_values,
                                    evaluate_global_errors=False,
                                    evaluate_lc=True,
                                )
                                self.inames = original_inames
                            else:
                                # If not, set them only for the instrument of interest:

                                if (
                                    self.dictionary[instrument]["TransitFit"]
                                    or self.dictionary[instrument]["TransitFitCatwoman"]
                                    or self.dictionary[instrument]["EclipseFit"]
                                    or self.dictionary[instrument]["TranEclFit"]
                                ):
                                    (
                                        self.model[instrument]["params"],
                                        self.model[instrument]["m"],
                                    ) = supersample_params, supersample_m

                                if self.lm_boolean[instrument]:
                                    self.lm_arguments[instrument] = LMregressors
                                self.model[instrument]["ones"] = np.ones(nt)
                                self.ndatapoints_per_instrument[instrument] = nt
                                # Generate lightcurve model:

                                self.generate_lc_model(
                                    current_parameter_values,
                                    evaluate_global_errors=False,
                                    evaluate_lc=True,
                                )

                        else:
                            # As with the lc case, RV model set-up depends on whether the model is global or not:
                            self.t = t
                            if self.global_model:
                                # If global, in the model evaluation part (generate_rv_model function), the model for each instrument is evaluated at
                                # certain indexes self.instrument_indexes[instrument]. We here decide that on each instrument we will evaluate the model
                                # at all the input times t (this is what the dummy_index variable does), so we fill up this dictionary with that.
                                self.model["global"] = np.ones(len(t))
                                for ginstrument in instruments:
                                    if self.lm_boolean[ginstrument]:
                                        self.lm_arguments[ginstrument] = LMregressors[
                                            ginstrument
                                        ]
                                    self.times[ginstrument] = t
                                    self.instrument_indexes[ginstrument] = dummy_indexes
                                # Generate RV model only for the instrument under consideration:
                                original_inames = copy.deepcopy(self.inames)
                                self.inames = [instrument]
                                self.generate_rv_model(
                                    current_parameter_values,
                                    evaluate_global_errors=False,
                                )
                                self.inames = original_inames
                            else:
                                self.times[instrument] = t
                                self.instrument_indexes[instrument] = dummy_indexes
                                if self.lm_boolean[instrument]:
                                    self.lm_arguments[instrument] = LMregressors
                                # Generate RV model:
                                self.generate_rv_model(
                                    current_parameter_values,
                                    evaluate_global_errors=False,
                                )

                    if self.global_model:
                        if self.dictionary["global_model"]["GPDetrend"]:
                            (
                                output_modelDET_samples[counter, :],
                                output_modelGP_samples[counter, :],
                                output_model_samples[counter, :],
                            ) = self.get_GP_plus_deterministic_model(
                                current_parameter_values, instrument=instrument
                            )
                        else:
                            output_model_samples[counter, :] = (
                                self.get_GP_plus_deterministic_model(
                                    current_parameter_values, instrument=instrument
                                )
                            )
                    else:
                        if self.dictionary[instrument]["GPDetrend"]:
                            (
                                output_modelDET_samples[counter, :],
                                output_modelGP_samples[counter, :],
                                output_model_samples[counter, :],
                            ) = self.get_GP_plus_deterministic_model(
                                current_parameter_values, instrument=instrument
                            )
                        else:
                            output_model_samples[counter, :] = (
                                self.get_GP_plus_deterministic_model(
                                    current_parameter_values, instrument=instrument
                                )
                            )

                    # Now, if user wants component back, again all depends if global model is on or not but only for the lightcurves
                    # (which depend on limb-darkening). For the RVs it doesn't matter except for 'mu' (the systemic velocity), which
                    # for global models is actually a dictionary:
                    if return_components:
                        if self.modeltype == "lc":
                            if self.global_model:
                                # If it is, then the components['p'+str(i)] dictionary will have to be a dictionary on itself,
                                # such that we return the global transit model for each of the instruments. Same thing for the
                                # components['transit'] dictionary.
                                for ginstrument in instruments:
                                    transit = 0.0
                                    for i in self.numbering:
                                        components["p" + str(i)][ginstrument][
                                            counter, :
                                        ] = self.model[ginstrument]["p" + str(i)]
                                        transit += (
                                            components["p" + str(i)][ginstrument][
                                                counter, :
                                            ]
                                            - 1.0
                                        )
                                    components["transit"][ginstrument][counter, :] = (
                                        1.0 + transit
                                    )
                            else:
                                transit = 0.0
                                for i in self.numbering:
                                    components["p" + str(i)][counter, :] = self.model[
                                        instrument
                                    ]["p" + str(i)]
                                    transit += (
                                        components["p" + str(i)][counter, :] - 1.0
                                    )
                                components["transit"][counter, :] = 1.0 + transit
                        else:
                            for i in self.numbering:
                                components["p" + str(i)][counter, :] = self.model[
                                    "p" + str(i)
                                ]

                            components["trend"][counter, :] = (
                                self.model["Keplerian+Trend"] - self.model["Keplerian"]
                            )
                            components["keplerian"][counter, :] = self.model[
                                "Keplerian"
                            ]

                            if self.global_model:
                                for ginstrument in instruments:
                                    components["mu"][ginstrument][counter] = (
                                        current_parameter_values[f"mu_{ginstrument}"]
                                    )
                            else:
                                components["mu"][counter] = current_parameter_values[
                                    f"mu_{instrument}"
                                ]
                        if self.global_model:
                            for ginstrument in instruments:
                                if self.lm_boolean[ginstrument]:
                                    components["lm"][ginstrument][counter, :] = (
                                        self.model[ginstrument]["LM"]
                                    )
                        else:
                            if self.lm_boolean[instrument]:
                                components["lm"][counter, :] = self.model[instrument][
                                    "LM"
                                ]

                    # Rollback in case t is not None:
                    if t is not None:
                        if self.global_model:
                            self.instrument_indexes = copy.deepcopy(
                                original_instrument_indexes
                            )

                            for ginstrument in instruments:
                                self.times[ginstrument] = original_instrument_times[
                                    ginstrument
                                ]
                                if self.modeltype == "lc":
                                    if (
                                        self.dictionary[ginstrument]["TransitFit"]
                                        or self.dictionary[ginstrument][
                                            "TransitFitCatwoman"
                                        ]
                                        or self.dictionary[ginstrument]["EclipseFit"]
                                        or self.dictionary[ginstrument]["TranEclFit"]
                                    ):
                                        (
                                            self.model[ginstrument]["params"],
                                            self.model[ginstrument]["m"],
                                        ) = (
                                            sample_params[ginstrument],
                                            sample_m[ginstrument],
                                        )

                                    if self.lm_boolean[ginstrument]:
                                        self.lm_arguments[ginstrument] = (
                                            original_lm_arguments[ginstrument]
                                        )
                                    self.model[ginstrument]["ones"] = np.ones(
                                        nt_original[ginstrument]
                                    )
                                else:
                                    self.t = original_t
                                    self.model["global"] = np.ones(len(original_t))
                                self.ndatapoints_per_instrument[ginstrument] = (
                                    nt_original[ginstrument]
                                )
                        else:
                            self.times[instrument] = original_instrument_times
                            if self.modeltype == "lc":
                                if (
                                    self.dictionary[instrument]["TransitFit"]
                                    or self.dictionary[instrument]["EclipseFit"]
                                    or self.dictionary[instrument]["TranEclFit"]
                                ):
                                    (
                                        self.model[instrument]["params"],
                                        self.model[instrument]["m"],
                                    ) = sample_params, sample_m

                                if self.lm_boolean[instrument]:
                                    self.lm_arguments[instrument] = (
                                        original_lm_arguments
                                    )
                                self.model[instrument]["ones"] = np.ones(nt_original)
                            else:
                                self.t = original_t

                                self.instrument_indexes[instrument] = (
                                    original_instrument_index
                                )

                            self.ndatapoints_per_instrument[instrument] = nt_original

                    counter += 1
                # If return_error is on, return upper and lower sigma (alpha x 100% CI) of the model(s):
                if return_err:
                    m_output_model, u_output_model, l_output_model = (
                        np.zeros(output_model_samples.shape[1]),
                        np.zeros(output_model_samples.shape[1]),
                        np.zeros(output_model_samples.shape[1]),
                    )
                    if self.global_model:
                        if self.dictionary["global_model"]["GPDetrend"]:
                            mDET_output_model, uDET_output_model, lDET_output_model = (
                                np.copy(m_output_model),
                                np.copy(u_output_model),
                                np.copy(l_output_model),
                            )

                            mGP_output_model, uGP_output_model, lGP_output_model = (
                                np.copy(m_output_model),
                                np.copy(u_output_model),
                                np.copy(l_output_model),
                            )
                        for i in range(output_model_samples.shape[1]):
                            m_output_model[i], u_output_model[i], l_output_model[i] = (
                                get_quantiles(output_model_samples[:, i], alpha=alpha)
                            )
                            if self.dictionary["global_model"]["GPDetrend"]:
                                (
                                    mDET_output_model[i],
                                    uDET_output_model[i],
                                    lDET_output_model[i],
                                ) = get_quantiles(
                                    output_modelDET_samples[:, i], alpha=alpha
                                )
                                (
                                    mGP_output_model[i],
                                    uGP_output_model[i],
                                    lGP_output_model[i],
                                ) = get_quantiles(
                                    output_modelGP_samples[:, i], alpha=alpha
                                )
                        if self.dictionary["global_model"]["GPDetrend"]:
                            self.model["deterministic"], self.model["GP"] = (
                                mDET_output_model,
                                mGP_output_model,
                            )
                            (
                                self.model["deterministic_uerror"],
                                self.model["GP_uerror"],
                            ) = uDET_output_model, uGP_output_model
                            (
                                self.model["deterministic_lerror"],
                                self.model["GP_lerror"],
                            ) = lDET_output_model, lGP_output_model
                    else:
                        if self.dictionary[instrument]["GPDetrend"]:
                            mDET_output_model, uDET_output_model, lDET_output_model = (
                                np.copy(m_output_model),
                                np.copy(u_output_model),
                                np.copy(l_output_model),
                            )

                            mGP_output_model, uGP_output_model, lGP_output_model = (
                                np.copy(m_output_model),
                                np.copy(u_output_model),
                                np.copy(l_output_model),
                            )
                        for i in range(output_model_samples.shape[1]):
                            m_output_model[i], u_output_model[i], l_output_model[i] = (
                                get_quantiles(output_model_samples[:, i], alpha=alpha)
                            )

                            if self.dictionary[instrument]["GPDetrend"]:
                                (
                                    mDET_output_model[i],
                                    uDET_output_model[i],
                                    lDET_output_model[i],
                                ) = get_quantiles(
                                    output_modelDET_samples[:, i], alpha=alpha
                                )
                                (
                                    mGP_output_model[i],
                                    uGP_output_model[i],
                                    lGP_output_model[i],
                                ) = get_quantiles(
                                    output_modelGP_samples[:, i], alpha=alpha
                                )

                        if self.dictionary[instrument]["GPDetrend"]:
                            (
                                self.model[instrument]["deterministic"],
                                self.model[instrument]["GP"],
                            ) = mDET_output_model, mGP_output_model
                            (
                                self.model[instrument]["deterministic_uerror"],
                                self.model[instrument]["GP_uerror"],
                            ) = uDET_output_model, uGP_output_model
                            (
                                self.model[instrument]["deterministic_lerror"],
                                self.model[instrument]["GP_lerror"],
                            ) = lDET_output_model, lGP_output_model

                else:
                    output_model = np.nanmedian(output_model_samples, axis=0)
                    if self.global_model:
                        if self.dictionary["global_model"]["GPDetrend"]:
                            self.model["deterministic"], self.model["GP"] = (
                                np.nanmedian(output_modelDET_samples, axis=0),
                                np.nanmedian(output_modelGP_samples, axis=0),
                            )
                    else:
                        if self.dictionary[instrument]["GPDetrend"]:
                            (
                                self.model[instrument]["deterministic"],
                                self.model[instrument]["GP"],
                            ) = (
                                np.nanmedian(output_modelDET_samples, axis=0),
                                np.nanmedian(output_modelGP_samples, axis=0),
                            )

                # If return_components is true, generate the median models for each part of the full model:
                if return_components:
                    if self.modeltype == "lc":
                        if self.global_model:
                            for k in components.keys():
                                for ginstrument in instruments:
                                    components[k][ginstrument] = np.median(
                                        components[k][ginstrument], axis=0
                                    )

                        else:
                            for k in components.keys():
                                components[k] = np.median(components[k], axis=0)
                    else:
                        for i in self.numbering:
                            components["p" + str(i)] = np.median(
                                components["p" + str(i)], axis=0
                            )

                        components["trend"] = np.median(components["trend"], axis=0)
                        components["keplerian"] = np.median(
                            components["keplerian"], axis=0
                        )

                        if self.global_model:
                            for ginstrument in instruments:
                                components["mu"][ginstrument] = np.median(
                                    components["mu"][ginstrument]
                                )

                        else:
                            components["mu"] = np.median(components["mu"], axis=0)
            else:
                if self.modeltype == "lc":
                    self.generate_lc_model(parameter_values, evaluate_lc=True)
                else:
                    self.generate_rv_model(parameter_values)

                if self.global_model:
                    self.residuals = self.y - self.model["global"]
                    self.variances = self.model["global_variances"]
                    if self.dictionary["global_model"]["GPDetrend"]:
                        self.model["deterministic"], self.model["GP"], output_model = (
                            self.get_GP_plus_deterministic_model(parameter_values)
                        )
                    else:
                        output_model = self.get_GP_plus_deterministic_model(
                            parameter_values
                        )
                    if return_components:
                        if self.modeltype == "lc":
                            for ginstrument in instruments:
                                transit = 0.0
                                for i in self.numbering:
                                    components["p" + str(i)][ginstrument] = self.model[
                                        ginstrument
                                    ]["p" + str(i)]
                                    transit += (
                                        components["p" + str(i)][ginstrument] - 1.0
                                    )
                                components["transit"][ginstrument] = 1.0 + transit
                        else:
                            for i in self.numbering:
                                components["p" + str(i)] = self.model["p" + str(i)]
                            components["trend"] = (
                                self.model["Keplerian+Trend"] - self.model["Keplerian"]
                            )
                            components["keplerian"] = self.model["Keplerian"]
                            for ginstrument in instruments:
                                components["mu"][ginstrument] = parameter_values[
                                    f"mu_{instrument}"
                                ]
                        for ginstrument in instruments:
                            if self.lm_boolean[ginstrument]:
                                components["lm"][ginstrument] = self.model[ginstrument][
                                    "LM"
                                ]
                else:
                    self.residuals = (
                        self.data[instrument] - self.model[instrument]["deterministic"]
                    )
                    if self.dictionary[instrument]["GPDetrend"]:
                        self.model["deterministic"], self.model["GP"], output_model = (
                            self.get_GP_plus_deterministic_model(
                                parameter_values, instrument=instrument
                            )
                        )
                    else:
                        output_model = self.get_GP_plus_deterministic_model(
                            parameter_values, instrument=instrument
                        )
                    if return_components:
                        if self.modeltype == "lc":
                            transit = 0.0
                            for i in self.numbering:
                                components["p" + str(i)] = self.model[instrument][
                                    "p" + str(i)
                                ]
                                transit += components["p" + str(i)] - 1.0
                            components["transit"] = 1.0 + transit
                        else:
                            for i in self.numbering:
                                components["p" + str(i)] = self.model["p" + str(i)]
                            components["trend"] = (
                                self.model["Keplerian+Trend"] - self.model["Keplerian"]
                            )
                            components["keplerian"] = self.model["Keplerian"]
                            components["mu"] = parameter_values[f"mu_{instrument}"]
                        if self.lm_boolean[instrument]:
                            components["lm"] = self.model[instrument]["LM"]
        else:
            x = self.evaluate_model(
                instrument=instrument,
                parameter_values=self.posteriors,
                all_samples=all_samples,
                nsamples=nsamples,
                return_samples=return_samples,
                t=t,
                GPregressors=GPregressors,
                LMregressors=LMregressors,
                return_err=return_err,
                return_components=return_components,
                alpha=alpha,
                evaluate_transit=evaluate_transit,
            )

            if return_samples:
                if return_err:
                    if return_components:
                        (
                            output_model_samples,
                            m_output_model,
                            u_output_model,
                            l_output_model,
                            components,
                        ) = x
                    else:
                        (
                            output_model_samples,
                            m_output_model,
                            u_output_model,
                            l_output_model,
                        ) = x
                else:
                    if return_components:
                        output_model_samples, output_model, components = x
                    else:
                        output_model_samples, output_model = x
            else:
                if return_err:
                    if return_components:
                        m_output_model, u_output_model, l_output_model, components = x
                    else:
                        m_output_model, u_output_model, l_output_model = x
                else:
                    if return_components:
                        output_model, components = x
                    else:
                        output_model = x

        if not self.global_model:
            # Return original inames back in case of non-global models:
            self.inames = original_inames

        else:
            if t is not None and self.dictionary["global_model"]["GPDetrend"]:
                # Return GP regressors back:
                self.dictionary["global_model"][
                    "noise_model"
                ].X = self.original_GPregressors

        if evaluate_transit:
            # Turn LM and GPs back on:
            self.lm_boolean[instrument] = true_lm_boolean
            if self.global_model:
                self.dictionary["global_model"]["GPDetrend"] = true_gp_boolean
            else:
                self.dictionary[instrument]["GPDetrend"] = true_gp_boolean

        if return_samples:
            if return_err:
                if return_components:
                    return (
                        output_model_samples,
                        m_output_model,
                        u_output_model,
                        l_output_model,
                        components,
                    )
                else:
                    return (
                        output_model_samples,
                        m_output_model,
                        u_output_model,
                        l_output_model,
                    )
            else:
                if return_components:
                    return output_model_samples, output_model, components
                else:
                    return output_model_samples, output_model
        else:
            if return_err:
                if return_components:
                    return m_output_model, u_output_model, l_output_model, components
                else:
                    return m_output_model, u_output_model, l_output_model
            else:
                if return_components:
                    return output_model, components
                else:
                    return output_model

    def generate_lc_model(
        self, parameter_values, evaluate_global_errors=True, evaluate_lc=False
    ):
        self.modelOK = True

        def _get_rho(params):
            # Allow rho either directly or computed from m_star and r_star
            if "rho" in params:
                return params["rho"]
            elif ("m_star" in params) and ("r_star" in params):
                # convert to SI using astropy constants imported in utils
                return (params["m_star"] * const.M_sun.value) / (
                    (4.0 / 3.0) * np.pi * (params["r_star"] * const.R_sun.value) ** 3
                )
            else:
                raise Exception(
                    "No stellar density (rho) or (m_star and r_star) provided in parameter_values"
                )

        # If TTV parametrization is 'T' for planet i, store transit times. Check only if the noTflag is False (which implies
        # at least one planet uses the T-parametrization):
        if self.Tflag:
            planet_t0, planet_P = {}, {}
            all_Ts, all_ns = {}, {}

            for i in self.numbering:
                if self.Tparametrization[i]:
                    all_Ts[i], all_ns[i] = np.array([]), np.array([])

                    for instrument in self.inames:
                        for transit_number in self.dictionary[instrument]["TTVs"][
                            int(i)
                        ]["transit_number"]:
                            all_Ts[i] = np.append(
                                all_Ts[i],
                                parameter_values[
                                    f"T_p{i}_{instrument}_{str(transit_number)}"
                                ],
                            )

                            all_ns[i] = np.append(all_ns[i], transit_number)

                    # If evaluate_lc flag is on, this means user is evaluating lightcurve. Here we do some tricks as to only evaluate
                    # models in the user-defined instrument (to speed up evaluation), so in that case we use the posterior t0 and P
                    # actually taken from the T-samples:
                    if not evaluate_lc:
                        XY, Y, X, X2 = (
                            np.sum(all_Ts[i] * all_ns[i]) / self.N_TTVs[i],
                            np.sum(all_Ts[i]) / self.N_TTVs[i],
                            np.sum(all_ns[i]) / self.N_TTVs[i],
                            np.sum(all_ns[i] ** 2) / self.N_TTVs[i],
                        )

                        # Get slope:
                        planet_P[i] = (XY - X * Y) / (X2 - (X**2))

                        # Intercept:
                        planet_t0[i] = Y - planet_P[i] * X

                    else:
                        planet_t0[i], planet_P[i] = (
                            parameter_values["t0_p" + str(i)],
                            parameter_values["P_p" + str(i)],
                        )

        # Start loop to populate the self.model[instrument]['deterministic_model'] array, which will host the complete lightcurve for a given
        # instrument (including flux from all the planets). Do the for loop per instrument for the parameter extraction, so in the
        # future we can do, e.g., wavelength-dependant rp/rs.
        for instrument in self.inames:
            # Set full array to ones by copying:
            self.model[instrument]["M"] = np.copy(self.model[instrument]["ones"])

            # If transit fit is on, then model the transit lightcurve:
            if (
                self.dictionary[instrument]["TransitFit"]
                or self.dictionary[instrument]["EclipseFit"]
                or self.dictionary[instrument]["TranEclFit"]
            ):
                # Extract and set the limb-darkening coefficients for the instrument:
                if (
                    self.dictionary[instrument]["ldlaw"] != "linear"
                    and self.dictionary[instrument]["ldlaw"] != "none"
                ):
                    if (
                        self.dictionary[instrument]["ldparametrization"]
                        == "kipping2013"
                    ):
                        coeff1, coeff2 = reverse_ld_coeffs(
                            self.dictionary[instrument]["ldlaw"],
                            parameter_values["q1_" + self.ld_iname[instrument]],
                            parameter_values["q2_" + self.ld_iname[instrument]],
                        )

                    elif self.dictionary[instrument]["ldparametrization"] == "normal":
                        if self.dictionary[instrument]["ldlaw"] != "nonlinear":
                            coeff1, coeff2 = (
                                parameter_values["u1_" + self.ld_iname[instrument]],
                                parameter_values["u2_" + self.ld_iname[instrument]],
                            )

                        else:
                            coeff1, coeff2, coeff3, coeff4 = (
                                parameter_values["c1_" + self.ld_iname[instrument]],
                                parameter_values["c2_" + self.ld_iname[instrument]],
                                parameter_values["c3_" + self.ld_iname[instrument]],
                                parameter_values["c4_" + self.ld_iname[instrument]],
                            )

                elif self.dictionary[instrument]["ldlaw"] == "none":
                    coeff1, coeff2 = 0.1, 0.3

                else:
                    if (
                        self.dictionary[instrument]["ldparametrization"]
                        == "kipping2013"
                    ):
                        coeff1 = parameter_values["q1_" + self.ld_iname[instrument]]

                    elif self.dictionary[instrument]["ldparametrization"] == "normal":
                        coeff1 = parameter_values["u1_" + self.ld_iname[instrument]]

                # First (1) check if TTV mode is activated. If it is not, simply save the sampled planet periods and time-of transit centers for check
                # in the next round of iteration (see below). If it is, depending on the parametrization, either shift the time-indexes accordingly (see below
                # comments for details).
                cP, ct0 = {}, {}

                for i in self.numbering:
                    # Check if we will be fitting for TTVs. If not, all goes as usual. If we are, check which parametrization (dt or T):
                    if not self.dictionary[instrument]["TTVs"][i]["status"]:
                        t0, P = (
                            parameter_values["t0_p" + str(i)],
                            parameter_values["P_p" + str(i)],
                        )

                        cP[i], ct0[i] = P, t0

                    else:
                        # If TTVs is on for planet i, compute the expected time of transit, and shift it. For this, use information encoded in the prior
                        # name; if, e.g., dt_p1_TESS1_-2, then n = -2 and the time of transit (with TTV) = t0 + n*P + dt_p1_TESS1_-2 in the case of the dt
                        # parametrization. In the case of the T-parametrization, the time of transit with TTV would be T_p1_TESS1_-2, and the period and t0
                        # will be derived from there from the least-squares slope and intercept, respectively, to the T's. Compute transit
                        # model assuming that time-of-transit; repeat for all the transits. Generally users will not do TTV analyses, so set this latter
                        # case to be the most common one by default in the if-statement:
                        dummy_time = np.copy(self.times[instrument])

                        if (
                            self.dictionary[instrument]["TTVs"][i]["parametrization"]
                            == "dt"
                        ):
                            t0, P = (
                                parameter_values["t0_p" + str(i)],
                                parameter_values["P_p" + str(i)],
                            )

                            cP[i], ct0[i] = P, t0

                            for transit_number in self.dictionary[instrument]["TTVs"][
                                int(i)
                            ]["transit_number"]:
                                transit_time = (
                                    t0
                                    + transit_number * P
                                    + parameter_values[
                                        f"dt_p{i}_{instrument}_{str(transit_number)}"
                                    ]
                                )

                                # This implicitly sets maximum transit duration to P/2 days:
                                idx = np.where(
                                    np.abs(self.times[instrument] - transit_time)
                                    < P / 4.0
                                )[0]

                                dummy_time[idx] = (
                                    self.times[instrument][idx]
                                    - parameter_values[
                                        f"dt_p{i}_{instrument}_{str(transit_number)}"
                                    ]
                                )

                        else:
                            t0, P = planet_t0[i], planet_P[i]

                            for transit_number in self.dictionary[instrument]["TTVs"][
                                int(i)
                            ]["transit_number"]:
                                dt = parameter_values[
                                    f"T_p{i}_{instrument}_{str(transit_number)}"
                                ] - (t0 + transit_number * P)

                                # This implicitly sets maximum transit duration to P/2 days:
                                idx = np.where(
                                    np.abs(
                                        self.times[instrument]
                                        - parameter_values[
                                            f"T_p{i}_{instrument}_{str(transit_number)}"
                                        ]
                                    )
                                    < P / 4.0
                                )[0]

                                dummy_time[idx] = self.times[instrument][idx] - dt

                            cP[i], ct0[i] = P, t0

                # Whether there are TTVs or not, and before anything continues, check the periods are chronologically ordered (this is to avoid multiple modes
                # due to periods "jumping" between planet numbering):
                first_time = True
                for i in self.numbering:
                    if first_time:
                        ccP = cP[i]  # parameter_values['P_p'+str(i)]
                        first_time = False

                    else:
                        if ccP < cP[i]:  # parameter_values['P_p'+str(i)]:
                            ccP = cP[i]  # parameter_values['P_p'+str(i)]

                        else:
                            self.modelOK = False
                            return False

                # Once all is OK with the periods and time-of-transit centers, loop through all the planets, getting the lightcurve model for each:
                for i in self.numbering:
                    P, t0 = cP[i], ct0[i]

                    ### For instrument dependent eclipse depth:
                    ### We only want to make eclipse depth instrument depended, not the time correction factor
                    if (
                        self.dictionary[instrument]["EclipseFit"]
                        or self.dictionary[instrument]["TranEclFit"]
                    ):
                        fp = parameter_values[
                            "fp_p" + str(i) + self.fp_iname["p" + str(i)][instrument]
                        ]

                        if self.dictionary[instrument]["PhaseCurveFit"]:
                            phase_offset = parameter_values[
                                "phaseoffset_p"
                                + str(i)
                                + self.phaseoffset_iname["p" + str(i)][instrument]
                            ]

                        if not self.light_travel_delay:
                            t_secondary = parameter_values["t_secondary_p" + str(i)]

                    if self.dictionary["efficient_bp"][i]:
                        if not self.dictionary["fitrho"]:
                            a, r1, r2 = (
                                parameter_values["a_p" + str(i)],
                                parameter_values["r1_p" + str(i)],
                                parameter_values["r2_p" + str(i)],
                            )
                        else:
                            rho = _get_rho(parameter_values)
                            r1, r2 = (
                                parameter_values["r1_p" + str(i)],
                                parameter_values["r2_p" + str(i)],
                            )
                            a = (
                                (rho * const.G.value * ((P * 24.0 * 3600.0) ** 2)) / (3.0 * np.pi)
                            ) ** (1.0 / 3.0)
                        if r1 > self.Ar:
                            b, p = (
                                (1 + self.pl) * (1.0 + (r1 - 1.0) / (1.0 - self.Ar)),
                                (1 - r2) * self.pl + r2 * self.pu,
                            )
                        else:
                            b, p = (
                                (1.0 + self.pl)
                                + np.sqrt(r1 / self.Ar) * r2 * (self.pu - self.pl),
                                self.pu
                                + (self.pl - self.pu)
                                * np.sqrt(r1 / self.Ar)
                                * (1.0 - r2),
                            )
                    else:
                        if not self.dictionary["fitrho"]:
                            if not self.dictionary[instrument]["TransitFitCatwoman"]:
                                a, b, p = (
                                    parameter_values["a_p" + str(i)],
                                    parameter_values["b_p" + str(i)],
                                    parameter_values[
                                        "p_p"
                                        + str(i)
                                        + self.p_iname["p" + str(i)][instrument]
                                    ],
                                )
                            else:
                                a, b, p1, p2, phi = (
                                    parameter_values["a_p" + str(i)],
                                    parameter_values["b_p" + str(i)],
                                    parameter_values[
                                        "p1_p"
                                        + str(i)
                                        + self.p1_iname["p" + str(i)][instrument]
                                    ],
                                    parameter_values[
                                        "p2_p"
                                        + str(i)
                                        + self.p1_iname["p" + str(i)][instrument]
                                    ],
                                    parameter_values["phi_p" + str(i)],
                                )

                                p = np.min([p1, p2])

                        else:
                            if not self.dictionary[instrument]["TransitFitCatwoman"]:
                                rho = _get_rho(parameter_values)
                                b, p = (
                                    parameter_values["b_p" + str(i)],
                                    parameter_values[
                                        "p_p"
                                        + str(i)
                                        + self.p_iname["p" + str(i)][instrument]
                                    ],
                                )

                            else:
                                rho = _get_rho(parameter_values)
                                b, p1, p2, phi = (
                                    parameter_values["b_p" + str(i)],
                                    parameter_values[
                                        "p1_p"
                                        + str(i)
                                        + self.p1_iname["p" + str(i)][instrument]
                                    ],
                                    parameter_values[
                                        "p2_p"
                                        + str(i)
                                        + self.p1_iname["p" + str(i)][instrument]
                                    ],
                                    parameter_values["phi_p" + str(i)],
                                )

                                p = np.min([p1, p2])

                            a = (
                                (rho * const.G.value * ((P * 24.0 * 3600.0) ** 2)) / (3.0 * np.pi)
                            ) ** (1.0 / 3.0)

                    # Now extract eccentricity and omega depending on the used parametrization for each planet:
                    if self.dictionary["ecc_parametrization"][i] == 0:
                        ecc, omega = (
                            parameter_values["ecc_p" + str(i)],
                            parameter_values["omega_p" + str(i)],
                        )
                    elif self.dictionary["ecc_parametrization"][i] == 1:
                        ecc = np.sqrt(
                            parameter_values["ecosomega_p" + str(i)] ** 2
                            + parameter_values["esinomega_p" + str(i)] ** 2
                        )
                        omega = (
                            np.arctan2(
                                parameter_values["esinomega_p" + str(i)],
                                parameter_values["ecosomega_p" + str(i)],
                            )
                            * 180.0
                            / np.pi
                        )
                    else:
                        ecc = (
                            parameter_values["secosomega_p" + str(i)] ** 2
                            + parameter_values["sesinomega_p" + str(i)] ** 2
                        )
                        omega = (
                            np.arctan2(
                                parameter_values["sesinomega_p" + str(i)],
                                parameter_values["secosomega_p" + str(i)],
                            )
                            * 180.0
                            / np.pi
                        )

                    # Generate lightcurve for the current planet if ecc is OK:
                    if ecc > self.ecclim:
                        self.modelOK = False
                        return False

                    else:
                        ecc_factor = (1.0 + ecc * np.sin(omega * np.pi / 180.0)) / (
                            1.0 - ecc**2
                        )
                        inc_inv_factor = (b / a) * ecc_factor

                        if not (b > 1.0 + p or inc_inv_factor >= 1.0):
                            self.model[instrument]["params"].t0 = t0
                            self.model[instrument]["params"].per = P
                            self.model[instrument]["params"].a = a

                            self.model[instrument]["params"].inc = (
                                np.arccos(inc_inv_factor) * 180.0 / np.pi
                            )
                            self.model[instrument]["params"].ecc = ecc
                            self.model[instrument]["params"].w = omega

                            if (
                                self.dictionary[instrument]["EclipseFit"]
                                or self.dictionary[instrument]["TranEclFit"]
                            ):
                                self.model[instrument]["params"].fp = fp

                                if not self.light_travel_delay:
                                    self.model[instrument][
                                        "params"
                                    ].t_secondary = t_secondary

                                else:
                                    # If light-travel time is activated, self-consistently calculate time of secondary eclipse:
                                    self.model[instrument][
                                        "params"
                                    ].Rs = self.stellar_radius

                                    if self.dictionary[instrument]["EclipseFit"]:
                                        self.model[instrument][
                                            "params"
                                        ].t_secondary = self.model[instrument][
                                            "m"
                                        ].get_t_secondary(
                                            self.model[instrument]["params"]
                                        )

                                    elif self.dictionary[instrument]["TranEclFit"]:
                                        self.model[instrument][
                                            "params"
                                        ].t_secondary = self.model[instrument]["m"][
                                            1
                                        ].get_t_secondary(
                                            self.model[instrument]["params"]
                                        )

                                    # Get time-delayed times:
                                    corrected_t = correct_light_travel_time(
                                        self.times[instrument],
                                        self.model[instrument]["params"],
                                    )

                                    # Dynamically modify the batman model for the eclipse part:
                                    if self.dictionary[instrument]["EclipseFit"]:
                                        if self.dictionary[instrument]["resampling"]:
                                            _, [_, self.model[instrument]["m"]] = (
                                                init_batman(
                                                    corrected_t,
                                                    self.dictionary[instrument][
                                                        "ldlaw"
                                                    ],
                                                    nresampling=self.dictionary[
                                                        instrument
                                                    ]["nresampling"],
                                                    etresampling=self.dictionary[
                                                        instrument
                                                    ]["exptimeresampling"],
                                                )
                                            )

                                        else:
                                            _, [_, self.model[instrument]["m"]] = (
                                                init_batman(
                                                    corrected_t,
                                                    self.dictionary[instrument][
                                                        "ldlaw"
                                                    ],
                                                )
                                            )

                                    elif self.dictionary[instrument]["TranEclFit"]:
                                        if self.dictionary[instrument]["resampling"]:
                                            _, [_, self.model[instrument]["m"][1]] = (
                                                init_batman(
                                                    corrected_t,
                                                    self.dictionary[instrument][
                                                        "ldlaw"
                                                    ],
                                                    nresampling=self.dictionary[
                                                        instrument
                                                    ]["nresampling"],
                                                    etresampling=self.dictionary[
                                                        instrument
                                                    ]["exptimeresampling"],
                                                )
                                            )

                                        else:
                                            _, [_, self.model[instrument]["m"][1]] = (
                                                init_batman(
                                                    corrected_t,
                                                    self.dictionary[instrument][
                                                        "ldlaw"
                                                    ],
                                                )
                                            )

                            if not self.dictionary[instrument]["TransitFitCatwoman"]:
                                self.model[instrument]["params"].rp = p

                            else:
                                self.model[instrument]["params"].rp = p1
                                self.model[instrument]["params"].rp2 = p2
                                self.model[instrument]["params"].phi = phi

                            if self.dictionary[instrument]["ldlaw"] == "nonlinear":
                                self.model[instrument]["params"].u = [
                                    coeff1,
                                    coeff2,
                                    coeff3,
                                    coeff4,
                                ]

                            elif self.dictionary[instrument]["ldlaw"] == "linear":
                                self.model[instrument]["params"].u = [coeff1]

                            else:
                                self.model[instrument]["params"].u = [coeff1, coeff2]

                            # If TTVs is on for planet i, compute the expected time of transit, and shift it. For this, use information encoded in the prior
                            # name; if, e.g., dt_p1_TESS1_-2, then n = -2 and the time of transit (with TTV) = t0 + n*P + dt_p1_TESS1_-2. Compute transit
                            # model assuming that time-of-transit; repeat for all the transits. Generally users will not do TTV analyses, so set this latter
                            # case to be the most common one by default in the if-statement:

                            if not self.dictionary[instrument]["TTVs"][i]["status"]:
                                # If log_like_calc is True (by default during juliet.fit), don't bother saving the lightcurve of planet p_i:

                                if self.log_like_calc:
                                    if not self.dictionary[instrument]["TranEclFit"]:
                                        self.model[instrument]["M"] += (
                                            self.model[instrument]["m"].light_curve(
                                                self.model[instrument]["params"]
                                            )
                                            - 1.0
                                        )

                                    else:
                                        # In combined transit + eclipse models, assume either (a) by default any phase-curve variations are being modelled externally
                                        # (by, e.g., systematics models, phase-curve variations added later, etc.). To this end, note batman has out-of-eclipse
                                        # model variations 1 + fp --- with in-eclipse always being 1. Subtract fp then::
                                        transit_model = self.model[instrument]["m"][
                                            0
                                        ].light_curve(self.model[instrument]["params"])
                                        eclipse_model = self.model[instrument]["m"][
                                            1
                                        ].light_curve(self.model[instrument]["params"])

                                        # Now, figure out if a phase-curve model is being fit or not:
                                        if not self.dictionary[instrument][
                                            "PhaseCurveFit"
                                        ]:
                                            eclipse_model = (
                                                eclipse_model
                                                - self.model[instrument]["params"].fp
                                            )
                                            self.model[instrument]["M"] += (
                                                transit_model * eclipse_model - 1.0
                                            )

                                        else:
                                            orbital_phase = (
                                                (
                                                    self.model[instrument]["m"][1].t
                                                    - self.model[instrument][
                                                        "params"
                                                    ].t0
                                                )
                                                / self.model[instrument]["params"].per
                                            ) % 1
                                            center_phase = -np.pi / 2.0

                                            # Build model. First, the basis sine function:
                                            sine_model = np.sin(
                                                2.0 * np.pi * (orbital_phase)
                                                + center_phase
                                                + phase_offset * (np.pi / 180.0)
                                            )
                                            # Scale to be 1 at secondary eclipse, 0 at transit:
                                            sine_model = (sine_model + 1) * 0.5
                                            # Amplify by phase-amplitude:
                                            sine_model = (
                                                self.model[instrument]["params"].fp
                                            ) * sine_model
                                            # Multiply by normed eclipse model:
                                            sine_model = 1.0 + sine_model * (
                                                (eclipse_model - 1.0)
                                                / self.model[instrument]["params"].fp
                                            )

                                            # And get all together:
                                            self.model[instrument]["M"] += (
                                                transit_model * sine_model - 1.0
                                            )

                                else:
                                    if not self.dictionary[instrument]["TranEclFit"]:
                                        self.model[instrument]["p" + str(i)] = (
                                            self.model[instrument]["m"].light_curve(
                                                self.model[instrument]["params"]
                                            )
                                        )
                                        self.model[instrument]["M"] += (
                                            self.model[instrument]["p" + str(i)] - 1.0
                                        )

                                    else:
                                        # In combined transit + eclipse models, assume by default any phase-curve variations are being modelled externally
                                        # by default (by, e.g., systematics models, phase-curve variations added later, etc.). To this end, note batman has out-of-eclipse
                                        # model variations 1 + fp --- with in-eclipse always being 1. Subtract fp then::
                                        transit_model = self.model[instrument]["m"][
                                            0
                                        ].light_curve(self.model[instrument]["params"])
                                        eclipse_model = self.model[instrument]["m"][
                                            1
                                        ].light_curve(self.model[instrument]["params"])

                                        # Now, figure out if a phase-curve model is being fit or not:
                                        if not self.dictionary[instrument][
                                            "PhaseCurveFit"
                                        ]:
                                            eclipse_model = (
                                                eclipse_model
                                                - self.model[instrument]["params"].fp
                                            )
                                            self.model[instrument]["p" + str(i)] = (
                                                transit_model * eclipse_model
                                            )

                                        else:
                                            orbital_phase = (
                                                (
                                                    self.model[instrument]["m"][1].t
                                                    - self.model[instrument][
                                                        "params"
                                                    ].t0
                                                )
                                                / self.model[instrument]["params"].per
                                            ) % 1
                                            center_phase = -np.pi / 2.0

                                            # Build model. First, the basis sine function:
                                            sine_model = np.sin(
                                                2.0 * np.pi * (orbital_phase)
                                                + center_phase
                                                + phase_offset * (np.pi / 180.0)
                                            )
                                            # Scale to be 1 at secondary eclipse, 0 at transit:
                                            sine_model = (sine_model + 1) * 0.5
                                            # Amplify by phase-amplitude:
                                            sine_model = (
                                                self.model[instrument]["params"].fp
                                            ) * sine_model
                                            # Multiply by normed eclipse model:
                                            sine_model = 1.0 + sine_model * (
                                                (eclipse_model - 1.0)
                                                / self.model[instrument]["params"].fp
                                            )

                                            self.model[instrument]["p" + str(i)] = (
                                                transit_model * sine_model
                                            )

                                        self.model[instrument]["M"] += (
                                            self.model[instrument]["p" + str(i)] - 1.0
                                        )

                            else:
                                if not self.dictionary[instrument][
                                    "TransitFitCatwoman"
                                ]:
                                    if self.dictionary[instrument]["resampling"]:
                                        if self.dictionary[instrument]["TransitFit"]:
                                            pm, [m, _] = init_batman(
                                                dummy_time,
                                                self.dictionary[instrument]["ldlaw"],
                                                nresampling=self.dictionary[instrument][
                                                    "nresampling"
                                                ],
                                                etresampling=self.dictionary[
                                                    instrument
                                                ]["exptimeresampling"],
                                            )
                                        elif self.dictionary[instrument]["EclipseFit"]:
                                            pm, [_, m] = init_batman(
                                                dummy_time,
                                                self.dictionary[instrument]["ldlaw"],
                                                nresampling=self.dictionary[instrument][
                                                    "nresampling"
                                                ],
                                                etresampling=self.dictionary[
                                                    instrument
                                                ]["exptimeresampling"],
                                            )
                                        elif self.dictionary[instrument]["TranEclFit"]:
                                            pm, m = init_batman(
                                                dummy_time,
                                                self.dictionary[instrument]["ldlaw"],
                                                nresampling=self.dictionary[instrument][
                                                    "nresampling"
                                                ],
                                                etresampling=self.dictionary[
                                                    instrument
                                                ]["exptimeresampling"],
                                            )
                                    else:
                                        if self.dictionary[instrument]["TransitFit"]:
                                            pm, [m, _] = init_batman(
                                                dummy_time,
                                                self.dictionary[instrument]["ldlaw"],
                                            )
                                        elif self.dictionary[instrument]["EclipseFit"]:
                                            pm, [_, m] = init_batman(
                                                dummy_time,
                                                self.dictionary[instrument]["ldlaw"],
                                            )
                                        elif self.dictionary[instrument]["TranEclFit"]:
                                            pm, m = init_batman(
                                                dummy_time,
                                                self.dictionary[instrument]["ldlaw"],
                                            )

                                else:
                                    if self.dictionary[instrument]["resampling"]:
                                        pm, m = init_catwoman(
                                            dummy_time,
                                            self.dictionary[instrument]["ldlaw"],
                                            nresampling=self.dictionary[instrument][
                                                "nresampling"
                                            ],
                                            etresampling=self.dictionary[instrument][
                                                "exptimeresampling"
                                            ],
                                        )
                                    else:
                                        pm, m = init_catwoman(
                                            dummy_time,
                                            self.dictionary[instrument]["ldlaw"],
                                        )

                                # If log_like_calc is True (by default during juliet.fit), don't bother saving the lightcurve of planet p_i:
                                if self.log_like_calc:
                                    if not self.dictionary[instrument]["TranEclFit"]:
                                        self.model[instrument]["M"] += (
                                            m.light_curve(
                                                self.model[instrument]["params"]
                                            )
                                            - 1.0
                                        )

                                    else:
                                        # In combined transit + eclipse models, assume by default any phase-curve variations are being modelled externally
                                        # by default (by, e.g., systematics models, phase-curve variations added later, etc.). To this end, note batman has out-of-eclipse
                                        # model variations 1 + fp --- with in-eclipse always being 1. Subtract fp then::
                                        transit_model = m[0].light_curve(
                                            self.model[instrument]["params"]
                                        )
                                        eclipse_model = m[1].light_curve(
                                            self.model[instrument]["params"]
                                        )

                                        # Now, figure out if a phase-curve model is being fit or not:
                                        if not self.dictionary[instrument][
                                            "PhaseCurveFit"
                                        ]:
                                            eclipse_model = (
                                                eclipse_model
                                                - self.model[instrument]["params"].fp
                                            )
                                            self.model[instrument]["M"] += (
                                                transit_model * eclipse_model - 1.0
                                            )

                                        else:
                                            orbital_phase = (
                                                (
                                                    self.model[instrument]["m"][1].t
                                                    - self.model[instrument][
                                                        "params"
                                                    ].t0
                                                )
                                                / self.model[instrument]["params"].per
                                            ) % 1
                                            center_phase = -np.pi / 2.0

                                            # Build model. First, the basis sine function:
                                            sine_model = np.sin(
                                                2.0 * np.pi * (orbital_phase)
                                                + center_phase
                                                + phase_offset * (np.pi / 180.0)
                                            )
                                            # Scale to be 1 at secondary eclipse, 0 at transit:
                                            sine_model = (sine_model + 1) * 0.5
                                            # Amplify by phase-amplitude:
                                            sine_model = (
                                                self.model[instrument]["params"].fp
                                            ) * sine_model
                                            # Multiply by normed eclipse model:
                                            sine_model = 1.0 + sine_model * (
                                                (eclipse_model - 1.0)
                                                / self.model[instrument]["params"].fp
                                            )

                                            self.model[instrument]["M"] += (
                                                transit_model * sine_model - 1.0
                                            )

                                else:
                                    if not self.dictionary[instrument]["TranEclFit"]:
                                        self.model[instrument]["p" + str(i)] = (
                                            m.light_curve(
                                                self.model[instrument]["params"]
                                            )
                                        )
                                        self.model[instrument]["M"] += (
                                            self.model[instrument]["p" + str(i)] - 1.0
                                        )

                                    else:
                                        # In combined transit + eclipse models, assume by default any phase-curve variations are being modelled externally
                                        # by default (by, e.g., systematics models, phase-curve variations added later, etc.). To this end, note batman has out-of-eclipse
                                        # model variations 1 + fp --- with in-eclipse always being 1. Subtract fp then:
                                        transit_model = m[0].light_curve(
                                            self.model[instrument]["params"]
                                        )
                                        eclipse_model = m[1].light_curve(
                                            self.model[instrument]["params"]
                                        )

                                        # Now, figure out if a phase-curve model is being fit or not:
                                        if not self.dictionary[instrument][
                                            "PhaseCurveFit"
                                        ]:
                                            eclipse_model = (
                                                eclipse_model
                                                - self.model[instrument]["params"].fp
                                            )
                                            self.model[instrument]["p" + str(i)] = (
                                                transit_model * eclipse_model
                                            )

                                        else:
                                            orbital_phase = (
                                                (
                                                    self.model[instrument]["m"][1].t
                                                    - self.model[instrument][
                                                        "params"
                                                    ].t0
                                                )
                                                / self.model[instrument]["params"].per
                                            ) % 1
                                            center_phase = -np.pi / 2.0

                                            # Build model. First, the basis sine function:
                                            sine_model = np.sin(
                                                2.0 * np.pi * (orbital_phase)
                                                + center_phase
                                                + phase_offset * (np.pi / 180.0)
                                            )
                                            # Scale to be 1 at secondary eclipse, 0 at transit:
                                            sine_model = (sine_model + 1) * 0.5
                                            # Amplify by phase-amplitude:
                                            sine_model = (
                                                self.model[instrument]["params"].fp
                                            ) * sine_model
                                            # Multiply by normed eclipse model:
                                            sine_model = 1.0 + sine_model * (
                                                (eclipse_model - 1.0)
                                                / self.model[instrument]["params"].fp
                                            )

                                            self.model[instrument]["p" + str(i)] = (
                                                transit_model * sine_model
                                            )

                                        self.model[instrument]["M"] += (
                                            self.model[instrument]["p" + str(i)] - 1.0
                                        )

                        else:
                            self.modelOK = False
                            return False

            # Once either the transit model is generated or after populating the full_model with ones if no transit fit is on,
            # convert the lightcurve so it complies with the juliet model accounting for the dilution and the mean out-of-transit flux:
            D, M = (
                parameter_values["mdilution_" + self.mdilution_iname[instrument]],
                parameter_values["mflux_" + self.mflux_iname[instrument]],
            )
            self.model[instrument]["M"] = (
                self.model[instrument]["M"] * D + (1.0 - D)
            ) * (1.0 / (1.0 + D * M))

            # Now, if a linear model was defined, generate it and add it to the full model:
            if self.lm_boolean[instrument]:
                self.model[instrument]["LM"] = np.zeros(
                    self.ndatapoints_per_instrument[instrument]
                )
                for i in range(self.lm_n[instrument]):
                    self.model[instrument]["LM"] += (
                        parameter_values[
                            "theta"
                            + str(i)
                            + "_"
                            + self.theta_iname[f"{i}{instrument}"]
                        ]
                        * self.lm_arguments[instrument][:, i]
                    )

                self.model[instrument]["deterministic"] = (
                    self.model[instrument]["M"] + self.model[instrument]["LM"]
                )

            else:
                self.model[instrument]["deterministic"] = self.model[instrument]["M"]

            # Now, if a non-linear model was defined, generate it and add it to the full model:
            if self.nlm_boolean[instrument]:
                self.model[instrument]["NLM"] = self.non_linear_functions[instrument][
                    "function"
                ](
                    self.non_linear_functions[instrument]["regressor"],
                    parameter_values,
                )

                if self.multiplicative_non_linear_function[instrument]:
                    self.model[instrument]["deterministic"] *= self.model[instrument][
                        "NLM"
                    ]

                else:
                    self.model[instrument]["deterministic"] += self.model[instrument][
                        "NLM"
                    ]

            self.model[instrument]["deterministic_variances"] = (
                self.errors[instrument] ** 2
                + (parameter_values["sigma_w_" + self.sigmaw_iname[instrument]] * 1e-6)
                ** 2
            )

            # Finally, if the model under consideration is a global model, populate the global model dictionary:
            if self.global_model:
                self.model["global"][self.instrument_indexes[instrument]] = self.model[
                    instrument
                ]["deterministic"]
                if evaluate_global_errors:
                    self.model["global_variances"][
                        self.instrument_indexes[instrument]
                    ] = (
                        self.yerr[self.instrument_indexes[instrument]] ** 2
                        + (parameter_values[f"sigma_w_{instrument}"] * 1e-6) ** 2
                    )

    def gaussian_log_likelihood(self, residuals, variances):
        taus = 1.0 / variances
        return -0.5 * (
            len(residuals) * LOG_2_PI
            + np.sum(-np.log(taus.astype(float)) + taus * (residuals**2))
        )

    def get_log_likelihood(self, parameter_values):
        if self.global_model:
            residuals = self.y - self.model["global"]
            if self.dictionary["global_model"]["GPDetrend"]:
                self.dictionary["global_model"]["noise_model"].set_parameter_vector(
                    parameter_values
                )
                self.dictionary["global_model"]["noise_model"].yerr = np.sqrt(
                    self.model["global_variances"]
                )
                self.dictionary["global_model"]["noise_model"].compute_GP()
                return self.dictionary["global_model"]["noise_model"].GP.log_likelihood(
                    residuals
                )
            else:
                self.gaussian_log_likelihood(residuals, self.model["global_variances"])
        else:
            log_like = 0.0

            for instrument in self.inames:
                residuals = (
                    self.data[instrument] - self.model[instrument]["deterministic"]
                )
                if self.dictionary[instrument]["GPDetrend"]:
                    self.dictionary[instrument]["noise_model"].set_parameter_vector(
                        parameter_values
                    )
                    # Catch possible GP evaluation errors:
                    try:
                        log_like += self.dictionary[instrument][
                            "noise_model"
                        ].GP.log_likelihood(residuals)
                    except Exception:
                        log_like = -np.inf
                        break
                else:
                    log_like += self.gaussian_log_likelihood(
                        residuals, self.model[instrument]["deterministic_variances"]
                    )

            return log_like

    def set_posterior_samples(self, posterior_samples):
        self.posteriors = posterior_samples
        self.median_posterior_samples = {}
        for parameter in self.posteriors.keys():
            if parameter != "unnamed":
                self.median_posterior_samples[parameter] = np.median(
                    self.posteriors[parameter]
                )

        for parameter in self.priors:
            if self.priors[parameter]["distribution"] == "fixed":
                self.median_posterior_samples[parameter] = self.priors[parameter][
                    "hyperparameters"
                ]
        try:
            self.generate(self.median_posterior_samples)
        except:
            print(
                "Warning: model evaluated at the posterior median did not compute properly."
            )
