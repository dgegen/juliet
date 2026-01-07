import os

import numpy as np

from .fit import fit
from .gaussian_process import gaussian_process
from .utils import input_error_catcher, read_data, readGPeparams, readpriors

__all__ = ["load"]


class load(object):
    """
    Given a dictionary with priors (or a filename pointing to a prior file) and data either given through arrays
    or through files containing the data, this class loads data into a juliet object which holds all the information
    about the dataset. Example usage:

               >>> data = juliet.load(priors=priors,t_lc=times,y_lc=fluxes,yerr_lc=fluxes_errors)

    Or, also,

               >>> data = juliet.load(input_folder = folder)

    :param priors: (optional, dict or string)
        This can be either a python ``string`` or a python ``dict``. If a ``dict``, this has to contain each of
        the parameters to be fit, along with their respective prior distributions and hyperparameters. Each key
        of this dictionary has to have a parameter name (e.g., ``r1_p1``, ``sigma_w_TESS``), and each of
        those elements are, in turn, dictionaries as well containing two keys: a ``distribution``
        key which defines the prior distribution of the parameter and a ``hyperparameters`` key,
        which contains the hyperparameters of that distribution.

        Example setup of the ``priors`` dictionary:
            >>> priors = {}
            >>> priors['r1_p1'] = {}
            >>> priors['r1_p1']['distribution'] = 'Uniform'
            >>> priors['r1_p1']['hyperparameters'] = [0.,1.]

        If a ``string``, this has to contain the filename to a proper juliet prior file; the prior ``dict`` will
        then be generated from there. A proper prior file has in the first column the name of the parameter,
        in the second the name of the distribution, and in the third the hyperparameters of that distribution for
        the parameter.

        Note that this along with either lightcurve or RV data or a ``input_folder`` has to be given in order to properly
        load a juliet data object.

    :param starting_point: (mandatory if using MCMC, useless if using nested samplers, dict)
        Dictionary indicating the starting value of each of the parameters for the MCMC run (i.e., currently only of use for ``emcee``). Keys should be consistent with the ``prior`` namings above;
        each key should have an associated float with the starting value. This is of no use if using nested samplers (which sample directly from the prior).

    :param input_folder: (optional, string)
        Python ``string`` containing the path to a folder containing all the input data --- this will thus be load into a
        juliet data object. This input folder has to contain at least a ``priors.dat`` file with the priors and either a ``lc.dat``
        file containing lightcurve data or a ``rvs.dat`` file containing radial-velocity data. If in this folder a ``GP_lc_regressors.dat``
        file or a ``GP_rv_regressors.dat`` file is found, data will be loaded into the juliet object as well.

        Note that at least this or a ``priors`` string or dictionary, along with either lightcurve or RV data has to be given
        in order to properly load a juliet data object.

    :param t_lc: (optional, dictionary)
        Dictionary whose keys are instrument names; each of those keys is expected to have arrays with the times corresponding to those instruments.
        For example,
                                    >>> t_lc = {}
                                    >>> t_lc['TESS'] = np.linspace(0,100,100)

        Is a valid input dictionary for ``t_lc``.

    :param y_lc: (optional, dictionary)
        Similarly to ``t_lc``, dictionary whose keys are instrument names; each of those keys is expected to have arrays with the fluxes corresponding to those instruments.
        These are expected to be consistent with the ``t_lc`` dictionaries.

    :param yerr_lc: (optional, dictionary)
        Similarly to ``t_lc``, dictionary whose keys are instrument names; each of those keys is expected to have arrays with the errors on the fluxes corresponding to those instruments.
        These are expected to be consistent with the ``t_lc`` dictionaries.

    :param GP_regressors_lc: (optional, dictionary)
        Dictionary whose keys are names of instruments where a GP is to be fit. On each name/element, an array of
        regressors of shape ``(m,n)`` containing in each column the ``n`` GP regressors to be used for
        ``m`` photometric measurements has to be given. Note that ``m`` for a given instrument has to be of the same length
        as the corresponding ``t_lc`` for that instrument. Also, note the order of each regressor of each instrument has to match
        the corresponding order in the ``t_lc`` array.
        For example,

                                    >>> GP_regressors_lc = {}
                                    >>> GP_regressors_lc['TESS'] = np.linspace(-1,1,100)

        If a global model wants to be used, then the instrument should be ``rv``, and each of the ``m`` rows should correspond to the ``m`` times.

    :param linear_regressors_lc: (optional, dictionary)
        Similarly as for ``GP_regressors_lc``, this is a dictionary whose keys are names of instruments where a linear regression is to be fit.
        On each name/element, an array of shape ``(q,p)`` containing in each column the ``p`` linear regressors to be used for the ``q``
        photometric measurements. Again, note the order of each regressor of each instrument has to match the corresponding order in the ``t_lc`` array.

    :param GP_regressors_rv: (optional, dictionary)
        Same as ``GP_regressors_lc`` but for the radial-velocity data. If a global model wants to be used, then the instrument should be ``lc``, and each of the ``m`` rows should correspond to the ``m`` times.

    :param linear_regressors_rv: (optional, dictionary)
        Same as ``linear_regressors_lc``, but for the radial-velocities.

    :param t_rv: (optional, dictionary)
        Same as ``t_lc``, but for the radial-velocities.

    :param y_rv: (optional, dictionary)
        Same as ``y_lc``, but for the radial-velocities.

    :param yerr_rv: (optional, dictionary)
        Same as ``yerr_lc``, but for the radial-velocities.

    :param out_folder: (optional, string)
        If a path is given, results will be saved to that path as a ``pickle`` file, along with all inputs in the standard juliet format.

    :param lcfilename:  (optional, string)
        If a path to a lightcurve file is given, ``t_lc``, ``y_lc``, ``yerr_lc`` and ``instruments_lc`` will be read from there. The basic file format is a pure
        ascii file where times are in the first column, relative fluxes in the second, errors in the third and instrument names in the fourth. If more columns are given for
        a given instrument, those will be identified as linear regressors for those instruments.

    :param rvfilename: (optional, string)
        Same as ``lcfilename``, but for the radial-velocities.

    :param GPlceparamfile: (optional, string)
        If a path to a file is given, the columns of that file will be used as GP regressors for the lightcurve fit. The file format is a pure ascii file
        where regressors are given in different columns, and the last column holds the instrument name. The order of this file has to be consistent with
        ``t_lc`` and/or the ``lcfilename`` file. If a global model wants to be used, set the instrument names of all regressors to ``lc``.

    :param GPrveparamfile: (optional, string)
        Same as ``GPlceparamfile`` but for the radial-velocities. If a global model wants to be used, set the instrument names of all regressors to ``rv``.

    :param LMlceparamfile: (optional, string)
        If a path to a file is given, the columns of that file will be used as linear regressors for the lightcurve fit. The file format is a pure ascii file
        where regressors are given in different columns, and the last column holds the instrument name. The order of this file has to be consistent with
        ``t_lc`` and/or the ``lcfilename`` file. If a global model wants to be used, set the instrument names of all regressors to ``lc``.

    :param LMrveparamfile: (optional, string)
        Same as ``LMlceparamfile`` but for the radial-velocities. If a global model wants to be used, set the instrument names of all regressors to ``rv``.

    :param lctimedef: (optional, string)
        Time definitions for each of the lightcurve instruments. Default is to assume all instruments (in lcs and rvs) have the same time definitions. If more than one instrument is given, this string
        should have instruments and time-definitions separated by commas, e.g., ``TESS-TDB, LCOGT-UTC``, etc.

    :param rvtimedef: (optional, string)
        Time definitions for each of the radial-velocity instruments. Default is to assume all instruments (in lcs and rvs) have the same time definitions. If more than one instrument is given,
        this string should have instruments and time-definitions separated by commas, e.g., ``FEROS-TDB, HARPS-UTC``, etc.

    :param ld_laws: (optional, string)
        Limb-darkening law to be used for each instrument. Default is ``quadratic`` for all instruments. If more than one instrument is given,
        this string should have instruments and limb-darkening laws separated by commas, e.g., ``TESS-quadratic, LCOGT-linear``.

    :param priorfile: (optional, string)
        If a path to a file is given, it will be assumed this is a prior file. The ``priors`` dictionary will be overwritten by the data in this
        file. The file structure is a plain ascii file, with the name of the parameters in the first column, name of the prior distribution in the
        second column and hyperparameters in the third column.

    :param lc_instrument_supersamp: (optional, array of strings)
        Define for which lightcurve instruments super-sampling will be applied (e.g., in the case of long-cadence integrations). e.g., ``lc_instrument_supersamp = ['TESS','K2']``

    :param lc_n_supersamp: (optional, array of ints)
        Define the number of datapoints to supersample. Order should be consistent with order in ``lc_instrument_supersamp``. e.g., ``lc_n_supersamp = [20,30]``.

    :param lc_exptime_supersamp: (optional, array of floats)
        Define the exposure-time of the observations for the supersampling. Order should be consistent with order in ``lc_instrument_supersamp``. e.g., ``lc_exptime_supersamp = [0.020434,0.020434]``

    :param verbose: (optional, boolean)
        If True, all outputs of the code are printed to terminal. Default is False.

    :param matern_eps: (optional, float)
        Epsilon parameter for the Matern approximation (see celerite documentation).

    :param george_hodlr: (optional, bool)
        Flag to define if you want to use the HODLR solver for george GP's or not (see http://dfm.io/george/current/user/solvers/).

    :param pickle_encoding: (optional, string)
        Define pickle encoding in case fit was done with Python 2.7 and results are read with Python 3.

    :param non_linear_functions: (optional, dict)
        Dictionary containing any non-linear functions (`non_linear_functions['function']`) and regressors (`non_linear_functions['regressor']`) that want to be fit.

    """

    @property
    def priors(self) -> dict:
        if not hasattr(self, "_priors") or not isinstance(self._priors, dict):
            raise AttributeError("Priors have not been set.")
        return self._priors

    def data_preparation(
        self, times, instruments, linear_regressors, non_linear_functions
    ):
        """
        This function generates f useful internal arrays for this class: inames which saves the instrument names, ``global_times``
        which is a "flattened" array of the ``times`` dictionary where all the times for all instruments are stacked, instrument_indexes,
        which is a dictionary that has, for each instrument the indexes of the ``global_times`` corresponding to each instrument, lm_boolean which saves booleans for each
        instrument to indicate if there are linear regressors and lm_arguments which are the linear-regressors for each instrument.
        """
        inames = []
        for i in range(len(times)):
            if instruments[i] not in inames:
                inames.append(instruments[i])

        instrument_indexes = {}
        for instrument in inames:
            instrument_indexes[instrument] = np.where(instruments == instrument)[0]

        # Also generate lm_lc_boolean in case linear regressors were passed:
        lm_boolean = {}

        if linear_regressors is not None:
            linear_instruments = linear_regressors.keys()
            for instrument in inames:
                if instrument in linear_instruments:
                    lm_boolean[instrument] = True
                else:
                    lm_boolean[instrument] = False
        else:
            for instrument in inames:
                lm_boolean[instrument] = False

        nlm_boolean = {}
        for instrument in inames:
            if instrument in list(non_linear_functions.keys()):
                nlm_boolean[instrument] = True

            else:
                nlm_boolean[instrument] = False

        return inames, instrument_indexes, lm_boolean, nlm_boolean

    def convert_input_data(self, t, y, yerr):
        """
        This converts the input dictionaries to arrays (this is easier to handle internally within juliet; input dictionaries are just asked because
        it is easier for the user to pass them).
        """
        instruments = list(t.keys())
        all_times = np.array([])
        all_y = np.array([])
        all_yerr = np.array([])
        all_instruments = np.array([])
        for instrument in instruments:
            for i in range(len(t[instrument])):
                all_times = np.append(all_times, t[instrument][i])
                all_y = np.append(all_y, y[instrument][i])
                all_yerr = np.append(all_yerr, yerr[instrument][i])
                all_instruments = np.append(all_instruments, instrument)
        return all_times, all_y, all_yerr, all_instruments

    def convert_to_dictionary(self, t, y, yerr, instrument_indexes):
        """
        Convert data given in arrays to dictionaries for easier user usage
        """
        times = {}
        data = {}
        errors = {}
        for instrument in instrument_indexes.keys():
            times[instrument] = t[instrument_indexes[instrument]]
            data[instrument] = y[instrument_indexes[instrument]]
            errors[instrument] = yerr[instrument_indexes[instrument]]
        return times, data, errors

    def save_regressors(self, fname, GP_arguments):
        """
        This function saves the GP regressors to fname.
        """
        fout = open(fname, "w")
        for GP_instrument in GP_arguments.keys():
            GP_regressors = GP_arguments[GP_instrument]
            multi_dimensional = False
            if len(GP_regressors.shape) == 2:
                multi_dimensional = True
            if multi_dimensional:
                for i in range(GP_regressors.shape[0]):
                    for j in range(GP_regressors.shape[1]):
                        fout.write("{0:.10f} ".format(GP_regressors[i, j]))
                    fout.write("{0:}\n".format(GP_instrument))
            else:
                for i in range(GP_regressors.shape[0]):
                    fout.write(
                        "{0:.10f} {1:}\n".format(GP_regressors[i], GP_instrument)
                    )
        fout.close()

    def save_data(self, fname, t, y, yerr, instruments, lm_boolean, lm_arguments):
        """
        This function saves t,y,yerr,instruments,lm_boolean and lm_arguments data to fname.
        """
        fout = open(fname, "w")
        lm_counters = {}
        for i in range(len(t)):
            fout.write(
                "{0:.10f} {1:.10f} {2:.10f} {3:}".format(
                    t[i], y[i], yerr[i], instruments[i]
                )
            )

            if lm_boolean[instruments[i]]:
                if instruments[i] not in lm_counters.keys():
                    lm_counters[instruments[i]] = 0

                for j in range(lm_arguments[instruments[i]].shape[1]):
                    fout.write(
                        " {0:.10f}".format(
                            lm_arguments[instruments[i]][lm_counters[instruments[i]]][j]
                        )
                    )

                lm_counters[instruments[i]] += 1

            fout.write("\n")

        fout.close()

    def save_priorfile(self, fname):
        """
        This function saves a priorfile file out to fname
        """
        fout = open(fname, "w")
        for pname in self.priors.keys():
            if self.priors[pname]["distribution"].lower() != "fixed":
                value = ",".join(
                    np.array(self.priors[pname]["hyperparameters"]).astype(str)
                )
                if self.starting_point is not None:
                    fout.write(
                        "{0: <20} {1: <20} {2: <20} {3: <20}\n".format(
                            pname,
                            self.priors[pname]["distribution"],
                            value,
                            self.starting_point[pname],
                        )
                    )
                else:
                    fout.write(
                        "{0: <20} {1: <20} {2: <20}\n".format(
                            pname, self.priors[pname]["distribution"], value
                        )
                    )
            else:
                value = str(self.priors[pname]["hyperparameters"])
                fout.write(
                    "{0: <20} {1: <20} {2: <20}\n".format(
                        pname, self.priors[pname]["distribution"], value
                    )
                )
        fout.close()

    def check_global(self, name):
        for pname in self.priors.keys():
            if name in pname.split("_")[1:]:
                return True
        return False

    def append_GP(self, ndata, instrument_indexes, GP_arguments, inames):
        """
        This function appends all the GP regressors into one --- useful for the global models.
        """
        # First check if GP regressors are multi-dimensional --- check this just for the first instrument:
        if len(GP_arguments[inames[0]].shape) == 2:
            nregressors = GP_arguments[inames[0]].shape[1]
            multidimensional = True
            out = np.zeros([ndata, nregressors])
        else:
            multidimensional = False
            out = np.zeros(ndata)
        for instrument in inames:
            if multidimensional:
                out[instrument_indexes[instrument], :] = GP_arguments[instrument]
            else:
                out[instrument_indexes[instrument]] = GP_arguments[instrument]
        return out

    def sort_GP(self, dictype):
        if dictype == "lc":
            # Sort first times, fluxes, errors and the GP regressor:
            idx_sort = np.argsort(self.GP_lc_arguments["lc"][:, 0])
            self.t_lc = self.t_lc[idx_sort]
            self.y_lc = self.y_lc[idx_sort]
            self.yerr_lc = self.yerr_lc[idx_sort]

            self.GP_lc_arguments["lc"][:, 0] = self.GP_lc_arguments["lc"][idx_sort, 0]

            # Now with the sorted indices, iterate through the instrument indexes and change them according to the new
            # ordering:
            for instrument in self.inames_lc:
                new_instrument_indexes = np.zeros(
                    len(self.instrument_indexes_lc[instrument])
                )
                instrument_indexes = self.instrument_indexes_lc[instrument]
                counter = 0
                for i in instrument_indexes:
                    new_instrument_indexes[counter] = np.where(i == idx_sort)[0][0]
                    counter += 1
                self.instrument_indexes_lc[instrument] = new_instrument_indexes.astype(
                    "int"
                )
        elif dictype == "rv":
            # Sort first times, rvs, errors and the GP regressor:
            idx_sort = np.argsort(self.GP_rv_arguments["rv"][:, 0])
            self.t_rv = self.t_rv[idx_sort]
            self.y_rv = self.y_rv[idx_sort]
            self.yerr_rv = self.yerr_rv[idx_sort]

            self.GP_rv_arguments["rv"][:, 0] = self.GP_rv_arguments["rv"][idx_sort, 0]

            # Now with the sorted indices, iterate through the instrument indexes and change them according to the new
            # ordering:
            for instrument in self.inames_rv:
                new_instrument_indexes = np.zeros(
                    len(self.instrument_indexes_rv[instrument])
                )

                instrument_indexes = self.instrument_indexes_rv[instrument]

                counter = 0
                for i in instrument_indexes:
                    new_instrument_indexes[counter] = np.where(i == idx_sort)[0][0]
                    counter += 1

                self.instrument_indexes_rv[instrument] = new_instrument_indexes.astype(
                    "int"
                )

    def generate_datadict(self, dictype):
        """
        This generates the options dictionary for lightcurves, RVs, and everything else you want to fit. Useful for the
        fit, as it separaters options per instrument.

        :param dictype: (string)
            Defines the type of dictionary type. It can either be 'lc' (for the lightcurve dictionary) or 'rv' (for the
            radial-velocity one).
        """

        dictionary = {}

        if dictype == "lc":
            inames = self.inames_lc
            ninstruments = self.ninstruments_lc
            instrument_supersamp = self.lc_instrument_supersamp
            n_supersamp = self.lc_n_supersamp
            exptime_supersamp = self.lc_exptime_supersamp
            numbering_planets = self.numbering_transiting_planets
            # Check if model is global based on the input prior names. If they include as instrument "rv", set to global model:
            self.global_lc_model = self.check_global("lc")
            global_model = self.global_lc_model
            # if global_model and (self.GP_lc_arguments is not None):
            #    self.GP_lc_arguments['lc'] = self.append_GP(len(self.t_lc), self.instrument_indexes_lc, self.GP_lc_arguments, inames)
            GP_regressors = self.GP_lc_arguments

        elif dictype == "rv":
            inames = self.inames_rv
            ninstruments = self.ninstruments_rv
            instrument_supersamp = None
            n_supersamp = None
            exptime_supersamp = None
            numbering_planets = self.numbering_rv_planets
            # Check if model is global based on the input prior names. If they include as instrument "lc", set to global model:
            self.global_rv_model = self.check_global("rv")
            global_model = self.global_rv_model
            # If global_model is True, create an additional key in the GP_regressors array that will have all the GP regressors appended:
            # if global_model and (self.GP_rv_arguments is not None):
            #    self.GP_rv_arguments['rv'] = self.append_GP(len(self.t_rv), self.instrument_indexes_rv, self.GP_rv_arguments, inames)
            GP_regressors = self.GP_rv_arguments

        else:
            raise Exception(
                "INPUT ERROR: dictype not understood. Has to be either lc or rv."
            )

        assert ninstruments is not None, (
            "Number of instruments could not be determined."
        )
        assert inames is not None, "Instrument names could not be determined."
        if not hasattr(numbering_planets, "__iter__"):
            raise Exception("Numbering of planets could not be determined.")
        for i in range(ninstruments):
            instrument = inames[i]
            dictionary[instrument] = {}
            # Save if a given instrument will receive resampling (initialize this as False):
            dictionary[instrument]["resampling"] = False
            # Save if a given instrument has GP fitting ON (initialize this as False):
            dictionary[instrument]["GPDetrend"] = False
            # Save if transit fitting will be done for a given dataset/instrument (this is so users can fit photometry with, e.g., GPs):
            if dictype == "lc":
                dictionary[instrument]["TransitFit"] = False
                dictionary[instrument]["TransitFitCatwoman"] = False
                dictionary[instrument]["EclipseFit"] = False
                dictionary[instrument]["PhaseCurveFit"] = False
                dictionary[instrument]["TranEclFit"] = False

        if dictype == "lc":
            # Extract limb-darkening law and parametrization to be used to explore limb-darkeining. If no limb-darkening law was given by the user,
            # assume LD law depending on whether the user defined a prior for q1/u1 only for a given instrument (in which that instrument is set to
            # the linear law) or a prior for q1/u1 and q2/u2, in which case we assume the user wants to use a quadratic law for that instrument.
            # If user gave one limb-darkening law, assume that law for all instruments that have priors for q1/u1 and q2/u2 (if only q1/u1 is given,
            # assume linear for those instruments). If LD laws given for every instrument, extract them:

            all_ld_laws = self.ld_laws.split(",")

            if len(all_ld_laws) == 1:
                for i in range(ninstruments):
                    instrument = inames[i]
                    coeff1_given = False
                    parametrization = "kipping2013"
                    coeff2_given = False

                    for parameter in self.priors.keys():
                        if (
                            parameter[0:2] == "q1"
                            or parameter[0:2] == "u1"
                            or parameter[0:2] == "c1"
                        ):
                            if instrument in parameter.split("_")[1:]:
                                coeff1_given = True

                                # Check which parametrization/law the user is choosing:
                                if parameter[0:2] == "u1":
                                    parametrization = "normal"

                                if parameter[0:2] == "c1":
                                    parametrization = "normal-nonlinear"

                        if parameter[0:2] == "q2" or parameter[0:2] == "u2":
                            if instrument in parameter.split("_")[1:]:
                                coeff2_given = True

                    if coeff1_given and (not coeff2_given):
                        if parametrization == "normal-nonlinear":
                            dictionary[instrument]["ldlaw"] = "nonlinear"
                            dictionary[instrument]["ldparametrization"] = "normal"

                        else:
                            dictionary[instrument]["ldlaw"] = "linear"
                            dictionary[instrument]["ldparametrization"] = (
                                parametrization
                            )

                    elif coeff1_given and coeff2_given:
                        dictionary[instrument]["ldlaw"] = (
                            (all_ld_laws[0].split("-")[-1]).split()[0].lower()
                        )

                        dictionary[instrument]["ldparametrization"] = parametrization

                    elif (not coeff1_given) and coeff2_given:
                        raise Exception(
                            "INPUT ERROR: it appears q1/u1 for instrument "
                            + instrument
                            + " was not defined (but q2/u2 was) in the prior file."
                        )

                    elif (not coeff1_given) and (not coeff2_given):
                        dictionary[instrument]["ldlaw"] = "none"
                        dictionary[instrument]["ldparametrization"] = "none"

            else:
                # Extract limb-darkening law from user-input:
                for ld_law in all_ld_laws:
                    instrument, ld = ld_law.split("-")

                    dictionary[instrument.split()[0]]["ldlaw"] = ld.split()[0].lower()

                # Now extract parametrization for each instrument depending on user priors file/dictionary:
                for i in range(ninstruments):
                    instrument = inames[i]
                    coeff1_given = False
                    parametrization = "kipping2013"
                    coeff2_given = False

                    for parameter in self.priors.keys():
                        if (
                            parameter[0:2] == "q1"
                            or parameter[0:2] == "u1"
                            or parameter[0:2] == "c1"
                        ):
                            if instrument in parameter.split("_")[1:]:
                                coeff1_given = True

                                # Check which parametrization the user is choosing:
                                if parameter[0:2] == "u1":
                                    parametrization = "normal"

                                if parameter[0:2] == "c1":
                                    parametrization = "normal-nonlinear"

                        if parameter[0:2] == "q2" or parameter[0:2] == "u2":
                            if instrument in parameter.split("_")[1:]:
                                coeff2_given = True

                    if coeff1_given and (not coeff2_given):
                        if parametrization == "normal-nonlinear":
                            dictionary[instrument]["ldparametrization"] = "normal"

                        else:
                            dictionary[instrument]["ldparametrization"] = (
                                parametrization
                            )

                    elif coeff1_given and coeff2_given:
                        dictionary[instrument]["ldparametrization"] = parametrization

                    elif (not coeff1_given) and coeff2_given:
                        raise Exception(
                            "INPUT ERROR: it appears q1/u1 for instrument "
                            + instrument
                            + " was not defined (but q2/u2 was) in the prior file."
                        )

                    elif (not coeff1_given) and (not coeff2_given):
                        dictionary[instrument]["ldlaw"] = "none"
                        dictionary[instrument]["ldparametrization"] = "none"

        # Extract supersampling parameters if given.
        # For now this only allows inputs from lightcurves; TODO: add supersampling for RVs.
        if instrument_supersamp is not None and dictype == "lc":
            assert n_supersamp is not None, (
                "Number of supersamplings not given for lightcurve resampling."
            )
            assert exptime_supersamp is not None, (
                "Exposure time for supersampling not given for lightcurve resampling."
            )
            for i in range(len(instrument_supersamp)):
                if self.verbose:
                    print(
                        "\t Resampling detected for instrument ",
                        instrument_supersamp[i],
                    )
                dictionary[instrument_supersamp[i]]["resampling"] = True
                dictionary[instrument_supersamp[i]]["nresampling"] = n_supersamp[i]
                dictionary[instrument_supersamp[i]]["exptimeresampling"] = (
                    exptime_supersamp[i]
                )

        # Check that user gave periods in chronological order. If not, raise an exception, tell the user and stop this madness.
        # Note we only check if fixed or normal/truncated normal. In the uniform or log-uniform cases, we trust the user knows
        # what they are doing. We don't touch the Beta case because that would be nuts to put in a prior anyways most of the time.
        cp_pnumber = np.array([])
        cp_period = np.array([])
        for pri in self.priors.keys():
            if pri[0:2] == "P_":
                if self.priors[pri]["distribution"].lower() in [
                    "normal",
                    "truncated normal",
                ]:
                    cp_pnumber = np.append(cp_pnumber, int(pri.split("_")[-1][1:]))
                    cp_period = np.append(
                        cp_period, self.priors[pri]["hyperparameters"][0]
                    )
                elif self.priors[pri]["distribution"].lower() == "fixed":
                    cp_pnumber = np.append(cp_pnumber, int(pri.split("_")[-1][1:]))
                    cp_period = np.append(
                        cp_period, self.priors[pri]["hyperparameters"]
                    )
        if len(cp_period) > 1:
            idx = np.argsort(cp_pnumber)
            cP = cp_period[idx[0]]
            cP_idx = cp_pnumber[idx[0]]
            for cidx in idx[1:]:
                P = cp_period[cidx]
                if P > cP:
                    cP = P
                    cP_idx = cp_pnumber[cidx]
                else:
                    print("\n")
                    raise Exception(
                        "INPUT ERROR: planetary periods in the priors are not ordered in chronological order. "
                        + "Planet p{0:} has a period of {1:} days, while planet p{2:} has a period of {3:} days (P_p{0:}<P_p{2:}).".format(
                            int(cp_pnumber[cidx]), P, int(cP_idx), cP
                        )
                    )

        # Now, if generating lightcurve dict, check whether for some photometric instruments only photometry, and not a
        # transit, will be fit. This is based on whether the user gave limb-darkening coefficients for a given photometric
        # instrument or not. If given, transit is fit. If not, no transit is fit. At the same time check if user wants to
        # fit TTVs for the desired instrument. For this latter, initialize as false for each instrument and only change to
        # true if the priors are found:
        if dictype == "lc":
            for i in range(ninstruments):
                dictionary[inames[i]]["TTVs"] = {}
                for pi in numbering_planets:
                    dictionary[inames[i]]["TTVs"][pi] = {}
                    dictionary[inames[i]]["TTVs"][pi]["status"] = False
                    dictionary[inames[i]]["TTVs"][pi]["parametrization"] = "dt"
                    dictionary[inames[i]]["TTVs"][pi]["transit_number"] = []

                for pri in self.priors.keys():
                    if pri[0:2] == "q1" or pri[0:2] == "u1" or pri[0:2] == "c1":
                        if inames[i] in pri.split("_"):
                            dictionary[inames[i]]["TransitFit"] = True

                            if self.verbose:
                                print(
                                    "\t Transit fit detected for instrument ", inames[i]
                                )

                    if pri[0:2] == "p1":
                        # If CW defined on instrument, or, there's a single CW for all instruments:
                        if (inames[i] in pri.split("_")) or (len(pri.split("_")) == 2):
                            dictionary[inames[i]]["TransitFit"] = True
                            dictionary[inames[i]]["TransitFitCatwoman"] = True

                            if self.verbose:
                                print(
                                    "\t Transit (catwoman) fit detected for instrument ",
                                    inames[i],
                                )

                    if pri[0:11] == "phaseoffset":
                        # If phase-offset defined on instrument, or, there's a single one for all instruments:
                        if (inames[i] in pri.split("_")) or (len(pri.split("_")) == 2):
                            dictionary[inames[i]]["PhaseCurveFit"] = True

                            if self.verbose:
                                print(
                                    "\t Phase curve fit detected for instrument ",
                                    inames[i],
                                )

                    if pri[0:2] == "fp":
                        # If an eclipse for instrument or there's a single depth for all instruments:
                        if (inames[i] in pri.split("_")) or (len(pri.split("_")) == 2):
                            dictionary[inames[i]]["EclipseFit"] = True

                            if self.verbose:
                                print(
                                    "\t Eclipse fit detected for instrument ", inames[i]
                                )

                    if pri[0:2] == "dt" or pri[0:2] == "T_":
                        planet_number, instrument, ntransit = pri.split("_")[1:]

                        if inames[i] == instrument:
                            if pri[0:2] == "T_":
                                dictionary[inames[i]]["TTVs"][int(planet_number[1:])][
                                    "parametrization"
                                ] = "T"

                            dictionary[inames[i]]["TTVs"][int(planet_number[1:])][
                                "status"
                            ] = True
                            dictionary[inames[i]]["TTVs"][int(planet_number[1:])][
                                "transit_number"
                            ].append(int(ntransit))

                if (
                    dictionary[inames[i]]["TransitFit"]
                    and dictionary[inames[i]]["EclipseFit"]
                ):
                    dictionary[inames[i]]["TranEclFit"] = True
                    dictionary[inames[i]]["TransitFit"] = False
                    dictionary[inames[i]]["EclipseFit"] = False

                    if self.verbose:
                        print(
                            "\t Joint Transit and Eclipse fit detected for instrument ",
                            inames[i],
                        )

            for pi in numbering_planets:
                for i in range(ninstruments):
                    if dictionary[inames[i]]["TTVs"][pi]["status"]:
                        dictionary[inames[i]]["TTVs"][pi]["totalTTVtransits"] = len(
                            dictionary[inames[i]]["TTVs"][pi]["transit_number"]
                        )

        # Now, implement noise models for each of the instrument. First check if model should be global or instrument-by-instrument,
        # based on the input instruments given for the GP regressors.
        if global_model:
            dictionary["global_model"] = {}
            if GP_regressors is not None:
                dictionary["global_model"]["GPDetrend"] = True
                dictionary["global_model"]["noise_model"] = gaussian_process(
                    self,
                    model_type=dictype,
                    instrument=dictype,
                    matern_eps=self.matern_eps,
                    george_hodlr=self.george_hodlr,
                )
                if not dictionary["global_model"]["noise_model"].isInit:
                    # If not initiated, most likely kernel is a celerite one. Reorder times, values, etc. This is OK --- is expected:
                    if dictype == "lc":
                        self.sort_GP("lc")
                    elif dictype == "rv":
                        self.sort_GP("rv")
                    # Try again:
                    dictionary["global_model"]["noise_model"] = gaussian_process(
                        self,
                        model_type=dictype,
                        instrument=dictype,
                        matern_eps=self.matern_eps,
                        george_hodlr=self.george_hodlr,
                    )
                    if not dictionary["global_model"]["noise_model"].isInit:
                        # Check, blame the user:
                        raise Exception(
                            "INPUT ERROR: GP initialization for object for "
                            + dictype
                            + " global kernel failed."
                        )
            else:
                dictionary["global_model"]["GPDetrend"] = False
        else:
            for i in range(ninstruments):
                instrument = inames[i]

                if (GP_regressors is not None) and (instrument in GP_regressors.keys()):
                    dictionary[instrument]["GPDetrend"] = True
                    dictionary[instrument]["noise_model"] = gaussian_process(
                        self,
                        model_type=dictype,
                        instrument=instrument,
                        matern_eps=self.matern_eps,
                        george_hodlr=self.george_hodlr,
                    )
                    if not dictionary[instrument]["noise_model"].isInit:
                        # Blame the user, although perhaps we could simply solve this as for the global modelling?:
                        raise Exception(
                            "INPUT ERROR: GP regressors for instrument "
                            + instrument
                            + " use celerite, and are not in ascending or descending order. Please, give the input in those orders --- it will not work othersie."
                        )

        # Check which eccentricity parametrization is going to be used for each planet in the juliet numbering scheme.
        # 0 = ecc, omega  1: ecosomega,esinomega  2: sqrt(e)cosomega, sqrt(e)sinomega
        dictionary["ecc_parametrization"] = {}
        if dictype == "lc":
            dictionary["efficient_bp"] = {}
        for i in numbering_planets:
            if "ecosomega_p" + str(i) in self.priors.keys():
                dictionary["ecc_parametrization"][i] = 1
                if self.verbose:
                    print(
                        "\t >> ecosomega,esinomega parametrization detected for "
                        + dictype
                        + " planet p"
                        + str(i)
                    )
            elif "secosomega_p" + str(i) in self.priors.keys():
                dictionary["ecc_parametrization"][i] = 2
                if self.verbose:
                    print(
                        "\t >> sqrt(e)cosomega, sqrt(e)sinomega parametrization detected for "
                        + dictype
                        + " planet p"
                        + str(i)
                    )
            else:
                dictionary["ecc_parametrization"][i] = 0
                if self.verbose:
                    print(
                        "\t >> ecc,omega parametrization detected for "
                        + dictype
                        + " planet p"
                        + str(i)
                    )
            if dictype == "lc":
                # Check if Espinoza (2018), (b,p) parametrization is on:
                if "r1_p" + str(i) in self.priors.keys():
                    dictionary["efficient_bp"][i] = True
                    if self.verbose:
                        print(
                            "\t >> (b,p) parametrization detected for "
                            + dictype
                            + " planet p"
                            + str(i)
                        )
                else:
                    dictionary["efficient_bp"][i] = False

        # Check if stellar density is in the prior. Allow providing r_star+m_star
        # instead of rho: if both are present we will compute rho from them.
        if dictype == "lc":
            dictionary["fitrho"] = False
            if ("rho" in self.priors.keys()) or (
                ("r_star" in self.priors.keys()) and ("m_star" in self.priors.keys())
            ):
                dictionary["fitrho"] = True

        # For RV dictionaries, check if RV trend will be fitted:
        if dictype == "rv":
            dictionary["fitrvline"] = False
            dictionary["fitrvquad"] = False
            if "rv_slope" in self.priors.keys():
                if "rv_quad" in self.priors.keys():
                    dictionary["fitrvquad"] = True
                    if self.verbose:
                        print("\t Fitting quadratic trend to RVs.")
                else:
                    dictionary["fitrvline"] = True
                    if self.verbose:
                        print("\t Fitting linear trend to RVs.")

        # Save dictionary to self:
        if dictype == "lc":
            self.lc_options = dictionary
        elif dictype == "rv":
            self.rv_options = dictionary
        else:
            raise Exception(
                "INPUT ERROR: dictype not understood. Has to be either lc or rv."
            )

    def set_lc_data(
        self,
        t_lc,
        y_lc,
        yerr_lc,
        instruments_lc,
        instrument_indexes_lc,
        ninstruments_lc,
        inames_lc,
        lm_lc_boolean,
        lm_lc_arguments,
        nlm_lc_boolean,
    ):
        self.t_lc = t_lc.astype("float64")
        self.y_lc = y_lc
        self.yerr_lc = yerr_lc
        self.inames_lc = inames_lc
        self.instruments_lc = instruments_lc
        self.ninstruments_lc = ninstruments_lc
        self.instrument_indexes_lc = instrument_indexes_lc
        self.lm_lc_boolean = lm_lc_boolean
        self.lm_lc_arguments = lm_lc_arguments
        self.nlm_lc_boolean = nlm_lc_boolean
        self.lc_data = True

    def set_rv_data(
        self,
        t_rv,
        y_rv,
        yerr_rv,
        instruments_rv,
        instrument_indexes_rv,
        ninstruments_rv,
        inames_rv,
        lm_rv_boolean,
        lm_rv_arguments,
        nlm_rv_boolean,
    ):
        self.t_rv = t_rv.astype("float64")
        self.y_rv = y_rv
        self.yerr_rv = yerr_rv
        self.inames_rv = inames_rv
        self.instruments_rv = instruments_rv
        self.ninstruments_rv = ninstruments_rv
        self.instrument_indexes_rv = instrument_indexes_rv
        self.lm_rv_boolean = lm_rv_boolean
        self.lm_rv_arguments = lm_rv_arguments
        self.nlm_rv_boolean = nlm_rv_boolean
        self.rv_data = True

    def save(self):
        if self.out_folder[-1] != "/":
            self.out_folder = self.out_folder + "/"

        # First, save lightcurve data:
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder, exist_ok=True)
        if not os.path.exists(self.out_folder + "lc.dat"):
            if self.lcfilename is not None:
                os.system("cp " + self.lcfilename + " " + self.out_folder + "lc.dat")
            elif self.t_lc is not None:
                self.save_data(
                    self.out_folder + "lc.dat",
                    self.t_lc,
                    self.y_lc,
                    self.yerr_lc,
                    self.instruments_lc,
                    self.lm_lc_boolean,
                    self.lm_lc_arguments,
                )

        # Now radial-velocity data:
        if not os.path.exists(self.out_folder + "rvs.dat"):
            if self.rvfilename is not None:
                os.system("cp " + self.rvfilename + " " + self.out_folder + "rvs.dat")
            elif self.t_rv is not None:
                self.save_data(
                    self.out_folder + "rvs.dat",
                    self.t_rv,
                    self.y_rv,
                    self.yerr_rv,
                    self.instruments_rv,
                    self.lm_rv_boolean,
                    self.lm_rv_arguments,
                )

        # Next, save GP regressors:
        if not os.path.exists(self.out_folder + "GP_lc_regressors.dat"):
            if self.GPlceparamfile is not None:
                os.system(
                    "cp "
                    + self.GPlceparamfile
                    + " "
                    + self.out_folder
                    + "GP_lc_regressors.dat"
                )
            elif self.GP_lc_arguments is not None:
                self.save_regressors(
                    self.out_folder + "GP_lc_regressors.dat", self.GP_lc_arguments
                )
        if not os.path.exists(self.out_folder + "GP_rv_regressors.dat"):
            if self.GPrveparamfile is not None:
                os.system(
                    "cp "
                    + self.GPrveparamfile
                    + " "
                    + self.out_folder
                    + "GP_rv_regressors.dat"
                )
            elif self.GP_rv_arguments is not None:
                self.save_regressors(
                    self.out_folder + "GP_rv_regressors.dat", self.GP_rv_arguments
                )

        # Finally, save LM regressors if any:
        if not os.path.exists(self.out_folder + "LM_lc_regressors.dat"):
            if self.LMlceparamfile is not None:
                os.system(
                    "cp "
                    + self.LMlceparamfile
                    + " "
                    + self.out_folder
                    + "LM_lc_regressors.dat"
                )
            elif self.LM_lc_arguments is not None:
                self.save_regressors(
                    self.out_folder + "LM_lc_regressors.dat", self.LM_lc_arguments
                )
        if not os.path.exists(self.out_folder + "LM_rv_regressors.dat"):
            if self.LMrveparamfile is not None:
                os.system(
                    "cp "
                    + self.LMrveparamfile
                    + " "
                    + self.out_folder
                    + "LM_rv_regressors.dat"
                )
            elif self.LM_rv_arguments is not None:
                self.save_regressors(
                    self.out_folder + "LM_rv_regressors.dat", self.LM_rv_arguments
                )

        # Save priors:
        if not os.path.exists(self.out_folder + "priors.dat"):
            self.prior_fname = self.out_folder + "priors.dat"
            self.save_priorfile(self.out_folder + "priors.dat")

    def fit(self, **kwargs):
        """
        Perhaps the most important function of the juliet data object. This function fits your data using the nested
        sampler of choice. This returns a results object which contains all the posteriors information.
        """
        # Note this return call creates a fit *object* with the current data object. The fit class definition is below.
        return fit(self, **kwargs)

    def __init__(
        self,
        priors=None,
        starting_point=None,
        input_folder=None,
        t_lc=None,
        y_lc=None,
        yerr_lc=None,
        t_rv=None,
        y_rv=None,
        yerr_rv=None,
        GP_regressors_lc=None,
        linear_regressors_lc=None,
        GP_regressors_rv=None,
        linear_regressors_rv=None,
        out_folder=None,
        lcfilename=None,
        rvfilename=None,
        GPlceparamfile=None,
        GPrveparamfile=None,
        LMlceparamfile=None,
        LMrveparamfile=None,
        lctimedef="TDB",
        rvtimedef="UTC",
        ld_laws="quadratic",
        priorfile=None,
        lc_n_supersamp=None,
        lc_exptime_supersamp=None,
        lc_instrument_supersamp=None,
        mag_to_flux=True,
        verbose=False,
        matern_eps=0.01,
        george_hodlr=True,
        pickle_encoding=None,
        non_linear_functions={},
        extra_loglikelihood=None,
    ):
        self.lcfilename = lcfilename
        self.rvfilename = rvfilename
        self.GPlceparamfile = GPlceparamfile
        self.GPrveparamfile = GPrveparamfile
        self.LMlceparamfile = LMlceparamfile
        self.LMrveparamfile = LMrveparamfile
        self.verbose = verbose
        self.pickle_encoding = pickle_encoding
        self.starting_point = starting_point

        # GP options:
        self.matern_eps = matern_eps  # Epsilon parameter for celerite Matern32Term
        self.george_hodlr = george_hodlr  # Wheter to use HODLR solver or not (see: http://dfm.io/george/current/user/solvers/)

        # Non-linear function options:
        self.non_linear_functions = non_linear_functions
        self.extra_loglikelihood = extra_loglikelihood

        if extra_loglikelihood is None:
            self.extra_loglikelihood_boolean = False

        else:
            self.extra_loglikelihood_boolean = True

        # Initialize data options for lightcurves:
        self.t_lc = None
        self.y_lc = None
        self.yerr_lc = None
        self.instruments_lc = None
        self.ninstruments_lc = None
        self.inames_lc = None
        self.instrument_indexes_lc = None
        self.lm_lc_boolean = None
        self.lm_lc_arguments = None
        self.GP_lc_arguments = None
        self.LM_lc_arguments = None
        self.lctimedef = lctimedef
        self.ld_laws = ld_laws
        self.lc_n_supersamp = lc_n_supersamp
        self.lc_exptime_supersamp = lc_exptime_supersamp
        self.lc_instrument_supersamp = lc_instrument_supersamp
        self.lc_data = False
        self.global_lc_model = False
        self.lc_options = {}

        # Initialize data options for RVs:
        self.t_rv = None
        self.y_rv = None
        self.yerr_rv = None
        self.instruments_rv = None
        self.ninstruments_rv = None
        self.inames_rv = None
        self.instrument_indexes_rv = None
        self.lm_rv_boolean = None
        self.lm_rv_arguments = None
        self.GP_rv_arguments = None
        self.LM_rv_arguments = None
        self.rvtimedef = rvtimedef
        self.rv_data = False
        self.global_rv_model = False
        self.rv_options = {}

        self.out_folder = None

        if input_folder is not None:
            if input_folder[-1] != "/":
                self.input_folder = input_folder + "/"
            else:
                self.input_folder = input_folder
            if os.path.exists(self.input_folder + "lc.dat"):
                lcfilename = self.input_folder + "lc.dat"
            if os.path.exists(self.input_folder + "rvs.dat"):
                rvfilename = self.input_folder + "rvs.dat"
            if (not os.path.exists(self.input_folder + "lc.dat")) and (
                not os.path.exists(self.input_folder + "rvs.dat")
            ):
                raise Exception(
                    "INPUT ERROR: No lightcurve data file (lc.dat) or radial-velocity data file (rvs.dat) found in folder "
                    + self.input_folder
                    + ". \n Create them and try again. For details, check juliet.load?"
                )
            if os.path.exists(self.input_folder + "GP_lc_regressors.dat"):
                GPlceparamfile = self.input_folder + "GP_lc_regressors.dat"
            if os.path.exists(self.input_folder + "GP_rv_regressors.dat"):
                GPrveparamfile = self.input_folder + "GP_rv_regressors.dat"
            if os.path.exists(self.input_folder + "LM_lc_regressors.dat"):
                LMlceparamfile = self.input_folder + "LM_lc_regressors.dat"
            if os.path.exists(self.input_folder + "LM_rv_regressors.dat"):
                LMrveparamfile = self.input_folder + "LM_rv_regressors.dat"
            if os.path.exists(self.input_folder + "priors.dat"):
                priors = self.input_folder + "priors.dat"
            else:
                raise Exception(
                    "INPUT ERROR: Prior file (priors.dat) not found in folder "
                    + self.input_folder
                    + "."
                    + "Create it and try again. For details, check juliet.load?"
                )
            # If there is an input folder and no out_folder, then simply set the out_folder as the input_folder
            # for ease in the later functions (more for replotting purposes)
            # So, one can simply do this to obtain the posteriors:
            # > dataset = juliet.load(input_folder=folder) # to reload the priors, data, etc.
            # > results = dataset.fit() # to obtain the results already found in the input_folder
            # > posteriors = results.posteriors
            if out_folder is None:
                self.out_folder = self.input_folder
        else:
            self.input_folder = None

        if isinstance(priors, str):
            self.prior_fname = priors
            (
                priors,
                n_transit,
                n_rv,
                numbering_transit,
                numbering_rv,
                n_params,
                starting_point,
            ) = readpriors(priors)
            # Save information stored in the prior: the dictionary, number of transiting planets,
            # number of RV planets, numbering of transiting and rv planets (e.g., if p1 and p3 transit
            # and all of them are RV planets, numbering_transit = [1,3] and numbering_rv = [1,2,3]).
            # Save also number of *free* parameters (FIXED don't count here).
            self._priors = priors
            self.n_transiting_planets = n_transit
            self.n_rv_planets = n_rv
            self.numbering_transiting_planets = numbering_transit
            self.numbering_rv_planets = numbering_rv
            self.nparams = n_params
            self.starting_point = starting_point

        elif isinstance(priors, dict):
            # Dictionary was passed, so save it.
            self._priors = priors
            # Extract same info as above if-statement but using only the dictionary:
            n_transit, n_rv, numbering_transit, numbering_rv, n_params = readpriors(
                priors
            )
            # Save information:
            self.n_transiting_planets = n_transit
            self.n_rv_planets = n_rv
            self.numbering_transiting_planets = numbering_transit
            self.numbering_rv_planets = numbering_rv
            self.nparams = n_params
            self.prior_fname = None
        else:
            raise Exception(
                "INPUT ERROR: Prior file is not a string or a dictionary (and it has to). Do juliet.load? for details."
            )

        # Define cases in which data is given through files:
        if t_lc is None:
            if lcfilename is not None:
                (
                    t_lc,
                    y_lc,
                    yerr_lc,
                    instruments_lc,
                    instrument_indexes_lc,
                    ninstruments_lc,
                    inames_lc,
                    lm_lc_boolean,
                    lm_lc_arguments,
                ) = read_data(lcfilename)

                # Set null boolean for now for non-linear in data:
                nlm_lc_boolean = {}
                for k in inames_lc:
                    nlm_lc_boolean[k] = False

                # Save data to object:

                self.set_lc_data(
                    t_lc,
                    y_lc,
                    yerr_lc,
                    instruments_lc,
                    instrument_indexes_lc,
                    ninstruments_lc,
                    inames_lc,
                    lm_lc_boolean,
                    lm_lc_arguments,
                    nlm_lc_boolean,
                )

        if t_rv is None:
            if rvfilename is not None:
                (
                    t_rv,
                    y_rv,
                    yerr_rv,
                    instruments_rv,
                    instrument_indexes_rv,
                    ninstruments_rv,
                    inames_rv,
                    lm_rv_boolean,
                    lm_rv_arguments,
                ) = read_data(rvfilename)

                # Set null boolean for now for non-linear in data:
                nlm_rv_boolean = {}
                for k in inames_rv:
                    nlm_rv_boolean[k] = False

                # Save data to object:
                self.set_rv_data(
                    t_rv,
                    y_rv,
                    yerr_rv,
                    instruments_rv,
                    instrument_indexes_rv,
                    ninstruments_rv,
                    inames_rv,
                    lm_rv_boolean,
                    lm_rv_arguments,
                    nlm_rv_boolean,
                )

        if t_lc is None and t_rv is None:
            if (lcfilename is None) and (rvfilename is None):
                raise Exception(
                    "INPUT ERROR: No complete dataset (photometric or radial-velocity) given.\n"
                    + " Make sure to feed times (t_lc and/or t_rv), values (y_lc and/or y_rv), \n"
                    + " errors (yerr_lc and/or yerr_rv)."
                )

        # Read GP regressors if given through files or arrays. The former takes priority. First lightcurve:
        if GPlceparamfile is not None:
            self.GP_lc_arguments, self.global_lc_model = readGPeparams(GPlceparamfile)
        elif GP_regressors_lc is not None:
            self.GP_lc_arguments = GP_regressors_lc
            instruments = set(list(self.GP_lc_arguments.keys()))

        # Same thing for RVs:
        if GPrveparamfile is not None:
            self.GP_rv_arguments, self.global_rv_model = readGPeparams(GPrveparamfile)
        elif GP_regressors_rv is not None:
            self.GP_rv_arguments = GP_regressors_rv
            instruments = set(list(self.GP_rv_arguments.keys()))

        # Same thing for linear regressors in case they were given in a separate file:
        if LMlceparamfile is not None:
            LM_lc_arguments, _ = readGPeparams(LMlceparamfile)
            for lmi in list(LM_lc_arguments.keys()):
                lm_lc_boolean[lmi] = True
                lm_lc_arguments[lmi] = LM_lc_arguments[lmi]

        # Same thing for RVs:
        if LMrveparamfile is not None:
            LM_rv_arguments, _ = readGPeparams(LMrveparamfile)
            for lmi in list(LM_rv_arguments.keys()):
                lm_rv_boolean[lmi] = True
                lm_rv_arguments[lmi] = LM_rv_arguments[lmi]

        # If data given through direct arrays (i.e., not data files), generate some useful internal lightcurve arrays: inames_lc, which have the different lightcurve instrument names,
        # instrument_indexes_lc (dictionary that holds, for each instrument, the indexes that have the time/lightcurve data for that particular instrument), lm_lc_boolean (dictionary of
        # booleans; True for an instrument if it has linear regressors), lm_lc_arguments (dictionary containing the linear regressors for each instrument), etc.:
        if (lcfilename is None) and (t_lc is not None):
            # First check user gave all data:
            input_error_catcher(t_lc, y_lc, yerr_lc, "lightcurve")
            # Convert times to float64 (batman really hates non-float64 inputs):
            for instrument in t_lc.keys():
                t_lc[instrument] = t_lc[instrument].astype("float64")
            # Create global arrays:
            tglobal_lc, yglobal_lc, yglobalerr_lc, instruments_lc = (
                self.convert_input_data(t_lc, y_lc, yerr_lc)
            )
            # Save data in a format useful for global modelling:
            inames_lc, instrument_indexes_lc, lm_lc_boolean, nlm_lc_boolean = (
                self.data_preparation(
                    tglobal_lc,
                    instruments_lc,
                    linear_regressors_lc,
                    non_linear_functions,
                )
            )
            lm_lc_arguments = linear_regressors_lc
            ninstruments_lc = len(inames_lc)

            # Save data to object:
            self.set_lc_data(
                tglobal_lc,
                yglobal_lc,
                yglobalerr_lc,
                instruments_lc,
                instrument_indexes_lc,
                ninstruments_lc,
                inames_lc,
                lm_lc_boolean,
                lm_lc_arguments,
                nlm_lc_boolean,
            )

            # Save input dictionaries:
            self.times_lc = t_lc
            self.data_lc = y_lc
            self.errors_lc = yerr_lc
        elif t_lc is not None:
            # In this case, convert data in array-form to dictionaries, save them so user can easily use them:
            times_lc, data_lc, errors_lc = self.convert_to_dictionary(
                t_lc, y_lc, yerr_lc, instrument_indexes_lc
            )
            self.times_lc = times_lc
            self.data_lc = data_lc
            self.errors_lc = errors_lc

        # Same for radial-velocity data:
        if (rvfilename is None) and (t_rv is not None):
            input_error_catcher(t_rv, y_rv, yerr_rv, "radial-velocity")
            tglobal_rv, yglobal_rv, yglobalerr_rv, instruments_rv = (
                self.convert_input_data(t_rv, y_rv, yerr_rv)
            )
            inames_rv, instrument_indexes_rv, lm_rv_boolean, nlm_rv_boolean = (
                self.data_preparation(
                    tglobal_rv,
                    instruments_rv,
                    linear_regressors_rv,
                    non_linear_functions,
                )
            )
            lm_rv_arguments = linear_regressors_rv
            ninstruments_rv = len(inames_rv)

            # Save data to object:

            self.set_rv_data(
                tglobal_rv,
                yglobal_rv,
                yglobalerr_rv,
                instruments_rv,
                instrument_indexes_rv,
                ninstruments_rv,
                inames_rv,
                lm_rv_boolean,
                lm_rv_arguments,
                nlm_rv_boolean,
            )

            # Save input dictionaries:
            self.times_rv = t_rv
            self.data_rv = y_rv
            self.errors_rv = yerr_rv
        elif t_rv is not None:
            # In this case, convert data in array-form to dictionaries, save them so user can easily use them:
            times_rv, data_rv, errors_rv = self.convert_to_dictionary(
                t_rv, y_rv, yerr_rv, instrument_indexes_rv
            )
            self.times_rv = times_rv
            self.data_rv = data_rv
            self.errors_rv = errors_rv

        # If out_folder does not exist, create it, and save data to it:
        if out_folder is not None:
            self.out_folder = out_folder
            self.save()

        # Finally, generate datadicts, that will save information about the fits, including gaussian_process objects for each instrument that requires it
        # (including the case of global models):
        if t_lc is not None:
            self.generate_datadict("lc")
        if t_rv is not None:
            self.generate_datadict("rv")
