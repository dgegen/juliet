import numpy as np


try:
    import george
except ImportError:
    print(
        "Warning: no george installation found. No non-celerite GPs will be able to be used"
    )

try:
    import celerite
    from celerite import terms

    # This class was written by Daniel Foreman-Mackey for his paper:
    # https://github.com/dfm/celerite/blob/master/paper/figures/rotation/rotation.ipynb
    class RotationTerm(terms.Term):
        parameter_names = ("log_amp", "log_timescale", "log_period", "log_factor")

        def get_real_coefficients(self, params):
            log_amp, log_timescale, log_period, log_factor = params
            f = np.exp(log_factor)
            return (
                np.exp(log_amp) * (1.0 + f) / (2.0 + f),
                np.exp(-log_timescale),
            )

        def get_complex_coefficients(self, params):
            log_amp, log_timescale, log_period, log_factor = params
            f = np.exp(log_factor)
            return (
                np.exp(log_amp) / (2.0 + f),
                0.0,
                np.exp(-log_timescale),
                2 * np.pi * np.exp(-log_period),
            )
except ImportError:
    print(
        "Warning: no celerite installation found. No celerite GPs will be able to be used"
    )


__all__ = ["gaussian_process"]

class gaussian_process(object):
    """
    Given a juliet data object (created via juliet.load), a model type (i.e., is this a GP for a RV or lightcurve dataset) and
    an instrument name, this object generates a Gaussian Process (GP) object to use within the juliet library. Example usage:

               >>> GPmodel = juliet.gaussian_process(data, model_type = 'lc', instrument = 'TESS')

    :param data (juliet.load object)
        Object containing all the information about the current dataset. This will help in determining the type of kernel
        the input instrument has and also if the instrument has any errors associated with it to initialize the kernel.

    :param model_type: (string)
        A string defining the type of data the GP will be modelling. Can be either ``lc`` (for photometry) or ``rv`` (for radial-velocities).

    :param instrument: (string)
        A string indicating the name of the instrument the GP is being applied to. This string simplifies cross-talk with juliet's ``posteriors``
        dictionary.

    :param george_hodlr: (optional, boolean)
        If True, this uses George's HODLR solver (faster).

    """

    def get_kernel_name(self, priors):
        # First, check all the GP variables in the priors file that are of the form GP_variable_instrument1_instrument2_...:
        variables_that_match = []
        for pname in priors.keys():
            vec = pname.split("_")
            if (vec[0] == "GP") and (self.instrument in vec):
                variables_that_match = variables_that_match + [vec[1]]
        # Now we have all the variables that match the current instrument in variables_that_match. Check which of the
        # implemented GP models gives a perfect match to all the variables; that will give us the name of the kernel:
        n_variables_that_match = len(variables_that_match)
        if n_variables_that_match == 0:
            raise Exception(
                "Input error: it seems instrument "
                + self.instrument
                + " has no defined priors in the prior file for a Gaussian Process. Check the prior file and try again."
            )

        for kernel_name in self.all_kernel_variables.keys():
            counter = 0
            for variable_name in self.all_kernel_variables[kernel_name]:
                if variable_name in variables_that_match:
                    counter += 1
            if (n_variables_that_match == counter) and (
                len(self.all_kernel_variables[kernel_name]) == n_variables_that_match
            ):
                return kernel_name

    def init_GP(self):
        if self.use_celerite:
            self.GP = celerite.GP(self.kernel, mean=0.0)
        else:
            if self.global_GP:
                if self.george_hodlr:
                    self.GP = george.GP(
                        self.kernel,
                        mean=0.0,
                        fit_mean=False,
                        fit_white_noise=False,
                        solver=george.HODLRSolver,
                    )
                else:
                    self.GP = george.GP(
                        self.kernel, mean=0.0, fit_mean=False, fit_white_noise=False
                    )
            else:
                # (Note no jitter kernel is given, as with george one defines this in the george.GP call):
                jitter_term = george.modeling.ConstantModel(1.0)
                if self.george_hodlr:
                    self.GP = george.GP(
                        self.kernel,
                        mean=0.0,
                        fit_mean=False,
                        white_noise=jitter_term,
                        fit_white_noise=True,
                        solver=george.HODLRSolver,
                    )
                else:
                    self.GP = george.GP(
                        self.kernel,
                        mean=0.0,
                        fit_mean=False,
                        white_noise=jitter_term,
                        fit_white_noise=True,
                    )
        self.compute_GP()

    def compute_GP(self, X=None):
        if self.yerr is not None:
            if X is None:
                self.GP.compute(self.X, yerr=self.yerr)
            else:
                self.GP.compute(X, yerr=self.yerr)
        else:
            if X is None:
                self.GP.compute(self.X)
            else:
                self.GP.compute(X)

    def set_input_instrument(self, input_variables):
        # This function sets the "input instrument" (self.input_instrument) name for each variable (self.variables).
        # If, for example, GP_Prot_TESS_K2_rv and GP_Gamma_TESS, and self.variables = ['Prot','Gamma'],
        # then self.input_instrument = ['TESS_K2_rv','TESS'].
        for i in range(len(self.variables)):
            GPvariable = self.variables[i]
            for pnames in input_variables.keys():
                vec = pnames.split("_")
                if (
                    (vec[0] == "GP")
                    and (GPvariable in vec[1])
                    and (self.instrument in vec)
                ):
                    self.input_instrument.append("_".join(vec[2:]))

    def set_parameter_vector(self, parameter_values):
        # To update the parameters, we have to transform the juliet inputs to celerite/george inputs. Update this
        # depending on the kernel under usage. For this, we first define a base_index variable that will define the numbering
        # of the self.parameter_vector. The reason for this is that the dimensions of the self.parameter_vector array is
        # different if the GP is global (i.e., self.global_GP is True --- meaning a unique GP is fitted to all instruments) or
        # not (self.global_GP is False --- meaning a different GP per instrument is fitted). If the former, the jitter terms are
        # modified directly by changing the self.yerr vector; in the latter, we have to manually add a jitter term in the GP parameter
        # vector. This base_index is only important for the george kernels though --- an if statement suffices for the celerite ones.
        base_index = 0
        if (self.kernel_name == "SEKernel") or (self.kernel_name == "M32Kernel"):
            if not self.global_GP:
                self.parameter_vector[base_index] = np.log(
                    (parameter_values["sigma_w_" + self.instrument] * self.sigma_factor)
                    ** 2
                )
                base_index += 1
            self.parameter_vector[base_index] = np.log(
                (
                    parameter_values["GP_sigma_" + self.input_instrument[0]]
                    * self.sigma_factor
                )
                ** 2.0
            )
            alpha_name = "alpha" if self.kernel_name == "SEKernel" else "malpha"
            for i in range(self.nX):
                self.parameter_vector[base_index + 1 + i] = np.log(
                    1.0
                    / parameter_values[
                        f"GP_{alpha_name}" + str(i) + "_" + self.input_instrument[1 + i]
                    ]
                )
        elif self.kernel_name == "ExpSineSquaredSEKernel":
            if not self.global_GP:
                self.parameter_vector[base_index] = np.log(
                    (parameter_values["sigma_w_" + self.instrument] * self.sigma_factor)
                    ** 2
                )
                base_index += 1
            self.parameter_vector[base_index] = np.log(
                (
                    parameter_values["GP_sigma_" + self.input_instrument[0]]
                    * self.sigma_factor
                )
                ** 2.0
            )
            self.parameter_vector[base_index + 1] = np.log(
                1.0 / (parameter_values["GP_alpha_" + self.input_instrument[1]])
            )
            self.parameter_vector[base_index + 2] = np.log(
                parameter_values["GP_Gamma_" + self.input_instrument[2]]
            )
            self.parameter_vector[base_index + 3] = np.log(
                parameter_values["GP_Prot_" + self.input_instrument[3]]
            )
        elif self.kernel_name == "CeleriteQPKernel":
            self.parameter_vector[0] = np.log(
                parameter_values["GP_B_" + self.input_instrument[0]]
            )
            self.parameter_vector[1] = np.log(
                parameter_values["GP_L_" + self.input_instrument[1]]
            )
            self.parameter_vector[2] = np.log(
                parameter_values["GP_Prot_" + self.input_instrument[2]]
            )
            self.parameter_vector[3] = np.log(
                parameter_values["GP_C_" + self.input_instrument[3]]
            )
            if not self.global_GP:
                self.parameter_vector[4] = np.log(
                    parameter_values["sigma_w_" + self.instrument] * self.sigma_factor
                )
        elif self.kernel_name == "CeleriteExpKernel":
            self.parameter_vector[0] = np.log(
                parameter_values["GP_sigma_" + self.input_instrument[0]]
            )
            self.parameter_vector[1] = np.log(
                parameter_values["GP_timescale_" + self.input_instrument[1]]
            )
            if not self.global_GP:
                self.parameter_vector[2] = np.log(
                    parameter_values["sigma_w_" + self.instrument] * self.sigma_factor
                )
        elif self.kernel_name == "CeleriteMaternKernel":
            self.parameter_vector[0] = np.log(
                parameter_values["GP_sigma_" + self.input_instrument[0]]
            )
            self.parameter_vector[1] = np.log(
                parameter_values["GP_rho_" + self.input_instrument[1]]
            )
            if not self.global_GP:
                self.parameter_vector[2] = np.log(
                    parameter_values["sigma_w_" + self.instrument] * self.sigma_factor
                )
        elif self.kernel_name == "CeleriteMaternExpKernel":
            self.parameter_vector[0] = np.log(
                parameter_values["GP_sigma_" + self.input_instrument[0]]
            )
            self.parameter_vector[1] = np.log(
                parameter_values["GP_timescale_" + self.input_instrument[1]]
            )
            self.parameter_vector[3] = np.log(
                parameter_values["GP_rho_" + self.input_instrument[2]]
            )
            if not self.global_GP:
                self.parameter_vector[4] = np.log(
                    parameter_values["sigma_w_" + self.instrument] * self.sigma_factor
                )
        elif self.kernel_name == "CeleriteSHOKernel":
            self.parameter_vector[0] = np.log(
                parameter_values["GP_S0_" + self.input_instrument[0]]
            )
            self.parameter_vector[1] = np.log(
                parameter_values["GP_Q_" + self.input_instrument[1]]
            )
            self.parameter_vector[2] = np.log(
                parameter_values["GP_omega0_" + self.input_instrument[2]]
            )

            if not self.global_GP:
                self.parameter_vector[3] = np.log(
                    parameter_values["sigma_w_" + self.instrument] * self.sigma_factor
                )

        elif self.kernel_name == "CeleriteDoubleSHOKernel":
            # The parametrization follows the "RotationTerm" implemented in
            # celerite2 https://celerite2.readthedocs.io/en/latest/api/python/#celerite2.terms.RotationTerm
            sigma = parameter_values["GP_sigma_" + self.input_instrument[0]]
            Q0 = parameter_values["GP_Q0_" + self.input_instrument[1]]
            P = parameter_values["GP_period_" + self.input_instrument[2]]
            f = parameter_values["GP_f_" + self.input_instrument[3]]
            dQ = parameter_values["GP_dQ_" + self.input_instrument[4]]

            Q1 = 1 / 2 + Q0 + dQ
            omega1 = 4 * np.pi * Q1 / (P * np.sqrt(4 * Q1**2 - 1))
            S1 = sigma**2 / ((1 + f) * omega1 * Q1)

            Q2 = 1 / 2 + Q0
            omega2 = 8 * np.pi * Q1 / (P * np.sqrt(4 * Q1**2 - 1))
            S2 = f * sigma**2 / ((1 + f) * omega2 * Q2)

            self.parameter_vector[0] = np.log(S1)
            self.parameter_vector[1] = np.log(Q1)
            self.parameter_vector[2] = np.log(omega1)
            self.parameter_vector[3] = np.log(S2)
            self.parameter_vector[4] = np.log(Q2)
            self.parameter_vector[5] = np.log(omega2)

            if not self.global_GP:
                self.parameter_vector[6] = np.log(
                    parameter_values["sigma_w_" + self.instrument] * self.sigma_factor
                )

        elif self.kernel_name == "CeleriteTripleSHOKernel":
            self.parameter_vector[0] = np.log(
                parameter_values["GP_S0_" + self.input_instrument[0]]
            )
            self.parameter_vector[1] = np.log(
                parameter_values["GP_Q0_" + self.input_instrument[1]]
            )
            self.parameter_vector[2] = np.log(
                2 * np.pi / parameter_values["GP_period_" + self.input_instrument[2]]
            )
            self.parameter_vector[3] = np.log(
                parameter_values["GP_f_" + self.input_instrument[3]]
                * parameter_values["GP_S0_" + self.input_instrument[0]]
            )
            self.parameter_vector[4] = np.log(
                parameter_values["GP_Q0_" + self.input_instrument[1]]
                - parameter_values["GP_dQ_" + self.input_instrument[4]]
            )
            self.parameter_vector[5] = np.log(
                np.pi / parameter_values["GP_period_" + self.input_instrument[2]]
            )
            self.parameter_vector[6] = np.log(
                parameter_values["GP_S0sc_" + self.input_instrument[5]]
            )
            self.parameter_vector[7] = np.log(
                parameter_values["GP_omega0sc_" + self.input_instrument[6]]
            )

            if not self.global_GP:
                self.parameter_vector[8] = np.log(
                    parameter_values["sigma_w_" + self.instrument] * self.sigma_factor
                )

        # For Matern+SHO kernel
        elif self.kernel_name == "CeleriteMaternSHOKernel":
            self.parameter_vector[0] = np.log(
                parameter_values["GP_sigma_" + self.input_instrument[0]]
            )
            self.parameter_vector[1] = np.log(
                parameter_values["GP_rho_" + self.input_instrument[1]]
            )
            self.parameter_vector[2] = np.log(
                parameter_values["GP_S0_" + self.input_instrument[2]]
            )
            self.parameter_vector[3] = np.log(
                parameter_values["GP_Q_" + self.input_instrument[3]]
            )
            self.parameter_vector[4] = np.log(
                parameter_values["GP_omega0_" + self.input_instrument[4]]
            )
            if not self.global_GP:
                self.parameter_vector[5] = np.log(
                    parameter_values["sigma_w_" + self.instrument] * self.sigma_factor
                )
        assert self.GP is not None, "gaussian_process.GP is not defined"
        self.GP.set_parameter_vector(self.parameter_vector)

    def __init__(
        self, data, model_type, instrument, george_hodlr=True, matern_eps=0.01
    ):
        self.isInit = False
        self.model_type = model_type.lower()
        # Perform changes that define the model_type. For example, the juliet input sigmas (both jitters and GP amplitudes) are
        # given in ppm in the input files, whereas for RVs they have the same units as the input RVs. This conversion factor is
        # defined by the model_type:
        if self.model_type == "lc":
            if instrument is None:
                instrument = "lc"
            self.sigma_factor = 1e-6
        elif self.model_type == "rv":
            if instrument is None:
                instrument = "rv"
            self.sigma_factor = 1.0
        else:
            raise Exception(
                "Model type "
                + model_type
                + ' currently not supported. Only "lc" or "rv" can serve as inputs for now.'
            )

        # Name of input instrument if given:
        self.instrument = instrument

        # Initialize global model variable:
        self.global_GP = False

        # Extract information from the data object:
        if self.model_type == "lc":
            # Save input predictor:
            if instrument == "lc":
                self.X = data.GP_lc_arguments["lc"]
                self.global_GP = True
            else:
                self.X = data.GP_lc_arguments[instrument]
            # Save errors (if any):
            if data.yerr_lc is not None:
                if instrument != "lc":
                    self.yerr = data.yerr_lc[data.instrument_indexes_lc[instrument]]
                else:
                    self.yerr = data.yerr_lc
            else:
                self.yerr = None
        elif self.model_type == "rv":
            # Save input predictor:
            if instrument == "rv":
                self.X = data.GP_rv_arguments["rv"]
                self.global_GP = True
            else:
                self.X = data.GP_rv_arguments[instrument]
            # Save errors (if any):
            if data.yerr_rv is not None:
                if instrument != "rv":
                    self.yerr = data.yerr_rv[data.instrument_indexes_rv[instrument]]
                else:
                    self.yerr = data.yerr_rv
            else:
                self.yerr = None

        # Fix sizes of regressors if wrong:
        if len(self.X.shape) == 2:
            if self.X.shape[1] != 1:
                self.nX = self.X.shape[1]
            else:
                self.X = self.X[:, 0]
                self.nX = 1
        else:
            self.nX = 1

        # Define all possible kernels available by the object:
        self.all_kernel_variables = {}
        self.all_kernel_variables["SEKernel"] = ["sigma"]
        self.all_kernel_variables["M32Kernel"] = ["sigma"]
        for i in range(self.nX):
            self.all_kernel_variables["SEKernel"] = self.all_kernel_variables[
                "SEKernel"
            ] + ["alpha" + str(i)]
            self.all_kernel_variables["M32Kernel"] = self.all_kernel_variables[
                "M32Kernel"
            ] + ["malpha" + str(i)]
        self.all_kernel_variables["ExpSineSquaredSEKernel"] = [
            "sigma",
            "alpha",
            "Gamma",
            "Prot",
        ]
        self.all_kernel_variables["CeleriteQPKernel"] = ["B", "L", "Prot", "C"]
        self.all_kernel_variables["CeleriteExpKernel"] = ["sigma", "timescale"]
        self.all_kernel_variables["CeleriteMaternKernel"] = ["sigma", "rho"]
        self.all_kernel_variables["CeleriteMaternExpKernel"] = [
            "sigma",
            "timescale",
            "rho",
        ]
        self.all_kernel_variables["CeleriteSHOKernel"] = ["S0", "Q", "omega0"]
        self.all_kernel_variables["CeleriteDoubleSHOKernel"] = [
            "sigma",
            "Q0",
            "period",
            "f",
            "dQ",
        ]
        # For Matern+SHO kernel
        self.all_kernel_variables["CeleriteMaternSHOKernel"] = [
            "sigma",
            "rho",
            "S0",
            "Q",
            "omega0",
        ]

        # Find kernel name (and save it to self.kernel_name):
        self.kernel_name = self.get_kernel_name(data.priors)
        # Initialize variable for the GP object:
        self.GP = None
        # Are we using celerite?
        self.use_celerite = False
        # Are we using george_hodlr?
        if george_hodlr:
            self.george_hodlr = True
        else:
            self.george_hodlr = False
        # Initialize variable that sets the "instrument" name for each variable (self.variables below). If, for example,
        # GP_Prot_TESS_K2_RV and GP_Gamma_TESS, and self.variables = [Prot,Gamma], then self.instrument_variables = ['TESS_K2_RV','TESS'].
        self.input_instrument = []

        # Initialize each kernel on the GP object. First, set the variables to the ones defined above. Then initialize the
        # actual kernel:
        self.variables = self.all_kernel_variables[self.kernel_name]
        phantomvariable = 0
        if self.kernel_name == "SEKernel":
            # Generate GPExpSquared base kernel:
            self.kernel = 1.0 * george.kernels.ExpSquaredKernel(
                np.ones(self.nX), ndim=self.nX, axes=range(self.nX)
            )
            # (Note no jitter kernel is given, as with george one defines this in the george.GP call):
        elif self.kernel_name == "M32Kernel":
            # Generate GPMatern32 base kernel:
            self.kernel = 1.0 * george.kernels.Matern32Kernel(
                np.ones(self.nX), ndim=self.nX, axes=range(self.nX)
            )
            # (Note no jitter kernel is given, as with george one defines this in the george.GP call):
        elif self.kernel_name == "ExpSineSquaredSEKernel":
            # Generate the kernels:
            K1 = 1.0 * george.kernels.ExpSquaredKernel(metric=1.0)
            K2 = george.kernels.ExpSine2Kernel(gamma=1.0, log_period=1.0)
            self.kernel = K1 * K2
            # (Note no jitter kernel is given, as with george one defines this in the george.GP call):
        elif self.kernel_name == "CeleriteQPKernel":
            # Generate rotational kernel:
            rot_kernel = terms.TermSum(
                RotationTerm(
                    log_amp=np.log(10.0),
                    log_timescale=np.log(10.0),
                    log_period=np.log(3.0),
                    log_factor=np.log(1.0),
                )
            )
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100 * 1e-6))
            # Wrap GP kernel and object:

            if self.instrument in ["rv", "lc"]:
                self.kernel = rot_kernel
            else:
                self.kernel = rot_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == "CeleriteExpKernel":
            # Generate exponential kernel:
            exp_kernel = terms.RealTerm(log_a=np.log(10.0), log_c=np.log(10.0))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100 * 1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ["rv", "lc"]:
                self.kernel = exp_kernel
            else:
                self.kernel = exp_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == "CeleriteMaternKernel":
            # Generate matern kernel:
            matern_kernel = terms.Matern32Term(
                log_sigma=np.log(10.0), log_rho=np.log(10.0), eps=matern_eps
            )
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100 * 1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ["rv", "lc"]:
                self.kernel = matern_kernel
            else:
                self.kernel = matern_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == "CeleriteMaternExpKernel":
            # Generate matern and exponential kernels:
            matern_kernel = terms.Matern32Term(
                log_sigma=np.log(10.0), log_rho=np.log(10.0), eps=matern_eps
            )
            exp_kernel = terms.RealTerm(log_a=np.log(10.0), log_c=np.log(10.0))
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100 * 1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ["rv", "lc"]:
                self.kernel = exp_kernel * matern_kernel
            else:
                self.kernel = exp_kernel * matern_kernel + kernel_jitter

            # We add a phantom variable because we want to leave index 2 without value ON PURPOSE: the idea is
            # that here, that is always 0 (because this defines the log(sigma) of the matern kernel in the
            # multiplication, which we set to 1).
            phantomvariable = 1
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == "CeleriteSHOKernel":
            # Generate kernel:
            sho_kernel = terms.SHOTerm(
                log_S0=np.log(10.0), log_Q=np.log(10.0), log_omega0=np.log(10.0)
            )
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100 * 1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ["rv", "lc"]:
                self.kernel = sho_kernel
            else:
                self.kernel = sho_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        elif self.kernel_name == "CeleriteDoubleSHOKernel":
            # Generate kernel
            # This kernel is adapted from the "RotationTerm" in celerite2 https://celerite2.readthedocs.io/en/latest/api/python/#celerite2.terms.RotationTerm
            sho_kernel1 = terms.SHOTerm(
                log_S0=np.log(10.0), log_Q=np.log(10.0), log_omega0=np.log(10.0)
            )
            sho_kernel2 = terms.SHOTerm(
                log_S0=np.log(10.0), log_Q=np.log(10.0), log_omega0=np.log(10.0)
            )

            double_sho_kernel = sho_kernel1 + sho_kernel2

            phantomvariable = 1
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100 * 1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ["rv", "lc"]:
                self.kernel = double_sho_kernel
            else:
                self.kernel = double_sho_kernel + kernel_jitter
            # We are using celerite:
            self.use_celerite = True
        ## For Matern+SHO kernel
        elif self.kernel_name == "CeleriteMaternSHOKernel":
            # Matern kernel:
            matern_kernel = terms.Matern32Term(
                log_sigma=np.log(10.0), log_rho=np.log(10.0), eps=matern_eps
            )
            # SHO kernel:
            sho_kernel = terms.SHOTerm(
                log_S0=np.log(10.0), log_Q=np.log(10.0), log_omega0=np.log(10.0)
            )
            # Jitter term:
            kernel_jitter = terms.JitterTerm(np.log(100 * 1e-6))
            # Wrap GP kernel and object:
            if self.instrument in ["rv", "lc"]:
                self.kernel = matern_kernel + sho_kernel
            else:
                self.kernel = matern_kernel + sho_kernel + kernel_jitter
            # And we are using celerite
            self.use_celerite = True
        # Check if use_celerite is True; if True, check that the regressor is ordered. If not, don't do the self.init_GP():
        if self.use_celerite:
            idx_sorted = np.argsort(self.X)
            diff1 = np.count_nonzero(self.X - self.X[idx_sorted])
            diff2 = np.count_nonzero(self.X - self.X[idx_sorted[::-1]])
            if diff1 == 0 or diff2 == 0:
                self.init_GP()
                self.isInit = True
        else:
            self.init_GP()
            self.isInit = True

        if self.global_GP:
            # If instrument is 'rv' or 'lc', assume GP object will fit for a global GP
            # (e.g., global photometric signal, or global RV signal) that assumes a given
            # GP realization for all instruments (but allows different jitters for each
            # instrument, added in quadrature to the self.yerr):
            self.parameter_vector = np.zeros(len(self.variables) + phantomvariable)
        else:
            # If GP per instrument, then there is one jitter term per instrument directly added in the model:
            self.parameter_vector = np.zeros(len(self.variables) + 1 + phantomvariable)
        self.set_input_instrument(data.priors)
