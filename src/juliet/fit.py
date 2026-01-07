import pickle

# Check existence of different samplers. First, import dynesty for (dynamic) nested sampling:
try:
    import dynesty
    from dynesty.utils import resample_equal

    force_pymultinest = False
except ImportError:
    force_pymultinest = True

# Import multinest for (importance) nested sampling:
try:
    import pymultinest

    force_dynesty = False
except Exception:
    force_dynesty = True

# Emcee for MCMC-ing:
try:
    import emcee
except Exception:
    print(
        "Warning: no emcee installation found. Will not be able to sample using sampler = 'emcee'."
    )

# Import zeus for fast MCMC:
try:
    import zeus
except ImportError:
    print(
        "Warning: no zeus installation found. Will not be able to sample using sampler = 'zeus'."
    )

# Import generic useful classes:
import contextlib

# Useful imports for parallelization:
import multiprocessing as mp
import os
from multiprocessing import Pool

import numpy as np

from .model import model
from .utils import (
    evaluate_beta,
    evaluate_exponential,
    evaluate_loguniform,
    evaluate_modifiedjeffreys,
    evaluate_normal,
    evaluate_truncated_normal,
    evaluate_uniform,
    transform_beta,
    transform_exponential,
    transform_loguniform,
    transform_modifiedjeffreys,
    transform_normal,
    transform_truncated_normal,
    transform_uniform,
    writepp,
)

__all__ = ["fit"]

MCMC_SAMPLERS = ["emcee", "zeus"]


class fit(object):
    """
    Given a juliet data object, this class performs a fit to the data and returns a results object to explore the
    results. Example usage:

               >>> results = juliet.fit(data)

    :param data: (juliet object)
        An object containing all the information regarding the data to be fitted, including options of the fit.
        Generated via juliet.load().

    On top of ``data``, a series of extra keywords can be included:

    :param sampler: (optional, string)
        String defining the sampler to be used on the fit. Current possible options include ``multinest`` to use `PyMultiNest <https://github.com/JohannesBuchner/PyMultiNest>`_ (via importance nested sampling),
        ``dynesty`` to use `Dynesty <https://github.com/joshspeagle/dynesty>`_'s importance nested sampling, ``dynamic_dynesty`` to use Dynesty's dynamic nested sampling algorithm, ``ultranest`` to use
        `Ultranest <https://github.com/JohannesBuchner/UltraNest/>`_, ``slicesampler_ultranest`` to use Ultranest's slice sampler and ``emcee`` to use `emcee <https://github.com/dfm/emcee>`_. Default is
        ``multinest`` if PyMultiNest is installed; ``dynesty`` if not.

    :param n_live_points: (optional, int)
        Number of live-points to use on the nested sampling samplers. Default is 500.

    :param nwalkers: (optional if using emcee, int)
        Number of walkers to use by emcee. Default is 100.

    :param nsteps: (optional if using MCMC, int)
        Number of steps/jumps to perform on the MCMC run. Default is 300.

    :param nburnin: (optional if using MCMC, int)
        Number of burnin steps/jumps when performing the MCMC run. Default is 500.

    :param emcee_factor: (optional, for emcee only, float)
        Factor multiplying the standard-gaussian ball around which the initial position is perturbed for each walker. Default is 1e-4.

    :param ecclim: (optional, float)
        Upper limit on the maximum eccentricity to sample. Default is ``1``.

    :param pl: (optional, float)
        If the ``(r1,r2)`` parametrization for ``(b,p)`` is used, this defines the lower limit of the planet-to-star radius ratio to be sampled.
        Default is ``0``.

    :param pu: (optional, float)
        Same as ``pl``, but for the upper limit. Default is ``1``.

    :param ta: (optional, float)
        Time to be substracted to the input times in order to generate the linear and/or quadratic trend to be added to the model.
        Default is 2458460.

    :param nthreads: (optinal, int)
        Define the number of threads to use within dynesty or emcee. Default is to use just 1. Note this will not impact PyMultiNest or UltraNest runs --- these can be parallelized via MPI only.

    :param light_travel_delay: (optinal, bool)
        Boolean indicating if light travel time delay wants to be included on eclipse time calculations.

    :param stellar_radius: (optional, float)
        Stellar radius in units of solar-radii to use for the light travel time corrections.

    In addition, any number of extra optional keywords can be given to the call, which will be directly ingested into the sampler of choice. For a full list of optional keywords for...

    - ...PyMultiNest, check the docstring of ``PyMultiNest``'s ``run`` `function <https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest/run.py>`_.

    - ...any of the nested sampling algorithms in ``dynesty``, see the docstring on the ``run_nested`` `function <https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynamicsampler.DynamicSampler.run_nested>`_.

    - ...the non-dynamic nested sampling algorithm implemented in ``dynesty``, see the docstring on ``dynesty.dynesty.NestedSampler`` in `dynesty's documentation <https://dynesty.readthedocs.io/en/latest/api.html>`_.

    - ...the dynamic nested sampling in ``dynesty`` check the docstring for ``dynesty.dynesty.DynamicNestedSampler`` in `dynesty's documentation <https://dynesty.readthedocs.io/en/latest/api.html>`_.

    - ...the ``ultranest`` sampler, see the docstring for `ultranest.integrationr.ReactiveNestedSampler` in `ultranest's documentation <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler>`_

    Finally, since ``juliet`` version 2.0.26, the following keywords have been deprecated, and are recommended to be removed from code using ``juliet`` as they
    will be removed sometime in the future:

    :param use_dynesty: (optional, boolean)
        If ``True``, use dynesty instead of `MultiNest` for posterior sampling and evidence evaluation. Default is
        ``False``, unless `MultiNest` via ``pymultinest`` is not working on the system.

    :param dynamic: (optional, boolean)
        If ``True``, use dynamic Nested Sampling with dynesty. Default is ``False``.

    :param dynesty_bound: (optional, string)
        Define the dynesty bound method to use (currently either ``single`` or ``multi``, to use either single ellipsoids or multiple
        ellipsoids). Default is ``multi`` (for details, see the `dynesty API <https://dynesty.readthedocs.io/en/latest/api.html>`_).

    :param dynesty_sample: (optional, string)
        Define the sampling method for dynesty to use. Default is ``rwalk``. Accorfing to the `dynesty API <https://dynesty.readthedocs.io/en/latest/api.html>`_,
        this should be changed depending on the number of parameters being fitted. If smaller than about 20, ``rwalk`` is optimal. For larger dimensions,
        ``slice`` or ``rslice`` should be used.

    :param dynesty_nthreads: (optional, int)
        Define the number of threads to use within dynesty. Default is to use just 1.

    :param dynesty_n_effective: (optional, int)
        Minimum number of effective posterior samples when using ``dynesty``. If the estimated “effective sample size” exceeds this number, sampling will terminate. Default is ``None``.

    :param dynesty_use_stop: (optional, boolean)
        Whether to evaluate the ``dynesty`` stopping function after each batch. Disabling this can improve performance if other stopping criteria such as maxcall are already specified.
        Default is ``True``.

    :param dynesty_use_pool: (optional, dict)
        A dictionary containing flags indicating where a pool in ``dynesty`` should be used to execute operations in parallel. These govern whether ``prior_transform`` is executed in parallel during
        initialization (``'prior_transform'``), loglikelihood is executed in parallel during initialization (``'loglikelihood'``), live points are proposed in parallel during a run
        (``'propose_point'``), and bounding distributions are updated in parallel during a run (``'update_bound'``). Default is True for all options.

    """

    def set_prior_transform(self):
        for pname in self.model_parameters:
            dist_name = self.data.priors[pname]["distribution"].lower()
            if dist_name != "fixed":
                if dist_name == "uniform":
                    self.transform_prior[pname] = transform_uniform
                if dist_name == "normal":
                    self.transform_prior[pname] = transform_normal
                if dist_name == "truncatednormal":
                    self.transform_prior[pname] = transform_truncated_normal

                if (
                    self.data.priors[pname]["distribution"] == "jeffreys"
                    or self.data.priors[pname]["distribution"] == "loguniform"
                ):
                    self.transform_prior[pname] = transform_loguniform
                if dist_name == "beta":
                    self.transform_prior[pname] = transform_beta
                if dist_name == "exponential":
                    self.transform_prior[pname] = transform_exponential
                if dist_name == "modjeffreys":
                    self.transform_prior[pname] = transform_modifiedjeffreys

    def set_logpriors(self):
        for pname in self.model_parameters:
            dist_name = self.data.priors[pname]["distribution"].lower()
            if dist_name != "fixed":
                if dist_name == "uniform":
                    self.evaluate_logprior[pname] = evaluate_uniform
                if dist_name == "normal":
                    self.evaluate_logprior[pname] = evaluate_normal
                if dist_name == "truncatednormal":
                    self.evaluate_logprior[pname] = evaluate_truncated_normal

                if (
                    self.data.priors[pname]["distribution"] == "jeffreys"
                    or self.data.priors[pname]["distribution"] == "loguniform"
                ):
                    self.evaluate_logprior[pname] = evaluate_loguniform
                if dist_name == "beta":
                    self.evaluate_logprior[pname] = evaluate_beta
                if dist_name == "exponential":
                    self.evaluate_logprior[pname] = evaluate_exponential
                if dist_name == "modjeffreys":
                    self.evaluate_logprior[pname] = evaluate_modifiedjeffreys

    # Prior transform for nested samplers:
    def prior_transform(self, cube, ndim=None, nparams=None):
        pcounter = 0
        for pname in self.model_parameters:
            if self.data.priors[pname]["distribution"].lower() != "fixed":
                cube[pcounter] = self.transform_prior[pname](
                    cube[pcounter], self.data.priors[pname]["hyperparameters"]
                )
                pcounter += 1

    # Prior transform for nested samplers (this one spits the transformed priors from the unit cube):
    def prior_transform_r(self, cube):
        pcounter = 0
        transformed_priors = np.copy(self.transformed_priors)
        for pname in self.model_parameters:
            if self.data.priors[pname]["distribution"].lower() != "fixed":
                transformed_priors[pcounter] = self.transform_prior[pname](
                    cube[pcounter], self.data.priors[pname]["hyperparameters"]
                )
                pcounter += 1
        return transformed_priors

    # Log-prior for MCMCs (returns evaluated prior):
    def logprior(self, theta):
        pcounter = 0
        total_logprior = 0.0
        for pname in self.model_parameters:
            if self.data.priors[pname]["distribution"].lower() != "fixed":
                total_logprior += self.evaluate_logprior[pname](
                    theta[pcounter], self.data.priors[pname]["hyperparameters"]
                )
                pcounter += 1
        return total_logprior

    def loglike(self, cube, ndim=None, nparams=None):
        # Evaluate the joint log-likelihood. For this, first extract all inputs:
        pcounter = 0
        for pname in self.model_parameters:
            if self.data.priors[pname]["distribution"].lower() != "fixed":
                self.posteriors[pname] = cube[pcounter]
                pcounter += 1
        # Initialize log-likelihood:
        log_likelihood = 0.0

        # Evaluate photometric model first:
        if self.data.t_lc is not None:
            self.lc.generate(self.posteriors)
            if self.lc.modelOK:
                log_likelihood += self.lc.get_log_likelihood(self.posteriors)
            else:
                return -1e101
        # Now RV model:
        if self.data.t_rv is not None:
            self.rv.generate(self.posteriors)
            if self.rv.modelOK:
                log_likelihood += self.rv.get_log_likelihood(self.posteriors)
            else:
                return -1e101

        # Evaluate any extra likelihoods:
        if self.extra_loglikelihood_boolean:
            log_likelihood += extra_loglikelihood["loglikelihood"](self.posteriors)

        # Return total log-likelihood:
        return log_likelihood

    # Log-probability for MCMC samplers:
    def logprob(self, theta):
        lp = self.logprior(theta) + self.loglike(theta)
        if np.isnan(lp):
            return -np.inf
        else:
            return lp

    def __init__(
        self,
        data,
        sampler="multinest",
        n_live_points=500,
        nwalkers=100,
        nsteps=300,
        nburnin=500,
        emcee_factor=1e-4,
        ecclim=1.0,
        pl=0.0,
        pu=1.0,
        ta=2458460.0,
        nthreads=None,
        light_travel_delay=False,
        stellar_radius=None,
        use_ultranest=False,
        use_dynesty=False,
        dynamic=False,
        dynesty_bound="multi",
        dynesty_sample="rwalk",
        dynesty_nthreads=None,
        dynesty_n_effective=np.inf,
        dynesty_use_stop=True,
        dynesty_use_pool=None,
        **kwargs,
    ):
        # Define output results object:
        self.results = None

        # Save sampler inputs in case users are still using old definitions
        self.use_ultranest = use_ultranest
        self.use_dynesty = use_dynesty
        self.dynamic = dynamic
        self.dynesty_bound = dynesty_bound
        self.dynesty_sample = dynesty_sample
        self.dynesty_nthreads = dynesty_nthreads
        self.dynesty_n_effective = dynesty_n_effective
        self.dynesty_use_stop = dynesty_use_stop
        self.dynesty_use_pool = dynesty_use_pool

        # Now extract sampler options:
        self.sampler = sampler
        self.n_live_points = n_live_points
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.emcee_factor = emcee_factor
        self.nburnin = nburnin
        self.nthreads = nthreads

        # Extract physical model details:
        self.light_travel_delay = light_travel_delay
        if self.light_travel_delay and (stellar_radius is None):
            raise Exception(
                "Error: if light_travel_delay is activated, a stellar radius needs to be given as well via stellar_radius = yourvalue; e.g., dataset.fit(..., light_travel_delay = True, stellar_radius = 1.1234)."
            )

        self.stellar_radius = stellar_radius

        # Update sampler inputs in case user still using deprecated inputs. We'll remove this in some future. First, define standard pre-fix and sufix for the warnings:
        ws1 = "WARNING: use of the "
        ws2 = " argument is deprecated and will be removed in future juliet versions. Use the "
        ws3 = " instead; for more information, check the API of juliet.fit: https://juliet.readthedocs.io/en/latest/user/api.html#juliet.fit"
        if self.use_ultranest:
            self.sampler = "ultranest"
            print(ws1 + "use_ultranest" + ws2 + '"sampler" string' + ws3)
        if self.use_dynesty:
            print(ws1 + "use_dynesty" + ws2 + '"sampler" string' + ws3)
            self.sampler = "dynesty"
            if self.dynamic:
                self.sampler = "dynamic_dynesty"
                print(ws1 + "dynamic" + ws2 + '"sampler" string' + ws3)
            # Add the other deprecated flags to the kwargs:
            if self.dynesty_bound != "multi":
                print(ws1 + "dynesty_bound" + ws2 + '"bound" argument' + ws3)
                # kwargs take presedence:
                if "bound" not in kwargs.keys():
                    kwargs["bound"] = self.dynesty_bound
            if self.dynesty_sample != "rwalk":
                print(ws1 + "dynesty_sample" + ws2 + '"sample" argument' + ws3)
                # kwargs take presedence:
                if "sample" not in kwargs.keys():
                    kwargs["sample"] = self.dynesty_sample
            if self.dynesty_nthreads is not None:
                print(ws1 + "dynesty_nthreads" + ws2 + '"nthreads" argument' + ws3)
                # The nthreads argument takes presedence now:
                if nthreads is None:
                    self.nthreads = self.dynesty_nthreads
            if self.dynesty_n_effective is not np.inf:
                print(
                    ws1 + "dynesty_n_effective" + ws2 + '"n_effective" argument' + ws3
                )
                # kwargs take presedence:
                if "n_effective" not in kwargs.keys():
                    kwargs["n_effective"] = self.dynesty_n_effective
            if not self.dynesty_use_stop:
                print(ws1 + "dynesty_use_stop" + ws2 + '"use_stop" argument' + ws3)
                # kwargs take presedence:
                if "use_stop" not in kwargs.keys():
                    kwargs["use_stop"] = self.dynesty_use_stop
            if self.dynesty_use_pool is not None:
                print(ws1 + "dynesty_use_pool" + ws2 + '"use_pool" argument' + ws3)
                # kwargs take presedence:
                if "use_pool" not in kwargs.keys():
                    kwargs["use_pool"] = self.dynesty_use_pool

        # Define (exo-)algorithmic options:
        self.ecclim = ecclim
        self.pl = pl
        self.pu = pu
        self.ta = ta

        # Inhert data object:
        self.data = data

        # Inhert the output folder:
        self.out_folder = data.out_folder
        self.transformed_priors = np.zeros(self.data.nparams)

        # Inhert extra likelihood:
        self.extra_loglikelihood = data.extra_loglikelihood
        self.extra_loglikelihood_boolean = data.extra_loglikelihood_boolean

        # Define prefixes in case saving is turned on (i.e., user passed an out_folder). PyMultiNest and dynesty ones are set by hand. For the rest, use the new
        # sampler string directly:
        if self.sampler == "multinest":
            self.sampler_prefix = ""
        elif self.sampler == "dynesty":
            self.sampler_prefix = "_dynesty_NS_"
        elif self.sampler == "dynamic_dynesty":
            self.sampler_prefix = "_dynesty_DNS_"
        else:
            self.sampler_prefix = self.sampler + "_"

        # Before starting, check if force_dynesty or force_pymultinest is on; change options accordingly:
        if force_dynesty and (self.sampler == "multinest"):
            print(
                "PyMultinest installation not detected. Forcing dynesty as the sampler."
            )
            self.sampler = "dynesty"
            self.sampler_prefix = "_dynesty_NS_"

        if force_pymultinest and (self.sampler == "dynesty"):
            print(
                "dynesty installation not detected. Forcing PyMultinest as the sampler."
            )

            self.sampler = "multinest"
            self.sampler_prefix = ""

        # Generate a posteriors self that will save the current values of each of the parameters. Initialization value is unimportant for nested samplers;
        # if MCMC, this saves the initial parameter values:
        self.posteriors = {}
        self.model_parameters = list(self.data.priors.keys())
        self.paramnames = []
        for pname in self.model_parameters:
            if self.data.priors[pname]["distribution"] == "fixed":
                self.posteriors[pname] = self.data.priors[pname]["hyperparameters"]

            else:
                if self.sampler in MCMC_SAMPLERS:
                    self.posteriors[pname] = self.data.starting_point[pname]
                else:
                    self.posteriors[pname] = 0.0  # self.data.priors[pname]['cvalue']
                self.paramnames.append(pname)
        # For each of the variables in the prior that is not fixed, define an internal dictionary that will save the
        # corresponding transformation function to the prior corresponding to that variable. Idea is that with this one
        # simply does self.transform_prior[variable_name](value) and you get the transformed value to the 0,1 prior.
        # This avoids having to keep track of the prior distribution on each of the iterations. This is only useful for
        # nested samplers:
        if self.sampler not in MCMC_SAMPLERS:
            self.transform_prior = {}
            self.set_prior_transform()
        else:
            self.evaluate_logprior = {}
            self.set_logpriors()

        # Generate light-curve and radial-velocity models:
        if self.data.t_lc is not None:
            self.lc = model(
                self.data,
                modeltype="lc",
                pl=self.pl,
                pu=self.pu,
                ecclim=self.ecclim,
                light_travel_delay=self.light_travel_delay,
                stellar_radius=self.stellar_radius,
                log_like_calc=True,
            )
        if self.data.t_rv is not None:
            self.rv = model(
                self.data,
                modeltype="rv",
                ecclim=self.ecclim,
                ta=self.ta,
                log_like_calc=True,
            )

        # First, check if a run has already been performed with the user-defined sampler. If it hasn't, run it.
        # If it has (detected through its output filename), skip running again and jump straight to loading the
        # data:
        out = {}
        runSampler = False
        if self.out_folder is None:
            self.out_folder = os.getcwd() + "/"
        if not os.path.exists(self.out_folder + self.sampler_prefix + "posteriors.pkl"):
            runSampler = True

        # If runSampler is True, then run the sampler of choice:
        if runSampler:
            if "ultranest" in self.sampler:
                from ultranest import ReactiveNestedSampler

                # Match kwargs to possible ReactiveNestedSampler keywords. First, extract possible arguments of ReactiveNestedSampler:
                args = ReactiveNestedSampler.__init__.__code__.co_varnames
                rns_args = {}
                # First, define some standard ones:
                rns_args["transform"] = self.prior_transform_r
                rns_args["log_dir"] = self.out_folder
                rns_args["resume"] = True
                # Now extract arguments from kwargs; they take presedence over the standard ones above:
                for arg in args:
                    if arg in kwargs:
                        rns_args[arg] = kwargs[arg]
                # ...and load the sampler:
                sampler = ReactiveNestedSampler(
                    self.paramnames, self.loglike, **rns_args
                )

                if "slicesampler" in self.sampler:
                    import ultranest.stepsampler

                    # Match kwarfs to possible args in RegionSliceSampler:
                    args = (
                        ultranest.stepsampler.SliceSampler.__init__.__code__.co_varnames
                    )
                    rss_args = {}
                    # First, define standard ones:
                    rss_args["nsteps"] = 400
                    rss_args["adaptive_nsteps"] = "move-distance"
                    # Extract kwargs, add them in:
                    for arg in args:
                        if arg in kwargs:
                            rss_args[arg] = kwargs[arg]

                    # Apply stepsampler:
                    sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
                        **rss_args
                    )

                # Now do the same for ReactiveNestedSampler.run --- load any kwargs the user has given as input:
                args = ReactiveNestedSampler.run.__code__.co_varnames
                rns_run_args = {}
                # Define some standard ones:
                rns_run_args["frac_remain"] = 0.1
                rns_run_args["min_num_live_points"] = self.n_live_points
                rns_run_args["max_num_improvement_loops"] = 1
                # Load the ones from the kwargs:
                for arg in args:
                    if arg in kwargs:
                        rns_run_args[arg] = kwargs[arg]
                # Run the sampler:
                results = sampler.run(**rns_run_args)
                sampler.print_results()
                sampler.plot()
                assert results is not None, "Ultranest sampler failed to run."

                # Save ultranest outputs:
                out["ultranest_output"] = results
                # Get weighted posterior:
                posterior_samples = results["samples"]
                # Get lnZ:
                out["lnZ"] = results["logz"]
                out["lnZerr"] = results["logzerr"]

            elif "multinest" in self.sampler:
                # As done for ultranest above, scan possible arguments for pymultinest.run:
                args = pymultinest.run.__code__.co_varnames
                mn_args = {}
                # Define some standard ones:
                mn_args["n_live_points"] = self.n_live_points
                mn_args["max_modes"] = 100
                mn_args["outputfiles_basename"] = self.out_folder + "jomnest_"
                mn_args["resume"] = False
                mn_args["verbose"] = self.data.verbose
                # Now extract arguments from kwargs:
                for arg in args:
                    if arg in kwargs:
                        mn_args[arg] = kwargs[arg]
                # Define the sampler:
                pymultinest.run(
                    self.loglike, self.prior_transform, self.data.nparams, **mn_args
                )

                # Now, with the sampler defined, repeat the same as above for the pymultinest.Analyzer object:
                args = pymultinest.Analyzer.__init__.__code__.co_varnames
                mna_args = {}
                # Define standard ones:
                mna_args["outputfiles_basename"] = self.out_folder + "jomnest_"
                mna_args["n_params"] = self.data.nparams
                # Load the ones from kwargs:
                for arg in args:
                    if arg in kwargs:
                        mna_args[arg] = kwargs[arg]
                # Run and get output:
                output = pymultinest.Analyzer(**mna_args)
                # Get out parameters: this matrix has (samples,n_params+1):
                posterior_samples = output.get_equal_weighted_posterior()[:, :-1]
                # Get INS lnZ:
                out["lnZ"] = output.get_stats()["global evidence"]
                out["lnZerr"] = output.get_stats()["global evidence error"]

            elif "dynesty" in self.sampler:
                if self.sampler == "dynamic_dynesty":
                    DynestySampler = dynesty.DynamicNestedSampler

                elif self.sampler == "dynesty":
                    DynestySampler = dynesty.NestedSampler

                # To run dynesty, we do it a little bit different depending if we are doing multithreading or not:
                if self.nthreads is None:
                    # As with the other samplers, first extract list of possible args (try-except for back-compatibility with prior dynesty versions):
                    try:
                        args = vars(DynestySampler)["__init__"].__code__.co_varnames

                    except Exception:
                        args = vars(DynestySampler).keys()

                    d_args = {}

                    # Define some standard ones (for back-compatibility with previous juliet versions):
                    d_args["bound"] = "multi"
                    d_args["sample"] = "rwalk"
                    d_args["nlive"] = self.n_live_points

                    # Match them with kwargs (kwargs take preference):
                    for arg in args:
                        if arg in kwargs:
                            d_args[arg] = kwargs[arg]

                    # Define the sampler:
                    sampler = DynestySampler(
                        self.loglike,
                        self.prior_transform_r,
                        self.data.nparams,
                        **d_args,
                    )

                    # Now do the same for the actual sampler:
                    try:
                        args = sampler.run_nested.__func__.__code__.co_varnames

                    except Exception:
                        args = vars(sampler).keys()

                    ds_args = {}

                    # Load ones from kwargs:
                    for arg in args:
                        if arg in kwargs:
                            ds_args[arg] = kwargs[arg]

                    # Now run:
                    sampler.run_nested(**ds_args)

                    # And extract results
                    results = sampler.results

                else:
                    # Before running the whole multithread magic, match kwargs with functional arguments (try-except
                    # for back-compatibility with prior dynesty versions):
                    try:
                        args = vars(DynestySampler)["__init__"].__code__.co_varnames

                    except Exception:
                        args = vars(DynestySampler).keys()

                    d_args = {}

                    # Define some standard ones (for back-compatibility with previous juliet versions):
                    d_args["bound"] = "multi"
                    d_args["sample"] = "rwalk"
                    d_args["nlive"] = self.n_live_points

                    # Match them with kwargs:
                    for arg in args:
                        if arg in kwargs:
                            d_args[arg] = kwargs[arg]

                    # Now define a mock sampler to retrieve variable names:
                    mock_sampler = DynestySampler(
                        self.loglike,
                        self.prior_transform_r,
                        self.data.nparams,
                        **d_args,
                    )
                    # Extract args:
                    try:
                        args = mock_sampler.run_nested.__func__.__code__.co_varnames

                    except Exception:
                        args = vars(mock_sampler).keys()

                    ds_args = {}

                    # Load ones from kwargs:
                    for arg in args:
                        if arg in kwargs:
                            ds_args[arg] = kwargs[arg]

                    # Now run all with multiprocessing:
                    """
                    with contextlib.closing(Pool(processes=self.nthreads -
                                                 1)) as executor:
                        sampler = DynestySampler(self.loglike,
                                                 self.prior_transform_r,
                                                 self.data.nparams,
                                                 pool=executor,
                                                 queue_size=self.nthreads,
                                                 **d_args)
                        sampler.run_nested(**ds_args)
                        results = sampler.results

                    """
                    with mp.Pool(self.nthreads) as pool:
                        sampler = DynestySampler(
                            self.loglike,
                            self.prior_transform_r,
                            self.data.nparams,
                            pool=pool,
                            queue_size=self.nthreads,
                            **d_args,
                        )

                        sampler.run_nested(**ds_args)

                    results = sampler.results

                # Extract dynesty outputs:
                out["dynesty_output"] = results

                # Get weighted posterior:
                weights = np.exp(results["logwt"] - results["logz"][-1])
                posterior_samples = resample_equal(results.samples, weights)

                # Get lnZ:
                out["lnZ"] = results.logz[-1]
                out["lnZerr"] = results.logzerr[-1]

            elif "emcee" in self.sampler:
                # Initiate starting point for each walker. To this end, first load starting values.
                initial_position = np.array([])
                for pname in self.model_parameters:
                    if self.data.priors[pname]["distribution"] != "fixed":
                        initial_position = np.append(
                            initial_position, self.posteriors[pname]
                        )

                # Perturb initial position for each of the walkers:
                pos = initial_position + self.emcee_factor * np.random.randn(
                    self.nwalkers, len(initial_position)
                )

                # Before performing the sampling, catch any kwargs that go to EnsembleSampler; rest of kwargs are assumed to go
                # to run_mcmc:
                args = emcee.EnsembleSampler.__init__.__code__.co_varnames
                ES_args = {}

                # Match them with kwargs (kwargs take preference):
                for arg in args:
                    if arg in kwargs:
                        ES_args[arg] = kwargs[arg]
                        kwargs.pop(arg)

                # Now perform the sampling. If nthreads is defined, parallelize. If not, go the serial way:
                if self.nthreads is None:
                    sampler = emcee.EnsembleSampler(
                        self.nwalkers, self.data.nparams, self.logprob, **ES_args
                    )
                    sampler.run_mcmc(pos, self.nsteps + self.nburnin, **kwargs)
                else:
                    with contextlib.closing(
                        Pool(processes=self.nthreads - 1)
                    ) as executor:
                        sampler = emcee.EnsembleSampler(
                            self.nwalkers,
                            self.data.nparams,
                            self.logprob,
                            pool=executor,
                            **ES_args,
                        )
                        sampler.run_mcmc(pos, self.nsteps + self.nburnin, **kwargs)

                # Store posterior samples. First, store the samples for each walker:
                out["posteriors_per_walker"] = sampler.get_chain()

                # And now store posteriors with all walkers flattened out:
                posterior_samples = sampler.get_chain(discard=self.nburnin, flat=True)

            elif "zeus" in self.sampler:
                # Identical implementation to emcee...?
                # Initiate starting point for each walker. To this end, first load starting values.
                initial_position = np.array([])
                for pname in self.model_parameters:
                    if self.data.priors[pname]["distribution"] != "fixed":
                        initial_position = np.append(
                            initial_position, self.posteriors[pname]
                        )

                # Perturb initial position for each of the walkers:
                pos = initial_position + self.emcee_factor * np.random.randn(
                    self.nwalkers, len(initial_position)
                )

                # Before performing the sampling, catch any kwargs that go to EnsembleSampler; rest of kwargs are assumed to go
                # to run_mcmc:
                args = zeus.EnsembleSampler.__init__.__code__.co_varnames
                zeus_args = {}

                # Match them with kwargs (kwargs take preference):
                for arg in args:
                    if arg in kwargs:
                        zeus_args[arg] = kwargs[arg]
                        kwargs.pop(arg)

                # Now perform the sampling. If nthreads is defined, parallelize. If not, go the serial way:
                if self.nthreads is None:
                    sampler = zeus.EnsembleSampler(
                        self.nwalkers, self.data.nparams, self.logprob, **zeus_args
                    )
                    sampler.run_mcmc(pos, self.nsteps + self.nburnin, **kwargs)
                else:
                    with contextlib.closing(
                        Pool(processes=self.nthreads - 1)
                    ) as executor:
                        sampler = zeus.EnsembleSampler(
                            self.nwalkers,
                            self.data.nparams,
                            self.logprob,
                            pool=executor,
                            **zeus_args,
                        )
                        sampler.run_mcmc(pos, self.nsteps + self.nburnin, **kwargs)

                # Store posterior samples. First, store the samples for each walker:
                out["posteriors_per_walker"] = sampler.get_chain()

                # And now store posteriors with all walkers flattened out:

                posterior_samples = sampler.get_chain(discard=self.nburnin, flat=True)

            # Save posterior samples as outputted by Multinest/Dynesty:
            out["posterior_samples"] = {}
            out["posterior_samples"]["unnamed"] = posterior_samples

            # Save log-likelihood of each of the samples:
            out["posterior_samples"]["loglike"] = np.zeros(posterior_samples.shape[0])
            for i in range(posterior_samples.shape[0]):
                out["posterior_samples"]["loglike"][i] = self.loglike(
                    posterior_samples[i, :]
                )

            pcounter = 0
            for pname in self.model_parameters:
                if data.priors[pname]["distribution"] != "fixed":
                    self.posteriors[pname] = np.median(posterior_samples[:, pcounter])
                    out["posterior_samples"][pname] = posterior_samples[:, pcounter]
                    pcounter += 1

            # Go through the posterior samples to see if dt or T, the TTV parameters, are present. If they are, add to the posterior dictionary
            # (.pkl) and file (.dat) the corresponding time-of-transit center, if the dt parametrization is being used, which is the actual
            # observable folks doing dynamics usually want. If the T parametrization is being used, write down the period and t0 implied by
            # those T's:
            fitted_parameters = list(out["posterior_samples"].keys())
            firstTime, Tparametrization = True, False
            for posterior_parameter in fitted_parameters:
                pvector = posterior_parameter.split("_")
                if pvector[0] == "dt":
                    # Extract planet number (pnum, e.g., 'p1'), instrument (ins, e.g., 'TESS') and transit number (tnum, e.g., '-1'):
                    pnum, ins, tnum = pvector[1:]
                    # Extract the period; check if it was fitted. If not, assume it was fixed:
                    if "P_" + pnum in fitted_parameters:
                        P = out["posterior_samples"]["P_" + pnum]
                    else:
                        P = data.priors["P_" + pnum]["hyperparameters"]
                    # Same for t0:
                    if "t0_" + pnum in fitted_parameters:
                        t0 = out["posterior_samples"]["t0_" + pnum]
                    else:
                        t0 = data.priors["t0_" + pnum]["hyperparameters"]
                    # Having extracted P and t0, generate the time-of-transit center for the current transit:
                    out["posterior_samples"]["T_" + pnum + "_" + ins + "_" + tnum] = (
                        t0
                        + np.double(tnum) * P
                        + out["posterior_samples"][posterior_parameter]
                    )
                if pvector[0] == "T":
                    if firstTime:
                        Tparametrization = True
                        Tdict = {}
                        firstTime = False
                    # Extract planet number (pnum, e.g., 'p1'), instrument (ins, e.g., 'TESS') and transit number (tnum, e.g., '-1'):
                    pnum, ins, tnum = pvector[1:]
                    if pnum not in list(Tdict.keys()):
                        Tdict[pnum] = {}
                    Tdict[pnum][int(tnum)] = out["posterior_samples"][
                        posterior_parameter
                    ]
            if Tparametrization:
                for pnum in list(Tdict.keys()):
                    all_ns = np.array(list(Tdict[pnum].keys()))
                    Nsamples = len(Tdict[pnum][all_ns[0]])
                    (
                        out["posterior_samples"]["P_" + pnum],
                        out["posterior_samples"]["t0_" + pnum],
                    ) = np.zeros(Nsamples), np.zeros(Nsamples)
                    N = len(all_ns)
                    for i in range(Nsamples):
                        all_Ts = np.zeros(N)
                        for j in range(len(all_ns)):
                            all_Ts[j] = Tdict[pnum][all_ns[j]][i]
                        XY, Y, X, X2 = (
                            np.sum(all_Ts * all_ns) / N,
                            np.sum(all_Ts) / N,
                            np.sum(all_ns) / N,
                            np.sum(all_ns**2) / N,
                        )
                        # Get slope:
                        out["posterior_samples"]["P_" + pnum][i] = (XY - X * Y) / (
                            X2 - (X**2)
                        )
                        # Intercept:
                        out["posterior_samples"]["t0_" + pnum][i] = (
                            Y - out["posterior_samples"]["P_" + pnum][i] * X
                        )
            if self.data.t_lc is not None:
                if True in self.data.lc_options["efficient_bp"].values():
                    out["pu"] = self.pu
                    out["pl"] = self.pl
            if self.data.t_rv is not None:
                if (
                    self.data.rv_options["fitrvline"]
                    or self.data.rv_options["fitrvquad"]
                ):
                    out["ta"] = self.ta

            # Finally, save juliet output to pickle file:
            pickle.dump(
                out,
                open(self.out_folder + self.sampler_prefix + "posteriors.pkl", "wb"),
            )
            """
            if 'dynesty' in self.sampler:
                if (self.sampler == 'dynamic_dynesty') and (self.out_folder is not None):
                    pickle.dump(out,open(self.out_folder+'_dynesty_DNS_posteriors.pkl','wb'))
                elif (self.sampler == 'dynesty') and (self.out_folder is not None):
                    pickle.dump(out,open(self.out_folder+'_dynesty_NS_posteriors.pkl','wb'))
            elif 'multinest' in self.sampler:
                if (self.sampler == 'multinest') and (self.out_folder is not None):
                    pickle.dump(out,open(self.out_folder+'posteriors.pkl','wb'))
            elif 'ultranest' in self.sampler:
                if (self.sampler == 'ultranest') and (self.out_folder is not None):
                    pickle.dump(out,open(self.out_folder+self.sampler_prefix+'posteriors.pkl','wb'))
            """
        else:
            # If the sampler was already ran, then user really wants to extract outputs from previous fit:
            print(
                "Detected "
                + self.sampler
                + " sampler output files --- extracting from "
                + self.out_folder
                + self.sampler_prefix
                + "posteriors.pkl"
            )
            if self.data.pickle_encoding is None:
                out = pickle.load(
                    open(self.out_folder + self.sampler_prefix + "posteriors.pkl", "rb")
                )
            else:
                out = pickle.load(
                    open(
                        self.out_folder + self.sampler_prefix + "posteriors.pkl", "rb"
                    ),
                    encoding=self.data.pickle_encoding,
                )
            """
            if (self.use_dynesty) and (self.out_folder is not None):
                if self.dynamic:
                    if os.path.exists(self.out_folder +
                                      '_dynesty_DNS_posteriors.pkl'):
                        if self.data.verbose:
                            print(
                                'Detected (dynesty) Dynamic NS output files --- extracting...'
                            )
                        if self.data.pickle_encoding is None:
                            out = pickle.load(
                                open(
                                    self.out_folder +
                                    '_dynesty_DNS_posteriors.pkl', 'rb'))
                        else:
                            out = pickle.load(
                                open(
                                    self.out_folder +
                                    '_dynesty_DNS_posteriors.pkl', 'rb'),
                                encoding=self.data.pickle_encoding)
                else:
                    if os.path.exists(self.out_folder +
                                      '_dynesty_NS_posteriors.pkl'):
                        if self.data.verbose:
                            print(
                                'Detected (dynesty) NS output files --- extracting...'
                            )
                        if self.data.pickle_encoding is None:
                            out = pickle.load(
                                open(
                                    self.out_folder +
                                    '_dynesty_NS_posteriors.pkl', 'rb'))
                        else:
                            out = pickle.load(
                                open(
                                    self.out_folder +
                                    '_dynesty_NS_posteriors.pkl', 'rb'),
                                encoding=self.data.pickle_encoding)
            elif self.out_folder is not None:
                if self.data.verbose:
                    print(
                        'Detected (MultiNest) NS output files --- extracting...'
                    )
                if self.data.pickle_encoding is None:
                    out = pickle.load(open(self.out_folder+'posteriors.pkl','rb'))
                else:
                    out = pickle.load(open(self.out_folder+'posteriors.pkl','rb'), encoding = self.data.pickle_encoding)
            """
            if len(out.keys()) == 0:
                print(
                    "Warning: no output generated or extracted. Check the fit options given to juliet.fit()."
                )
            else:
                # For retro-compatibility, check for sigma_w_rv_instrument and add an extra variable on out
                # for sigma_w_instrument:
                out_temp = dict()
                for pname in out["posterior_samples"].keys():
                    if "sigma_w_rv" == pname[:10]:
                        instrument = pname.split("_")[-1]
                        out_temp["sigma_w_" + self.sigmaw_iname[instrument]] = out[
                            "posterior_samples"
                        ][pname]
                for pname in out_temp.keys():
                    out["posterior_samples"][pname] = out_temp[pname]
                # Extract parameters:
                for pname in self.posteriors.keys():
                    if data.priors[pname]["distribution"] != "fixed":
                        self.posteriors[pname] = np.median(
                            out["posterior_samples"][pname]
                        )
                posterior_samples = out["posterior_samples"]["unnamed"]
                if "pu" in out.keys():
                    self.pu = out["pu"]
                    self.pl = out["pl"]
                    self.Ar = (self.pu - self.pl) / (2.0 + self.pl + self.pu)
                if "ta" in out.keys():
                    self.ta = out["ta"]

        # Either fit done or extracted. If doesn't exist, create the posteriors.dat file:
        if self.out_folder is not None:
            if not os.path.exists(self.out_folder + "posteriors.dat"):
                outpp = open(self.out_folder + "posteriors.dat", "w")
                writepp(outpp, out, data.priors)

        # Save all results (posteriors) to the self.results object:
        self.posteriors = out

        # Save posteriors to lc and rv:
        if self.data.t_lc is not None:
            self.lc.set_posterior_samples(out["posterior_samples"])
        if self.data.t_rv is not None:
            self.rv.set_posterior_samples(out["posterior_samples"])
