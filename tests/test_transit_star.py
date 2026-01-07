import os

import batman
import matplotlib.pyplot as plt
import numpy as np

import juliet


# Create dataset:
def get_transit_model(t):
    params = batman.TransitParams()
    params.t0 = 0.0  # time of inferior conjunction
    params.per = 1.0  # orbital period (days)
    params.a = 3.6  # semi-major axis (in units of stellar radii)
    params.rp = 0.1  # rp/rs
    params.inc = 90.0  # orbital inclination (in degrees)
    params.ecc = 0.0  # eccentricity
    params.w = 90.0  # longitude of periastron (in degrees) p
    params.limb_dark = "quadratic"  # limb darkening profile to use
    params.u = [0.2, 0.3]  # limb darkening coefficients
    tmodel = batman.TransitModel(params, t.astype("float64"))
    return tmodel.light_curve(params)


def standarize_variable(x):
    return (x - np.mean(x)) / np.sqrt(np.var(x))


# Generate times (over an hour) and fake flat fluxes:
times = np.linspace(-0.1, 0.1, 300)
fluxes = get_transit_model(times)

# Add noise (if not already added):
if not os.path.exists("transit_test.txt"):
    sigma = 100  # ppm
    noise = np.random.normal(0.0, sigma * 1e-6, len(times))

    dataset = fluxes + noise

    fout = open("transit_test.txt", "w")
    for i in range(len(dataset)):
        fout.write("{0:.10f}\n".format(dataset[i]))

else:
    dataset = np.loadtxt("transit_test.txt", unpack=True)

# Fit:
jtimes, jfluxes, jfluxes_error = {}, {}, {}
jtimes["instrument"], jfluxes["instrument"], jfluxes_error["instrument"] = (
    times,
    dataset,
    np.ones(len(dataset)) * 1e-6,
)

priors = {
    "mdilution_instrument": {"distribution": "fixed", "hyperparameters": 1.0},
    "mflux_instrument": {"distribution": "normal", "hyperparameters": [0.0, 0.1]},
    "sigma_w_instrument": {"distribution": "loguniform", "hyperparameters": [1e1, 1e3]},
    "P_p1": {"distribution": "fixed", "hyperparameters": 1.0},
    "t0_p1": {"distribution": "fixed", "hyperparameters": 0.0},
    "p_p1": {"distribution": "uniform", "hyperparameters": [0, 0.2]},
    "a_p1": {"distribution": "fixed", "hyperparameters": 3.6},
    "b_p1": {"distribution": "fixed", "hyperparameters": 0.0},
    "q1_instrument": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
    "q2_instrument": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
    "ecc_p1": {"distribution": "fixed", "hyperparameters": 0.0},
    "omega_p1": {"distribution": "fixed", "hyperparameters": 90.0},
}


# Fit with hodlr:
jdataset = juliet.load(
    priors=priors,
    t_lc=jtimes,
    y_lc=jfluxes,
    yerr_lc=jfluxes_error,
    out_folder="transit",
    verbose=True,
)

results = jdataset.fit(sampler="multinest", progress=True)

# Plot:
model = results.lc.evaluate("instrument")

plt.plot(times, dataset, ".", label="data")
plt.plot(times, model, label="Fitted model (transit)")
plt.plot(times, fluxes, label="True model")

plt.legend()
plt.show()
