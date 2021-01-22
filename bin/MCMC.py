"""Markov-Chain Monte Carlo (MCMC) Module of BANAnA pipeline
Copyright (c) 2020 Eric Zinn

Contains definitions relating to fitting curves to data by MCMC through pystan

This code is free software;  you can redstribute it and/or modify it under the terms of the AGPL
license (see LICENSE file included in the distribution)
"""
import pystan
import numpy as np


def doseCurve(x, EC50, Hillslope):
    """Define a function for the dose-response curve (sourced from the Graphpad
       Prism documentation?)"""
    return (x**Hillslope) / (x**Hillslope + EC50**Hillslope)


def initializeInvLogitModel():
    # Stan Code to fit the dose-response curve
    doseResponseCode = """
	data {
	    int<lower=0> N; // number of rows (dilutions)
	    vector[N] logConcentration;
	    vector[N] neutralization; //
	}
	parameters {
	    real logEC50;
	    real Hillslope;
	    real<lower=0> sigma;
	}
	model {
	    vector[N] numerator;
	    vector[N] denominator;
	    vector[N] mu;
	    
	    real EC50;
	    vector[N] Concentration;
	    
	    EC50 = pow(10, logEC50);
	    
	    for (n in 1:N){
	        Concentration[n] = pow(10, logConcentration[n]);
	    }
	    
	    for (n in 1:N){
	        numerator[n] = pow(Concentration[n],Hillslope);
	    }
	    for (n in 1:N){
	        denominator[n] = pow(Concentration[n], Hillslope)+pow(EC50, Hillslope);
	    }
	    for (n in 1:N){
	        mu[n] = numerator[n] / denominator[n];
	    }   
	    
	    sigma ~ cauchy(0, 0.1);
	    //logEC50 ~ normal(0, 5);
	    logEC50 ~ uniform(-5, -0.001);
	    //Hillslope ~ normal(-1, 5);
	    Hillslope ~ uniform(-10, -0.1);
	    
	    neutralization ~ normal(mu, sigma);
	}
	"""
    model = pystan.StanModel(model_code = doseResponseCode)

    return model


def fitData(neutralizationSeries, concentrationSeries, model):
    print("Fitting " + neutralizationSeries.name + "...")
    fit = model.sampling(data = dict(
        N = len(neutralizationSeries),
        logConcentration = np.log10(concentrationSeries),
        neutralization = neutralizationSeries,
    ), iter = 100000, chains = 1, thin = 10,
        control = {"adapt_delta": 0.99})

    return fit
