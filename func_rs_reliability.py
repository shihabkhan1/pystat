#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:08:00 2020
    This script contains a function for evaluating the probability of
    failure for an R-S problem with gaussnian random variables
@author: shihab
"""
import numpy as np
# from logn_params import *
import PyRe as pr

def rel(param_lnR, param_lnS, rate=0.01, time=0.0, degtype=1.0):
    """
    This function estimates the probability of failure for an R-S problem
    with lognormally distributed random variables.

    The problem under consideration is:
        R0 * (1-rate*t**degtype) - S

    Parameters
    ----------
    param_lnR : LIST
        [mean, cov] for lnR0
    param_lnS : LIST
        [mean, cov] for lnS
    rate : float, optional
        Rate of degradation. The default is 0.01.
    time : float, optional
        DESCRIPTION. The default is 0.0.
    degtype : float, optional
        degradation type. The default is 1.0.

    Returns
    -------
    PrRe analysis object
        Analysis object of PyRe library.

    """
    def LSF_ln_RS(X1, X2):
        """
        lnR0 + ln(1-kt^nu) - lnS
        """
        return X1 + np.log(1-k*t**nu) - X2

    k = rate
    t = time
    nu = degtype
    lnR00, lnR0cv = param_lnR
    lnR0s = lnR00*lnR0cv
    lnS00, lnS0cv = param_lnS
    lnS0s = lnS00*lnS0cv

    limit_state = pr.LimitState(LSF_ln_RS)

    # Set some options (optional)
    options = pr.AnalysisOptions()
    # options.printResults(False)

    stochastic_model = pr.StochasticModel()
    # Define random variables
    stochastic_model.addVariable(pr.Normal('X1', lnR00, lnR0s))
    stochastic_model.addVariable(pr.Normal('X2', lnS00, lnS0s))

    # Performe FORM analysis
    Analysis = pr.Form(analysis_options=options,
                    stochastic_model=stochastic_model, limit_state=limit_state)

    return Analysis

# if __name__ == '__main__':
# # =============================================================================
# #     Initialize inputs
# # =============================================================================
#     R00 = 2.5
#     R0cv = 0.1
#     S0 = 1
#     Scv = 0.3
#     # k = 0.01
#     # t = 0
#     # nu = 1
#     ln_params_R = logn_params(R00, R0cv)
#     ln_params_S = logn_params(S0, Scv)
#     input_lnR = [ln_params_R[0], ln_params_R[3]]
#     input_lnS = [ln_params_S[0], ln_params_S[3]]

#     results = rel(input_lnR, input_lnS)

#     results.showDetailedOutput()
#         # Some single results:
#     beta = results.getBeta()
#     pf = results.getFailure()