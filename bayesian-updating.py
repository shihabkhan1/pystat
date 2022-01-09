#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:46:07 2021

@author: shihab
"""

import numpy as np
from norm_update import *
import scipy.stats as st
import matplotlib.pyplot as plt

def updt_pyplot_params():
    SMALL_SIZE = 11
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 15

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    return

if __name__ == "__main__":
    param_R = [3.5, 0.35]
    obs = [4.0]
    sig_eps = 0.5
    param_R_updt = norm_upd(param_R, obs, sig_eps)

    vect_r = np.arange(1.0, 6.0, 0.1)
    likelihood = [st.norm.pdf(obs, xx, sig_eps) for xx in vect_r]
    prior = [st.norm.pdf(xx, param_R[0], param_R[1]) for xx in vect_r]
    post = [st.norm.pdf(xx, param_R_updt[0], param_R_updt[1]) for xx in vect_r]

    updt_pyplot_params()
    plt.figure(1)
    plt.plot(vect_r, prior, label='Prior')
    plt.plot(vect_r, post, '-.',label='Posterior')
    plt.plot(vect_r, likelihood, '--',label='Likelihood')
    plt.legend()
    plt.xlabel('Resistance [units]')
    plt.ylabel('Probability density')
    plt.tight_layout()
    plt.savefig('norm-updt-obs-4.pdf', format='pdf')
