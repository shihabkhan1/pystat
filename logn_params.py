#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:32:13 2020
    This script contains a function for calculating the equivalent normal
    parameters of a lognormally distributed random variable
@author: shihab
"""
import numpy as np

def logn_params(mean=0.0, cov=0.1):
    """
    This function calculates the equivalent normal
    parameters of a lognormally distributed random variable

    Parameters
    ----------
    mean : float, optional
        mean of RV. The default is 0.
    cov : float, optional
        Coefficient of variation of RV. The default is 0.1.

    Returns
    -------
    list
        [ln0, lns, lnv, lncv].

    """
    std = mean*cov
    ln0 = np.log(mean/np.sqrt(1+cov**2))
    lnv = np.log(1+cov**2)
    lns = np.sqrt(lnv)
    lncv = lns/ln0
    return [ln0, lns, lnv, lncv]