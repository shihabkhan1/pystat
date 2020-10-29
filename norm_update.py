#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:18:14 2020
    This script calculates the posterior distribution of normally distributed
    random variable with a normal likelihood
@author: shihab
"""
import numpy as np


def norm_upd(prior_dist, obs, err):
    """
    This function estimates the posterior distribution for a normal random
    variable when observed with an unbiased normal random likelihood function

    Parameters
    ----------
    prior_dist : LIST
        Takes a list with the mean and std deviation of prior distribution
        as given by [mu, std].
    obs : LIST
        a 1-D list of observations.
    err : float
        standard deviation (random error) of the normal likelihood function.

    Returns
    -------
    post_dist : LIST
        Mean and std deviation of preposterior distribution.
        [mu_, std_]

    """
    mu, std = prior_dist
    n = len(obs)
    std_ = (std**-2 + n/err**2)**-0.5
    mu_ = (mu/std**2 + np.sum(obs)/err**2)*std_**2
    post_dist = [mu_, std_]
    return post_dist