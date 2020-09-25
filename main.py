#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:22:51 2020

@author: shihab
"""

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import statsmodels.api as sm
import linreg


if __name__ == "__main__":
    t0 = time.time()
# =============================================================================
#     Import Data
# =============================================================================
    f = open('./data_regression.pkl', 'rb')
    dict_sens = pickle.load(f)
    f.close()
    design_transf = dict_sens["design_transf"]
    design_num = len(dict_sens["vect_EVI"])
    vect_EVI = np.array(dict_sens["vect_EVI"]).reshape(design_num,1)
    data_X = pd.DataFrame(data=design_transf, columns=['Cf', 'Cfl', 'CB',
                                                       'Cr', 'Crm'])
    data_Y = pd.DataFrame(data=vect_EVI/1e10, columns=['EVI/1e10'])

# =============================================================================
#     Test class linreg
# =============================================================================

    linreg_obj = linreg(data_X, data_Y)
# # =============================================================================
# #     Fit second degree polynomial
# # =============================================================================
#     order = 2
#     # mmscaler = preprocessing.MinMaxScaler()
#     # data_X_mmscaled = mmscaler.fit_transform(data_X)
#     # design_X_poly = poly_features(data_X_mmscaled, order)
#     scaler = preprocessing.StandardScaler()
#     data_X_scaled = scaler.fit_transform(data_X)
#     design_X_poly = poly_features(data_X_scaled, order)
#     num_pts, num_feat = design_X_poly.shape
#     pol_reg = LinearRegression(fit_intercept=False)
#     pol_reg.fit(design_X_poly, data_Y)
#     predictions = np.squeeze(pol_reg.predict(design_X_poly))
# # =============================================================================
# #     Fit first degree polynomial
# # =============================================================================
#     # design_X_poly_1 = poly_features(data_X, 1)
#     # lin_reg = sm.OLS(data_Y, design_X_poly_1)
#     # lin_fit = lin_reg.fit()
#     # print(lin_fit.summary())
# # =============================================================================
# #    Visualizing the Polymonial Regression results
# # =============================================================================
#     viz_predictions(design_X_poly, np.array(data_Y), pol_reg)
# # =============================================================================
# #    P-value signficance
# # =============================================================================
#     mod = sm.OLS(data_Y, design_X_poly)
#     fii = mod.fit()
#     print(fii.summary())
# # =============================================================================
# #     Manual Hypothesis testing
# # =============================================================================
#     y = np.array(data_Y.iloc[:,0])
#     err = y-predictions
#     # mse = sum(err**2)
#     mse = 3.062254619388217e-37
#     dof = num_pts - num_feat
#     v_statistic = []
#     t_statistic = []
#     for i in np.arange(0,num_feat,1):
#         B = fii.params[i]
#         x = np.array(design_X_poly.iloc[:,i])
#         # pred = B*x
#         # err = y-pred
#         # mse = sum(err**2)    
#         Sxx = square_diff(x,x)
#         print(Sxx)
#         v_statistic.append(np.sqrt(dof*Sxx/mse)*abs(B))
#         t_statistic.append(2*(1-sp.stats.t.cdf(abs(B),dof)))
        
#     v_statistic = np.array(v_statistic)
#     t_statistic = np.array(t_statistic)
#     # SSR = square_diff()
# =============================================================================
#   End
# =============================================================================
    tt = time.time()
    print('Time taken =', tt-t0, 'seconds')