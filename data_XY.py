#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:37:40 2020

This file contains a class data_XY which takes up X-Y data and returns
appropriate manipulated X-Y data for further analysis.

@author: shihab
"""

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

class data_XY:

# =============================================================================
#     Initiatlize attributes
# =============================================================================
    # Class attributes
    # Object attributes
    num_fig = 0

# =============================================================================
#    Define methods
# =============================================================================
    def __init__(self, dataX, dataY):
        self.df_X = dataX
        self.df_Y = dataY

    def preprocessing_features(self, sl, sly):
        if sl == None:
            self.df_X_scaled = self.df_X
            self.scaler = preprocessing.StandardScaler()
        elif sl == "standard":
            self.scaler = preprocessing.StandardScaler()
            self.df_X_scaled = self.scaler.fit_transform(self.df_X)
        else:
            raise Exception("Sorry, unrecognized scaler. Going for standard \
                              scaler")
            self.scaler = preprocessing.StandardScaler()
            self.df_X_scaled = self.scaler.fit_transform(self.df_X)
        if sly == True:
            self.df_Y_scaled = self.scaler.fit_transform(self.df_Y)
        else: self.df_Y_scaled = self.df_Y

    def transf_features(self, X):
        X_transf = self.scaler.transform(X)
        return X_transf

    def poly_features(self, od):

        poly_reg = PolynomialFeatures(degree=od)
        X_poly = poly_reg.fit_transform(self.df_X_scaled)
        feat_names = poly_reg.get_feature_names()
        self.df_X_poly = pd.DataFrame(data=X_poly, columns=feat_names)

    def returnXY(self, order=1, scale=None, scaley=False):
        self.preprocessing_features(scale, scaley)
        self.poly_features(order)
        return [self.df_X_poly, self.df_Y_scaled]