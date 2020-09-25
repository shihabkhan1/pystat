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
from sklearn.model_selection import train_test_split
import pandas as pd
import statsmodels.api as sm

class linreg:
    """ Linear Regression Class
    
    """
# =============================================================================
#     Initiatlize attributes
# =============================================================================
    # Class attributes
    # Object attributes
    num_fig = 0

# =============================================================================
#    Define methods
# =============================================================================
    def __init__(self, dataX, dataY, order=1, split=0.2):
        self.df_X = dataX
        self.df_Y = dataY
        self.order = order

    def preprocessing_features(self):
        self.scaler = preprocessing.StandardScaler()
        self.df_X_scaled = scaler.fit_transform(self.df_X)

    def transf_features(self, X):
        X_transf = self.scaler.transform(X)
        return X_transf

    def poly_features(self):

        poly_reg = PolynomialFeatures(degree=order)
        X_poly = poly_reg.fit_transform(self.df_X_scaled)
        feat_names = poly_reg.get_feature_names()
        self.df_X_poly = pd.DataFrame(data=X_poly, columns=feat_names)

    def split_features(self):
        self.df_X_train, self.df_X_test, self.df_Y_train, self.df_Y_test = ...
        train_test_split(df_X_poly, df_Y, test_size=split, random_state=
                         time.time())
        self.num_pts_total, self.num_feat = self.df_X_poly.shape
        self.num_pts_train, _ = self.df_X_train.shape
        self.num_pts_test, _ = self.df_X_test.shape

    def linregression(self):
        self.preprocessing_features()
        self.poly_features()
        self.split_features()
        self.pol_reg = LinearRegression(fit_intercept=False)
        self.pol_reg.fit(self.df_X_train, self.df_Y_train)
        self.pred_test = self.predictions_linreg(self.df_X_test)
        self.pred_train = self.predictions_linreg(self.df_X_train)
        print("Linear regression done")
        return self.pol_reg

    def predictions_linreg(self, X, transf=0):
        X = self.transf_features(X) if transf==1 else X
        pred = self.pol_reg.predict(X)
        return pred

    def viz_predictions(self):
        self.num_fig += 1
        label = list(self.df_Y.columns)
        y = self.df_Y[label].values
        y_train = self.df_Y_train[label].values
        y_test = self.df_Y_test[label].values
        plt.figure(self.num_fig)
        plt.plot(y, y, color='blue', label='True')
        plt.scatter(self.pred_train, y_train, color='red', label='Pred--train')
        plt.scatter(self.pred_test, y_test, color='yellow', label='Pred--test')
        plt.title('Truth or Bluff (Linear Regression)')
        plt.xlim(min(y)*0.9, max(y)*1.1)
        plt.ylim(min(max(y_train), max(y_test))*0.9, max(max(y_train),
                                                         max(y_test))*1.1)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    def viz_scatter1D(X, y, label=['Parameter', 'True']):
        plt.scatter(X, y, color='red')
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.xlim(min(X)*0.9, max(X)*1.1)
        plt.ylim(min(y)*0.9, max(y)*1.1)
        plt.title('Relationship')
        plt.tight_layout()
        plt.show()
        return

    def square_diff(X, Y):
        mean_X = np.mean(X)
        mean_Y = np.mean(Y)
        diff_X = X-mean_X
        diff_Y = Y-mean_Y
        S = np.sum(diff_X*diff_Y)
        return S
    
    
