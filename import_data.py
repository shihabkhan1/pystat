#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:38:15 2022

@author: shihab
"""

import pandas as pd
import numpy as np
import pickle as pkl

def import_data(file, *args, **kwargs):
    """
    This function can import data from an excel or a csv file and return a
    pandas dataframe.

    Parameters
    ----------
    file : string
        Complete path to the file.
    skipr : list of int, optional
        The rows that need to be skipped beginning from 0. The default is None.
    headr : list of int, optional
        Specify the header rows. The default is None.
    colindex : list of int, optional
        Specify the indices of the columns. The default is None.

    Raises
    ------
    Exception
        If the filetype is not recognized, an exception is raised.

    Returns
    -------
    dfs : Dictionary of dataframes
        Pandas dataframe from the read file.

    """
    if file.endswith('.xlsx'):
        dfs = pd.read_excel(file, *args, **kwargs)
        # dfs = pd.read_excel(file, skiprows=skipr, header=headr,\
                             # index_col=colindex, sheet_name=None)
    elif file.endswith('.csv'):
        pass
    else:
        raise Exception("Sorry, filetype not recognized")


    return dfs