# install libraries

%pip install pygam

from pygam import LinearGAM, s, f, l
import pandas as pd
import patsy as pt
import numpy as np

# get data

training_data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/assignment_data_train.csv")

test_data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/assignment_data_test.csv")

eqn = "trips ~ hour + day + month + year"

y, x = pt.dmatrices(eqn, data = training_data)

model = LinearGAM(s(0) + s(1) + s(2) + s(3)) # init and fit model to spline based nonlinear for all params
"""
y = training_data['trips'].values # dep var vals
x = training_data[['hour', 'day', 'month', 'year']].values # indep vars vals
"""

modelFit = model.gridsearch(np.asarray(x), y) # fits model

#xPred = test_data[['hour', 'day', 'month', 'year']].values # indep vars vals

#xPred = x.append(xPred)

xPred = pt.build_design_matrices([x.design_info], test_data)

pred = model.predict(xPred[0]) # prediction for each hour of a year after final year of data