from Lab1.minimization_full import *
import pandas as pd
import numpy as np
import math
import copy


data = pd.read_csv("parkinsons_updrs.csv")  # Import CSV into a dataframe
data = data.drop(columns=['subject#', 'age', 'sex', 'test_time'])  # Drop the first columns
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle rows and reset index


# F0 is the feature that we want to estimate (regress) from the other features(regressand).
F0 = 1

# Submatrices: training, validation and test.
data_train = data[0:math.ceil(data.shape[0]/2)-1]
data_test = data[math.floor(3/4*data.shape[0]):data.shape[0]]
data_val = data[math.floor(data.shape[0]/2):math.floor(3/4*data.shape[0])]

# Data normalization.
data_train_norm = copy.deepcopy(data_train)  # To preserve original data
data_test_norm = copy.deepcopy(data_test)  # To preserve original data
data_val_norm = copy.deepcopy(data_val)  # To preserve original data

for i in range(data_train.shape[1]):
    mean = np.mean(data_train.iloc[:, i])  # Calculate mean for data_train
    data_train_norm.iloc[:, i] -= mean
    data_val_norm.iloc[:, i] -= mean
    data_test_norm.iloc[:, i] -= mean
    std = np.std(data_train.iloc[:, i])  # Calculate standard deviation for data_train
    data_train_norm.iloc[:, i] /= std
    data_val_norm.iloc[:, i] /= std
    data_test_norm.iloc[:, i] /= std

# Mean and standard deviation in order to de-standardize data for the plots.
m = np.mean(data_train.iloc[:, F0])
s = np.std(data_train.iloc[:, F0])

if __name__ == "__main__":
    y_train = data_train_norm.iloc[:, F0]  # F0 column vector
    y_test = data_test_norm.iloc[:, F0]  # F0 column vector
    y_val = data_val_norm.iloc[:, F0]  # F0 column vector
    X_train = data_train_norm.drop(columns='total_UPDRS')  # Remove column F0
    X_test = data_test_norm.drop(columns='total_UPDRS')  # Remove column F0
    X_val = data_val_norm.drop(columns='total_UPDRS')  # Remove column F0

    logx = 0
    logy = 0

    # lls = SolveLLS("LLS", X_train.values, X_test.values, X_val.values, y_train.values, y_test.values, y_val.values, m, s)
    # lls.run()
    # lls.print_result(title='LLS')
    # lls.plot_w(title='LLS pseudoinverse: Weight vector')
    # lls.plot_y(title='LLS pseudoinverse: Estimated Y versus Real Y')
    # lls.plot_hist(title="LLS pseudoinverse: Estimation Error")
    #
    cong = SolveConjGrad("conjugate_gradient", X_train.values, X_test.values, X_val.values, y_train.values, y_test.values, y_val.values, m, s)
    cong.run()
    cong.print_result(title='Conjugate Gradient Algorithm')
    cong.plot_err(title='Conjugate Gradient: Mean Square error', logy=logy, logx=logx)
    cong.plot_y(title='Conjugate Gradient: Estimated Y versus Real Y')
    cong.plot_hist(title="Conjugate Gradient: Estimation Error")
    cong.plot_w(title='Conjugate Gradient: Weight vector')
    #
    # Nit = 200
    # gamma = 1e-5
    # # #'''
    # # #  It’s important with this algorithm to set appropriate values of Nit and γ : in order to choose
    # # #  right values, the program has to be run several times. Moreover, if γ is too small,
    # # #  it could take a lot of time to the program to find the optimum value of w; if γ is too large,
    # # #  the algorithm could not converge.
    # # # '''
    # g = SolveGrad("gradient", X_train.values, X_test.values, X_val.values, y_train.values, y_test.values, y_val.values, m, s)
    # g.run(gamma=gamma, Nit=Nit)
    # g.print_result(title='Gradient Algorithm')
    # g.plot_err(title='Gradient: Mean Square Error', logy=logy, logx=logx)  # find stop point: over fitting?
    # g.plot_y(title='Gradient: Estimated Y versus Real Y')
    # g.plot_hist(title="Gradient: Estimation Error")
    # g.plot_w(title="Gradient: Weight vector")
    #
    # Nit = 600
    # gamma = 1e-6
    # stg = SolveStochGrad("Stochastic_gradient", X_train.values, X_test.values, X_val.values, y_train.values, y_test.values, y_val.values, m, s)
    # stg.run(gamma=gamma, Nit=Nit)
    # stg.print_result(title="Stochastic Gradient Algorithm")
    # stg.plot_err(title="Stochastic Gradient: Mean Square Error", logy=logy, logx=logx)
    # stg.plot_y(title='Stochastic Gradient: Estimated Y versus Real Y')
    # stg.plot_hist(title="Stochastic Gradient: Estimation Error")
    # stg.plot_w(title='Stochastic Gradient: Weight vector')
    #
    # Nit = 300
    # gamma = 1e-3
    # sd = SolveSteepestDec("Steepest_descent", X_train.values, X_test.values, X_val.values, y_train.values, y_test.values, y_val.values, m, s)
    # sd.run(gamma=gamma, Nit=Nit)
    # sd.print_result(title="Steepest Decent Algorithm")
    # sd.plot_err(title='Steepest decent: Mean Square Error', logy=logy, logx=logx)
    # sd.plot_y(title='Steepest Descent: Estimated Y versus Real Y')
    # sd.plot_hist(title="Steepest Descent: Estimation Error")
    # sd.plot_w(title='Steepest Descent: Weight vector')

    # #Ridge regression. when lamba = 0 , we are again in the case of LLS, VERIFIED.
    # sr = SolveRidge("Ridge_regression", X_train.values, X_test.values, X_val.values, y_train.values, y_test.values, y_val.values, m, s)
    # sr.run()
    # sr.print_result(title='Ridge Regression Algorithm')
    # sr.plot_err(title='Ridge Regression: Mean Square Error', logy=logy, logx=logx)
    # sr.plot_y(title='Ridge Regression: Estimated Y versus Real Y')
    # sr.plot_hist(title="Ridge Regression: Estimation Error")
    # sr.plot_w(title='Ridge Regression: Weight vector')

