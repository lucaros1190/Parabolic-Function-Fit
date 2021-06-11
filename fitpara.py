
# fitpara.py Python script to fit data with a second order polinomial function
# Created by Luca Rossini on 27 February 2021
# E-mail luca.rossini@unitus.it
# Last update 27 Febryary 2021

import pandas as pd
import plotly.graph_objs as go
from math import *
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats.distributions import chi2
from scipy import odr
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import sys

host='0.0.0.0'
port='8080'

# Read the data to fit and plot

Data = pd.read_csv("data.txt", sep="\t", header=None)


# Set the header of the dataset

Data.columns = ["x", "y", "err_x", "err_y"]

x = Data['x']
y = Data['y']
err_x = Data['err_x']
err_y = Data['err_y']

# Fit data with a parabolic function
    # Definition of the parabolic function
    
def parabfun(y, x):
    a, b, c = y
    return a * x * x + b * x + c

    # Create an object for odr
    
parab = odr.Model(parabfun)

    # Create a RealData object

dataset = odr.RealData(x, y, sx=err_x, sy=err_y)

    # Set ODR with data and model
    
fit = odr.ODR(dataset, parab, beta0=[1., 1., 1.])

    # Fit

out = fit.run()

# Best fit line and errors

best_fit_line = out.beta
err = out.sd_beta

# Ask how many sigma do you want to include in the confidence band

print('\n How many sigma do you want to include in the confidence band? (2 sigma is 95%): \n')

num_sigma = float(input())

# Upper ad lower confidence bands

    # Create the linespace to plot the best fit curve
x_fun = np.linspace(0, 60, 1000)

    # Calculations
fitted_fun = parabfun(best_fit_line, x)
fitted_plot = parabfun(best_fit_line, x_fun)
parabfun_up = best_fit_line + num_sigma * err
parabfun_low = best_fit_line - num_sigma * err

upper_fit = parabfun(parabfun_up, x_fun)
lower_fit = parabfun(parabfun_low, x_fun)

# Calculating R-squared

resid = y - fitted_fun
ss_res = np.sum(resid**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)


# Number of degrees of freedom (NDF)

ndf = len(x) - 3

# Calculate the chi-squared (with error below the fraction)

chi_sq = 0
for i in range(len(x)):
    chi_sq = pow((y[i] - fitted_fun[i]), 2)/err_y[i]

# Calculate the P-value from chi-square

Pvalue = 1 - chi2.sf(chi_sq, ndf)

# Calculate AIC and BIC

AIC = 2 * 3 - 2 * np.log(ss_res/len(x))
BIC = 3 * np.log(len(x)) - 2 * np.log(ss_res/len(x))

# Extrapolate the covariance matrix

covmat = out.cov_beta

# Print the results

print('\n Parabolic fit (a * x * x + b * x + c) results: \n')

    # Define the parameters' name
parname = (' a = ', ' b = ', ' c = ')

for i in range(len(best_fit_line)):
    print(parname[i] + str(round(best_fit_line[i], 7)) + ' +/- ' + str(round(err[i],7)))

print(' R-squared = ', round(r_squared, 5))
print(' Chi-squared = ', round(chi_sq, 5))
print(' P-value = ', round(Pvalue, 7))
print(' Number of degrees of freedom (NDF) =', ndf)
print(' Akaike Information Criterion (AIC):', round(AIC, 5))
print(' Bayesian Information Criterion (BIC)', round(BIC, 5))
print('\n')

print(' Covariance matrix: \n')
    # Define the row names to print
rowname = (' ', 'a ', 'b ', 'c ')
print(rowname[0] + '  \t' + rowname[1] + '  \t\t' + rowname[2]+ '  \t\t' + rowname[3])

for i in range(len(covmat)):
    print(rowname[i+1] + ' ' + str(covmat[i]))

print(' ')

# Plot the data

app = dash.Dash()

trace_points = go.Scatter(
    x=x,
    y=y,
    mode='markers',
    name='Experimental points',
    error_y=dict(
        type='data',
        array=err_y,
        visible=True,
        color='purple',
        thickness=1,
        width=3
        ),
    error_x=dict(
        type='data',
        array=err_x,
        visible=True,
        color='purple',
        thickness=1,
        width=3
        ),
    marker=dict(
        color='purple',
        size=7
        ),
)

trace_fit = go.Scatter(
    x=x_fun,
    y=fitted_plot,
    mode='lines',
    name='Best fit function'
)
    
trace_UpConfBand = go.Scatter(
    x=x_fun,
    y=upper_fit,
    mode='lines',
    marker=dict(color="#444"),
    line=dict(width=0),
    name='Upper bound',
    fillcolor='rgba(68, 122, 219, 0.25)',
    fill='tonexty',
    showlegend=False
)

band_legend = str(num_sigma) + ' sigma interval data'

trace_LowConfBand = go.Scatter(
    x=x_fun,
    y=lower_fit,
    mode='lines',
    marker=dict(color="#444"),
    line=dict(width=0),
    name=band_legend,
    fillcolor='rgba(68, 122, 219, 0.25)',
    fill='tonexty',
)

fig = go.Figure(data=[trace_points, trace_fit, trace_LowConfBand, trace_UpConfBand])


# Set plot tile and layout

fig.update_layout(
    title='Best fit function',
    yaxis_title='y-axis',
    xaxis_title='x-axis',
    autosize=True,
    height=650,
    font=dict(
        size=15
    ),
    plot_bgcolor='rgba(0,0,0,0)',

)

# Set x-axis options
fig.update_xaxes(
    showline=True,
    linewidth=2,
    linecolor='black',
    showgrid=True,
    gridwidth=1,
    gridcolor='LightBlue'
)

# Set y-axis options
fig.update_yaxes(
    showline=True,
    linewidth=2,
    linecolor='black',
    showgrid=True,
    gridwidth=1,
    gridcolor='LightBlue'
)

# Plotting..!
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(port=port, host=host, debug=False, use_reloader=False)

