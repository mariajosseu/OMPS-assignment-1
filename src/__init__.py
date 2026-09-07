"""Starter code for 46750 Assignment 1 - Demand-Side Flexibility in Active Distribution Grids.

Modules
-------
data_loader : read the JSON files of one question folder into an ``InputData`` object
model       : gurobipy model of the flexible consumer (build / solve / extract primal & dual values)
scenarios   : helpers to derive sensitivity scenarios (prices, tariffs, PV, preferences) from base data
plotting    : matplotlib figures for input data and optimisation results
"""
