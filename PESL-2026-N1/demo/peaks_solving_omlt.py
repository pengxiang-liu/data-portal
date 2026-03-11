# ----------------------------------------------------------------------- #
# MIT license
# Copyright (c) 2026, 
# Data Analytics and Computational Intelligence (DACI) Laboratory,
# Department of Data Science, College of Computing,
# City University of Hong Kong,
# Dr. Pengxiang Liu, All Rights Reserved.
# ----------------------------------------------------------------------- #

"""
Description:
    This script demonstrates the pipeline for constraint learning using 
    Adaptive Sigmoid Partitioning (ASP) algorithm.

Mathematical Formulation:
    min   z
    s.t.  peaks(x, y) == z
          -3 <= x <= 3, -3 <= y <= 3
          x is binary; y and z are continuous

Matrix Formulation:
    min   c * x
    s.t.  A * x == e,
          B * x <= f,
          x[q_n] = 1 / (1 + exp(-x[p_n])), for all (p_n, q_n) in W,
          lb <= x <= ub,
          x_i is binary, for all (i) in Int
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import onnx
import pandas as pd
import numpy as np
import pyomo.environ as pyo

from omlt import OffsetScaling, OmltBlock
from omlt.neuralnet import FullSpaceSmoothNNFormulation
from omlt.io import load_onnx_neural_network


# ----------------------------------------------------------------------- #
# optimize by OMLT
def optimize_by_omlt(fp_csv, fp_net):

    # load the dataset
    col_x, col_y = ["x", "y"], ["z"]
    df = pd.read_csv(fp_csv, usecols = col_x + col_y)
    df_x, df_y = df[col_x], df[col_y]
    # get the scaling parameters to use later in optimization formulation
    x_offset, x_factor = df_x.mean().to_dict(), df_x.std().to_dict()
    y_offset, y_factor = df_y.mean().to_dict(), df_y.std().to_dict()
    # capture the minimum and maximum values of the scaled inputs
    scaled_lb = df_x.min()[col_x].to_numpy()
    scaled_ub = df_x.max()[col_x].to_numpy()

    # create the pyomo model
    m = pyo.ConcreteModel()

    # create the scaling object
    scaler = OffsetScaling(
        offset_inputs  = {i: x_offset[col_x[i]] for i in range(len(col_x))},
        factor_inputs  = {i: x_factor[col_x[i]] for i in range(len(col_x))},
        offset_outputs = {i: y_offset[col_y[i]] for i in range(len(col_y))},
        factor_outputs = {i: y_factor[col_y[i]] for i in range(len(col_y))},
    )
    scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) 
                           for i in range(len(col_x))}
    # load the model with bounds
    net = load_onnx_neural_network(
        onnx.load(fp_net),
        scaling_object = scaler, input_bounds = scaled_input_bounds
    )
    m.peaks = OmltBlock()
    m.peaks.build_formulation(FullSpaceSmoothNNFormulation(net))

    # now add the objective and the constraints
    m.obj = pyo.Objective(expr = -m.peaks.outputs[0], 
                          sense = pyo.minimize)
    m.con = pyo.ConstraintList()
    m.con.add(expr = m.peaks.inputs[0] >= -3)
    m.con.add(expr = m.peaks.inputs[0] <=  3)
    m.con.add(expr = m.peaks.inputs[1] >= -3)
    m.con.add(expr = m.peaks.inputs[1] <=  3)
    
    sf = pyo.SolverFactory("scip")
    status = sf.solve(m, tee = True)
    if status.solver.termination_condition == "optimal":
        sol = [pyo.value(m.peaks.inputs[i]) for i in range(len(col_x))]
        obj = pyo.value(m.peaks.outputs[0])
        print("Optimal solution:")
        print(f"sol: {sol}, obj: {obj}")
    else:
        print("No optimal solution found.")


# ----------------------------------------------------------------------- #
# main function
if __name__ == "__main__":

    # set the paths
    fp_dir = os.path.dirname(os.path.abspath(__file__))
    fp_csv = os.path.join(fp_dir, "peaks_data.csv")
    fp_net = os.path.join(fp_dir, "peaks_neural_approx.onnx")

    # solve the problem using OMLT
    optimize_by_omlt(fp_csv, fp_net)