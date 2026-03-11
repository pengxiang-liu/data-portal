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
import onnx
import time
import pandas as pd
import numpy as np
import pyomo.environ as pyo

from onnx import numpy_helper

# ----------------------------------------------------------------------- #
# create the neural approximation
class define_metadata(object):

    def __init__(self, fp_csv, fp_net):

        self.net = define_neural_network(fp_csv, fp_net)
        self.var = define_variable_indices(self.net)
        self.mat = define_coefficient_matrices(self.net, self.var)


# ----------------------------------------------------------------------- #
# define the neural network parameters
class define_neural_network(object):
    
    # initialization
    def __init__(self, fp_csv, fp_net):  

        self.dataset_normalization(fp_csv) 
        self.get_onnx_forward_propagation(fp_net)

    def dataset_normalization(self, fp_csv): 
        # load the dataset
        col_x, col_y = ["x", "y"], ["z"]
        df = pd.read_csv(fp_csv, usecols = col_x + col_y)
        df_x, df_y = df[col_x], df[col_y]
        # get the scaling parameters to use later in optimization formulation
        x_offset, x_factor = df_x.mean().values, df_x.std().values
        y_offset, y_factor = df_y.mean().values, df_y.std().values
        # capture the minimum and maximum values of the scaled inputs
        scaled_lb = df_x.min()[col_x].to_numpy()
        scaled_ub = df_x.max()[col_x].to_numpy()
        # store the parameters
        self.x_offset, self.x_factor = x_offset, x_factor
        self.y_offset, self.y_factor = y_offset, y_factor
        self.scaled_lb, self.scaled_ub = scaled_lb, scaled_ub
    
    def get_onnx_forward_propagation(self, fp_net):
        onnx_model = onnx.load(fp_net)
        nodes, weights = {}, {}
        for node in onnx_model.graph.node:
            nodes[node.name] = {
                "input": node.input, "output": node.output,
                "op_type": node.op_type
            }
        for initializer in onnx_model.graph.initializer:
            weight_key = initializer.name
            weight_val = numpy_helper.to_array(initializer)
            weights[weight_key] = weight_val
        self.onnx = {}
        for name, info in nodes.items():
            data = {}
            data["input_vec"] = info["input"][0]
            data["output_vec"] = info["output"][0]
            match info["op_type"]:
                case "Gemm":
                    data["weight"] = weights[info["input"][1]]
                    data["bias"] = weights[info["input"][2]]
                    data["input_dim"] = data["weight"].shape[1]
                    data["output_dim"] = data["weight"].shape[0]
                    data["type"] = "gemm"
                case "Sigmoid":
                    upstream = self.onnx[f"node_{info['input'][0]}"]
                    data["input_dim"] = upstream["output_dim"]
                    data["output_dim"] = data["input_dim"]
                    data["type"] = "sigmoid"
                case _:
                    raise NotImplementedError("Unsupported ONNX op_type.")
            self.onnx[name] = data


# ----------------------------------------------------------------------- #
# define the indices of variables
class define_variable_indices(object):
    # initialization
    def __init__(self, net):
        self.get_net_variables(net)
        # create variable dictionary (shape)
        # 1) binary variables
        self.binvar = {}
        # 2) continuous variables
        self.sdpvar = {
            k: v for k, v in self.onnx_vars.items()
        }
        # set variable information
        self.set_var_indices()
        self.set_var_types()
        self.set_var_bounds(net)
    
    # forward propagation variables
    def get_net_variables(self, net):
        self.onnx_vars = {}
        # extract variable dimensions
        for node in net.onnx.values():
            for k in ["input", "output"]:
                key = node[f"{k}_vec"]
                val = node[f"{k}_dim"]
                if key in self.onnx_vars:
                    assert self.onnx_vars[key] == val, "Error!"
                else:
                    self.onnx_vars[key] = val
        # denormalized input and output variables
        self.onnx_vars["input_original"] = self.onnx_vars["input"]
        self.onnx_vars["output_original"] = self.onnx_vars["output"]
    
    # set variable indices
    def set_var_indices(self):
        variables = dict(self.binvar, **self.sdpvar)
        n_vars = 0
        for key, val in variables.items():
            indices = np.arange(np.prod(val)).reshape(val) + n_vars
            n_vars += np.prod(val)
            setattr(self, f"n_{key}", indices.tolist())
        self.n_vars = n_vars
    
    # set variable types
    def set_var_types(self):
        n_bin_var = sum([np.prod(i) for i in self.binvar.values()])
        n_sdp_var = sum([np.prod(i) for i in self.sdpvar.values()])
        self.type = ["B"] * n_bin_var + ["C"] * n_sdp_var
    
    # set variable bounds
    def set_var_bounds(self, net):
        # initialize the bounds
        self.lb = np.ones(self.n_vars) * -20
        self.ub = np.ones(self.n_vars) *  20
        # set bounds for denormalized input variables
        input_original_indices = getattr(self, "n_input_original")
        for i in range(len(input_original_indices)):
            self.lb[input_original_indices[i]] = net.scaled_lb[i]
            self.ub[input_original_indices[i]] = net.scaled_ub[i]


# ----------------------------------------------------------------------- #
# define the coefficients of the model
class define_coefficient_matrices(object):
    # initialization
    def __init__(self, net, var):
        # initialize matrices
        names = "c, A, e, B, f, W, lb, ub".split(", ")
        mat = {i: [] for i in names}
        mat["lb"], mat["ub"] = var.lb, var.ub
        # set model coefficients
        # 1) objective function
        mat = self.add_obj(net, var, mat)
        # 2) constraints
        mat = self.add_constr_denormalization(net, var, mat)
        for info in net.onnx.values():
            match info["type"]:
                case "gemm":
                    mat = self.add_constr_gemm(info, var, mat)
                case "sigmoid":
                    mat = self.add_constr_sigmoid(info, var, mat)
        # formatting
        for i in names:
            setattr(self, i, mat[i])
    
    # add objective function
    def add_obj(self, net, var, mat):

        # initialize empty matrices
        c = {}

        # create cost coefficient
        for i in range(len(var.n_output)):
            c[var.n_output_original[i]] = -1  # maximize the output

        # update format and return
        mat = self.update_coef(mat, sp = {"c": c})
        return mat

    # add constraint for denormalization
    def add_constr_denormalization(self, net, var, mat):

        # get info
        x_offset, x_factor = net.x_offset, net.x_factor
        y_offset, y_factor = net.y_offset, net.y_factor

        # add equality constraints (A * x == e)
        # 1) input denormalization
        #    -> x_original = factor * x + offset
        for i in range(len(var.n_input)):
            # initialize empty matrices
            A = {}
            # update elements
            A[eval(f"var.n_input_original[i]")] = 1
            A[eval(f"var.n_input[i]")] = -x_factor[i]
            e = x_offset[i]
            # update coefficient
            mat = self.update_coef(mat, sp = {"A": A}, ap = {"e": e})
        # 2) output denormalization
        #    -> y_original = factor * y + offset
        for i in range(len(var.n_output_original)):
            # initialize empty matrices
            A = {}
            # update elements
            A[eval(f"var.n_output_original[i]")] = 1
            A[eval(f"var.n_output[i]")] = -y_factor[i]
            e = y_offset[i]
            # update coefficient
            mat = self.update_coef(mat, sp = {"A": A}, ap = {"e": e})
        
        # return
        return mat
    
    # add constraint for general matrix multiplication
    def add_constr_gemm(self, info, var, mat):
        # get info
        weight, bias = info["weight"], info["bias"]
        input_vec, output_vec = info["input_vec"], info["output_vec"]
        input_dim, output_dim = info["input_dim"], info["output_dim"]
        
        # add equality constraints (A * x == e)
        # 1) linear forward propagation
        #    -> y = weight * x + bias
        for i in range(output_dim):
            # initialize empty matrices
            A = {}
            # update elements
            for j in range(input_dim):
                A[eval(f"var.n_{input_vec}[j]")] = weight[i][j]
            A[eval(f"var.n_{output_vec}[i]")] = -1
            e = -bias[i]
            # update coefficient
            mat = self.update_coef(mat, sp = {"A": A}, ap = {"e": e})
        
        # return
        return mat

    # add constraint for sigmoid activation function
    def add_constr_sigmoid(self, info, var, mat):
        # get info
        input_vec, output_vec = info["input_vec"], info["output_vec"]
        input_dim, output_dim = info["input_dim"], info["output_dim"]
        assert input_dim == output_dim, "Inconsistent dimensions!"

        # add sigmoid constraints
        # 1) linear forward propagation
        #    -> y = 1 / (1 + exp(-x))
        for i in range(input_dim):
            # initialize empty matrices
            W = []
            # update elements
            W.append(eval(f"var.n_{input_vec}[i]"))
            W.append(eval(f"var.n_{output_vec}[i]"))
            # update coefficient
            mat = self.update_coef(mat, ap = {"W": W})
        
        # return
        return mat

    # update coefficient format
    def update_coef(self, mat, sp = {}, ap = {}):
    
        # 1. update coefficient
        # 1.1) sparse
        for n in sp:
            ind, val = list(sp[n].keys()), list(sp[n].values())
            mat[n].append({"ind": ind, "val": val})
        # 1.2) append
        for n in ap:
            mat[n].append(ap[n])
        
        # 2. return
        return mat


# ----------------------------------------------------------------------- #
# sigmoid function
def func_sigmoid(x, decimals = 8):
    """Calculates the sigmoid function."""
    y = 1 / (1 + np.exp(-x))
    if decimals is not None:
        return np.round(y, decimals = decimals)
    else:
        return y


# ----------------------------------------------------------------------- #
# derivative of sigmoid function
def func_sigmoid_derivative(x, decimals = 8):
    """Calculates the derivative of the sigmoid function."""
    s = func_sigmoid(x, decimals = None)
    y = s * (1 - s)
    if decimals is not None:
        return np.round(y, decimals = decimals)
    else:
        return y


# ----------------------------------------------------------------------- #
# build pyomo model
def build_pyomo_model(meta):

    # initialize pyomo
    model = pyo.ConcreteModel()
    model.linear_constr = pyo.ConstraintList()
    model.neural_constr = pyo.ConstraintList()
    model.active_constr = pyo.ConstraintList()
    
    # get the variable and matrix from metadata
    var, mat = meta.var, meta.mat

    # 2. add variables and objective
    # 2.1) binary and continuous variables
    x = pyo.Var(range(var.n_vars), domain = pyo.Reals)
    model.x = x
    for n, vtype in enumerate(var.type):
        x[n].domain = pyo.Binary if vtype == "B" else pyo.Reals
    # 2.2) set objective coefficient
    expr = 0
    for n, c in enumerate(mat.c):
        ind, val = c["ind"], c["val"]
        temp = [val[i] * x[ind[i]] for i in range(len(ind))]
        expr = expr + sum(temp)
    model.obj = pyo.Objective(expr = expr)
    # 2.3) specify input and output
    model.input = [x[i] for i in var.n_input_original]
    model.output = [x[i] for i in var.n_output_original]
    
    # 3. add constraints
    # 3.1) linear constraints
    for n in range(len(mat.A)):
        A, e = mat.A[n], mat.e[n]
        ind, val = A["ind"], A["val"]
        expr = np.sum([val[i] * x[ind[i]] for i in range(len(ind))])
        model.linear_constr.add(expr == e)
    for n in range(len(mat.B)):
        B, f = mat.B[n], mat.f[n]
        ind, val = B["ind"], B["val"]
        expr = sum([val[i] * x[ind[i]] for i in range(len(ind))])
        model.linear_constr.add(expr <= f)
    # 3.2) lower and upper bounds
    for n in range(var.n_vars):
        lb, ub = mat.lb, mat.ub
        model.linear_constr.add(x[n] >= lb[n])
        model.linear_constr.add(x[n] <= ub[n])
    # 3.3) neural constraints
    for n in range(len(mat.W)):
        W = mat.W[n]
        expr = 0
        expr = x[W[1]] - 1 / (1 + pyo.exp(-x[W[0]]))
        model.neural_constr.add(expr == 0)

    # return model
    return model


# ----------------------------------------------------------------------- #
# solve pyomo model
def solve_pyomo_model(model, solver = "scip", tee = True, time_limit = 600):

    # specify solver options
    solver = solver
    match solver:
        case "gurobi":
            sf = pyo.SolverFactory("gurobi")
            sf.options["TimeLimit"] = time_limit
            assert sf.available(), f"{solver} is unavailable!"
        case "scip":
            sf = pyo.SolverFactory("scip")
            sf.options["limits/time"] = time_limit
            assert sf.available(), f"{solver} is unavailable!"
        case "ipopt":
            sf = pyo.SolverFactory("ipopt")
            sf.options["max_cpu_time"] = time_limit
            assert sf.available(), f"{solver} is unavailable!"
        case _:
            raise ValueError(f"Unknown solver: {solver}")

    # solve model
    diag = sf.solve(model, tee = tee)
    state_0 = diag.solver.termination_condition == "optimal"
    state_1 = diag.solver.message == "gap limit reached"
    optimal = True if state_0 or state_1 else False
    if optimal:
        res = {
            "sol": np.array([pyo.value(model.x[i]) for i in model.x]),
            "obj": pyo.value(model.obj)
        }
    else:
        print("No optimal solution found.")
        res = None
    return res


# ----------------------------------------------------------------------- #
# adaptive sigmoid partitioning
def adaptive_sigmoid_partitioning(meta, fp_npz):

    # get the attributes
    net, var, mat = meta.net, meta.var, meta.mat
    # set tolerance for partition refinement
    tol_partition = 1e-3

    # pre-solve the model
    root = build_pyomo_model(meta)
    base = solve_pyomo_model(root, solver = "ipopt")
    
    # optimization-based bound-tightening
    if not os.path.exists(fp_npz):
        [lb, ub] = sequential_bound_tightening(meta, root, base)
        np.savez_compressed(fp_npz, lb = lb, ub = ub)

    peaks_obbt = np.load(fp_npz, allow_pickle = True)
    lb, ub = peaks_obbt["lb"], peaks_obbt["ub"]
    partition = initialize_partition(mat.W, lb, ub)
    mdl_tight = root.clone()
    for n in range(var.n_vars):
        mdl_tight.linear_constr.add(mdl_tight.x[n] >= lb[n])
        mdl_tight.linear_constr.add(mdl_tight.x[n] <= ub[n])
    
    # best incumbent
    res = solve_pyomo_model(mdl_tight, solver = "ipopt")
    inc_sol, inc_obj = res["sol"], res["obj"]
    result = {"sol": [pyo.value(i) for i in mdl_tight.input], 
              "obj": pyo.value(mdl_tight.output[0])}
    
    # best bound
    model = piecewise_sigmoid_relaxation(mdl_tight, partition)
    model.neural_constr.clear()
    res = solve_pyomo_model(model, solver = "gurobi")
    bnd_sol, bnd_obj = res["sol"], res["obj"]
    
    # main loop
    incumbent = True
    while not np.allclose(inc_obj, bnd_obj, rtol = 1e-2, atol = 1e-2):
        
        # refine partition and solve
        for (p, q) in mat.W:
            part = partition[(p, q)][0]  # partition for x[p]
            # find the active partition
            s = inc_sol[p] if incumbent else bnd_sol[p]
            for i in range(len(part) - 1):
                if part[i] <= s <= part[i + 1]:
                    idx = i
                    break
            # if the current one is too small, set to the larger one
            if part[idx + 1] - part[idx] <= tol_partition:
                sec = [part[i + 1] - part[i] for i in range(len(part) - 1)]
                idx = int(np.argmax(sec))
            # refine the partition
            x_opt = optimal_sigmoid_partitioning(part[idx], part[idx + 1])
            partition[(p, q)][0].insert(idx + 1, x_opt)
            partition[(p, q)][1].insert(idx + 1, func_sigmoid(x_opt))
        # rebuild model with refined partition
        mdl_relax = piecewise_sigmoid_relaxation(mdl_tight, partition)
        mdl_relax.neural_constr.clear()
        # solve
        res_relax = solve_pyomo_model(mdl_relax, solver = "gurobi")
        # update best bound
        bnd_sol, bnd_obj = res_relax["sol"], res_relax["obj"]
        
        # get active partition bounds
        lb_act, ub_act = lb.copy(), ub.copy()
        for (p, q) in mat.W:
            z = getattr(mdl_relax, f"z_{p}_{q}")
            i = [round(x) for x in list(z.get_values().values())].index(1)
            lb_act[p] = partition[(p, q)][0][i]
            ub_act[p] = partition[(p, q)][0][i + 1]
            lb_act[q] = partition[(p, q)][1][i]
            ub_act[q] = partition[(p, q)][1][i + 1]
        
        # local search around the bnd_obj solution
        mdl_local = mdl_tight.clone()
        for n in range(var.n_vars):
            mdl_local.linear_constr.add(mdl_local.x[n] >= lb_act[n])
            mdl_local.linear_constr.add(mdl_local.x[n] <= ub_act[n])
        # solve
        res_local = solve_pyomo_model(mdl_local, solver = "ipopt")
        # update best solution and objective
        incumbent = False
        if res_local is not None:
            if res_local["obj"] < inc_obj:
                inc_sol, inc_obj = res_local["sol"], res_local["obj"]
                result = {"sol": [pyo.value(i) for i in mdl_local.input], 
                          "obj": pyo.value(mdl_local.output[0])}
                incumbent = True

        # print current status
        print(f"Current incumbent obj: {inc_obj}, bound obj: {bnd_obj}")
        print(f"Current gap: {(inc_obj - bnd_obj) / abs(inc_obj)}")
    
    # retrieve final solution
    print("Optimal solution:")
    print(f"sol: {result['sol']}, obj: {result['obj']}")


# ----------------------------------------------------------------------- #
# piecewise sigmoid relaxation
def piecewise_sigmoid_relaxation(mdl, partition):
    # copy the model
    model = mdl.clone()
    # add binary variables indicating the active partition
    z = {}
    for (p, q), [bound_p, bound_q] in partition.items():
        n_part = len(bound_p) - 1
        z[(p, q)] = pyo.Var(range(n_part), domain = pyo.Binary)
        model.add_component(f"z_{p}_{q}", z[(p, q)])
        # add partitioning constraints
        model.active_constr.add(
            expr = sum(z[(p, q)][i] for i in range(n_part)) == 1
        )
        # add piecewise linear relaxations
        for i in range(n_part):
            lb_p, ub_p = bound_p[i], bound_p[i + 1]
            assert lb_p * ub_p >= 0, "Partition should not cross zero!"
            if lb_p <= 0 and ub_p <= 0:
                model = add_lhs_branch(model, lb_p, ub_p, p, q, z, i)
            if lb_p >= 0 and ub_p >= 0:
                model = add_rhs_branch(model, lb_p, ub_p, p, q, z, i)
    # return model
    return model


# ----------------------------------------------------------------------- #
# add left-hand side branch
def add_lhs_branch(model, lb_p, ub_p, p, q, z, i):
    # tangent at lb_p
    slope = func_sigmoid_derivative(lb_p)
    intercept = func_sigmoid(lb_p) - slope * lb_p
    model.active_constr.add(
        expr = model.x[q] >= slope * model.x[p] + intercept - 
                        (1 - z[(p, q)][i]) * 1e4
    )
    # tangent at ub_p
    slope = func_sigmoid_derivative(ub_p)
    intercept = func_sigmoid(ub_p) - slope * ub_p
    model.active_constr.add(
        expr = model.x[q] >= slope * model.x[p] + intercept - 
                        (1 - z[(p, q)][i]) * 1e4
    )
    # secant between (lb_p, f(lb_p)) and (ub_p, f(ub_p))
    slope = (func_sigmoid(ub_p) - func_sigmoid(lb_p)) / (ub_p - lb_p)
    intercept = func_sigmoid(lb_p) - slope * lb_p
    model.active_constr.add(
        expr = model.x[q] <= slope * model.x[p] + intercept + 
                        (1 - z[(p, q)][i]) * 1e4
    )
    # return model
    return model


# ----------------------------------------------------------------------- #
# add right-hand side branch
def add_rhs_branch(model, lb_p, ub_p, p, q, z, i):
    # tangent at lb_p
    slope = func_sigmoid_derivative(lb_p)
    intercept = func_sigmoid(lb_p) - slope * lb_p
    model.active_constr.add(
        expr = model.x[q] <= slope * model.x[p] + intercept + 
                        (1 - z[(p, q)][i]) * 1e4
    )
    # tangent at ub_p
    slope = func_sigmoid_derivative(ub_p)
    intercept = func_sigmoid(ub_p) - slope * ub_p
    model.active_constr.add(
        expr = model.x[q] <= slope * model.x[p] + intercept + 
                        (1 - z[(p, q)][i]) * 1e4
    )
    # secant between (lb_p, f(lb_p)) and (ub_p, f(ub_p))
    slope = (func_sigmoid(ub_p) - func_sigmoid(lb_p)) / (ub_p - lb_p)
    intercept = func_sigmoid(lb_p) - slope * lb_p
    model.active_constr.add(
        expr = model.x[q] >= slope * model.x[p] + intercept - 
                        (1 - z[(p, q)][i]) * 1e4
    )
    # return model
    return model


# ----------------------------------------------------------------------- #
# sequential optimization-based bound tightening
def sequential_bound_tightening(meta, root, base, max_iter = 5):

    # get the attributes
    net, var, mat = meta.net, meta.var, meta.mat

    # optimization-based bound-tightening
    bounds = {"lb": [mat.lb.copy()], "ub": [mat.ub.copy()]}
    k = 0  # iteration counter
    while True:
        k += 1
        # set current bounds
        lb = bounds["lb"][-1].copy()
        ub = bounds["ub"][-1].copy()
        # initialize partition
        partition = initialize_partition(mat.W, lb, ub)
        
        # modify model
        model = root.clone()
        # remove neural constraints
        model.neural_constr.clear()
        # add new bounds
        for n in range(var.n_vars):
            model.linear_constr.add(model.x[n] >= lb[n])
            model.linear_constr.add(model.x[n] <= ub[n])
        # add piecewise sigmoid relaxations
        model = piecewise_sigmoid_relaxation(model, partition)
        # set best known solution as upper bound
        expr = 0
        for n, c in enumerate(mat.c):
            ind, val = c["ind"], c["val"]
            temp = [val[i] * model.x[ind[i]] for i in range(len(ind))]
            expr = expr + sum(temp)
        model.linear_constr.add(expr <= base["obj"])
        
        # sequential OBBT
        for (p, q) in mat.W:
            m = model.clone()
            m.del_component(m.obj)
            m.obj = pyo.Objective(expr = m.x[p])
            m.obj.sense = pyo.minimize
            r_lb = solve_pyomo_model(m, solver = "gurobi")
            lb[p] = max(lb[p], r_lb["obj"])
            lb[q] = max(lb[q], func_sigmoid(lb[p]))
            m.obj.sense = pyo.maximize
            r_ub = solve_pyomo_model(m, solver = "gurobi")
            ub[p] = min(ub[p], r_ub["obj"])
            ub[q] = min(ub[q], func_sigmoid(ub[p]))
        
        # update bounds
        bounds["lb"].append(lb)
        bounds["ub"].append(ub)

        # check convergence
        lb_chg = np.max(np.abs(bounds["lb"][-1] - bounds["lb"][-2]))
        ub_chg = np.max(np.abs(bounds["ub"][-1] - bounds["ub"][-2]))
        lb_cvg = np.allclose(bounds["lb"][-1], bounds["lb"][-2], 
                             rtol = 1e-2, atol = 1e-2)
        ub_cvg = np.allclose(bounds["ub"][-1], bounds["ub"][-2], 
                             rtol = 1e-2, atol = 1e-2)
        if (lb_cvg and ub_cvg) or (k >= max_iter):
            break
    
    # return final bounds and partition
    return [bounds["lb"][-1], bounds["ub"][-1]]


# ----------------------------------------------------------------------- #
# initialize partition
def initialize_partition(W, lb, ub):
    partition = {}
    for (p, q) in W:
        lb_p, ub_p = float(lb[p]), float(ub[p])
        lb_q, ub_q = float(lb[q]), float(ub[q])
        # partitioning
        if lb_p < 0 and ub_p > 0:
            bound_p = [lb_p, 0, ub_p]
            bound_q = [max(lb_q, func_sigmoid(lb_p)),
                       func_sigmoid(0),
                       min(ub_q, func_sigmoid(ub_p))]
            partition[(p, q)] = [bound_p, bound_q]
        else:
            bound_p = [lb_p, ub_p]
            bound_q = [max(lb_q, func_sigmoid(lb_p)),
                       min(ub_q, func_sigmoid(ub_p))]
            partition[(p, q)] = [bound_p, bound_q]
    return partition


# ----------------------------------------------------------------------- #
# optimal sigmoid partitioning
def optimal_sigmoid_partitioning(lb, ub):
    x1, y1 = lb, func_sigmoid(lb)
    x2, y2 = ub, func_sigmoid(ub)
    k = (y2 - y1) / (x2 - x1)
    p1 = -np.log(2 / (1 + np.sqrt(1 - 4 * k)) - 1)
    p2 = -np.log(2 / (1 - np.sqrt(1 - 4 * k)) - 1)
    match (p1, p2):
        case _ if lb <= p1 <= ub:
            return p1
        case _ if lb <= p2 <= ub:
            return p2
        case _:
            raise ValueError("No valid partition point found!")


# ----------------------------------------------------------------------- #
# main function
if __name__ == "__main__":

    # set the paths
    fp_dir = os.path.dirname(os.path.abspath(__file__))
    fp_csv = os.path.join(fp_dir, "peaks_data.csv")
    fp_net = os.path.join(fp_dir, "peaks_neural_approx.onnx")
    fp_npz = os.path.join(fp_dir, "peaks_obbt.npz")

    # solve the problem
    meta = define_metadata(fp_csv, fp_net)
    t0 = time.time()
    opts = adaptive_sigmoid_partitioning(meta, fp_npz)
    t1 = time.time()
    print(f"Total time: {t1 - t0} seconds.")