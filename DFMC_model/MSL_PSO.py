"""
@author: Nicolò Perello
"""

import numpy as np
from typing import Dict, Tuple
from smt.sampling_methods import LHS
from evaluate_model import evaluate_swarm

#####################################################################
# MSL-PSO ALGORITHM - functions
#####################################################################

EPSILON_CALIB = 10**(-8)


def initialize_pos_vel(N_parts: int, bounds: np.array, how: str = 'LHS'):
    dimension = bounds.shape[1]
    if how == 'random':
        # position initialization - RANDOM
        X = np.random.uniform(low=np.tile(bounds[0], (N_parts, 1)),
                              high=np.tile(bounds[1], (N_parts, 1)),
                              size=(N_parts, dimension))
    elif how == 'LHS':
        # position initialization - LATIN HYPERCUBE SAMPLING
        xlimits = np.array([[x, y] for x, y in zip(bounds[0], bounds[1])])
        sampling = LHS(xlimits=xlimits)
        X = sampling(N_parts)
    # velocity initialization - UNIFORM SAMPLING
    V = np.random.uniform(low=0, high=1, size=(N_parts, dimension))
    return X, V


def random_selection(idxs: np.array, dims: int, X: np.array):
    """Select random probes in different dimensions"""
    if idxs.shape[0] > 0:
        rand = np.zeros(dims)
        rand_idx = np.random.choice(idxs, dims)
        for dd, ii in enumerate(rand_idx):
            rand[dd] = X[ii, dd]
    else:
        rand = np.array([])
    return rand


def update(x_part: np.array, v_part: np.array,
           x_dem_1: np.array, x_dem_2: np.array,
           phi: float):
    """Update position and velocity"""
    dims = x_part.shape[0]
    r1, r2, r3 = np.random.uniform(size=(3, dims))
    v_part_new = r1*v_part
    if x_dem_1.shape[0] != 0:
        tmp1 = r2*(x_dem_1-x_part)
        v_part_new += tmp1
    if x_dem_2.shape[0] != 0:
        tmp2 = phi*r3*(x_dem_2-x_part)
        v_part_new += tmp2
    x_part_new = x_part+v_part_new
    return x_part_new, v_part_new


def in_bounds(X: np.array, V: np.array,
              bounds: np.array,
              how: str = 'edge'):
    """Put inside the domain again"""
    mins = np.tile(bounds[0], (X.shape[0], 1))
    maxs = np.tile(bounds[1], (X.shape[0], 1))
    idx_no_min = X < mins
    idx_no_max = X > maxs
    idx_no = idx_no_min | idx_no_max
    if how == 'random':
        # particles are positioned inside the bounds randomly
        X_rand, V_rand = initialize_pos_vel(N_parts=X.shape[0],
                                            bounds=bounds,
                                            how='random')
        X_in = np.where(idx_no, X_rand, X)
        V_in = np.where(idx_no, 0, V)
    elif how == 'edge':
        # the particles are positione in the edge with reversed velocity
        X_in = np.where(idx_no_max, maxs, X)
        X_in = np.where(idx_no_min, mins, X_in)
        V_in = np.where(idx_no, -V, V)
    return X_in, V_in


def check_bounds(X: np.array, bounds: np.array):
    mins = np.tile(bounds[0], (X.shape[0], 1))
    maxs = np.tile(bounds[1], (X.shape[0], 1))
    return np.any(((X < mins) | (X > maxs)))


#####################################################################
# MSL-PSO ALGORITHM
#####################################################################
def MSL_PSO_algorithm(N_parts: int, N_iters: int, dimension: int,
                      bounds: np.array, params: Dict,
                      args_OF: Tuple):
    # parameters
    phi_param = params['phi']
    N_probe = params['N_probe']

    # INITIALIZATION ################
    print('   - Swarm initialization...')
    X, V = initialize_pos_vel(N_parts=N_parts, bounds=bounds, how='LHS')
    if check_bounds(X=X, bounds=bounds):
        print('Particles out of bounds')
    # INITIAL COST - evaluate Objective Function for all particles
    cost = evaluate_swarm(X, *args_OF)
    order = np.flip(np.argsort(cost))  # order particles - desceding order
    centroid = (1/N_parts)*np.sum(X, axis=0)  # centroid of particles

    # OPTIMIZATION ################
    print('   - Calibration...')
    history_swarm = np.full((N_iters, N_parts, dimension), np.nan)
    opt_vals = np.full(N_iters, np.nan)
    history_best = np.full((N_iters, dimension), np.nan)
    pos_best = None
    cost_old = 0
    cost_best = 10000000
    make_iters = True

    i = 0
    while (i < N_iters) and (make_iters):  # per ogni iterazione
        print('      - iter {} - '.format(i), end=' ')

        # STEP 1: search for probes - one per each particle
        X_probes = np.zeros((N_parts, dimension))
        V_probes = np.zeros((N_parts, dimension))
        for ii, pp in enumerate(order):  # particles IN DESCENDING ORDER
            # indexes of possible demonstrators - particles with less cost
            idx_dems = order[ii+1:]
            xx_probes = np.zeros((N_probe, dimension))
            vv_probes = np.zeros((N_probe, dimension))
            for kk in range(N_probe):  # make multiple probes
                # for each dimension, selection of random demonstrator
                rand_dem = random_selection(idxs=idx_dems,
                                            dims=dimension,
                                            X=X)
                xx_probes[kk, :], vv_probes[kk, :] = update(
                                                        x_part=X[pp, :],
                                                        v_part=V[pp, :],
                                                        x_dem_1=rand_dem,
                                                        x_dem_2=centroid,
                                                        phi=phi_param)
            # put inside out-of-the-box particles
            xx_probes, vv_probes = in_bounds(X=xx_probes,
                                             V=vv_probes,
                                             bounds=bounds,
                                             how='edge')
            if check_bounds(X=xx_probes, bounds=bounds):
                print('Particles out of bounds')
            # evaluate costs of probes
            cost_probes_tmp = evaluate_swarm(xx_probes,
                                             *args_OF)
            idx_probe_ok = np.argmin(cost_probes_tmp)  # find the best one
            # choice of the best probe position and velocity
            X_probes[pp, :] = xx_probes[idx_probe_ok, :]
            V_probes[pp, :] = vv_probes[idx_probe_ok, :]

        # STEP 2: update position and velocity
        # costs of probes
        cost_probes = evaluate_swarm(X_probes, *args_OF)
        # order probes in descending order
        order_probes = np.flip(np.argsort(cost_probes))
        for ii, pp in enumerate(order):
            # ranking of the respective probe -> find the index
            kk = np.where(order_probes == pp)[0][0]
            idx_dems_1 = order_probes[np.min([ii, kk])+1:np.max([ii, kk])]
            idx_dems_2 = order_probes[kk+1:]
            rand_dem_1 = random_selection(idxs=idx_dems_1,
                                          dims=dimension,
                                          X=X_probes)
            rand_dem_2 = random_selection(idxs=idx_dems_2,
                                          dims=dimension,
                                          X=X_probes)
            X[pp, :], V[pp, :] = update(x_part=X_probes[pp, :],
                                        v_part=V_probes[pp, :],
                                        x_dem_1=rand_dem_1,
                                        x_dem_2=rand_dem_2,
                                        phi=phi_param)
        # keep inside the bounds
        X, V = in_bounds(X=X, V=V,
                         bounds=bounds,
                         how='random')
        if check_bounds(X=X, bounds=bounds):
            print('Particles out of bounds')
        # UPDATE GLOBAL BEST
        cost = evaluate_swarm(X, *args_OF)
        order = np.flip(np.argsort(cost))
        centroid = (1/N_parts)*np.sum(X, axis=0)
        if cost[order[-1]] < cost_best:
            cost_old = cost_best
            pos_best = X[order[-1], :]
            cost_best = cost[order[-1]]

        # SAVING ITERATIONS
        history_swarm[i, :, :] = X
        opt_vals[i] = cost_best
        history_best[i, :] = pos_best

        print('     cost {:.6f}'.format(cost_best))

        # check if interrupt iterations
        if (np.abs(cost_best-cost_old) < EPSILON_CALIB) and (i > N_iters/2):
            make_iters = False

        i += 1

    return pos_best, cost_best, history_swarm, history_best, opt_vals
