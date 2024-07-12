"""
@author: Nicolò Perello
"""

import numpy as np
import pandas as pd
import argparse
import json
import sys
from MSL_PSO import MSL_PSO_algorithm
from evaluate_model import evaluate_swarm


#####################################################################
# parameters
#####################################################################
ALGORITHMS = {  # List of available optimization algorithms
    'MSL_PSO': MSL_PSO_algorithm,
}


DIMENSIONS = {  # dimension of the optimization problem
    'rain': 3,
    'no_rain': 13,
}


#####################################################################
# functions
#####################################################################
def parse_params():
    parser = argparse.ArgumentParser(description='''Info''')
    parser.add_argument('--path', dest='path', type=str,
                        help='Path of json configuration file')
    args = parser.parse_args()
    return args


#####################################################################
# ALGORITHM
#####################################################################
def main():
    args = parse_params()
    path_save = '/'.join(args.path.split('/')[:-1])

    # settings
    print('Loading settings...')
    with open(args.path) as f:
        config = json.load(f)
    for kk in config:
        print('     - {}:{}'.format(kk, config[kk]))
    print('----------------------------')
    type_ts = config['type_ts']
    algorithm = config['algorithm']
    params_PSO = config['params']
    min_val = config['min_val']
    max_val = config['max_val']
    N_parts = config['n_particles']
    N_iters = config['N_iters']
    N_epoch = config['N_epoch']

    # get dimension
    dimension = DIMENSIONS.get(type_ts)

    # get bounds
    if isinstance(min_val, float) or isinstance(min_val, int):
        min_val = np.tile(min_val, dimension)
        max_val = np.tile(max_val, dimension)
    elif (len(min_val) != dimension) | (len(max_val) != dimension):
        print('ERROR: bounds not correctly assigned')
        sys.exit()
    bounds = np.array([min_val, max_val])

    print('Loading calibration data...')
    print('----------------------------')
    # dataset calibration
    df_TS_calib = pd.read_pickle(path_save+'/TS_calibration.pkl')
    # dataset validation
    df_TS_valid = pd.read_pickle(path_save+'/TS_validation.pkl')

    # arguments of evaluate swarm function
    args_calib = (df_TS_calib, type_ts)
    args_valid = (df_TS_valid, type_ts)

    # data structures
    history_swarm_epoch = []  # all particles, all times
    opt_vals_epoch = []  # objective function values, all times
    history_best_epoch = []  # the best position, all times
    epoch_win = None  # the winning epoch
    pos_best_win = None  # best position ever
    cost_best_win = 10000000  # cost of the winner

    for EE in range(N_epoch):  # for each epoch #######################
        print('* EPOCH: {}'.format(EE))

        # OPTIMIZATION ALGORITHM
        pos_best, cost_best, history_swarm, \
            history_best, opt_vals = ALGORITHMS.get(algorithm)(
                        N_parts=N_parts,
                        N_iters=N_iters,
                        dimension=dimension,
                        bounds=bounds,
                        params=params_PSO,
                        args_OF=args_calib)

        # cost of EPOCH on validation dataset
        cost_epoch = evaluate_swarm(pos_best,
                                    *args_valid)

        print('     best cost: {:.4f}'.format(cost_best))
        print('     best position: {}'.format(pos_best))
        print('     validation cost: {:.4f}'.format(cost_epoch))
        # end of the epoch ############################################

        # UPDATE BEST WIN - the best on validation
        if cost_epoch < cost_best_win:
            pos_best_win = pos_best
            cost_best_win = cost_epoch
            epoch_win = EE

        # SAVING EPOCH
        history_swarm_epoch.append(history_swarm)
        opt_vals_epoch.append(opt_vals)
        history_best_epoch.append(history_best)

    print('----------------------------')
    print('best epoch: {}'.format(epoch_win))
    print('BEST COST EVER: {:.4f}'.format(cost_best_win))
    print('BEST POSITION EVER: {}'.format(pos_best_win))
    np.savez(path_save+f'/{type_ts}_calib.npz',
             params=pos_best_win,
             epoch_win=epoch_win,
             vals=np.array(opt_vals_epoch),
             history_swarm=np.array(history_swarm_epoch, dtype='object'),
             history_best=np.array(history_best_epoch, dtype='object'))


if __name__ == '__main__':
    main()
