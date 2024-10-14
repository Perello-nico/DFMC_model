"""
@author: Nicolò Perello
"""

import numpy as np
import pandas as pd
import argparse
import json
import sys
import os
import time
from MSL_PSO import MSL_PSO_algorithm
from evaluate_model import evaluate_swarm, DIMENSIONS


#####################################################################
# SETTINGS
#####################################################################

# paths to calibration and validation default datasets
dir_path = os.path.dirname(os.path.realpath(__file__))
path_data_calib = dir_path+'/data_calibration/ISR_dataset'

PATH_CALIBRATION = {
    'rain': {
        'calibration': path_data_calib+'/ISR_rain/TS_calibration.pkl',
        'validation': path_data_calib+'/ISR_rain/TS_validation.pkl',
    },
    'no_rain': {
        'calibration': path_data_calib+'/ISR_no_rain/TS_calibration.pkl',
        'validation': path_data_calib+'/ISR_no_rain/TS_validation.pkl'
    }
}

# List of available optimization algorithms
ALGORITHMS = {
    'MSL_PSO': MSL_PSO_algorithm,
}


#####################################################################
# FUNCTIONS
#####################################################################
def parse_params():
    parser = argparse.ArgumentParser(description='''Info''')
    parser.add_argument('--config', dest='config', type=str,
                        help='Path of json configuration file')
    parser.add_argument('--log', dest='log', type=lambda x: eval(x),
                        default=False, help='Log option [*False, True]')
    args = parser.parse_args()
    return args


class DualOutput:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


#####################################################################
# ALGORITHM
#####################################################################
def calibrate_model(config_path: str, logging: bool = False):

    path_save = '/'.join(config_path.split('/')[:-1])
    name_json = config_path.split('/')[-1].split('.')[0]

    # log file
    if logging:
        print('Logging option enabled')
        logfile = open(path_save+f'/{name_json}.txt', 'w')
        sys.stdout = DualOutput(logfile)

    print('############################')
    print('DFMC calibration')
    print('############################\n')

    # settings
    print('Loading settings')
    with open(config_path) as f:
        config = json.load(f)
    for kk in config:
        print('     - {}:{}'.format(kk, config[kk]))
    data_path = config.get('data_path', None)  # if not present, use default
    try:
        type_ts = config['type_ts']
        algorithm = config['algorithm']
        params_PSO = config['params']
        min_val = config['min_val']
        max_val = config['max_val']
        N_parts = config['n_particles']
        N_iters = config['N_iters']
        N_epoch = config['N_epoch']
    except Exception as e:
        print('ERROR: missing settings in the configuration file')
        print(e)
        sys.exit()
    # check the algorithm
    if algorithm not in ALGORITHMS.keys():
        print('ERROR: algorithm not available')
        sys.exit()
    print('----------------------------')

    # get optimization problem dimension
    dimension = DIMENSIONS.get(type_ts)

    # set bounds
    if isinstance(min_val, float) or isinstance(min_val, int):
        min_val = np.tile(min_val, dimension)
        max_val = np.tile(max_val, dimension)
    elif (len(min_val) != dimension) | (len(max_val) != dimension):
        print('ERROR: bounds not correctly assigned')
        sys.exit()
    bounds = np.array([min_val, max_val])

    print('Loading calibration data')
    if data_path is not None:
        try:
            df_TS_calib = pd.read_pickle(data_path+'/TS_calibration.pkl')
            df_TS_valid = pd.read_pickle(data_path+'/TS_validation.pkl')
        except Exception as e:
            print('ERROR: data not found')
            print(e)
            sys.exit()
    else:
        print('     - using default data')
        df_TS_calib = pd.read_pickle(PATH_CALIBRATION[type_ts]['calibration'])
        df_TS_valid = pd.read_pickle(PATH_CALIBRATION[type_ts]['validation'])
    print('----------------------------')

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

    time_start_all = time.time()
    print('\nStart calibration')
    # for each epoch #######################
    for EE in range(N_epoch):
        print('* epoch: {}'.format(EE))
        time_start_epoch = time.time()

        # OPTIMIZATION ALGORITHM
        pos_best, cost_best, history_swarm, \
            history_best, opt_vals = ALGORITHMS.get(algorithm)(
                        N_parts=N_parts,
                        N_iters=N_iters,
                        dimension=dimension,
                        bounds=bounds,
                        params=params_PSO,
                        args_OF=args_calib)

        # cost of epoch on validation dataset
        cost_epoch = evaluate_swarm(pos_best,
                                    *args_valid)

        print('     best cost: {:.4f}'.format(cost_best))
        print('     best position: {}'.format(pos_best))
        print('     validation cost: {:.4f}'.format(cost_epoch))

        # UPDATE BEST WIN - the best on validation
        if cost_epoch < cost_best_win:
            pos_best_win = pos_best
            cost_best_win = cost_epoch
            epoch_win = EE

        # SAVING EPOCH
        history_swarm_epoch.append(history_swarm)
        opt_vals_epoch.append(opt_vals)
        history_best_epoch.append(history_best)

        # compute the computational time of the epoch
        time_end_epoch = time.time()
        print('     computational time: {:.2f} s'.format(
            time_end_epoch-time_start_epoch))

        # end of the epoch ############################################

    print('End of the calibration\n')

    print('############################')
    print('best epoch: {}'.format(epoch_win))
    print('BEST COST EVER: {:.4f}'.format(cost_best_win))
    print('BEST POSITION EVER: {}'.format(pos_best_win))
    # compute the time required for the calibration
    time_end_all = time.time()
    print('Total computational time: {:.2f} s'.format(
        time_end_all-time_start_all))
    print('############################')

    # save the results
    np.savez(path_save+f'/{name_json}.npz',
             params=pos_best_win,
             epoch_win=epoch_win,
             vals=np.array(opt_vals_epoch),
             history_swarm=np.array(history_swarm_epoch, dtype='object'),
             history_best=np.array(history_best_epoch, dtype='object'))

    if logging:
        logfile.close()  # close the log file
        sys.stdout = sys.__stdout__
        print('Log saved in: {}'.format(path_save+f'/{name_json}.txt'))


if __name__ == '__main__':
    args = parse_params()
    calibrate_model(config_path=args.config, logging=args.log)
