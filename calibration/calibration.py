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
from evaluate_model import evaluate_swarm, RAIN_PARAMS, NO_RAIN_PARAMS


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


def unpack_params(params: np.array) -> tuple[list, dict, int]:
    """Unpack the parameters"""
    calib_params = []
    fixed_params = dict()
    for pp in params:
        if isinstance(params[pp], list):
            calib_params.append(pp)
        else:
            fixed_params[pp] = params[pp]
    dimension = len(calib_params)
    return calib_params, fixed_params, dimension


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
        hyper_params_PSO = config['hyper_params']
        params = config['params']
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
    # check the type of time series
    if type_ts not in ['rain', 'no_rain']:
        print('ERROR: wrong type of time series')
        sys.exit()
    print('----------------------------')

    # SET PARAMETERS
    print('Setting parameters')
    fixed_params = dict()
    calib_params = []
    min_val = []
    max_val = []
    try:
        calib_params, fixed_params, dimension = unpack_params(params)
        # set bounds
        for kk in calib_params:
            min_val.append(params[kk][0])
            max_val.append(params[kk][1])
        bounds = np.array([min_val, max_val])
        # print info
        print('     - dimension: {}'.format(dimension))
        print('     - parameters to calibrate:')
        for kk in calib_params:
            print('         {}: [{}, {}]'.format(kk,
                                                 params[kk][0],
                                                 params[kk][1]))
        print('     - fixed parameters:')
        for kk in fixed_params:
            print('         {}: {}'.format(kk, fixed_params[kk]))
    except Exception as e:
        print('ERROR: parameters not correctly assigned')
        print(e)
        sys.exit()
    if len(calib_params) == 0:
        print('WARNING: no parameters to calibrate')
        sys.exit()
    # check on parameters
    if type_ts == 'rain':
        if set(calib_params+list(fixed_params.keys())) != set(RAIN_PARAMS):
            print('ERROR: wrong parameters for the rain model')
            sys.exit()
    elif type_ts == 'no_rain':
        if set(calib_params+list(fixed_params.keys())) != set(NO_RAIN_PARAMS):
            print('ERROR: wrong parameters for the no rain model')
            sys.exit()
    print('----------------------------')

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
    args_calib = (df_TS_calib, type_ts, calib_params, fixed_params)
    args_valid = (df_TS_valid, type_ts, calib_params, fixed_params)

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
                        hyper_params=hyper_params_PSO,
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
