"""
@author: Nicolò Perello
"""

import numpy as np
import pandas as pd
from typing import Dict
from DFMC_model import (rain_phase, no_rain_phase, compute_DFMC, NODATAVAL)


###############################################################################
# METRICS #####################################################################
###############################################################################
def SSR(mod: float, obs: float, axis_time: int):
    """
    Sum of Squared Residuals

    :param mod: model values
    :param obs: observed values
    :param axis_time: axis identifying times of timeseries
    :return: array of norm of residuals
    """
    return np.sum(np.power(mod-obs, 2), axis=axis_time)


def MAE(mod: float, obs: float, axis_time: int):
    """
    Mean Absolute Error

    :param mod: model values
    :param obs: observed values
    :param axis_time: axis identifying times of timeseries
    :return: array of Mean Absolute Error
    """
    return (1/mod.shape[axis_time])*np.sum(np.abs(mod-obs),
                                           axis=axis_time)


def BIAS(mod: float, obs: float, axis_time: int):
    """
    Bias [from Tsai, 2021]

    :param mod: model values
    :param obs: observed values
    :param axis_time: axis identifying times of timeseries
    :return: array of bias
    """
    return (1/mod.shape[axis_time])*np.sum(mod-obs, axis=axis_time)


def RMSE(mod: float, obs: float, axis_time: int):
    """
    Root Mean Square Error

    :param mod: model values
    :param obs: observed values
    :param axis_time: axis identifying times of timeseries
    :return: array of Root Mean Square Error
    """
    return np.sqrt(np.divide(np.sum(np.power(mod-obs, 2), axis=axis_time),
                             mod.shape[axis_time]))


def NSE(mod: float, obs: float, axis_time: int):
    """
    Nash-Sutcliffe Efficiency

    :param mod: model values
    :param obs: observed values
    :param axis_time: axis identifying times of timeseries
    :return: array of NSE
    """
    num = np.sum(np.power(mod-obs, 2), axis=axis_time)
    mean = np.mean(obs, axis=axis_time)
    mean = np.repeat(mean, obs.shape[axis_time]).reshape(obs.shape)
    den = np.sum(np.power(obs-mean, 2), axis=axis_time)
    return 1 - num / den


def NNSE(mod: float, obs: float, axis_time: int):
    """
    Normalized Nash-Sutcliffe Efficiency

    :param mod: model values
    :param obs: observed values
    :param axis_time: axis identifying times of timeseries
    :return: array of NNSE
    """
    nse = NSE(mod=mod, obs=obs, axis_time=axis_time)
    return 1 / (2-nse)


def compute_metric(dfmc: np.array, dfmc_obs: np.array,
                   metric, time_lag: int = 0):
    """
    Compute a specific metric

    :param dfmc: DFMC value computed. Shape: N_parts x N_TS x N_times
    :param dfmc_obs: DFMC observed values. Shape: N_TS x N_times
    :param metric: metric to be computed
    :param time_lag: time lag to be applied before computing the metric
    :return: metric computed per each particle and time series.
             Shape: N_parts x N_TS
    """
    N_parts = dfmc.shape[0]
    # repeat ffmc_obs per each particle - same dimension of ffmc
    ffmc_obs_compute = np.tile(dfmc_obs, (N_parts, 1, 1))
    # results for each particle and time series
    res = metric(
            mod=dfmc[:, :, time_lag:],
            obs=ffmc_obs_compute[:, :, time_lag:],
            axis_time=2)
    return res


METRICS = {'MAE': MAE, 'BIAS': BIAS, 'RMSE': RMSE, 'NNSE': NNSE}
TIME_LAG = {
    'no_rain': 72,
    'rain': 0,
    'mixed': 72
}


def evaluate_model(X: np.array, df_TS: pd.DataFrame, type_ts: str):
    """Compute the run of the model and GoF for the parameters combination"""
    X = np.array([X])
    N_TS = len(df_TS)
    N_parts = X.shape[0]
    dfmc_obs = get_dfmc_obs(df_TS=df_TS)
    dfmc, phase, emc, k_const = run_model(df_TS=df_TS, X=X, type_ts=type_ts,
                                          mode='validation')
    df_TS_gof = pd.DataFrame(
        {
            'DFMC_model': [pd.Series(dfmc[0, x, :], name='DFMC_model')
                           for x in range(N_TS)],
            'phase': [pd.Series(phase[0, x, :], name='phase')
                      for x in range(N_TS)],
            'EMC': [pd.Series(emc[0, x, :], name='EMC')
                    for x in range(N_TS)],
            'K_const': [pd.Series(k_const[0, x, :], name='K_const')
                        for x in range(N_TS)],
        },
        index=range(N_TS)
    )
    gof = np.full((N_parts, N_TS, len(METRICS)), NODATAVAL, dtype='float')
    for mm, metr in enumerate(METRICS):
        gof[:, :, mm] = compute_metric(dfmc=dfmc,
                                       dfmc_obs=dfmc_obs,
                                       metric=METRICS[metr],
                                       time_lag=TIME_LAG[type_ts])
        df_TS_gof.loc[:, metr] = gof[0, :, mm]
    return df_TS_gof


def get_dfmc_obs(df_TS: pd.DataFrame) -> np.array:
    N_TS = len(df_TS)
    N_times = len(df_TS.iloc[0].DFMC.values)
    dfmc_obs = np.zeros((N_TS, N_times))
    for ts in range(N_TS):
        dfmc_obs[ts, :] = df_TS.iloc[ts].DFMC.values
    return dfmc_obs


###############################################################################
# EVALUATE SWARM ##############################################################
###############################################################################

RAIN_PARAMS = ['R1', 'R2', 'R3']
NO_RAIN_PARAMS = ['A1', 'A2', 'A3', 'A4', 'A5',
                  'Bd1', 'Bd2', 'Bd3', 'Cd1', 'Cd2', 'Cd3',
                  'Bw1', 'Bw2', 'Bw3', 'Cw1', 'Cw2', 'Cw3']


def build_params(x: np.array, calib_params: list, fixed_params: dict) -> Dict:
    if len(x) != len(calib_params):
        print('ERROR: wrong number of parameters')
        return None
    params = dict(zip(calib_params, x))
    params.update(fixed_params)
    return params


def run_model(df_TS: pd.DataFrame, X: np.array, type_ts: str,
              calib_params: list, fixed_params: dict,
              mode: str = 'calibration') -> np.array:
    """Comute FFMC values for all parameters settings and all Timeseries"""
    N_parts = X.shape[0]  # number of particles
    N_TS = len(df_TS)  # number of timeseries
    N_times = len(df_TS.iloc[0].DFMC.values)  # number of time steps
    dfmc = np.full((N_parts, N_TS, N_times), NODATAVAL, dtype='float64')
    phase = np.full((N_parts, N_TS, N_times), NODATAVAL, dtype='int32')
    emc = np.full((N_parts, N_TS, N_times), NODATAVAL, dtype='float64')
    k_const = np.full((N_parts, N_TS, N_times), NODATAVAL, dtype='float64')
    for pp in range(N_parts):  # for each particle (e.g. set of parameters)
        params = build_params(x=X[pp, :], calib_params=calib_params,
                              fixed_params=fixed_params)
        if type_ts == 'rain':
            for ts in range(N_TS):
                dfmc[pp, ts, 0] = df_TS.iloc[ts]['DFMC'].values[0]
                if mode == 'calibration':
                    dfmc[pp, ts, 1:] = rain_phase(
                        rain=df_TS.iloc[ts]['Rain'].values[1:],
                        moisture=df_TS.iloc[ts]['DFMC'].values[0:-1],
                        params=params)
                elif mode == 'validation':
                    for tt in range(1, N_times):
                        dfmc[pp, ts, tt] = rain_phase(
                            rain=df_TS.iloc[ts]['Rain'].values[tt],
                            moisture=dfmc[pp, ts, tt-1],
                            params=params)
        elif type_ts == 'no_rain':
            for ts in range(N_TS):
                dfmc[pp, ts, 0] = df_TS.iloc[ts]['DFMC'].values[0]
                if mode == 'calibration':
                    dfmc[pp, ts, 1:], phase[pp, ts, 1:], \
                        emc[pp, ts, 1:], k_const[pp, ts, 1:] = no_rain_phase(
                            moisture=df_TS.iloc[ts]['DFMC'].values[0:-1],
                            temp=df_TS.iloc[ts]['Temp'].values[1:],
                            wspeed=df_TS.iloc[ts]['Wspeed'].values[1:],
                            hum=df_TS.iloc[ts]['Hum'].values[1:],
                            params=params)
                elif mode == 'validation':
                    for tt in range(1, N_times):
                        dfmc[pp, ts, tt], phase[pp, ts, tt], \
                            emc[pp, ts, tt], k_const[pp, ts, tt] = \
                            no_rain_phase(
                                moisture=dfmc[pp, ts, tt-1],
                                temp=df_TS.iloc[ts]['Temp'].values[tt],
                                wspeed=df_TS.iloc[ts]['Wspeed'].values[tt],
                                hum=df_TS.iloc[ts]['Hum'].values[tt],
                                params=params)
        elif type_ts == 'mixed':
            for ts in range(N_TS):
                dfmc[pp, ts, 0] = df_TS.iloc[ts]['DFMC'].values[0]
                if mode == 'calibration':
                    dfmc[pp, ts, 1:], phase[pp, ts, 1:], \
                        emc[pp, ts, 1:], k_const[pp, ts, 1:] = compute_DFMC(
                        moisture=df_TS.iloc[ts]['DFMC'].values[0:-1],
                        rain=df_TS.iloc[ts]['Rain'].values[1:],
                        temp=df_TS.iloc[ts]['Temp'].values[1:],
                        hum=df_TS.iloc[ts]['Hum'].values[1:],
                        wspeed=df_TS.iloc[ts]['Wspeed'].values[1:],
                        params=params)
                elif mode == 'validation':
                    for tt in range(1, N_times):
                        dfmc[pp, ts, tt], phase[pp, ts, tt], \
                            emc[pp, ts, tt], k_const[pp, ts, tt] = \
                            compute_DFMC(
                                moisture=dfmc[pp, ts, tt-1],
                                rain=df_TS.iloc[ts]['Rain'].values[tt],
                                temp=df_TS.iloc[ts]['Temp'].values[tt],
                                hum=df_TS.iloc[ts]['Hum'].values[tt],
                                wspeed=df_TS.iloc[ts]['Wspeed'].values[tt],
                                params=params)
    return dfmc, phase, emc, k_const


def evaluate_swarm(X: np.array, *args):
    """Compute the OF for each particle"""
    df_TS, type_ts, calib_params, fixed_params = args
    single_part = False
    if len(X.shape) == 1:  # particle-based setting
        X = np.array([X])
        single_part = True
    dfmc_obs = get_dfmc_obs(df_TS=df_TS)
    dfmc, _, _, _ = run_model(df_TS=df_TS, X=X, type_ts=type_ts,
                              calib_params=calib_params,
                              fixed_params=fixed_params,
                              mode='calibration')
    # SSR - objective function
    ssr = compute_metric(dfmc=dfmc, dfmc_obs=dfmc_obs,
                         metric=SSR, time_lag=0)
    # sum results of all timeseries
    # axis=1 in the data structure...
    ssr_all = np.sum(ssr, axis=1)
    if single_part:  # particle-based setting
        ssr_all = ssr_all[0]
    return ssr_all
