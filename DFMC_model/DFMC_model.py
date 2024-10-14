"""
@author: Nicolò Perello
"""

import numpy as np
from typing import Dict

##########################################################################
# DFMC model PARAMETERS
##########################################################################
NODATAVAL = -9999

# Fuel Stick parameters
T0 = 10  # h
SAT = 45  # %

# data time step
DT = 1  # h

# standard weather values
T_STD = 27.0  # °C
H_STD = 20.0  # %
W_STD = 0.0  # m/s


DFMC_PARAMS = dict()
# rain phase
DFMC_PARAMS['MIN_RAIN'] = 0.1  # mm
DFMC_PARAMS['R1'] = 68.658964
DFMC_PARAMS['R2'] = 53.374067
DFMC_PARAMS['R3'] = 0.935953
# no rain phase
# EMC
DFMC_PARAMS['A1'] = 0.592789
DFMC_PARAMS['A2'] = 0.555
DFMC_PARAMS['A3'] = 10.6
DFMC_PARAMS['A4'] = 0.5022
DFMC_PARAMS['A5'] = 0.0133
# K - dry phase
DFMC_PARAMS['Bd1'] = 0.112756
DFMC_PARAMS['Bd2'] = 0.349820
DFMC_PARAMS['Bd3'] = 0.111055
DFMC_PARAMS['Cd1'] = 0.531471
DFMC_PARAMS['Cd2'] = 0.534400
DFMC_PARAMS['Cd3'] = 0.517728
# K - wet phase
DFMC_PARAMS['Bw1'] = 0.104363
DFMC_PARAMS['Bw2'] = 0.482954
DFMC_PARAMS['Bw3'] = 0.100061
DFMC_PARAMS['Cw1'] = 0.509857
DFMC_PARAMS['Cw2'] = 0.678900
DFMC_PARAMS['Cw3'] = 0.504871


# normalizing factor (drying) - to have 1 in standard conditions
# NOTE: depends on K dry formulation
def Dd(params: Dict = DFMC_PARAMS) -> float:
    return (1.0 +
            params['Bd1'] * (
                T_STD**params['Cd1']) +
            params['Bd2'] * (
                W_STD**params['Cd2'])
            ) / (
            1.0 +
            params['Bd3'] * (
                H_STD**params['Cd3']))


# normalizing factor - to have 1 in standard conditions
# # NOTE: depends on K wet formulation
def Dw(params: Dict = DFMC_PARAMS) -> float:
    return (1.0 +
            params['Bw3'] * (
                H_STD**params['Cw3'])
            ) / (
            1.0 +
            params['Bw1'] * (
                T_STD**params['Cw1']) +
            params['Bw2'] * (
                W_STD**params['Cw2'])
            )


##########################################################################
# DFMC model
##########################################################################
def compute_DFMC(moisture: float,
                 rain: float, temp: float, hum: float, wspeed: float,
                 saturation: float = SAT, T0: float = T0,
                 dt: float = DT,
                 params: Dict = DFMC_PARAMS
                 ) -> tuple[float, int, float, float]:
    """
    ## Dead Fuel Moisture Content (DFMC) [%]

    ### Args:
        1. moisture (float): DFMC of previous time step (%)
        2. rain (float): cumulated rain (mm)
        3. temp (float): air temperature (°C)
        4. wspeed (float): wind speed (m/h)
        5. hum (float): relative humidity (%)
        6. saturation (float): DFMC saturation value, fuel parameter (%)
        7. T0 (float): time constant in standard condition, fuel parameter (h)
        8. dt (float): temporal step (h)
        9: params (Dict): dictionary with model's parameters
    ### Return:
        1. (float): new moisture valure (%)
        2. (int): phase code (drying: -1, wetting: 1, rain: 2, no update: 0)
        3. (float): Equilibrium Moisture Content EMC (%)
        4. (float): drying/wetting time (h)
    """
    if check_no_data(rain=rain, temp=temp, hum=hum, wspeed=wspeed,
                     T0=T0, saturation=saturation):
        # no data
        moisture_new = moisture
        phase = NODATAVAL
        emc = NODATAVAL
        K_const = NODATAVAL
    elif check_no_update(temp=temp):
        # no update
        moisture_new = moisture
        phase = 0
        emc = NODATAVAL
        K_const = NODATAVAL
    else:
        if check_rain(rain=rain, params=params):
            # rain phase
            moisture_new = rain_phase(
                rain=rain,
                moisture=moisture,
                saturation=saturation,
                params=params)
            phase = 2
            emc = NODATAVAL
            K_const = NODATAVAL
        else:
            # no rain phase
            moisture_new, phase, emc, K_const = no_rain_phase(
                    moisture=moisture,
                    temp=temp,
                    wspeed=wspeed,
                    hum=hum,
                    saturation=saturation,
                    T0=T0,
                    dt=dt,
                    params=params)
    moisture_new = np.clip(moisture_new, 0, saturation)
    return moisture_new, phase, emc, K_const


##########################################################################
# DFMC phases
##########################################################################
def rain_phase(rain: float, moisture: float, saturation: float = SAT,
               params: Dict = DFMC_PARAMS) -> float:
    """
    ## Rain phase

    ### Args:
        1. rain (float): cumulated rainfall (mm)
        2. moisture (float): DFMC of previous time step (%)
        3. saturation (float): DFMC saturation value - fuel parameter (%)
        4: params (Dict): dictionary with model's parameters
    ### Return:
        1. (float): new moisture content value (%)
    """
    # update moisture
    moisture_new = moisture + delta_rain(rain=rain,
                                         moisture=moisture,
                                         saturation=saturation,
                                         params=params)
    # check range
    moisture_new = np.clip(moisture_new, 0, saturation)
    return moisture_new


def no_rain_phase(moisture: float,
                  temp: float, wspeed: float, hum: float,
                  saturation: float = SAT, T0: float = T0,
                  dt: float = DT, params: Dict = DFMC_PARAMS
                  ) -> tuple[float, int, float, float]:
    """
    ## No-rain phase

    ### Args:
        1. moisture (float): DFMC of previous time step (%)
        2. temp (float): air temperature (°C)
        3. wspeed (float): wind speed (m/s)
        4. hum (float): relative humidity (%)
        5. saturation (float): DFMC saturation value - fuel parameter (%)
        6. T0 (float): time constant in standard condition - fuel parameter (h)
        7. dt (float): temporal step (h)
        8: params (Dict): dictionary with model's parameters
    ### Return:
        1. (float): new moisture valure (%)
        2. (int): phase code (drying: -1, wetting: 1)
        3. (float): Equilibrium Moisture Content EMC (%)
        4. (float): drying/wetting time (h)
    """
    # compute EMC
    emc = EMC(hum=hum, temp=temp, params=params)
    # compute K
    K_const = np.where(
                    moisture >= emc,
                    K_dry(T0=T0, temp=temp, wspeed=wspeed, hum=hum,
                          params=params),
                    K_wet(T0=T0, temp=temp, wspeed=wspeed, hum=hum,
                          params=params)
    )
    # define phase
    phase = np.where(moisture >= emc, -1, 1)
    # update moisture
    moisture_new = update_no_rain(moisture=moisture, emc=emc,
                                  K_const=K_const, dt=dt)
    # check range
    moisture_new = np.clip(moisture_new, 0, saturation)
    return moisture_new, phase, emc, K_const


##########################################################################
# DFMC model - functions
##########################################################################
def update_no_rain(moisture: float, emc: float, K_const: float,
                   dt: float = DT) -> float:
    """
    ## Update moisture in the no-rain phase

    ### Args:
        1. moisture (float): DFMC of previous time step (%)
        2. emc (float): Equilibrium Moisture Content (%)
        3. K_const (float): Drying/Wetting response time (h)
        4. dt (float): time step (h)
    ### Return:
        1. (float): new moisture content value (%)
    """
    const = (emc - moisture) / (100.0 - moisture)
    num = emc - (100.0 * const * np.e**(-dt / K_const))
    den = 1.0 - (const * np.e**(-dt / K_const))
    return np.divide(num, den)


def EMC(hum: float, temp: float, params: Dict = DFMC_PARAMS) -> float:
    """
    ## Equilibrium Moisture Content (EMC) [%]

    ### Args:
        1. hum (float): relative humidity (%)
        2. temp (float): air temperature (°C)
        3: params (Dict): dictionary with model's parameters
    ### Return:
        1. (float): EMC (%)
    """
    return params['A1'] * (hum**params['A2']) + \
        params['A3'] * (np.e**((hum - 100.0) / 10.0)) + \
        params['A4'] * (30.0 - np.minimum(temp, 30.0)) * (
            1.0 - np.e**(-params['A5'] * hum))


def K_dry(temp: float, wspeed: float, hum: float,
          T0: float = T0, params: Dict = DFMC_PARAMS) -> float:
    """
    ## Drying time [h]

    ### Args:
        1. temp (float): air temperature (°C)
        2. wspeed (float): wind speed (m/s)
        3. hum (float): relative humidity (%)
        4. T0 (float): time constant in standard condition - fuel parameter (h)
        5: params (Dict): dictionary with model's parameters
    ### Return:
        1. (float): drying time (h)
    """
    return T0 * Dd(params) * (
                    (
                        1.0 +
                        params['Bd3'] * (hum**params['Cd3'])
                    ) / (
                        1.0 +
                        params['Bd1'] * (temp**params['Cd1']) +
                        params['Bd2'] * (wspeed**params['Cd2'])
                        )
                )


def K_wet(temp: float, wspeed: float, hum: float,
          T0: float = T0, params: Dict = DFMC_PARAMS) -> float:
    """
    ## Wetting time [h]

    ### Args:
        1. temp (float): air temperature (°C)
        2. wspeed (float): wind speed (m/s)
        3. hum (float): relative humidity (%)
        4. T0 (float): time constant in standard condition - fuel parameter (h)
        5: params (Dict): dictionary with model's parameters
    ### Return:
        1. (float): wetting time (h)
    """
    return T0 * Dw(params) * (
                    (
                        1.0 +
                        params['Bw1'] * (temp**params['Cw1']) +
                        params['Bw2'] * (wspeed**params['Cw2'])
                    ) / (
                        1.0 +
                        params['Bw3'] * (hum**params['Cw3'])
                    )
                )


def delta_rain(rain: float, moisture: float, saturation: float = SAT,
               params: Dict = DFMC_PARAMS) -> float:
    """
    ## Increase of fuel moisture

    ### Args:
        1. rain (float): cumulated rainfall (mm)
        2. moisture (float): DFMC of previous time step (%)
        3. saturation (float): DFMC saturation value - fuel parameter (%)
        4: params (Dict): dictionary with model's parameters
    ### Return:
        1. (float): increase of fuel moisture (%)
    """
    return params['R1'] * rain * (
                np.e**(-params['R2'] / (saturation + 1.0 - moisture))
            ) * (
                1.0 - np.e**(-params['R3'] / rain))


def check_rain(rain: float, params: Dict = DFMC_PARAMS) -> bool:
    """
    ## Check if rain phase occurs

    ### Args:
        1. rain (float): cumulated rainfall (mm)
        2: params (Dict): dictionary with model's parameters
    ### Return:
        1. (bool): True if rain phase
    """
    return (rain > params['MIN_RAIN'])


def check_no_data(rain: float, temp: float, hum: float, wspeed: float,
                  T0: float = T0, saturation: float = SAT) -> bool:
    """
    ## Check if all needed data are present

    ### Args:
        1. rain (float): cumulated rain (mm)
        2. temp (float): air temperature (°C)
        3. wspeed (float): wind speed (m/h)
        4. hum (float): relative humidity (%)
        5. saturation (float): DFMC saturation value, fuel parameter (%)
        6. T0 (float): time constant in standard condition, fuel parameter (h)
    ### Return:
        1. (bool): True if no data are present
    """
    return not ((rain != NODATAVAL) & (temp != NODATAVAL) &
                (hum != NODATAVAL) & (wspeed != NODATAVAL) &
                (T0 != NODATAVAL) & (saturation != NODATAVAL))


def check_no_update(temp: float) -> bool:
    """
    ## Check if moisture value is not updated

    ### Args:
        1. temp (float): air temperature (°C)
    ### Return:
        1. (bool): True if moisture can be updated
    """
    # only check on temperature
    return not (temp >= 0)
