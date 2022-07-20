#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:08:51 2021

@author: mike
"""
import xarray as xr
import pandas as pd
import numpy as np

###########################################
### Functions

param_height_mappings = {'sl1': -0.07,
                         'stl2': -0.28,
                         'stl3': -1,
                         'stl4': -2.89,
                         'swvl1': -0.07,
                         'swvl2': -0.28,
                         'swvl3': -1,
                         'swvl4': -2.89,
                         }


def calc_soil_temp(wrf_xr):
    """

    """
    vars1 = ['stl1', 'stl2', 'stl3', 'stl4']

    ## Assign heights
    ds_list = []
    for v in vars1:
        if v in wrf_xr:
            res1 = wrf_xr[v].assign_coords({'height': param_height_mappings[v]})
            res1 = res1.expand_dims('height')
            res1.name = 'temperature'
        ds_list.append(res1)
    ds = xr.concat(ds_list, dim='height') - 273.15

    return ds


def calc_soil_water(wrf_xr):
    """

    """
    vars1 = ['swvl1', 'swvl2', 'swvl3', 'swvl4']

    ## Assign heights
    ds_list = []
    for v in vars1:
        if v in wrf_xr:
            res1 = wrf_xr[v].assign_coords({'height': param_height_mappings[v]})
            res1 = res1.expand_dims('height')
            res1.name = 'volumetric_water_content'
            ds_list.append(res1)
    ds = xr.concat(ds_list, dim='height')

    return ds


def calc_rh2(wrf_xr):
    """
    Methods to calc relative humidity at 2 meters from the FAO56 manual. Uses 2m temperature and dew point temperature.

    Parameters
    ----------
    wrf_xr : xr.Dataset
        The complete WRF output dataset with Q2, T2, and PSFC.

    Returns
    -------
    xr.DataArray
    """
    ## Assign variables
    t = wrf_xr['t2m'] - 273.15
    dew = wrf_xr['d2m'] - 273.15

    eo = 0.6108*np.exp((17.27*t)/(t + 237.3))
    ea = 0.6108*np.exp((17.27*dew)/(dew + 237.3))

    rh = (ea/eo) * 100

    rh = xr.where(rh > 100, 100, rh).assign_coords(height=2).expand_dims('height')

    rh.name = 'relative_humidity'

    return rh


# def calc_wind_speed2(wrf_xr):
#     """
#     Estimate the mean wind speed at 10 or 2 m from the V and U WRF vectors of wind speed. The 2 m method is according to the FAO 56 paper.

#     Parameters
#     ----------
#     wrf_xr : xr.Dataset
#         The complete WRF output dataset with Q2, T2, and PSFC.
#     height : int
#         The height for the estimate.

#     Returns
#     -------
#     xr.DataArray
#     """
#     u10 = wrf_xr['u10']
#     v10 = wrf_xr['v10']

#     ws = np.sqrt(u10**2 + v10**2)

#     ws = (ws*4.87/(np.log(67.8*10 - 5.42))).assign_coords(height=2)

#     ws.name = 'wind_speed'

#     return ws


def calc_wind_speed10(wrf_xr):
    """
    Estimate the mean wind speed at 10 or 2 m from the V and U WRF vectors of wind speed. The 2 m method is according to the FAO 56 paper.

    Parameters
    ----------
    wrf_xr : xr.Dataset
        The complete WRF output dataset with Q2, T2, and PSFC.
    height : int
        The height for the estimate.

    Returns
    -------
    xr.DataArray
    """
    u10 = wrf_xr['u10']
    v10 = wrf_xr['v10']

    ws = np.sqrt(u10**2 + v10**2).assign_coords(height=10).expand_dims('height')

    ws.name = 'wind_speed'

    return ws


def calc_wind_direction10(wrf_xr):
    """

    """
    u10 = wrf_xr['u10']
    v10 = wrf_xr['v10']

    wd = (np.mod(180 + (180/np.pi)*np.arctan2(v10, u10), 360)).assign_coords(height=10).expand_dims('height')

    return wd


def calc_temp2(wrf_xr, units='degC'):
    """

    """
    t2 = wrf_xr['t2m'].assign_coords(height=2).expand_dims('height')

    if units == 'degC':
        t2 = t2 - 273.15
    elif units != 'K':
        raise ValueError('units must be either degC or K.')

    t2.name = 'temperature'

    return t2


def calc_dew2(wrf_xr):
    """

    """
    dew = (wrf_xr['d2m'] - 273.15).assign_coords(height=2).expand_dims('height')

    dew.name = 'temperature_dew_point'

    return dew


def calc_surface_pressure(wrf_xr, units='hPa'):
    """

    """
    pres = wrf_xr['sp'].assign_coords(height=0).expand_dims('height')

    if units == 'hPa':
        pres = pres * 0.01
    elif units == 'kPa':
        pres = pres * 0.001
    elif units != 'Pa':
        raise ValueError('units must be kPa, hPa, or Pa.')

    pres.name = 'barometric_pressure'

    return pres


def fix_accum(da):
    """

    """
    ## Convert from accumultion to cumultive
    ds2 = da.diff('time')
    ds3 = xr.where(ds2 < 0, da[1:], ds2)
    ds3['time'] = da['time'][:-1]

    return ds3


def calc_precip0(wrf_xr):
    """

    """
    ## Assign variables
    ds1 = wrf_xr['tp']

    ## Convert from accumultion to cumultive
    ds2 = (fix_accum(ds1) * 1000).assign_coords(height=0).expand_dims('height')

    ds2.name = 'precipitation'

    return ds2


def calc_snow0(wrf_xr):
    """

    """
    ## Assign variables
    ds1 = wrf_xr['sf']

    ## Convert from accumultion to cumultive
    ds2 = (fix_accum(ds1) * 1000).assign_coords(height=0).expand_dims('height')

    ds2.name = 'snow_depth'

    return ds2


def calc_snow_cover(data):
    """

    """
    ## Assign variables
    ds1 = data['snowc'].assign_coords(height=0).expand_dims('height') * 0.01

    return ds1


def calc_runoff0(wrf_xr):
    """

    """
    ## Assign variables
    ds1 = wrf_xr['sro']

    ## Convert from accumultion to cumultive
    ds2 = (fix_accum(ds1) * 1000).assign_coords(height=0).expand_dims('height')

    ds2.name = 'surface_runoff'

    return ds2


def calc_recharge0(wrf_xr):
    """

    """
    ## Assign variables
    ds1 = wrf_xr['ssro']

    ## Convert from accumultion to cumultive
    ds2 = (fix_accum(ds1) * 1000).assign_coords(height=0).expand_dims('height')

    ds2.name = 'recharge_groundwater'

    return ds2


def calc_longwave0(wrf_xr):
    """

    """
    ## Assign variables
    ds1 = wrf_xr['str']

    ## Convert from accumultion to cumultive
    ds2 = fix_accum(ds1).assign_coords(height=0).expand_dims('height')

    ds2.name = 'radiation_incoming_longwave'

    return ds2


def calc_shortwave0(wrf_xr):
    """

    """
    ## Assign variables
    ds1 = wrf_xr['ssr']

    ## Convert from accumultion to cumultive
    ds2 = fix_accum(ds1).assign_coords(height=0).expand_dims('height')

    ds2.name = 'radiation_incoming_shortwave'

    return ds2


def calc_heat_flux0(wrf_xr):
    """

    """
    ## Assign variables
    ds1 = wrf_xr['slhf']

    ## Convert from accumultion to cumultive
    ds2 = fix_accum(ds1).assign_coords(height=0).expand_dims('height')

    ds2.name = 'ground_heat_flux'

    return ds2


def calc_pet(wrf_xr):
    """

    """
    ## Assign variables
    ds1 = wrf_xr['pev']

    ## Convert from accumultion to cumultive
    # ds2 = fix_accum(ds1) * 1000
    ds2 = (ds1 * -1000).assign_coords(height=0).expand_dims('height')

    ds2.name = 'potential_et'

    return ds2


def calc_evap(wrf_xr):
    """

    """
    ## Assign variables
    ds1 = wrf_xr['e']

    ## Convert from accumultion to cumultive
    ds2 = (fix_accum(ds1) * -1000).assign_coords(height=0).expand_dims('height')

    ds2.name = 'evaporation'

    return ds2


def calc_eto(wrf_xr):
    """

    """
    ## Assign variables
    dew = wrf_xr['d2m'] - 273.15
    t2 = wrf_xr['t2m'] - 273.15
    pres = wrf_xr['sp']
    gamma = (0.665*10**-3)*pres/1000
    G = fix_accum(wrf_xr['slhf']) * 0.0036
    R_n = (fix_accum(wrf_xr['ssr']) + fix_accum(wrf_xr['str'])) * 0.0036
    u10 = wrf_xr['u10']
    v10 = wrf_xr['v10']
    ws2 = np.sqrt(u10**2 + v10**2)*4.87/(np.log(67.8*10 - 5.42))

    # Humidity
    e_mean = 0.6108*np.exp(17.27*t2/(t2+237.3))
    e_a = 0.6108*np.exp((17.27*dew)/(dew + 237.3))

    # Vapor pressure
    delta = 4098*(0.6108*np.exp(17.27*t2/(t2 + 237.3)))/((t2 + 237.3)**2)
    # R_ns = (1 - alb)*R_s

    # Calc ETo
    ETo = (((0.408*delta*(R_n - G) + gamma*37/(t2 + 273)*ws2*(e_mean - e_a))/(delta + gamma*(1 + 0.34*ws2))) * -1).assign_coords(height=0).expand_dims('height')

    ETo.name = 'reference_et'

    return ETo


##################################################
### dicts

# variables_dict = {'temp_at_2': ['t2m'],
#                     'precip_at_0': ['tp'],
#                     'snow_at_0': ['sf'],
#                     'runoff_at_0': ['sro'],
#                     'recharge_at_0': ['ssro'],
#                     'pressure_at_0': ['sp'],
#                     'shortwave_rad_at_0': ['ssr'],
#                     'longwave_rad_at_0': ['str'],
#                     'heat_flux_at_0': ['slhf'],
#                     'relative_humidity_at_2': ['t2m', 'd2m'],
#                     # 'wind_speed_at_2': ['u10', 'v10'],
#                     'wind_speed_at_10': ['u10', 'v10'],
#                     'reference_et_at_0': ['t2m', 'd2m', 'u10', 'v10', 'ssr', 'str', 'slhf', 'sp'],
#                     'pet_at_0': ['pev'],
#                     'evaporation_at_0': ['e'],
#                     'soil_temp': ['stl1', 'stl2', 'stl3', 'stl4'],
#                     'soil_water': ['stl1', 'stl2', 'stl3', 'stl4']
#                     }

func_dict = {
    'air_temperature': {'variables': ['t2m'],
             'function': calc_temp2,
             'metadata':
                 {'feature': 'atmosphere',
                  'parameter': 'temperature',
                  'aggregation_statistic': 'instantaneous',
                  'units': 'degC',
                  'cf_standard_name': 'air_temperature',
                  'wrf_standard_name': 'T',
                  'precision': 0.01,
                  'properties':
                    {'encoding':
                      {'temperature':
                        {'scale_factor': 0.01,
                        'dtype': 'int16',
                        '_FillValue': -9999}
                        }
                          }
                        }
                  },
    'relative_humidity': {'variables': ['t2m', 'd2m'],
                          'function': calc_rh2,
                          'metadata':
                              {'feature': 'atmosphere',
                               'parameter': 'relative_humidity',
                               'aggregation_statistic': 'instantaneous',
                               'units': '%',
                               'cf_standard_name': 'relative_humidity',
                               'wrf_standard_name': 'RH',
                               'precision': 0.01,
                               'properties':
                                 {'encoding':
                                   {'relative_humidity':
                                     {'scale_factor': 0.01,
                                     'dtype': 'int16',
                                     '_FillValue': -9999}
                                     }
                                       }
                                     }
                },
    'wind_speed': {'variables': ['u10', 'v10'],
                   'function': calc_wind_speed10,
                   'metadata':
                       {'feature': 'atmosphere',
                        'parameter': 'wind_speed',
                        'aggregation_statistic': 'instantaneous',
                        'units': 'm/s',
                        'cf_standard_name': 'wind_speed',
                        'wrf_standard_name': 'UV',
                        'precision': 0.01,
                        'properties':
                          {'encoding':
                            {'wind_speed':
                              {'scale_factor': 0.01,
                              'dtype': 'int16',
                              '_FillValue': -9999}
                              }
                                }
                              }
                        },
    'wind_direction': {'variables': ['u10', 'v10'],
                       'function': calc_wind_direction10,
                       'metadata':
                           {'feature': 'atmosphere',
                            'parameter': 'wind_direction',
                            'aggregation_statistic': 'instantaneous',
                            'units': 'deg',
                            'cf_standard_name': 'wind_from_direction',
                            'wrf_standard_name': 'UV',
                            'precision': 0.1,
                            'properties':
                              {'encoding':
                                {'wind_direction':
                                  {'scale_factor': 0.1,
                                  'dtype': 'int16',
                                  '_FillValue': -999}
                                  }
                                    }
                                  }
                        },
    'dew_temperature': {'variables': ['d2m'],
                 'function': calc_dew2,
                 'metadata':
                     {'feature': 'atmosphere',
                      'parameter': 'temperature_dew_point',
                      'aggregation_statistic': 'instantaneous',
                      'units': 'degC',
                      'cf_standard_name': 'dew_point_temperature',
                      'wrf_standard_name': 'TD',
                      'precision': 0.01,
                      'properties':
                        {'encoding':
                          {'temperature_dew_point':
                            {'scale_factor': 0.01,
                            'dtype': 'int16',
                            '_FillValue': -9999}
                            }
                              }
                            }
                      },
    'barometric_pressure': {'variables': ['sp'],
                         'function': calc_surface_pressure,
                         'metadata':
                             {'feature': 'atmosphere',
                              'parameter': 'barometric_pressure',
                              'aggregation_statistic': 'instantaneous',
                              'units': 'hPa',
                              'cf_standard_name': 'air_pressure',
                              'wrf_standard_name': 'P',
                              'precision': 0.1,
                              'properties':
                                {'encoding':
                                  {'barometric_pressure':
                                    {'scale_factor': 0.1,
                                    'dtype': 'int16',
                                    '_FillValue': -9999}
                                    }
                                      }
                                    }
                      },
    'precipitation': {'variables': ['tp'],
               'function': calc_precip0,
               'metadata':
                   {'feature': 'atmosphere',
                    'parameter': 'precipitation',
                    'aggregation_statistic': 'cumulative',
                    'units': 'mm',
                    'cf_standard_name': 'precipitation_amount',
                    'wrf_standard_name': 'RAINNC',
                    'precision': 0.1,
                    'properties':
                      {'encoding':
                        {'precipitation':
                          {'scale_factor': 0.1,
                          'dtype': 'int16',
                          '_FillValue': -9999}
                          }
                            }
                          }
                     },
    'snowfall': {'variables': ['sf'],
                 'function': calc_snow0,
                 'metadata':
                     {'feature': 'atmosphere',
                      'parameter': 'snow_depth',
                      'aggregation_statistic': 'cumulative',
                      'units': 'mm',
                      'cf_standard_name': 'thickness_of_snowfall_amount',
                      'wrf_standard_name': 'SNOWNC',
                      'precision': 0.1,
                      'properties':
                        {'encoding':
                          {'snow_depth':
                            {'scale_factor': 0.1,
                            'dtype': 'int16',
                            '_FillValue': -9999}
                            }
                              }
                            }
                   },
    'snow_cover': {'variables': ['snowc'],
                   'function': calc_snow_cover,
                   'metadata':
                       {'feature': 'atmosphere',
                        'parameter': 'snow_cover',
                        'aggregation_statistic': 'cumulative',
                        'units': 'm^2/m^2',
                        'cf_standard_name': 'surface_snow_area_fraction',
                        'wrf_standard_name': 'SNOWC',
                        'precision': 0.0001,
                        'properties':
                          {'encoding':
                            {'snow_cover':
                              {'scale_factor': 0.0001,
                              'dtype': 'int16',
                              '_FillValue': -9999}
                              }
                                }
                              }
                   },
    'downward_longwave': {'variables': ['str'],
                          'function': calc_longwave0,
                          'metadata':
                              {'feature': 'atmosphere',
                               'parameter': 'radiation_incoming_longwave',
                               'aggregation_statistic': 'cumulative',
                               'units': 'W/m^2',
                               'cf_standard_name': 'surface_downwelling_longwave_flux_in_air',
                               'wrf_standard_name': 'GLW',
                               'precision': 0.1,
                               'properties':
                                 {'encoding':
                                   {'radiation_incoming_longwave':
                                     {'scale_factor': 0.1,
                                     'dtype': 'int16',
                                     '_FillValue': -9999}
                                     }
                                       }
                                     }
                       },
    'downward_shortwave': {'variables': ['ssr'],
                           'function': calc_shortwave0,
                           'metadata':
                               {'feature': 'atmosphere',
                                'parameter': 'radiation_incoming_shortwave',
                                'aggregation_statistic': 'cumulative',
                                'units': 'W/m^2',
                                'cf_standard_name': 'surface_downwelling_shortwave_flux_in_air',
                                'wrf_standard_name': 'SWDOWN',
                                'precision': 0.1,
                                'properties':
                                  {'encoding':
                                    {'radiation_incoming_shortwave':
                                      {'scale_factor': 0.1,
                                      'dtype': 'int16',
                                      '_FillValue': -9999}
                                      }
                                        }
                                      }

                        },
    'surface_runoff': {'variables': ['sro'],
                       'function': calc_runoff0,
                       'metadata':
                           {'feature': 'pedosphere',
                            'parameter': 'runoff',
                            'aggregation_statistic': 'cumulative',
                            'units': 'mm',
                            'cf_standard_name': 'runoff_amount',
                            'wrf_standard_name': 'SFROFF',
                            'precision': 0.1,
                            'properties':
                              {'encoding':
                                {'runoff':
                                  {'scale_factor': 0.1,
                                  'dtype': 'int16',
                                  '_FillValue': -9999}
                                  }
                                    }
                                  }
                        },
    'gw_recharge': {'variables': ['ssro'],
                    'function': calc_recharge0,
                    'metadata':
                        {'feature': 'pedosphere',
                         'parameter': 'recharge_groundwater',
                         'aggregation_statistic': 'cumulative',
                         'units': 'mm',
                         'cf_standard_name': 'subsurface_runoff_amount',
                         'wrf_standard_name': 'UDROFF',
                         'precision': 0.1,
                         'properties':
                           {'encoding':
                             {'recharge_groundwater':
                               {'scale_factor': 0.1,
                               'dtype': 'int16',
                               '_FillValue': -9999}
                               }
                                 }
                               }
                        },
    'ground_heat_flux': {'variables': ['slhf'],
                         'function': calc_heat_flux0,
                         'metadata':
                             {'feature': 'pedosphere',
                              'parameter': 'ground_heat_flux',
                              'aggregation_statistic': 'cumulative',
                              'units': 'W/m^2',
                              'cf_standard_name': 'downward_heat_flux_in_soil',
                              'wrf_standard_name': 'GRDFLX',
                              'precision': 0.1,
                              'properties':
                                {'encoding':
                                  {'ground_heat_flux':
                                    {'scale_factor': 0.1,
                                    'dtype': 'int16',
                                    '_FillValue': -9999}
                                    }
                                      }
                                    }
                        },
    # 'latent_heat_flux': {'variables': ['latent_heat_flux'],
    #                      'function': latent_heat_flux,
    #                      'metadata':
    #                          {'feature': 'pedosphere',
    #                           'parameter': 'latent_heat_flux',
    #                           'aggregation_statistic': 'cumulative',
    #                           'units': 'W/m^2',
    #                           'cf_standard_name': 'downward_heat_flux_in_soil',
    #                           'wrf_standard_name': 'GRDFLX',
    #                           'precision': 0.1,
    #                           'properties':
    #                             {'encoding':
    #                               {'ground_heat_flux':
    #                                 {'scale_factor': 0.1,
    #                                 'dtype': 'int16',
    #                                 '_FillValue': -9999}
    #                                 }
    #                                   }
    #                                 }
    #                     },
    # 'upward_heat_flux': {'variables': ['ground_heat_flux'],
    #                      'function': ground_heat_flux,
    #                      'metadata':
    #                          {'feature': 'pedosphere',
    #                           'parameter': 'ground_heat_flux',
    #                           'aggregation_statistic': 'cumulative',
    #                           'units': 'W/m^2',
    #                           'cf_standard_name': 'downward_heat_flux_in_soil',
    #                           'wrf_standard_name': 'GRDFLX',
    #                           'precision': 0.1,
    #                           'properties':
    #                             {'encoding':
    #                               {'ground_heat_flux':
    #                                 {'scale_factor': 0.1,
    #                                 'dtype': 'int16',
    #                                 '_FillValue': -9999}
    #                                 }
    #                                   }
    #                                 }
    #                     },
    # 'downward_heat_flux': {'variables': ['ground_heat_flux'],
    #                      'function': ground_heat_flux,
    #                      'metadata':
    #                          {'feature': 'pedosphere',
    #                           'parameter': 'ground_heat_flux',
    #                           'aggregation_statistic': 'cumulative',
    #                           'units': 'W/m^2',
    #                           'cf_standard_name': 'downward_heat_flux_in_soil',
    #                           'wrf_standard_name': 'GRDFLX',
    #                           'precision': 0.1,
    #                           'properties':
    #                             {'encoding':
    #                               {'ground_heat_flux':
    #                                 {'scale_factor': 0.1,
    #                                 'dtype': 'int16',
    #                                 '_FillValue': -9999}
    #                                 }
    #                                   }
    #                                 }
    #                     },
    'soil_temperature': {'variables': ['stl1', 'stl2', 'stl3', 'stl4'],
                  'function': calc_soil_temp,
                  'metadata':
                      {'feature': 'pedosphere',
                       'parameter': 'temperature',
                       'aggregation_statistic': 'instantaneous',
                       'units': 'degC',
                       'cf_standard_name': 'soil_temperature',
                       'wrf_standard_name': 'TSLB',
                       'precision': 0.01,
                       'properties':
                         {'encoding':
                           {'temperature':
                             {'scale_factor': 0.01,
                             'dtype': 'int16',
                             '_FillValue': -9999}
                             }
                               }
                             }
                  },
    'soil_water': {'variables': ['swvl1', 'swvl2', 'swvl3', 'swvl4'],
                   'function': calc_soil_water,
                   'metadata':
                       {'feature': 'pedosphere',
                        'parameter': 'volumetric_water_content',
                        'aggregation_statistic': 'instantaneous',
                        'units': 'm^3/m^3',
                        'cf_standard_name': 'mass_content_of_water_in_soil',
                        'wrf_standard_name': 'SMOIS',
                        'precision': 0.0001,
                        'properties':
                          {'encoding':
                            {'volumetric_water_content':
                              {'scale_factor': 0.0001,
                              'dtype': 'int16',
                              '_FillValue': -9999}
                              }
                                }
                              }
                  },
    # 'avi': {'variables': ['u_wind', 'v_wind', 'pblh'],
    #         'function': avi,
    #         'metadata':
    #             {'feature': 'atmosphere',
    #              'parameter': 'air_ventilation_index',
    #              'aggregation_statistic': 'instantaneous',
    #              'units': 'm^2/s',
    #              # 'cf_standard_name': 'air_pressure',
    #              # 'wrf_standard_name': 'PSFC',
    #              'description': 'The air ventilation index is the product of the mixing height (m) and the transport wind speed (m/s) used as a tool for air quality forecasters to determine the potential of the atmosphere to disperse contaminants such as smoke or smog. We have used the product of the PBLH and 20m wind speed. This is comparible to the air ventilation index used by the University of Washington.',
    #              'precision': 1,
    #              'properties':
    #                {'encoding':
    #                  {'air_ventilation_index':
    #                    {'scale_factor': 1,
    #                    'dtype': 'int16',
    #                    '_FillValue': -9999}
    #                    }
    #                      }
    #                    }
    #               },
    # 'surface_emissivity': {'variables': ['surface_emissivity'],
    #                        'function': surface_emissivity,
    #                        'metadata':
    #                            {'feature': 'pedosphere',
    #                             'parameter': 'surface_emissivity',
    #                             'aggregation_statistic': 'instantaneous',
    #                             'units': '',
    #                             'cf_standard_name': 'surface_longwave_emissivity',
    #                             'wrf_standard_name': 'EMISS',
    #                             'precision': 0.0001,
    #                             'properties':
    #                               {'encoding':
    #                                 {'surface_emissivity':
    #                                   {'scale_factor': 0.0001,
    #                                   'dtype': 'int16',
    #                                   '_FillValue': -9999}
    #                                   }
    #                                     }
    #                                   }
    #               },
    # 'specific_humidity': {'variables': ['water_vapor_mixing_ratio'],
    #                        'function': specific_humidity,
    #                        'metadata':
    #                            {'feature': 'atmosphere',
    #                             'parameter': 'specific_humidity',
    #                             'aggregation_statistic': 'instantaneous',
    #                             'units': 'g/kg',
    #                             'cf_standard_name': 'specific_humidity',
    #                             # 'wrf_standard_name': 'QVAPOR',
    #                             'precision': 0.001,
    #                             'properties':
    #                               {'encoding':
    #                                 {'specific_humidity':
    #                                   {'scale_factor': 0.001,
    #                                   'dtype': 'int16',
    #                                   '_FillValue': -9999}
    #                                   }
    #                                     }
    #                                   }
    #               },
    # 'albedo': {'variables': ['albedo'],
    #                        'function': albedo,
    #                        'metadata':
    #                            {'feature': 'pedosphere',
    #                             'parameter': 'albedo',
    #                             'aggregation_statistic': 'instantaneous',
    #                             'units': '',
    #                             'cf_standard_name': 'surface_albedo',
    #                             'wrf_standard_name': 'ALBEDO',
    #                             'precision': 0.0001,
    #                             'properties':
    #                               {'encoding':
    #                                 {'albedo':
    #                                   {'scale_factor': 0.0001,
    #                                   'dtype': 'int16',
    #                                   '_FillValue': -9999}
    #                                   }
    #                                     }
    #                                   }
    #               },
    }