#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:59:16 2021

@author: mike
"""







##########################################3
### Objects

param_file_mappings = {'t2m': '2m_temperature_*.nc',
                       'd2m': '2m_dewpoint_temperature_*.nc',
                       'tp': 'total_precipitation_*.nc',
                       'sf': 'snowfall_*.nc',
                       'sro': 'surface_runoff_*.nc',
                       'ssro': 'sub_surface_runoff_*.nc',
                       'sp': 'surface_pressure_*.nc',
                       'ssr': 'surface_net_solar_radiation_*.nc',
                       'str': 'surface_net_thermal_radiation_*.nc',
                       'slhf': 'surface_latent_heat_flux_*.nc',
                       'u10': '10m_u_component_of_wind_*.nc',
                       'v10': '10m_v_component_of_wind_*.nc',
                       # 'reference_et_at_0': ['2m_temperature_*.nc', '2m_dewpoint_temperature_*.nc', '10m_u_component_of_wind_*.nc', '10m_v_component_of_wind_*.nc', 'surface_net_solar_radiation_*.nc', 'surface_net_thermal_radiation_*.nc', 'surface_latent_heat_flux_*.nc', 'surface_pressure_*.nc'],
                       'pev': 'potential_evaporation_*.nc',
                       'e': 'total_evaporation_*.nc',
                       'stl1': 'soil_temperature_*.nc',
                       'stl2': 'soil_temperature_*.nc',
                       'stl3': 'soil_temperature_*.nc',
                       'stl4': 'soil_temperature_*.nc',
                       'swvl1': 'volumetric_soil_water_*.nc',
                       'swvl2': 'volumetric_soil_water_*.nc',
                       'swvl3': 'volumetric_soil_water_*.nc',
                       'swvl4': 'volumetric_soil_water_*.nc',
                       }

# param_func_mappings = {'temp_at_2': ['t2m'],
#                        'precip_at_0': ['tp'],
#                        'snow_at_0': ['sf'],
#                        'runoff_at_0': ['sro'],
#                        'recharge_at_0': ['ssro'],
#                        'pressure_at_0': ['sp'],
#                        'shortwave_rad_at_0': ['ssr'],
#                        'longwave_rad_at_0': ['str'],
#                        'heat_flux_at_0': ['slhf'],
#                        'relative_humidity_at_2': ['t2m', 'd2m'],
#                        'wind_speed_at_2': ['u10', 'v10'],
#                        'wind_speed_at_10': ['u10', 'v10'],
#                        'reference_et_at_0': ['t2m', 'd2m', 'u10', 'v10', 'ssr', 'str', 'slhf', 'sp'],
#                        'pet_at_0': ['pev'],
#                        'evaporation_at_0': ['e'],
#                        'soil_temp': ['stl1', 'stl2', 'stl3', 'stl4'],
#                        # 'soil_water': ['swvl1', 'swvl2', 'swvl3', 'swvl4']
#                        }


param_descriptions = {'soil_temp': """The ECMWF Integrated Forecasting System (IFS) has a four-layer representation of soil, where the surface is at 0cm:

Layer 1: 0 - 7cm
Layer 2: 7 - 28cm
Layer 3: 28 - 100cm
Layer 4: 100 - 289cm

Soil temperature is set at the middle of each layer, and heat transfer is calculated at the interfaces between them. It is assumed that there is no heat transfer out of the bottom of the lowest layer.""",
                    'soil_water': """The ECMWF Integrated Forecasting System model has a four-layer representation of soil:
Layer 1: 0 - 7cm
Layer 2: 7 - 28cm
Layer 3: 28 - 100cm
Layer 4: 100 - 289cm

The volumetric soil water is associated with the soil texture (or classification), soil depth, and the underlying groundwater level."""}
