Matthew Ard 
Energy 224 - Flow Based Upscaling Reservior Simulator
11/23/2025

To run this code you only need to use 'config.py' and 'main.py'.

To change the parameters you will use 'config.py'
To run the simulation and see the results you will use 'main.py'

Use the following settings for each case.

Case A:

Simulation.SIM_CASE = 'A'
Upscaling.GRID_TYPE = 'structured'

   WELLS = {
        'well1': {
            'type': 'injector',
            'location': [30, 30],
            'control': 'rate',
            'rates': [900]  # [STB/day]
        }
    }


Case B:

Simulation.SIM_CASE = 'B'
Upscaling.GRID_TYPE = 'structured'

   WELLS = {
        'well1': {
            'type': 'injector',
            'location': [23, 41],
            'control': 'rate',
            'rates': [900]  # [STB/day]
        },
        'well2': {
            'type': 'injector',
            'location': [14, 23],
            'control': 'rate',
            'rates': [1400]  # [STB/day]
        },
        'well3': {
            'type': 'producer',
            'location': [43, 23],
            'control': 'bhp',
            'bhp': 2000  # [STB/day]
        }
    }


Case C:

Simulation.SIM_CASE = 'C'
Upscaling.GRID_TYPE = 'structured'

    WELLS = {
        'well1': {
            'type': 'injector',
            'location': [13, 28],
            'control': 'rate',
            'rates': [1400]  # [STB/day]
        },
        'well3': {
            'type': 'producer',
            'location': [38, 28],
            'control': 'bhp',
            'bhp': 2000  # [STB/day]
        }
    }