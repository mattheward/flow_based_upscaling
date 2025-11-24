Matthew Ard 
Energy 224 - Flow Based Upscaling Reservior Simulator
11/23/2025

To run this code you only need to use 'config.py' and 'main.py'.

To change the parameters you will use 'config.py'
To run the simulation and see the results you will use 'main.py'

Base Case

    To run the base case adjust these settings
    Boundary Condition = 0
    well_x = np.array([20])
    well_y = np.array([20])
    well_rate = np.array([1352])
    stop_injection = False

Constant Pressure Case

    To run the constant pressure case adjust these settings
    P_boundary = P_init
    Boundary Condition = 1
    well_x = np.array([20])
    well_y = np.array([20])
    well_rate = np.array([1352])
    stop_injection = False

Two Injection Well Case

    To run the two injection well case adjust these settings
    Boundary Condition = 1
    well_x = np.array([13, 26])
    well_y = np.array([13, 26])
    well_rate = np.array([1352, 1352])
    stop_injection = False

Stop Injection Case

    To run the two injection well case adjust these settings
    Boundary Condition = 0
    well_x = np.array([20])
    well_y = np.array([20])
    well_rate = np.array([1352])
    stop_injection = True
    stop_time = 400