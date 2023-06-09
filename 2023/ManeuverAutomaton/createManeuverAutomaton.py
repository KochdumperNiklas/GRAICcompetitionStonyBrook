import numpy as np
import pickle
from scipy.integrate import solve_ivp
from ManeuverAutomaton import MotionPrimitive
from ManeuverAutomaton import ManeuverAutomaton

# acceleration
accelerations = [-6, -3, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 2, 3]
accelerations = [-6, -3, -1.2, 0, 1.2, 3]

# desired final orientation
orientation1 = [-1, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
orientation2 = [-0.1, 0, 0.1]

# maximum steering angle
s_max = 1

# wheelbase of the car
WB = 2.7

# velocity range
v_start = 0
v_end = 10
v_diff = 0.2

# time of the motion primitives
tFinal = 0.4

# loop over all initial velocities
primitives = []
v_init = v_start

while v_init < v_end:

    # loop over all accelerations
    for acc in accelerations:

        # check if the motion primitive can be connected to other motion primitives
        if v_start <= v_init + acc*tFinal <= v_end:

            if v_init > 5:
                orientation = orientation1
            else:
                orientation = orientation2

            # loop over all final orientations
            for o in orientation:

                if abs(v_init * tFinal + 0.5*acc * tFinal**2) > 0:

                    # compute the required steering angle to achieve the desired final orientation
                    steer = np.arctan(WB * o / (v_init * tFinal + 0.5*acc * tFinal**2))

                    if abs(steer) < s_max:

                        # simulate the system
                        ode = lambda t, x, u1, u2: [x[2] * np.cos(x[3]) + WB/2 * np.sin(x[3]) * x[2] * np.tan(u2) / WB,
                                                    x[2] * np.sin(x[3]) + WB/2 * np.cos(x[3]) * x[2] * np.tan(u2) / WB,
                                                    u1,
                                                    x[2] * np.tan(u2) / WB]
                        sol = solve_ivp(ode, [0, tFinal], [0, 0, v_init, 0], args=(acc, steer), dense_output=True)
                        t = np.linspace(0, tFinal, 5)
                        x = sol.sol(t)

                        # construct the motion primitive
                        primitives.append(MotionPrimitive(x, np.array([acc, steer]), tFinal))

    v_init = v_init + v_diff

# construct and save maneuver automaton
MA = ManeuverAutomaton(primitives, v_end, v_diff)

filehandler = open('maneuverAutomaton.obj', 'wb')
pickle.dump(MA, filehandler)

