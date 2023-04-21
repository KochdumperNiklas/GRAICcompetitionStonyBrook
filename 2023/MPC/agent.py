import carla
import time
import pickle
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.affinity import affine_transform
from shapely.affinity import translate
from copy import deepcopy

DT = 0.1                        # time step size
N = 10                          # number of time steps

A_MAX = 6                       # maximum acceleration
A_MIN = -6.17                   # maximum deceleration
S_MAX = np.deg2rad(40.0)        # maximum steering
V_MAX = 20                      # maximum velocity

R = np.diag([0.000, 100.0])      # input cost matrix, penalty for inputs - [accel, steer]
RD = np.diag([0.000, 1000.0])    # input difference cost matrix, penalty for change of inputs - [accel, steer]
W = np.array([[13.5, 13.5, 1000, 13]])  # weights for difference to reference trajectory - [x, y, v, phi]

VISUALIZATION = False

class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Execute one step of navigation.

        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints 
            - Type:         List[[x,y,z], ...] 
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D 
            - Description:  Ego's current velocity in (x, y, z) in m/s
        transform
            - Type:         carla.Transform 
            - Description:  Ego's current transform
        boundary 
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.

        Return: carla.VehicleControl()
        """
        # Actions to take during each simulation step
        # Feel Free to use carla API; however, since we already provide info to you, using API will only add to your delay time
        # Currently the timeout is set to 10s

        # extract road boundaries
        left, right = extract_road_boundary(boundary)

        # extract dynamic obstacles
        obs = get_obstacles(filtered_obstacles)

        # compute state constraints
        con = state_constraints(left, right, obs)

        # assemble initial state
        x0 = get_initial_state(transform, vel)

        # switch to lower trajectory in a hard curve
        v_max = V_MAX
        if is_hard_curve(left, right):
            v_max = 15

        # compute reference trajectory
        ref_traj = reference_trajectory(left, right, x0, v_max, con)

        # plan trajectory
        x, u = optimal_control_problem(x0, ref_traj, con)

        if u is None:
            con = [None for i in range(N+1)]
            x, u = optimal_control_problem(x0, ref_traj, con)
            if u is None:
                u = np.array([[-6.17], [0]])

        # assign control inputs
        acc = u[0, 0]/6.17
        steer = u[1, 0]

        control = carla.VehicleControl()
        control.steer = steer

        if acc > 0:
            control.throttle = acc
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = -acc

        #control.manual_gear_shift = False
        #control.gear = 2

        # visualize planned trajectory
        if VISUALIZATION:
            visualize(left, right, ref_traj, x, obs, con)

        # debug
        """try:
            filehandler = open('trajectory.obj', 'rb')
            data = pickle.load(filehandler)
            data.append(x0)
        except:
            data = [x0]

        filehandler = open('trajectory.obj', 'wb')
        pickle.dump(data, filehandler)"""


        obj = {'left': left, 'right': right}
        filehandler = open('test.obj', 'wb')
        pickle.dump(obj, filehandler)

        return control

def extract_road_boundary(boundary):
    """extract the left and right road boundary"""

    # extract left road boundary
    left = []

    for p in boundary[0]:
        left.append([p.transform.location.x, p.transform.location.y])

    left = np.asarray(left)

    # extract right road boundary
    right = []

    for p in boundary[1]:
        right.append([p.transform.location.x, p.transform.location.y])

    right = np.asarray(right)

    return left, right

def get_initial_state(transform, vel):
    """assemble the initial state of the car as a vector"""

    x = transform.location.x
    y = transform.location.y
    orientation = np.deg2rad(transform.rotation.yaw)
    velocity = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    x0 = np.array([x, y, velocity, orientation])

    return x0

def is_hard_curve(left, right):
    """check if the track ahead represents a hard curve"""

    # compute center trajectory
    center = 0.5 * (left + right)

    diff = center[1, :] - center[0, :]
    orientationStart = np.arctan2(diff[1], diff[0])
    orientation = []

    for i in range(1, center.shape[0]):
        diff = center[i-1, :] - center[i, :]
        angle = np.arctan2(diff[1], diff[0]) - orientationStart
        orientation.append(np.mod(angle + np.pi, 2 * np.pi) - np.pi)
        #orientation.append(np.sign(angle) * np.mod(angle, np.pi))

    do = abs(np.diff(np.asarray(orientation)))

    if max(do[do < 6]) > 0.15:
        return True

    return False

def reference_trajectory(left, right, x0, v_max, con):
    """compute the reference trajectory from the left and right bounds of the road"""

    # compute center trajectory
    center = 0.5 * (left + right)

    length = np.sqrt(np.sum(np.diff(center, axis=0)**2, axis=1))
    dist = np.zeros((len(length)+1, ))

    for i in range(len(length)):
        dist[i+1] = dist[i] + length[i]

    # compute desired acceleration profile based on the state constraints
    a = np.ones((N, )) * A_MAX
    t = np.linspace(0, DT * N, N + 1)

    for i in range(1, N+1):
        if con[i] is not None:
            ind = con[i]['index']
            a_des = max(2*((dist[ind] - 3) - x0[2] * t[i])/(t[i]**2), A_MIN)
            a[0:ind] = np.minimum(a[0:ind], a_des)

    # compute expected trajectory based on desired acceleration profile
    x = np.zeros((N+1, ))
    v = np.zeros((N+1, ))
    v[0] = x0[2]

    for i in range(N):
        x[i+1] = x[i] + v[i]*DT + 0.5*a[i]*DT
        v[i+1] = min(v[i] + a[i]*DT, v_max)

    # find closest point on the reference trajectory
    ind = np.argmin(np.sum((center - x0[0:2])**2, axis=1))

    # compute reference trajectory
    ref_traj = np.zeros((2, N+1))
    ref_traj[:, 0] = center[ind, :]

    for i in range(1, N+1):
        for j in range(ind, center.shape[0]):
            if dist[j] > x[i]:
                ref_traj[:, i] = center[j-1, :]
                ind = j-1
                break

    # add velocity to reference trajectory
    ref_traj = np.concatenate((ref_traj, np.expand_dims(v, axis=0)))

    # add orientation for reference trajectory
    orientation = np.zeros((1, ref_traj.shape[1]))
    orientation[0, 0] = x0[3]

    for i in range(ref_traj.shape[1]-1):
        diff = ref_traj[0:2, i+1] - ref_traj[0:2, i]
        orientation[0, i+1] = np.arctan2(diff[1], diff[0])

    tmp = orientation - x0[3]
    orientation = x0[3] + np.mod(tmp + np.pi, 2*np.pi) - np.pi
    #orientation = np.sign(tmp) * np.mod(np.abs(tmp), np.pi)

    ref_traj = np.concatenate((ref_traj, orientation))

    return ref_traj

def get_obstacles(obs):
    """get dynamic obstacles (= the other traffic participants)"""

    obstacles = [[] for i in range(N+1)]

    for o in obs:

        # get shape and position of the vehicle
        transform = o.get_transform()
        vert = o.bounding_box.get_world_vertices(transform)
        vert_ = []

        for v in vert:
            vert_.append((v.x, v.y))

        car = Polygon(vert_).convex_hull
        obstacles[0].append(car)

        # predict future occupied space under the assumption that the vehicle keeps the same velocity
        vel = o.get_velocity()
        x = 0
        y = 0

        for i in range(N):
            x = x + vel.x * DT
            y = y + vel.y * DT
            obstacles[i+1].append(translate(car, x, y))

    return obstacles

def state_constraints(left, right, obs):
    """compute state constraints"""

    con = [None for i in range(N+1)]

    # compute relevant space around the reference trajectory
    left_vertices = []
    right_vertices = []
    cent_traj = 0.5 * (left + right)

    for i in range(left.shape[0]):

        # compute left vertex
        d = left[i, :] - cent_traj[i, :]
        d = d/np.linalg.norm(d)
        v = cent_traj[i, :] + d * 1
        left_vertices.append(Point(v[0], v[1]))

        # compute right vertex
        d = right[i, :] - cent_traj[i, :]
        d = d / np.linalg.norm(d)
        v = cent_traj[i, :] + d * 1
        right_vertices.append(Point(v[0], v[1]))

    r = deepcopy(right_vertices)
    l = deepcopy(left_vertices)

    r.reverse()
    l.extend(r)
    pgon = Polygon(l)

    # loop over all time steps
    for i in range(len(obs)):
        for o in obs[i]:

            if pgon.intersects(o):

                # loop over all reference trajectory segments
                for j in range(left.shape[0]-1):

                    tmp = Polygon([right_vertices[j], right_vertices[j+1], left_vertices[j+1], left_vertices[j]])

                    if tmp.intersects(o):

                        C = cent_traj[[j+1], :] - cent_traj[[j], :]
                        C = C / np.linalg.norm(C)
                        d = C @ np.transpose(cent_traj[[j], :])

                        if con[i] is None or con[i]['index'] > j:
                            con[i] = {'C': C, 'd': d[0, 0] - 2, 'index': j}

    return con

def optimal_control_problem(x0, ref_traj, con):
    """solve an optimal control problem to obtain a concrete trajectory"""

    # get vehicle model
    f, nx, nu = vehicle_model(x0[2])

    # initialize optimizer
    opti = casadi.Opti()

    # initialize variables
    x = opti.variable(nx, N+1)
    u = opti.variable(nu, N)

    # define cost function
    cost = 0

    for i in range(N+1):

        # minimize control inputs
        if i < N:
            cost += mtimes(mtimes(u[:, i].T, R), u[:, i])

        # minimize difference between consecutive control inputs
        if i < N - 1:
            cost += mtimes(mtimes((u[:, i] - u[:, i + 1]).T, RD), u[:, i] - u[:, i + 1])

        # minimize distance to reference trajectory
        cost += mtimes(mtimes((x[:, i] - ref_traj[:, i]).T, np.diag(W)), x[:, i] - ref_traj[:, i])

    opti.minimize(cost)

    # constraint (trajectory has to satisfy the differential equation)
    for i in range(N):
        opti.subject_to(x[:, i + 1] == f(x[:, i], u[:, i]))

    # state constraints
    """for i in range(N+1):
        if con[i] is not None:
            opti.subject_to(mtimes(con[i]['C'], x[0:2, i]) <= con[i]['d'])"""

    # constraints on the control input
    opti.subject_to(u[0, :] >= A_MIN)
    opti.subject_to(u[0, :] <= A_MAX)
    opti.subject_to(u[1, :] >= -S_MAX)
    opti.subject_to(u[1, :] <= S_MAX)
    opti.subject_to(x[2, :] >= -V_MAX)
    opti.subject_to(x[2, :] <= V_MAX)
    opti.subject_to(x[:, 0] == x0)

    # solver settings
    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0)
    opti.solver("ipopt", p_opts, s_opts)

    # solve optimal control problem
    try:
        sol = opti.solve()
        x_ = sol.value(x)
        u_ = sol.value(u)
    except:
        x_ = None
        u_ = None

    return x_, u_

def vehicle_model(v0):
    """differential equation describing the dynamic behavior of the car"""

    # states
    sx = MX.sym("sx")
    sy = MX.sym("sy")
    v = MX.sym("v")
    phi = MX.sym("phi")

    x = vertcat(sx, sy, v, phi)

    # control inputs
    acc = MX.sym("acc")
    steer = MX.sym("steer")

    u = vertcat(acc, steer)

    # dynamic function
    if v0 < 0.1:
        ode = vertcat(v * cos(phi),
                      v * sin(phi),
                      acc,
                      steer)
    else:
        ode = vertcat(v * cos(phi),
                      v * sin(phi),
                      acc,
                      v *tan(steer)/2.3)

    # define integrator
    options = {'tf': DT, 'simplify': True, 'number_of_finite_elements': 2}
    dae = {'x': x, 'p': u, 'ode': ode}

    intg = integrator('intg', 'rk', dae, options)

    # define a symbolic function x(k+1) = F(x(k),u(k)) representing the integration
    res = intg(x0=x, p=u)
    x_next = res['xf']

    F = Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

    return F, 4, 2

def intersection_polygon_constraint(pgon, con):
    """compue the intersection between a polygon and a constraint"""

    C = con['C']
    d = con['d']
    u = 1000
    C_ = np.array([[C[0, 1], -C[0, 0]]])
    pgon_ = Polygon([Point(d, -u), Point(d, u), Point(u, u), Point(u, -u)])
    pgon_ = affine_transform(pgon_, [C[0, 0], C_[0, 0], C[0, 1], C_[0, 1], 0, 0])
    pgon_ = pgon_.intersection(pgon)

    return pgon_

def visualize(left, right, ref_traj, x, obs, con):
    """visualize the planned trajectory"""

    # plot road
    cent_traj = 0.5 * (left + right)
    plt.plot(left[:,0], left[:,1],'b')
    plt.plot(right[:,0], right[:,1],'b')
    plt.plot(cent_traj[:, 0], cent_traj[:, 1], 'r')
    plt.plot(ref_traj[0, :], ref_traj[1, :], 'g')

    # plot planned trajectory
    if x is not None:
        plt.plot(x[0, :], x[1, :])

    # plot obstacles
    for ob in obs:
        for o in ob:
            plt.plot(*o.exterior.xy, 'k')

    # plot constraints
    x_max = max(np.max(left[:, 0]), np.max(right[:, 0]))
    x_min = min(np.min(left[:, 0]), np.min(right[:, 0]))
    y_max = max(np.max(left[:, 1]), np.max(right[:, 1]))
    y_min = min(np.min(left[:, 1]), np.min(right[:, 1]))

    bound = Polygon([Point(x_min, y_min), Point(x_min, y_max), Point(x_max, y_max), Point(x_max, y_min)])

    for i in range(N+1):
        if con[i] is not None:
            pgon = intersection_polygon_constraint(bound, con[i])
            plt.plot(*pgon.exterior.xy, 'r')

    plt.xlim(x_min - 2, x_max + 2)
    plt.ylim(y_min - 2, y_max + 2)

    plt.pause(0.1)
    plt.cla()
