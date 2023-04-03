import carla
import time
from ManeuverAutomaton import ManeuverAutomaton
import pickle
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from copy import deepcopy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.affinity import affine_transform
from shapely.affinity import translate
from scipy.linalg import block_diag

DT = 0.1                        # time step size
HORIZON = 2                     # number of time steps
N = 8
A_MAX = 3                    # maximum acceleration
V_MAX = 30

R = np.diag([0.000, 5000.0])
W = np.array([[13.5, 13.5, 0, 5]])

VISUALIZATION = True

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

        # load the maneuver automaton
        filehandler = open('maneuverAutomaton.obj', 'rb')
        MA = pickle.load(filehandler)

        # extract road boundaries
        left, right = extract_road_boundary(boundary)

        # extract dynamic obstacles
        obs = get_obstacles(filtered_obstacles)

        # assemble initial state
        x0 = get_initial_state(transform, vel)

        # compute polygon representing the road boundary
        road = road_boundary(left, right)

        # compute reference trajectory
        ref_traj = reference_trajectory(left, right, x0)

        # plan the trajectory using A*-search
        x, u = seach_problem(MA, x0, ref_traj, obs, road)

        if x is None:
            u = np.array([[-6.17], [0]])

        # assign control inputs
        acc = u[0, 0] / 6.17
        steer = u[1, 0]

        control = carla.VehicleControl()
        control.steer = steer

        if acc > 0:
            control.throttle = acc
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = -acc

        # visualize planned trajectory
        if VISUALIZATION:
            visualize(left, right, ref_traj, x, obs)

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

def road_boundary(left, right):
    """construct a polygon that represents the road boundary"""

    road = []
    cent_traj = 0.5 * (left + right)

    # left road boundary
    left_vertices = []
    right_vertices = []

    for i in range(left.shape[0]):

        d = left[i, :] - cent_traj[i, :]
        v = cent_traj[i, :] + d * 1.1
        left_vertices.append(Point(v[0], v[1]))
        right_vertices.append(Point(left[i, 0], left[i, 1]))

    right_vertices.reverse()
    left_vertices.extend(right_vertices)
    road.append(Polygon(left_vertices))

    # right road boundary
    left_vertices = []
    right_vertices = []

    for i in range(left.shape[0]):
        d = right[i, :] - cent_traj[i, :]
        v = cent_traj[i, :] + d * 1.1
        left_vertices.append(Point(v[0], v[1]))
        right_vertices.append(Point(right[i, 0], right[i, 1]))

    right_vertices.reverse()
    left_vertices.extend(right_vertices)
    road.append(Polygon(left_vertices))

    return road

def reference_trajectory(left, right, x0):
    """compute the reference trajectory from the left and right bounds of the road"""

    # compute center trajectory
    center = 0.5 * (left + right)

    length = np.sqrt(np.sum(np.diff(center, axis=0)**2, axis=1))
    dist = np.zeros((len(length)+1, ))

    for i in range(len(length)):
        dist[i+1] = dist[i] + length[i]

    # compute expected trajectory based on current velocity
    t = np.linspace(0, DT*N, N+1)
    v = x0[2] + A_MAX * t
    v = np.minimum(v, V_MAX)
    x = np.zeros(v.shape)

    for i in range(len(v)-1):
        x[i+1] = x[i] + v[i]*DT

    # find closest point on the reference trajectory
    ind = np.argmin(np.sum((center - x0[0:2])**2, axis=1))

    # compute reference trajectory
    ref_traj = np.zeros((2, N+1))
    ref_traj[:, 0] = center[ind, :]

    for i in range(1, N+1):
        for j in range(ind, center.shape[0]):
            if dist[j] > x[i]:
                ref_traj[:, i] = center[j, :]
                ind = j
                break

    # add velocity to reference trajectory
    ref_traj = np.concatenate((ref_traj, V_MAX * np.ones((1, ref_traj.shape[1]))))

    # add orientation for reference trajectory
    orientation = np.zeros((1, ref_traj.shape[1]))
    orientation[0, 0] = x0[3]

    for i in range(ref_traj.shape[1] - 1):
        diff = ref_traj[0:2, i + 1] - ref_traj[0:2, i]
        orientation[0, i + 1] = np.arctan2(diff[1], diff[0])

    tmp = orientation - x0[3]
    orientation = np.sign(tmp) * np.mod(np.abs(tmp), np.pi)

    ref_traj = np.concatenate((ref_traj, orientation))

    return ref_traj


def seach_problem(MA, x0, ref_traj, obs, road):
    """solve the A*-search problem for planning with a maneuver automaton"""

    # initialize queue for the search problem
    ind = MA.velocity2primitives(x0[2])
    node = Node([], np.expand_dims(x0, axis=1), 0)
    queue = []

    for i in ind:
        queue.append(expand_node(node, MA.primitives[i], i, ref_traj))

    # loop until goal set is reached or queue is empty
    while len(queue) > 0:

        # sort the queue
        queue.sort(key=lambda i: i.cost)

        # select node with the lowest costs
        node = queue.pop(0)
        primitive = MA.primitives[node.primitives[-1]]

        # check if motion primitive is collision free
        if collision_check(node, primitive, obs, road):

            # check if planning horizon has been reached
            if len(node.primitives) == HORIZON:
                u = extract_control_inputs(node, MA.primitives)
                return node.x, u

            # construct child nodes
            for i in primitive.successors:
                queue.append(expand_node(node, MA.primitives[i], i, ref_traj))

    return None, None

def collision_check(node, primitive, obs, road):
    """check if a motion primitive is collision free"""

    # get state at the before the last primitive
    ind = node.x.shape[1] - primitive.x.shape[1]
    x = node.x[:, ind]

    # loop over all time steps
    for o in primitive.occ:

        # transform motion primitive to the current state
        time = int(ind + o['time']/DT)
        pgon = affine_transform(o['space'], [np.cos(x[3]), -np.sin(x[3]), np.sin(x[3]), np.cos(x[3]), x[0], x[1]])

        # check if the motion primitive is inside the road
        for bound in road:
            if bound.intersects(pgon):
                return False

        # check if the motion primitive intersects any obstacles
        for obstacle in obs[time]:
            if pgon.intersects(obstacle):
                return False

    return True

def extract_control_inputs(node, primitives):
    """construct the sequence of control inputs for the given node"""

    u = []

    for i in range(len(node.primitives)):
        primitive = primitives[node.primitives[i]]
        u_new = np.expand_dims(primitive.u, axis=1) @ np.ones((1, primitive.x.shape[1]-1))
        if i == 0:
            u = u_new
        else:
            u = np.concatenate((u, u_new), axis=1)

    return u

def visualize(left, right, ref_traj, x, obs):
    """visualize the planned trajectory"""

    # plot road
    cent_traj = 0.5 * (left + right)
    plt.plot(left[:,0], left[:,1],'b')
    plt.plot(right[:,0], right[:,1],'b')
    plt.plot(cent_traj[:, 0], cent_traj[:, 1], 'r')
    plt.plot(ref_traj[0, :], ref_traj[1, :], 'g')

    # plot planned trajectory
    if x is not None:
        plot_trajectory(x)

    # plot obstacles
    for ob in obs:
        for o in ob:
            plt.plot(*o.exterior.xy, 'k')

    # formatting
    x_max = max(np.max(left[:, 0]), np.max(right[:, 0]))
    x_min = min(np.min(left[:, 0]), np.min(right[:, 0]))
    y_max = max(np.max(left[:, 1]), np.max(right[:, 1]))
    y_min = min(np.min(left[:, 1]), np.min(right[:, 1]))

    plt.xlim(x_min - 2, x_max + 2)
    plt.ylim(y_min - 2, y_max + 2)

    plt.pause(0.1)
    plt.cla()

def plot_trajectory(x):
    """plot the trajectory for the given node"""

    # shape of the car
    L = 4.3
    W = 1.7
    car = Polygon([(-L / 2, -W / 2), (-L / 2, W / 2), (L / 2, W / 2), (L / 2, -W / 2)])

    # plot car
    for i in range(x.shape[1]):
        phi = x[3, i]
        tmp = affine_transform(car, [np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi), x[0, i], x[1, i]])
        plt.plot(*tmp.exterior.xy, 'b')

def expand_node(node, primitive, ind, ref_traj):
    """add a new primitive to a node"""

    # add current primitive to the list of primitives
    primitives = node.primitives + [ind]

    # combine trajectories
    phi = node.x[3, -1]
    T = block_diag(np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]), np.eye(2))
    x_ = T @ primitive.x + np.array([[node.x[0, -1]], [node.x[1, -1]], [0], [phi]])
    x = np.concatenate((node.x[:, :-1], x_), axis=1)

    # compute costs
    ind = x.shape[1] - primitive.x.shape[1]
    if x.shape[1] <= ref_traj.shape[1]:
        index = range(ind, x.shape[1])
    else:
        index = range(ind, ref_traj.shape[1])

    cost = node.cost + np.sum(W @ (ref_traj[:, index] - x[:, index])**2) + np.transpose(primitive.u) @ R @ primitive.u

    return Node(primitives, x, cost)

class Node:
    """class representing a node for A*-search"""

    def __init__(self, primitives, x, cost):
        """class constructor"""

        self.primitives = primitives
        self.x = x
        self.cost = cost