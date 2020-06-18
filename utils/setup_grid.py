from grid_world import timeOpt_grid
from utils.custom_functions import my_meshgrid
from definition import ROOT_DIR
import scipy.io
import numpy as np
from os import getcwd
from os.path import join
import math


def get_filler_coords(traj, start_pos):
    x0, y0 = start_pos
    x, y = traj[0]
#     num_points = int(np.linalg.norm(traj[0] - np.array([x0, y0]), 2) // np.linalg.norm(traj[0] - traj[1], 2))
    num_points = int(80) # changed to specifically suit DG trajs

    filler_xy = np.linspace((x0, y0), (x, y), int(num_points), endpoint=False)
    return filler_xy


def prune_and_pad_paths(path_ndarray, start_xy, end_xy):
    xf, yf = end_xy
    # _, num_rzns = path_ndarray.shape
    num_rzns, _ = path_ndarray.shape

    for n in range(num_rzns):
        # prune path
        l = len(path_ndarray[n, 0])
        idx_list = []
        for i in range(l - 100, l):
            x, y = path_ndarray[n, 0][i]
            if x < xf or y > yf:
#             if x < xf and x>xf:
                idx_list.append(i)
            elif math.isnan(x) or math.isnan(y):
                idx_list.append(i)
        path_ndarray[n, 0] = np.delete(path_ndarray[n, 0], idx_list, axis=0)

        # pad path
        filler = get_filler_coords(path_ndarray[n, 0], start_xy)
        path_ndarray[n, 0] = np.append(filler, path_ndarray[n, 0], axis=0)
    return path_ndarray


# IMP: default values of nt, dt, F, startpos, endpos are taken from DG2. 
# startpos and enpos are based on coords start_coord = (0.1950, 0.2050), end_coord = (0.4, 0.8) / (0.41, 0.8)
#                                                                                     (20, 40) / (20, 41)

def setup_grid(num_actions =8, startpos = (4,2), endpos = (1,2)):

    gsize = 5

    #paths = load_DP_policy_paths
    traj = np.array([[2,0],
                    [2, 1],
                    [2, 2],
                    [2, 3]])
    paths = np.array([[traj]])
    xs = np.arange(gsize)
    ys = np.arange(gsize)
    X, Y = my_meshgrid(xs, ys)

    nt = gsize
    F = 1
    dt = 1

    nmodes = 1  
    nrzns = 1

    all_u_mat = np.zeros((nt, gsize, gsize))
    all_v_mat = np.zeros((nt, gsize, gsize, gsize))
    all_ui_mat = np.zeros(())
    all_vi_mat = np.zeros((nt, nmodes, gsize, gsize))
    all_yi = np.zeros((nt, nrzns, nmodes))
    vel_field_data = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_yi]
    # all_u_mat[:,4,:] = 1
    # all_ui_mat[:,4,:] = 1
    # for i in range(nrzns):
    #     all_yi[:,i,:] = np.random.normal(0,1)

    param_str = ['num_actions', 'nt', 'dt', 'F', 'startpos', 'endpos']
    params = [num_actions, nt, dt, F, startpos, endpos]

    g = timeOpt_grid(xs, ys, dt, nt, F, startpos, endpos, num_actions=num_actions)
    print("Grid Setup Complete !")

    # CHANGE RUNNER FILE TO GET PARAMS(9TH ARG) IF YOU CHANGE ORDER OF RETURNS HERE
    return g, xs, ys, X, Y, vel_field_data, nmodes, nrzns, paths, params, param_str


