from utils.setup_grid import setup_grid
g, xs, ys, X, Y, vel_field_data, nmodes, nrzns, paths, params, param_str = setup_grid()
print(X)
print(Y)
print(paths[0,0].shape)
