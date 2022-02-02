import numpy as np
# import os
from plot_snap import read_snap, plot_snap, get_para, rotate


def save_to_file(x, y, theta, para, adjust_rho0=False):
    n_new = int(para["rho0"] * para["Lx"] * para["Ly"])

    if n_new <= x.size:
        x = x[:n_new]
        y = y[:n_new]
        theta = theta[:n_new]
    else:
        n2 = n_new - x.size
        x2 = np.random.rand(n2) * para["Lx"]
        y2 = np.random.rand(n2) * para["Ly"]
        theta2 = np.random.rand(n2) * np.pi * 2
        x = np.hstack((x, x2))
        y = np.hstack((y, y2))
        theta = np.hstack((theta, theta2))
        print("Warning, add", n2, "particles")

    if adjust_rho0:
        para["rho0"] = x.size / (para["Lx"] * para["Ly"])
    elif x.size != int(para["rho0"] * para["Lx"] * para["Ly"]):
        print("Waring, particle number =", x.size)
    fout_subfix = "{eta:.3f}_{rho0:.3f}_{v0:.1f}_{seed}_{t:08d}.bin".format(
        **para)
    if para["Lx"] == para["Ly"]:
        fout = f"duplicated/s{para['Lx']}_{fout_subfix}"
    else:
        fout = f"duplicated/s{para['Lx']}_{para['Ly']}_{fout_subfix}"
    print("output new snapshot to", fout)
    data = np.zeros((3, n_new), np.float32)
    data[0] = x
    data[1] = y
    data[2] = theta
    data = data.T.flatten()
    data.tofile(fout)


def duplicate(fin, nx=1, ny=1, rot_angle=None):
    para0 = get_para(fin)
    x0, y0, theta0 = read_snap(fin)
    if rot_angle is not None:
        x0, y0, theta0 = rotate(x0, y0, theta0, rot_angle, para0["Lx"],
                                para0["Ly"])
    para_new = {key: para0[key] for key in para0}
    if rot_angle is None:
        para_new["Lx"] *= nx
        para_new["Ly"] *= ny
        dx = para0["Lx"]
        dy = para0["Ly"]
    elif rot_angle == 90 or rot_angle == -90:
        para_new["Lx"] = para0["Ly"] * nx
        para_new["Ly"] = para0["Lx"] * ny
        dx = para0["Ly"]
        dy = para0["Lx"]
    para_new["seed"] = int("2%d%d%d" % (nx, ny, para0["seed"]))
    para_new["t"] = 0
    N0 = x0.size

    x, y, theta = np.zeros((3, x0.size * nx * ny), np.float32)
    for row in range(ny):
        for col in range(nx):
            k = col + row * nx
            beg = k * N0
            end = beg + N0
            x[beg:end] = x0 + col * dx
            y[beg:end] = y0 + row * dy
            theta[beg:end] = theta0

    plot_snap(x0, y0, theta0, para0, frac=2e4 / x0.size)
    plot_snap(x, y, theta, para_new, frac=2e4 / x.size)
    save_to_file(x, y, theta, para_new)


def slice(x, y, theta, rect, shift=False):
    xmin, xmax, ymin, ymax = rect
    mask = y >= ymin
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    mask = y < ymax
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    mask = x >= xmin
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    mask = x < xmax
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    if shift:
        x -= xmin
        y -= ymin
    return x, y, theta


def inverse(x, y, theta, xc, yc):
    x_inv = 2 * xc - x
    y_inv = 2 * yc - y
    theta_inv = theta + np.pi
    return x_inv, y_inv, theta_inv


def make_band_lane(x, y, theta, n_lane, width, length, direction="y"):
    if direction == "y":
        x_inv, y_inv, theta_inv = inverse(x, y, theta, length / 2, width / 2)
    else:
        x_inv, y_inv, theta_inv = inverse(x, y, theta, width / 2, length / 2)

    n_new = x.size * n_lane
    x_new, y_new, theta_new = np.zeros((3, n_new), np.float32)
    for i_lane in range(n_lane):
        i = i_lane * x.size
        j = (i_lane + 1) * x.size
        if i_lane % 2 == 0:
            x_new[i:j] = x
            y_new[i:j] = y
            theta_new[i:j] = theta
        else:
            x_new[i:j] = x_inv
            y_new[i:j] = y_inv
            theta_new[i:j] = theta_inv
        if direction == "y":
            y_new[i:j] += i_lane * width
        else:
            x_new[i:j] += i_lane * width
    return x_new, y_new, theta_new


def create_band_lane_snap():
    # fin = "snap/s2400_0.290_1.000_0.5_133_47200000.bin"
    fin = "snap/s2400_0.350_1.000_0.5_411_26160000.bin"
    para = get_para(fin)
    if para["seed"] == 133:
        direction = "y"
        length = para["Lx"]
        # width, ymin, ymax = 400, 1500, 1900
        width, ymin, ymax = 600, 0, 600
        rect = [0, para["Lx"], ymin, ymax]
    else:
        direction = "x"
        length = para["Ly"]
        width, xmin, xmax = 400, 1500, 1900
        rect = [xmin, xmax, 0, para["Ly"]]
    x, y, theta = read_snap(fin)
    x, y, theta = slice(x, y, theta, rect, shift=True)
    n_lane = 6
    para["Ly"] = n_lane * width
    para["t"] = 0
    para["seed"] = int("%d%d%d" % (2, n_lane, para["seed"]))
    x, y, theta = make_band_lane(x, y, theta, n_lane, width, length, direction)
    save_to_file(x, y, theta, para, adjust_rho0=True)


def piece_together(f1, f2, rot_ang1=None, rot_ang2=None, save=True):
    para1 = get_para(f1)
    x, y, theta = read_snap(f1)
    x1, y1, theta1 = rotate(x, y, theta, rot_ang1, para1["Lx"], para1["Ly"])

    para2 = get_para(f2)
    x, y, theta = read_snap(f2)
    x2, y2, theta2 = rotate(x, y, theta, rot_ang2, para2["Lx"], para2["Ly"])

    x2 += para2["Ly"]
    x = np.hstack((x1, x2))
    y = np.hstack((y1, y2))
    theta = np.hstack((theta1, theta2))

    para = para1
    para["Lx"] = para1["Lx"] * 2
    para["seed"] = int(f"2{para1['seed']}{para2['seed']}")
    para["t"] = 0
    plot_snap(x, y, theta, para, show_relative_angle=False, frac=0.05)
    if save:
        save_to_file(x, y, theta, para)


def make_lane(fin, save=True):
    para0 = get_para(fin)
    x0, y0, theta0 = read_snap(fin)
    x1, y1, theta1 = rotate(x0.copy(), y0.copy(), theta0.copy(), 180,
                            para0["Lx"], para0["Ly"])
    x1 += para0["Lx"]
    x = np.hstack((x0, x1))
    y = np.hstack((y0, y1))
    theta = np.hstack((theta0, theta1))
    para = para0
    para["Lx"] = para0["Lx"] * 2
    para["t"] = 0
    plot_snap(x, y, theta, para, show_relative_angle=False, frac=0.01)
    if save:
        save_to_file(x, y, theta, para)
    

def add_particles(fin, rho_add, save=True):
    para0 = get_para(fin)
    x0, y0, theta0 = read_snap(fin)
    rho_new = para0["rho0"] + rho_add
    n_old = x0.size
    n_new = int(para0["Lx"] * para0["Ly"] * rho_new)
    x1, y1, theta1 = np.zeros((3, n_new), np.float32)
    x1[:n_old] = x0
    y1[:n_old] = y0
    theta1[:n_old] = theta0
    x1[n_old:n_new] = np.random.rand(n_new-n_old) * para0["Lx"]
    y1[n_old:n_new] = np.random.rand(n_new-n_old) * para0["Ly"]
    theta1[n_old:n_new] = np.random.rand(n_new-n_old) * np.pi * 2
    para = para0
    para["rho0"] = rho_new
    para["t"] = 0
    plot_snap(x1, y1, theta1, show_relative_angle=False, frac=0.01)
    if save:
        save_to_file(x1, y1, theta1, para)


if __name__ == "__main__":
    # fin = "snap/s1200_0.310_0.637_1.0_1014_00750000.bin"
    # fin = "duplicated/s2400_0.310_0.637_1.0_2221014_00000000.bin"
    # fin = "snap/s9600_2400_0.290_1.000_0.5_411_62720000.bin"
    # fin = "snap/s2400_0.280_1.000_0.5_134_01180000.bin"
    # duplicate(fin, 8, 1, rot_angle=-90)

    # x, y, theta = read_snap(fin)
    # para = get_para(fin)
    # plot_snap(x, y, theta, para, frac=0.01)

    # create_band_lane_snap()

    # f1 = "snap/s1200_0.290_0.637_0.5_1003_00650000.bin"
    # f2 = "snap/s1200_0.290_0.637_0.5_1003_00650000.bin"
    # piece_together(f1, f2, 90, 90, True)

    # fin = "snap/s1200_0.310_0.637_0.5_1011_00650000.bin"
    # duplicate(fin, 2, 2)

    fin = "duplicated/s4800_0.410_1.000_0.5_44125_00000000.bin"
    # duplicate(fin, 2, 1)
    make_lane(fin)

    # fin = "snap/s1200_0.380_1.000_0.5_1011_00540000.bin"
    # add_particles(fin, 0.25)

    # fin = "snap/s1200_0.410_1.000_0.5_125_00720000.bin"
    # duplicate(fin, 2, 4)
