import numpy as np
import matplotlib.pyplot as plt
import os
# import sys
from scipy.interpolate import splev, splrep
from read_fields import get_para, read_fields, get_nframe


def detect_lane(x, profile, max_lanes, show_smooth=False):
    def locate_zero(y):
        idx = []
        for i in range(x.size):
            a = y[i - 1]
            b = y[i]
            if a < 0 <= b or a > 0 >= b:
                idx.append(i)
        return idx

    edge = locate_zero(profile)
    s = 1
    while len(edge) > max_lanes:
        n_edges = len(edge)
        if show_smooth:
            plt.plot(x, profile)
            for i in edge:
                plt.axvline(x[i], linestyle="dashed")
        spl = splrep(x, profile, per=True, k=5, s=s)
        profile_s = splev(x, spl)
        edge = locate_zero(profile_s)
        if show_smooth:
            plt.plot(x, profile_s)
            for i in edge:
                plt.axvline(x[i], linestyle="dashed", c="tab:red")
            plt.show()
            plt.close()
        print("Warning, after smoothing, edge number:%d->%d" %
              (n_edges, len(edge)))
        s += 1
    return edge


def get_bands_profile(rho, lane_edges, band_ori="x"):
    nx = rho.shape[1]
    n_lane = len(lane_edges)
    if n_lane == 0:
        n_lane = 1
    if band_ori == "x":
        rho_y = np.zeros((n_lane, rho.shape[0]))
    else:
        rho_y = np.zeros((n_lane, rho.shape[1]))
    if n_lane == 1:
        if band_ori == "x":
            rho_y[0] = np.mean(rho, axis=1)
        else:
            rho_y[0] = np.mean(rho, axis=0)
    else:
        for i_lane in range(n_lane - 1):
            xi = lane_edges[i_lane]
            xj = lane_edges[i_lane + 1]
            if band_ori == "x":
                rho_y[i_lane] = np.mean(rho[:, xi:xj], axis=1)
            else:
                rho_y[i_lane] = np.mean(rho[xi:xj, :], axis=0)
        xi = lane_edges[n_lane - 1]
        xj = lane_edges[0]
        if band_ori == "x":
            rho_y[n_lane - 1] = np.sum(rho[:, xi:], axis=1) + np.sum(
                rho[:, :xj], axis=1)
        else:
            rho_y[n_lane - 1] = np.sum(rho[xi:, :], axis=0) + np.sum(
                rho[:xj, :], axis=0)
        rho_y[n_lane - 1] /= (nx - xi + xj)
    return rho_y


def count_bands(y,
                rho_y_arr,
                lane_edges,
                vy_x,
                Lx,
                rho_threshold=1.5,
                min_spacing=0,
                show_res=False):
    def add_root(roots, new_root, x, min_dx):
        if len(roots) == 0:
            roots.append(new_root)
        else:
            dis = np.abs(x[new_root] - x[roots[-1]])
            if dis >= min_dx:
                roots.append(new_root)

    def check_last_root(roots, x, min_dx, Lx, descending=False):
        xj = roots[-1]
        xi = roots[0]
        dis = Lx - np.abs(xi - xj)
        if dis < min_dx:
            if descending:
                del roots[-1]
            else:
                del roots[0]

    def find_roots(y, y0, x, descending=True):
        roots = []
        if descending:
            for j in range(y.size - 1, -1, -1):
                yj = y[j]
                if j == y.size - 1:
                    yi = y[0]
                else:
                    yi = y[j + 1]
                if yj >= y0 > yi:
                    add_root(roots, j, x, min_spacing)
        else:
            for j in range(y.size):
                yj = y[j]
                yi = y[j - 1]
                if yi < y0 <= yj:
                    add_root(roots, j, x, min_spacing)
        check_last_root(roots, x, min_spacing, Lx, descending)
        return roots

    n_lane = len(lane_edges)
    if n_lane == 0:
        n_bands = np.zeros(1, np.int32)
        sum_vy = np.sum(vy_x)
        if sum_vy > 0:
            descending = True
        else:
            descending = False
        roots = find_roots(rho_y_arr[0], rho_threshold, y, descending)
        n_bands[0] = len(roots)
    else:
        n_bands = np.zeros(n_lane, np.int32)
        for i_lane in range(n_lane):
            xi = lane_edges[i_lane]
            if i_lane == n_lane - 1:
                xj = lane_edges[0]
                sum_vy = np.sum(vy_x[xi:]) + np.sum(vy_x[:xj])
            else:
                xj = lane_edges[i_lane + 1]
                sum_vy = np.sum(vy_x[xi:xj])
            if sum_vy > 0:
                descending = True
            else:
                descending = False
            roots = find_roots(rho_y_arr[i_lane], rho_threshold, y, descending)
            n_bands[i_lane] = len(roots)
            if show_res:
                plt.plot(y, rho_y_arr[i_lane])
                for i_band in roots:
                    plt.axvline(y[i_band], linestyle="dashed", c="tab:red")
                    plt.axhline(rho_threshold,
                                linestyle="dotted",
                                c="tab:green")
                plt.title("%d bands" % len(roots))
                plt.show()
                plt.close()
    return n_bands


def get_band_num_profile(x, lane_edges, n_bands):
    nb_x = np.zeros(x.size, np.int32)
    n_lane = len(lane_edges)
    if n_lane > 0:
        for i_lane in range(n_lane - 1):
            nb_x[lane_edges[i_lane]:lane_edges[i_lane + 1]] = n_bands[i_lane]
        nb_x[lane_edges[n_lane - 1]:] = n_bands[-1]
        nb_x[:lane_edges[0]] = n_bands[-1]

        lane_interface = np.array([x[i] for i in lane_edges], np.float32)
    else:
        nb_x = n_bands[0]
        lane_interface = np.array([0], np.float32)
    return nb_x, lane_interface


def cal_space_time(f0,
                   beg_frame=None,
                   end_frame=None,
                   band_ori="x",
                   max_lanes=None,
                   rho_threshold=None):
    para = get_para(f0)
    fnpz = "space_time/" + \
        "{Lx}_{Ly}_{eta:.3f}_{rho0:.3f}_{v0:.1f}_{seed}.npz".format(
            **para)

    max_frames = np.sum(get_nframe(f0, match_all=True))
    if rho_threshold is None:
        rho_threshold = 1.5 * para["rho0"]
    min_band_spacing = 100
    x = np.linspace(para['dx'] / 2, para['Lx'] - para['dx'] / 2,
                    para['Lx'] // para['dx'])
    y = np.linspace(para['dx'] / 2, para['Ly'] - para['dx'] / 2,
                    para['Ly'] // para['dx'])
    if max_lanes is None:
        if band_ori == "x":
            max_lanes = para["Lx"] // 1200
        else:
            max_lanes = para["Ly"] // 1200

    nb_xt = np.zeros((max_frames, x.size), np.float32)
    lane_xt = np.zeros((max_frames, max_lanes), np.int32)
    rho_xt, vy_xt = np.zeros((2, max_frames, x.size), np.float32)
    t_arr = np.zeros(max_frames, np.int32)

    flag_old_data = False
    if beg_frame is None:
        if os.path.exists(fnpz):
            with np.load(fnpz) as data:
                flag_old_data = True
                beg_frame = data["t"][-1] // para['dt']
                t_old = data['t']
                rho_x_old = data['rho_x']
                vy_x_old = data['vy_x']
                nb_x_old = data['nb_x']
                lane_x_old = data['lane_x']
        else:
            beg_frame = 20

    frames = read_fields(f0, beg=beg_frame, end=end_frame)
    count = 0
    for i, (t, rho, vx, vy) in enumerate(frames):
        if band_ori == "x":
            v_T = np.mean(vy, axis=0)
            rho_T = np.mean(rho, axis=0)
            x_T = x
            x_L = y
            L = para["Ly"]
        else:
            v_T = np.mean(vx, axis=1)
            rho_T = np.mean(rho, axis=1)
            x_T = y
            x_L = x
            L = para["Lx"]
        edge = detect_lane(x_T, v_T, max_lanes, False)
        rho_L = get_bands_profile(rho, edge, band_ori=band_ori)
        n_bands = count_bands(x_L,
                              rho_L,
                              edge,
                              v_T,
                              L,
                              rho_threshold,
                              min_band_spacing,
                              show_res=False)
        nb_xt[i], lane_xt[i][:n_bands.size] = get_band_num_profile(
            x_T, edge, n_bands)
        vy_xt[i] = v_T
        rho_xt[i] = rho_T
        t_arr[i] = t
        count += 1

    t_arr = t_arr[:count]
    nb_xt = nb_xt[:count]
    lane_xt = lane_xt[:count]
    rho_xt = rho_xt[:count]
    vy_xt = vy_xt[:count]

    if flag_old_data:
        t_arr = np.hstack((t_old, t_arr))
        rho_xt = np.vstack((rho_x_old, rho_xt))
        vy_xt = np.vstack((vy_x_old, vy_xt))
        nb_xt = np.vstack((nb_x_old, nb_xt))
        lane_xt = np.vstack((lane_x_old, lane_xt))

    np.savez(fnpz,
             t=t_arr,
             x=x,
             rho_x=rho_xt,
             vy_x=vy_xt,
             nb_x=nb_xt,
             lane_x=lane_xt)


if __name__ == "__main__":
    # f0 = "fields/9600_2400_0.290_1.000_0.5_411_8_10000_0.bin"
    # # f0 = "fields/4800_2400_0.290_1.000_0.5_411_8_10000_0.bin"
    # # f0 = "fields/4800_2560_0.290_1.000_0.5_421_8_10000_0.bin"
    # # f0 = "fields/2400_0.350_1.000_0.5_411_8_10000_0.bin"
    # cal_space_time(f0, beg_frame=None, band_ori="x")

    # f0 = "fields/2400_0.290_1.000_0.5_133_8_10000_0.bin"
    # cal_space_time(f0, beg_frame=None, band_ori="y")

    # f0 = "2400_0.350_1.264_0.5_26411_8_10000_0.bin"
    # cal_space_time(f0,
    #                beg_frame=0,
    #                end_frame=None,
    #                band_ori="x",
    #                max_lanes=6,
    #                rho_threshold=2)

    # f0 = "9600_2400_0.280_1.000_0.5_44800_8_10000_0.bin"
    # # f0 = "9600_3600_0.290_1.000_0.5_42400_8_10000_0.bin"
    # cal_space_time(f0, beg_frame=None, max_lanes=2)

    # f0 = "9600_4800_0.290_1.000_0.5_212411_8_10000_0.bin"
    # cal_space_time(f0, beg_frame=None, max_lanes=2)

    # f0 = "2400_0.350_1.264_0.5_26411_8_10000_0.bin"
    # cal_space_time(f0, beg_frame=0, max_lanes=6)

    f0 = "9600_4800_0.290_1.000_0.5_2144140_8_10000_0.bin"
    cal_space_time(f0, beg_frame=None, max_lanes=4)
