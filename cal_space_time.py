import numpy as np
import matplotlib.pyplot as plt
import os
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
    if band_ori == "x":
        rho_y = np.zeros((n_lane, rho.shape[0]))
    else:
        rho_y = np.zeros((n_lane, rho.shape[1]))
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
        rho_y[n_lane -
              1] = np.sum(rho[:, xi:], axis=1) + np.sum(rho[:, :xj], axis=1)
    else:
        rho_y[n_lane -
              1] = np.sum(rho[xi:, :], axis=0) + np.sum(rho[:xj, :], axis=0)
    rho_y[n_lane - 1] /= (nx - xi + xj)
    return rho_y


def count_bands(y,
                rho_y_arr,
                lane_edges,
                vy_x,
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
        return roots

    n_lane = len(lane_edges)
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
                plt.axhline(rho_threshold, linestyle="dotted", c="tab:green")
            plt.title("%d bands" % len(roots))
            plt.show()
            plt.close()
    return n_bands


def get_band_num_profile(x, lane_edges, n_bands):
    nb_x = np.zeros(x.size, np.int32)
    n_lane = len(lane_edges)
    for i_lane in range(n_lane - 1):
        nb_x[lane_edges[i_lane]:lane_edges[i_lane + 1]] = n_bands[i_lane]
    nb_x[lane_edges[n_lane - 1]:] = n_bands[-1]
    nb_x[:lane_edges[0]] = n_bands[-1]

    lane_interface = np.array([x[i] for i in lane_edges], np.float32)
    return nb_x, lane_interface


def cal_space_time(f0,
                   beg_frame=None,
                   end_frame=None,
                   band_ori="x",
                   max_lanes=None):
    para = get_para(f0)
    fnpz = "space_time/" + \
        "{Lx}_{Ly}_{eta:.3f}_{rho0:.3f}_{v0:.1f}_{seed}.npz".format(
            **para)

    max_frames = np.sum(get_nframe(f0, match_all=True))
    rho_threshold = 1.5
    min_band_spacing = 100
    x = np.linspace(para['dx'] / 2, para['Lx'] - para['dx'] / 2,
                    para['Lx'] // para['dx'])
    y = np.linspace(para['dx'] / 2, para['Ly'] - para['dx'] / 2,
                    para['Ly'] // para['dx'])
    max_lanes = para["Lx"] // 1200
    nb_xt, lane_xt, rho_xt, vy_xt = np.zeros((4, max_frames, x.size),
                                             np.float32)
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
    if band_ori == "x":
        for i, (t, rho, vx, vy) in enumerate(frames):
            t_arr[i] = t
            vy_x = np.mean(vy, axis=0)
            edge = detect_lane(x, vy_x, max_lanes, False)
            rho_y = get_bands_profile(rho, edge)
            n_bands = count_bands(y,
                                  rho_y,
                                  edge,
                                  vy_x,
                                  rho_threshold,
                                  min_band_spacing,
                                  show_res=False)
            nb_xt[i], lane_xt[i] = get_band_num_profile(x, edge, n_bands)
            vy_xt[i] = vy_x
            rho_xt[i] = np.mean(rho, axis=0)
    else:
        for i, (t, rho, vx, vy) in enumerate(frames):
            t_arr[i] = t
            vy_x = np.mean(vx, axis=1)
            edge = detect_lane(y, vy_x, max_lanes, False)
            # if len(edge) != max_lanes:
            #     print("Warning, skipping frame", i)
            #     nb_xt[i] = nb_xt[i-1]
            #     lane_xt[i] = lane_xt[i-1]
            #     vy_xt[i] = vy_xt[i-1]
            #     rho_xt[i] = rho_xt[i-1]
            #     continue
            rho_y = get_bands_profile(rho, edge, band_ori)
            n_bands = count_bands(x,
                                  rho_y,
                                  edge,
                                  vy_x,
                                  rho_threshold,
                                  min_band_spacing,
                                  show_res=False)
            nb_xt[i], lane_xt[i] = get_band_num_profile(y, edge, n_bands)
            vy_xt[i] = vy_x
            rho_xt[i] = np.mean(rho, axis=1)

    t_arr = t_arr[:i + 1]
    nb_xt = nb_xt[:i + 1]
    lane_xt = lane_xt[:i + 1]
    rho_xt = rho_xt[:i + 1]
    vy_xt = vy_xt[:i + 1]

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
    f0 = "fields/9600_2400_0.290_1.000_0.5_411_8_10000_0.bin"
    # f0 = "fields/4800_2400_0.290_1.000_0.5_411_8_10000_0.bin"
    # f0 = "fields/4800_2560_0.290_1.000_0.5_421_8_10000_0.bin"
    # f0 = "fields/2400_0.350_1.000_0.5_411_8_10000_0.bin"
    cal_space_time(f0, beg_frame=None, band_ori="x")

    # f0 = "fields/2400_0.290_1.000_0.5_133_8_10000_0.bin"
    # cal_space_time(f0, beg_frame=20, band_ori="y")
