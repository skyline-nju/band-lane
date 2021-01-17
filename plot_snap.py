import numpy as np
import matplotlib.pyplot as plt
import struct
# import glob
import os


def read_snap(fin):
    with open(fin, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        f.seek(0)
        N = filesize // 12
        buf = f.read()
        data = struct.unpack("%df" % (N * 3), buf)
        x, y, theta = np.array(data).reshape(N, 3).T
    return x, y, theta


def zoom_in(x0, y0, theta0, xmin, xmax, ymin, ymax):
    mask = x0 > xmin
    x = x0[mask]
    y = y0[mask]
    theta = theta0[mask]
    mask = x < xmax
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    mask = y > ymin
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    mask = y < ymax
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    return x, y, theta


def random_select(x, y, theta, frac, idx_arr, rng=None):
    n = int(frac * x.size)
    if rng is not None:
        rng.shuffle(idx_arr)
    mask = idx_arr[:n]
    return x[mask], y[mask], theta[mask]


def get_para(fname):
    basename = os.path.basename(fname)
    s = basename.lstrip("s").rstrip(".bin").split("_")
    para = {}

    if len(s) == 6:
        para["Lx"] = int(s[0])
        para["Ly"] = para["Lx"]
        para["eta"] = float(s[1])
        para["rho0"] = float(s[2])
        para["v0"] = float(s[3])
        para["seed"] = int(s[4])
        para["t"] = int(s[5])
    elif len(s) == 7:
        para["Lx"] = int(s[0])
        para["Ly"] = int(s[1])
        para["eta"] = float(s[2])
        para["rho0"] = float(s[3])
        para["v0"] = float(s[4])
        para["seed"] = int(s[5])
        para["t"] = int(s[6])
    return para


def get_title(para):
    title_subfix = f"\\eta={para['eta']:g},\\rho_0={para['rho0']:g}," \
        f"v_0={para['v0']:g},{{\\rm seed}}={para['seed']},t={para['t']}$"
    if para["Lx"] == para["Ly"]:
        title = f"$L={para['Lx']},{title_subfix}"
    else:
        title = f"$L_x={para['Lx']}, L_y={para['Ly']},{title_subfix}"
    return title


def plot_snap(x, y, theta, para, frac=1., idx_arr=None):
    vx_m = np.mean(np.cos(theta))
    vy_m = np.mean(np.sin(theta))
    theta_m = np.arctan2(vy_m, vx_m)
    if frac < 1.:
        if idx_arr is None:
            rng = np.random.default_rng()
            idx_arr = np.arange(x.size)
            rng.shuffle(idx_arr)
        x1, y1, theta1 = random_select(x, y, theta, frac, idx_arr)
    else:
        x1, y1, theta1 = x, y, theta

    xmin, xmax = 0, para["Lx"]
    ymin, ymax = 0, para["Ly"]

    c = theta1 - theta_m
    c[c > np.pi] -= np.pi * 2
    c[c < -np.pi] += np.pi * 2

    plt.figure(figsize=(10, 8.5))
    plt.subplot(111, fc="k")
    plt.scatter(x1, y1, s=0.25, c=c, cmap="hsv")
    plt.axis("scaled")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cb = plt.colorbar()
    cb.set_label(r"$\theta - \theta_m$", fontsize="x-large")
    title = get_title(para)
    plt.suptitle(title, fontsize="xx-large")
    plt.tight_layout()
    plt.show()
    # plt.savefig("D:/data/tmp2/%04d.png" % i, dpi=300)
    plt.close()


if __name__ == "__main__":
    fin = "v1.0/s1216.300.0.1234.0.0015.bin"
    x, y, theta = read_snap(fin)
    plt.figure(figsize=(10, 8))
    xmin, xmax = 600, 800
    ymin, ymax = 800, 1000

    x, y, theta = zoom_in(x, y, theta, xmin, xmax, ymin, ymax)
    # plt.plot(x, y, ".", ms=0.1)
    # plt.scatter(x, y, s=0.5, c=theta, cmap="hsv")
    plt.subplot(111, fc="k")
    plt.scatter(x, y, s=0.5, c=np.sin(theta), vmin=-0.5, vmax=0.5, cmap="bwr")
    plt.axis("scaled")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cb = plt.colorbar(extend="both")
    cb.set_label(r"$\sin \theta$", fontsize="xx-large")
    # cb = plt.colorbar()
    # cb.set_label(r"$\theta$", fontsize="xx-large")
    plt.tight_layout()
    plt.show()
    plt.close()

    # os.chdir("snap")
    # files = glob.glob("s2400_0.230_0.637_0.5_2133_*.bin")
    # print(files)
    # idx_arr = None
    # frac = 0.01
    # # f = files[0]
    # for i, f in enumerate(files[:1800]):
    #     x, y, theta = read_snap(f)
    #     if idx_arr is None:
    #         rng = np.random.default_rng()
    #         idx_arr = np.arange(x.size)
    #         rng.shuffle(idx_arr)
    #     vx_m = np.mean(np.cos(theta))
    #     vy_m = np.mean(np.sin(theta))
    #     theta_m = np.arctan2(vy_m, vx_m)

    #     x, y, theta = random_select(x, y, theta, frac, idx_arr)

    #     para = get_para(f)
    #     xmin, xmax = 0, para["Lx"]
    #     ymin, ymax = 0, para["Ly"]
    #     # x, y, theta = zoom_in(x, y, theta, xmin, xmax, ymin, ymax)

    #     c = theta - theta_m
    #     c[c > np.pi] -= np.pi * 2
    #     c[c < -np.pi] += np.pi * 2

    #     plt.figure(figsize=(10, 8.5))
    #     plt.subplot(111, fc="k")
    #     plt.scatter(x, y, s=0.25, c=c, cmap="hsv")
    #     plt.axis("scaled")
    #     plt.xlim(xmin, xmax)
    #     plt.ylim(ymin, ymax)
    #     cb = plt.colorbar()
    #     cb.set_label(r"$\theta - \theta_m$", fontsize="x-large")
    #     title = r"$L=%d,\eta=%g,\rho_0=%.4f,v_0=%g,{\rm seed}=%d,t=%d$" % (
    #         para["L"], para["eta"], para["rho0"], para["v0"], para["seed"],
    #         para["t"])
    #     plt.suptitle(title, fontsize="xx-large")
    #     plt.tight_layout()
    #     plt.show()
    #     # plt.savefig("D:/data/tmp2/%04d.png" % i, dpi=300)
    #     plt.close()
