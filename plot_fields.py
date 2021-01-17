import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
from matplotlib.colors import hsv_to_rgb
from decode import read_field_series, \
    get_para_field, get_matched_f0, get_last_frame


def map_v_to_rgb(theta, module, m_max=None):
    """
    Transform orientation and magnitude of velocity into rgb.

    Parameters:
    --------
    theta: array_like
        Orietation of velocity field.
    module: array_like
        Magnitude of velocity field.
    m_max: float, optional
        Max magnitude to show.

    Returns:
    --------
    RGB: array_like
        RGB corresponding to velocity fields.
    """
    H = theta / 360
    V = module
    if m_max is not None:
        V[V > m_max] = m_max
    V /= m_max
    S = np.ones_like(H)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def add_colorbar(ax, mmin, mmax, theta_min=0, theta_max=360, orientation="h"):
    """ Add colorbar for the RGB image plotted by plt.imshow() """
    V, H = np.mgrid[0:1:50j, 0:1:180j]
    if orientation == "v":
        V = V.T
        H = H.T
        box = [mmin, mmax, theta_min, theta_max]
    else:
        box = [theta_min, theta_max, mmin, mmax]
    S = np.ones_like(V)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    ax.imshow(RGB, origin='lower', extent=box, aspect='auto')
    theta_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    if orientation == "h":
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'module $|{\bf m}|$', fontsize="large")
        ax.set_xlabel("orientation", fontsize="large")
    else:
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position("right")
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'orientation $\theta$', fontsize="large")
        # ax.set_xlabel(r"module $|{\bf m}|$", fontsize="large")
        ax.set_title(r"$|{\bf m}|$", fontsize="large")


def get_colobar_extend(vmin, vmax):
    if vmin is None or vmin == 0.:
        if vmax is None:
            ext = "neither"
        else:
            ext = "max"
    else:
        if vmax is None:
            ext = "min"
        else:
            ext = "both"
    return ext


def plot_density_momentum(rho, vx, vy, t, para, figsize=(12, 7.5), fout=None):
    theta = np.arctan2(vy, vx)
    theta[theta < 0] += np.pi * 2
    theta *= 180 / np.pi
    module = np.sqrt(vx**2 + vy**2)

    if para["Lx"] == para["Ly"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    box = [0, para["Lx"], 0, para["Ly"]]
    if para["rho0"] >= 1.5:
        vmin1, vmax1 = None, 6
        vmin2, vmax2 = 0, 6
    elif para["rho0"] >= 1:
        vmin1, vmax1 = None, 4
        vmin2, vmax2 = 0, 4
    elif para["rho0"] > 0.55:
        vmin1, vmax1 = None, 3
        vmin2, vmax2 = 0, 3
    else:
        vmin1, vmax1 = None, 2
        vmin2, vmax2 = 0, 2
    im1 = ax1.imshow(rho, origin="lower", extent=box, vmin=vmin1, vmax=vmax1)
    RGB = map_v_to_rgb(theta, module, m_max=vmax2)
    ax2.imshow(RGB, extent=box, origin="lower")

    title_suffix = r"\eta=%g,\rho_0=%.4f,v_0=%g,{\rm seed}=%d,t=%d" % (
        para["eta"], para["rho0"], para["v0"], para["seed"], t)
    if para["Lx"] == para["Ly"]:
        ax1.set_title(r"(a) density")
        ax2.set_title(r"(b) momentum")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        bbox1 = ax1.get_position().get_points().flatten()
        bbox2 = ax2.get_position().get_points().flatten()
        fig.subplots_adjust(bottom=0.24)
        bbox1[1], bbox1[3] = 0.14, 0.04
        bbox1[2] = bbox1[2] - bbox1[0] - 0.03
        bbox2[1], bbox2[3] = 0.08, 0.14
        bbox2[2] = bbox2[2] - bbox2[0]
        cb_ax1 = fig.add_axes(bbox1)
        cb_ax2 = fig.add_axes(bbox2)
        ext1 = get_colobar_extend(vmin1, vmax1)
        cb1 = fig.colorbar(im1,
                           ax=ax1,
                           cax=cb_ax1,
                           orientation="horizontal",
                           extend=ext1)
        cb1.set_label(r"density $\rho$", fontsize="x-large")
        add_colorbar(cb_ax2, vmin2, vmax2, 0, 360)
        title = r"$L=%d,%s$" % (para["Lx"], title_suffix)
        plt.suptitle(title, y=0.995, fontsize="x-large")
    else:
        title = r"$L_x=%d,L_y=%d,%s$" % (para["Lx"], para["Ly"], title_suffix)
        plt.tight_layout(rect=[0, -0.02, 1, 0.98])
        plt.suptitle(title, y=0.999, fontsize="x-large")
    if fout is None:
        plt.show()
    else:
        plt.savefig(fout)
        print(f"save frame at t={t}")
    plt.close()


def plot_momentum(rho, vx, vy, t, para, figsize=(12, 7.5), fout=None):
    theta = np.arctan2(vy, vx)
    theta[theta < 0] += np.pi * 2
    theta *= 180 / np.pi
    module = np.sqrt(vx**2 + vy**2)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    box = [0, para["Lx"], 0, para["Ly"]]
    if para["rho0"] >= 1.5:
        vmin, vmax = 0, 6
    elif para["rho0"] >= 1:
        vmin, vmax = 0, 4
    elif para["rho0"] > 0.55:
        vmin, vmax = 0, 3
    else:
        vmin, vmax = 0, 2
    RGB = map_v_to_rgb(theta, module, m_max=vmax)
    ax.imshow(RGB, extent=box, origin="lower")

    title_suffix = r"\eta=%g,\rho_0=%g,v_0=%g,{\rm seed}=%d,t=%d" % (
        para["eta"], para["rho0"], para["v0"], para["seed"], t)
    if para["Lx"] == para["Ly"]:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        title = r"$L=%d,%s$" % (para["Lx"], title_suffix)
        plt.suptitle(title, y=0.995, fontsize="x-large")
        fig.subplots_adjust(right=0.85)
        bbox = ax.get_position().get_points().flatten()
        # bbox = [xmin, ymin, xmax, ymax]
        bbox = ax.get_position().get_points().flatten()
        bbox[0] = 0.83
        bbox[2] = 0.08
        bbox[1] += 0.05
        bbox[3] = bbox[3] - bbox[1] - 0.1
    else:
        plt.tight_layout(rect=[-0.01, -0.05, 1, 0.98])
        title = r"$L_x=%d,L_y=%d,%s$" % (para["Lx"], para["Ly"], title_suffix)
        plt.suptitle(title, y=0.999, fontsize="x-large")
        fig.subplots_adjust(right=0.875)
        # bbox = [xmin, ymin, xmax, ymax]
        bbox = ax.get_position().get_points().flatten()
        bbox[0] = 0.885
        bbox[2] = 0.06
        bbox[3] = bbox[3] - bbox[1]
    cb_ax = fig.add_axes(bbox)
    add_colorbar(cb_ax, vmin, vmax, 0, 360, "v")

    if fout is None:
        plt.show()
    else:
        plt.savefig(fout, dpi=150)
        print(f"same frame at t={t}")
    plt.close()


def plot_frames(f0,
                save_fig=True,
                fmt="jpg",
                data_dir="fields",
                which="momentum"):
    para = get_para_field(f0)
    prefix = "D:/data/lane/%s" % data_dir
    if para["Lx"] == para["Ly"]:
        folder = "%s/%.1f_%d_%.3f_%.3f_%d" % (prefix, para["v0"], para["Lx"],
                                              para["eta"], para["rho0"],
                                              para["seed"])
        if which == "both":
            figsize = (12, 7.5)
        else:
            figsize = (7, 6)
    else:
        folder = "%s/%.1f_%d_%d_%.3f_%.3f_%d" % (
            prefix, para["v0"], para["Lx"], para["Ly"], para["eta"],
            para["rho0"], para["seed"])
        if which == "both":
            if para["Lx"] // para["Ly"] == 4:
                figsize = (12, 6)
            elif para["Lx"] // para["Ly"] == 2:
                figsize = (8, 8)
            elif para["Lx"] // para["Ly"] == 8:
                figsize = (12, 4)
        else:
            if para["Lx"] // para["Ly"] == 4:
                figsize = (12, 3)
            elif para["Lx"] // para["Ly"] == 8:
                figsize = (12, 2)
            elif para["Lx"] // para["Ly"] == 2:
                figsize = (8, 4)
            else:
                figsize = (8, 4.2)
    if not os.path.exists(folder):
        os.mkdir(folder)
    existed_snap = glob.glob("%s/t=*.%s" % (folder, fmt))
    beg = len(existed_snap)
    print(beg, "snapshots have existed")
    t_beg = (beg + 1) * para["dt"]
    frames = read_field_series(f0, beg=beg)
    for i, (rho, vx, vy) in enumerate(frames):
        if save_fig:
            fout = "%s/t=%04d.%s" % (folder, beg + i, fmt)
        else:
            fout = None
        t = t_beg + i * para["dt"]
        if which == "both":
            plot_density_momentum(rho, vx, vy, t, para, figsize, fout)
        elif which == "momentum":
            plot_momentum(rho, vx, vy, t, para, figsize, fout)


def plot_all_fields_series(pat="*.bin", fmt="jpg", data_dir="fields"):
    # os.chdir("fields")
    f0_list = get_matched_f0(pat)
    for f0 in f0_list:
        print(f0)
        plot_frames(f0, True, fmt=fmt, data_dir=data_dir)


def set_figsize(n):
    if n == 0:
        print("find no matched file")
        sys.exit()
    elif n == 1:
        nrows = 1
        ncols = 1
        figsize = (4.5, 3)
    elif n == 2:
        nrows = 1
        ncols = 2
        figsize = (6, 3)
    elif n == 3:
        nrows = 1
        ncols = 3
        figsize = (7, 3)
    elif n == 4:
        nrows = 1
        ncols = 4
        figsize = (9, 3)
    elif n == 5:
        nrows = 1
        ncols = 5
        figsize = (10, 3)
    elif n == 6:
        nrows = 1
        ncols = 6
        figsize = (12, 3)
    elif n <= 12:
        nrows = 2
        ncols = 6
        figsize = (12, 5)
    elif n <= 18:
        nrows = 3
        ncols = 6
        figsize = (12, 7)
    elif n <= 24:
        nrows = 4
        ncols = 6
        figsize = (12, 9)
    elif n <= 30:
        nrows = 5
        ncols = 6
        figsize = (12, 12)
    elif n <= 36:
        nrows = 6
        ncols = 6
        figsize = (12, 15)
    else:
        print("find %s file" % n)
        sys.exit()
    return nrows, ncols, figsize


def plot_last_frames(L, eta, rho0, v0, dx=8, save_fig=False):
    def plot_one_panel(f0, ax, show_xticks=True, show_yticks=True):
        t, rho, vx, vy = get_last_frame(f0)
        im = ax.imshow(rho, origin="lower", extent=box, vmin=vmin, vmax=vmax)
        b = int(np.log10(t))
        a = t / 10**b
        para = get_para_field(f0)
        ax.set_title(f"$S{para['seed']}, t={a:g}e{b}$")
        if not show_xticks:
            ax.set_xticks([])
        if not show_yticks:
            ax.set_yticks([])
        return im

    def get_seed(elem):
        return get_para_field(elem)["seed"]

    pat = "%d_%.3f_%.3f_%.1f_1*_%d_*.bin" % (L, eta, rho0, v0, dx)
    f0_list = get_matched_f0(pat)
    f0_list.sort(key=get_seed)
    n = len(f0_list)
    print(f"L={L}, eta={eta:.3f}, rho0={rho0:.3f}, v0={v0}, IC numbers={n}")
    nrows, ncols, figsize = set_figsize(n)
    box = [0, L, 0, L]
    if rho0 < 1.2:
        vmin, vmax = None, rho0 * 4
    else:
        vmin, vmax = None, rho0 * 3
    if True:
        fig, axes = plt.subplots(nrows,
                                 ncols,
                                 figsize=figsize,
                                 constrained_layout=True)
        if n == 1:
            plot_one_panel(f0_list[0], axes)
        else:
            for i, ax in enumerate(axes.flat):
                if i < n:
                    if i % ncols == 0:
                        show_yticks = True
                    else:
                        show_yticks = False
                    if i // ncols == nrows - 1:
                        show_xticks = True
                    else:
                        show_xticks = False
                    plot_one_panel(f0_list[i], ax, show_xticks, show_yticks)
                else:
                    ax.axis("off")

        title = r"$L=%d, \eta=%g, \rho_0=%g, v_0=%g$" % (L, eta, rho0, v0)
        # fig.colorbar(im, ax=axes, shrink=1)
        plt.suptitle(title, fontsize="xx-large")
        if save_fig:
            root = "D:/data/cross_sea2/last_frame"
            folder1 = "%s/v0=%.1f" % (root, v0)
            if not os.path.exists(folder1):
                os.mkdir(folder1)
            folder2 = "%s/rho0=%.3f" % (folder1, rho0)
            if not os.path.exists(folder2):
                os.mkdir(folder2)
            fout = "%s/%d_%.3f.jpg" % (folder2, L, eta)
            plt.savefig(fout)
        else:
            plt.show()
        plt.close()


def plot_all_last_frames(pat="*.bin", save_fig=True):
    files = glob.glob(pat)
    para_list = []
    for f in files:
        para = get_para_field(f)
        if para["Lx"] == para["Ly"] and int(str(para["seed"])[0]) == 1:
            my_para = [para["Lx"], para["eta"], para["rho0"], para["v0"]]
            if my_para not in para_list:
                para_list.append(my_para)

    for p in para_list:
        plot_last_frames(p[0], p[1], p[2], p[3], save_fig=save_fig)


if __name__ == "__main__":
    data_dir = "fields"
    # data_dir = "fields200"
    os.chdir(data_dir)
    plot_all_fields_series("*.bin", data_dir=data_dir, fmt="jpg")

    # plot_all_last_frames("4800_*_1.0_*_8_*.bin", True)
