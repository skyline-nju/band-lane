import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
# from matplotlib.colors import hsv_to_rgb
from read_fields import get_para, read_last_frame


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
    elif n <= 48:
        nrows = 6
        ncols = 8
        figsize = (16, 15)
    else:
        print("find %s file" % n)
        sys.exit()
    return nrows, ncols, figsize


def get_matched_f0(L, eta, rho0, v0, dx):
    pat = "%d_%.3f_%.3f_%.1f_1*_%d_*.bin" % (L, eta, rho0, v0, dx)
    files = glob.glob("fields/%s" % pat)
    seeds = []
    for f in files:
        seed = get_para[f]["seed"]
        if seed not in seeds:
            seeds.append(seed)
    f0_list = []
    for seed in sorted(seeds):
        pat = "%d_%.3f_%.3f_%.1f_%d_%d_*.bin" % (L, eta, rho0, v0, seed, dx)
        files = glob.glob("fields/%s" % pat)
        f0_list.append(files[0])
    return f0_list


def plot_last_frames(L, eta, rho0, v0, dx=8, save_fig=False):
    def plot_one_panel(f0, ax, show_xticks=True, show_yticks=True):
        t, rho, vx, vy = read_last_frame(f0)
        im = ax.imshow(rho, origin="lower", extent=box, vmin=vmin, vmax=vmax)
        b = int(np.log10(t))
        a = t / 10**b
        para = get_para(f0)
        ax.set_title(f"$S{para['seed']}, t={a:g}e{b}$")
        if not show_xticks:
            ax.set_xticks([])
        if not show_yticks:
            ax.set_yticks([])
        return im

    f0_list = get_matched_f0(L, eta, rho0, v0, dx)
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
        para = get_para(f)
        if para["Lx"] == para["Ly"] and int(str(para["seed"])[0]) == 1:
            my_para = [para["Lx"], para["eta"], para["rho0"], para["v0"]]
            if my_para not in para_list:
                para_list.append(my_para)

    for p in para_list:
        plot_last_frames(p[0], p[1], p[2], p[3], save_fig=save_fig)
