import numpy as np
import matplotlib.pyplot as plt


def plot_space_time(Lx,
                    eta=0.29,
                    rho0=1,
                    v0=0.5,
                    seed=411,
                    Ly=None,
                    save=False):
    if Ly is None:
        Ly = Lx
    fin = f"space_time/{Lx}_{Ly}_{eta:.3f}_{rho0:.3f}_{v0:.1f}_{seed}.npz"
    with np.load(fin) as data:
        t = data['t']
        # x = data['x']
        rho_x = data['rho_x']
        vy_x = data['vy_x']
        nb_x = data['nb_x']
        lane_x = data['lane_x']
        print(t.min(), t.max())

    if Lx == Ly == 2400:
        figsize = (6, 6)
    else:
        if t.max() > 1e8:
            figsize = (12, 12)
        else:
            figsize = (12, 8)
    fig, (ax1, ax2, ax3) = plt.subplots(1,
                                        3,
                                        figsize=figsize,
                                        constrained_layout=True,
                                        sharey=True)

    extent = [0, Lx, t[0], t[-1]]
    im1 = ax1.imshow(rho_x,
                     origin="lower",
                     aspect="auto",
                     extent=extent,
                     vmin=None,
                     vmax=None)
    im2 = ax2.imshow(vy_x / rho_x,
                     origin="lower",
                     aspect="auto",
                     extent=extent,
                     cmap="Spectral")
    nb_min, nb_max = nb_x.min(), nb_x.max()
    im3 = ax3.imshow(
        nb_x,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=nb_min - 0.05,
        #  vmin=21,
        vmax=nb_max + 0.05)
    for i_lane in range(lane_x.shape[1]):
        ax3.plot(lane_x[:, i_lane], t, "w.", ms=0.15)
    cb1 = plt.colorbar(im1, ax=ax1, orientation="horizontal", extend="both")
    cb2 = plt.colorbar(im2, ax=ax2, orientation="horizontal")
    cb3 = plt.colorbar(im3, ax=ax3, orientation="horizontal")

    cb1.set_label(r"$\langle \rho\rangle_y$", fontsize="x-large")
    cb2.set_label(r"$\langle v_y\rangle_y$", fontsize="x-large")
    cb3.set_label(r"$n_{\rm b}$", fontsize="x-large")
    ax1.set_title("(a) density", fontsize="x-large")
    ax2.set_title("(b) velocity", fontsize="x-large")
    ax3.set_title("(c) band number", fontsize="x-large")
    ax1.set_ylabel(r"$t$", fontsize="x-large")
    ax1.set_xlabel(r"$x$", fontsize="x-large")
    ax2.set_xlabel(r"$x$", fontsize="x-large")
    ax3.set_xlabel(r"$x$", fontsize="x-large")

    if Lx != Ly:
        title = f"$L_x={Lx},L_y={Ly},\\eta={eta},\\rho_0={rho0},v_0={v0}$"
    else:
        title = f"$L={Lx},\\eta={eta},\\rho_0={rho0},v_0={v0}$"
    plt.suptitle(title, fontsize="xx-large")
    if save:
        fout = fin.replace(".npz", ".png")
        plt.savefig(fout, dpi=300)
    else:
        plt.show()
    plt.close()

    # plt.plot(rho_x[-1])
    # plt.plot(rho_x[-10])
    # plt.plot(rho_x[-20])
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    # os.chdir("D:/data/lane/space_time")
    # plot_density_vel(9600, 0.29, 411, Ly=2400)

    # Lx = 2400
    # Ly = 2400
    # eta = 0.29
    # rho0 = 1
    # v0 = 0.5
    # seed = 133
    # plot_space_time(Lx, eta, rho0, v0, seed, Ly, True)

    # Lx = 2400
    # Ly = 2400
    # eta = 0.35
    # rho0 = 1.264
    # v0 = 0.5
    # seed = 26411
    # plot_space_time(Lx, eta, rho0, v0, seed, Ly, False)

    # Lx = 9600
    # Ly = 2400
    # eta = 0.29
    # rho0 = 1
    # v0 = 0.5
    # seed = 411
    # plot_space_time(Lx, eta, rho0, v0, seed, Ly, False)

    # Lx = 2400
    # Ly = 2400
    # eta = 0.35
    # rho0 = 1.264
    # v0 = 0.5
    # seed = 26411
    # plot_space_time(Lx, eta, rho0, v0, seed, Ly, False)

    # Lx = 9600
    # Ly = 4800
    # eta = 0.29
    # rho0 = 1
    # v0 = 0.5
    # seed = 212411
    # plot_space_time(Lx, eta, rho0, v0, seed, Ly, False)

    # Lx = 4800
    # Ly = 4800
    # eta = 0.29
    # rho0 = 1.
    # v0 = 0.5
    # seed = 44140
    # plot_space_time(Lx, eta, rho0, v0, seed, Ly, False)

    Lx = 9600
    Ly = 4800
    eta = 0.29
    rho0 = 1.
    v0 = 0.5
    seed = 2144140
    plot_space_time(Lx, eta, rho0, v0, seed, Ly, False)
