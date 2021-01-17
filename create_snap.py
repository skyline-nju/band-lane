import numpy as np
from plot_snap import read_snap, plot_snap, get_para


def clone(fin, nx=1, ny=1):
    para0 = get_para(fin)
    x0, y0, theta0 = read_snap(fin)
    para_new = {key: para0[key] for key in para0}
    para_new["Lx"] *= nx
    para_new["Ly"] *= ny
    para_new["seed"] = int("2%d" % para0["seed"])
    para_new["t"] = 0
    N0 = x0.size
    N_new = int(para0["rho0"] * para_new["Lx"] * para_new["Ly"])
    print("N_old=%d, N_new=%d, N_new%%N_old=%d" % (N0, N_new, N_new % N0))

    if N_new < N0 * nx * ny:
        size_new = N0 * nx * ny
    else:
        size_new = N_new
    x, y, theta = np.zeros((3, size_new))
    for row in range(ny):
        dy = row * para0["Ly"]
        for col in range(nx):
            dx = col * para0["Lx"]
            k = col + row * nx
            beg = k * N0
            end = beg + N0
            x[beg:end] = x0 + dx
            y[beg:end] = y0 + dy
            theta[beg:end] = theta0
    if N_new < size_new:
        x = x[:N_new]
        y = y[:N_new]
        theta = theta[:N_new]
    elif N_new > N0 * nx * ny:
        for j in range(N0 * nx * ny, N_new):
            x[j] = np.random.rand() * para_new["Lx"]
            y[j] = np.random.rand() * para_new["Ly"]
            theta[j] = np.random.rand() * np.pi * 2.
    plot_snap(x0, y0, theta0, para0, 0.01)
    plot_snap(x, y, theta, para_new, 0.01/(nx * ny))

    if para_new["Lx"] == para_new["Ly"]:
        fout = "snap/cloned/s%d_%.3f_%.3f_%.1f_%d_%08d.bin" % (
            para_new["Lx"], para_new["eta"], para_new["rho0"], para_new["v0"],
            para_new["seed"], para_new["t"])
    else:
        fout = "snap/cloned/s%d_%d_%.3f_%.3f_%.1f_%d_%08d.bin" % (
            para_new["Lx"], para_new["Ly"], para_new["eta"], para_new["rho0"],
            para_new["v0"], para_new["seed"], para_new["t"])

    print("output new snapshot to", fout)
    data = np.zeros((3, N_new), np.float32)
    data[0] = x
    data[1] = y
    data[2] = theta
    data = data.T.flatten()
    data.tofile(fout)


if __name__ == "__main__":
    fin = "snap/s1200_0.280_1.000_0.5_132_01440000.bin"
    # x, y, theta = read_snap(fin)
    # para = get_para(fin)
    # plot_snap(x, y, theta, para, frac=0.1)

    # plt.show()
    clone(fin, 8, 2)
