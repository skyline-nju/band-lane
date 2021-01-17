import numpy as np
# import matplotlib.pyplot as plt
import os
from decode import read_field_series, get_tot_frames


def cal_density_momentum():
    Lx = 9600
    Ly = 2400
    eta = 0.29
    rho0 = 1.
    seed = 411
    v0 = 0.5
    dx = 8
    dt = 10000
    subfix = f"{eta:.3f}_{rho0:.3f}_{v0:.1f}_{seed}_{dx}_{dt}_0.bin"
    if Lx == Ly:
        f0 = f"{Lx}_{subfix}"
    else:
        f0 = f"{Lx}_{Ly}_{subfix}"

    max_frame = None
    n_frame = get_tot_frames(f0)
    frames = read_field_series(f0)
    rho_x, vy_x = np.zeros((2, n_frame, Lx//dx), np.float32)
    t_arr = np.zeros(n_frame, int)
    if seed == 133:
        axis = 1
    else:
        axis = 0
    for i, (rho, vx, vy) in enumerate(frames):
        rho_x[i] = np.mean(rho, axis=axis)
        if axis == 0:
            vy_x[i] = np.mean(vy, axis=axis)
        else:
            vy_x[i] = np.mean(vx, axis=axis)
        t_arr[i] = (i+1) * dt
        if max_frame is not None and (i+1) >= max_frame:
            break
        if i % 100 == 0:
            print("%d/%d" % (i, n_frame))
    rho_x, vy_x, t_arr = rho_x[:i+1], vy_x[:i+1], t_arr[:i+1]

    dest_folder = "D:/data/lane/space_time"
    basename = f"{Lx}_{Ly}_{eta:.3f}_{rho0:.3f}_{v0:.1f}_{seed}.npz"
    fout = f"{dest_folder}/{basename}"
    np.savez(fout, t=t_arr, rho_x=rho_x, vy_x=vy_x)


if __name__ == "__main__":
    os.chdir("fields")
    cal_density_momentum()

    # Lx = 9600
    # Ly = 2400
    # eta = 0.29
    # rho0 = 1.
    # seed = 411
    # v0 = 0.5
    # dx = 8
    # dt = 10000
    # subfix = f"{eta:.3f}_{rho0:.3f}_{v0:.1f}_{seed}_{dx}_{dt}_0.bin"
    # if Lx == Ly:
    #     f0 = f"{Lx}_{subfix}"
    # else:
    #     f0 = f"{Lx}_{Ly}_{subfix}"

    # max_frame = 1
    # n_frame = get_tot_frames(f0)
    # frames = read_field_series(f0, beg=99)
    # rho_x, vy_x = np.zeros((2, n_frame, Lx//dx), np.float32)
    # t_arr = np.zeros(n_frame, int)
    # if seed == 133:
    #     axis = 1
    # else:
    #     axis = 0
    # x = np.linspace(dx/2, Lx-dx/2, Lx//dx)
    # for i, (rho, vx, vy) in enumerate(frames):
    #     rho_x[i] = np.mean(rho, axis=axis)
    #     if axis == 0:
    #         vy_x[i] = np.mean(vy, axis=axis)
    #     else:
    #         vy_x[i] = np.mean(vx, axis=axis)
    #     plt.plot(x, vy_x[i])
    #     plt.show()
    #     plt.close()
    #     t_arr[i] = (i+1) * dt
    #     if max_frame is not None and (i+1) >= max_frame:
    #         break
    #     if i % 100 == 0:
    #         print("%d/%d" % (i, n_frame))
    # rho_x, vy_x = rho_x[:i+1], vy_x[:i+1]

    # dest_folder = "D:/data/lane/space_time"
    # basename = f"{Lx}_{Ly}_{eta:.3f}_{rho0:.3f}_{v0:.1f}_{seed}.npz"
    # fout = f"{dest_folder}/{basename}"
    # np.savez(fout, t=t_arr, rho_x=rho_x, vy_x=vy_x)
