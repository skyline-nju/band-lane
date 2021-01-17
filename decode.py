import numpy as np
import struct
import os
import glob
import sys


def get_para_field(fin):
    s = os.path.basename(fin).rstrip(".bin").split("_")
    para = {}
    if len(s) == 8:
        para["Lx"] = int(s[0])
        para["Ly"] = para["Lx"]
        para["eta"] = float(s[1])
        para["rho0"] = float(s[2])
        para["v0"] = float(s[3])
        para["seed"] = int(s[4])
        para["dx"] = int(s[5])
        para["dt"] = int(s[6])
        para["t_beg"] = int(s[7])
    elif len(s) == 9:
        para["Lx"] = int(s[0])
        para["Ly"] = int(s[1])
        para["eta"] = float(s[2])
        para["rho0"] = float(s[3])
        para["v0"] = float(s[4])
        para["seed"] = int(s[5])
        para["dx"] = int(s[6])
        para["dt"] = int(s[7])
        para["t_beg"] = int(s[8])
    return para


def get_nframe(fname):
    with open(fname, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        para = get_para_field(fname)
        nx, ny = para["Lx"] // para["dx"], para["Ly"] // para["dx"]
        n = nx * ny
        frame_size = n * 12
        n_frame = filesize // frame_size
    return n_frame


def read_field(fname, beg=0, end=None, sep=1):
    """ Read coarse-grained density and momentum fields """
    with open(fname, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        para = get_para_field(fname)
        nx, ny = para["Lx"] // para["dx"], para["Ly"] // para["dx"]
        n = nx * ny
        framesize = n * 12
        f.seek(beg * framesize)
        if end is None:
            file_end = filesize
        else:
            file_end = end * framesize
        idx_frame = 0
        while f.tell() < file_end:
            if idx_frame % sep == 0:
                buf = f.read(framesize)
                data = np.array(struct.unpack("%df" % (n * 3), buf))
                rho_m, vx_m, vy_m = data.reshape(
                    3, ny, nx) / (para["dx"] * para["dx"])
                yield rho_m, vx_m, vy_m
            else:
                f.seek(framesize, 1)
            idx_frame += 1


def read_one_frame(fname, idx):
    with open(fname, "rb") as f:
        para = get_para_field(fname)
        nx, ny = para["Lx"] // para["dx"], para["Ly"] // para["dx"]
        n = nx * ny
        framesize = n * 12
        f.seek(0, 2)
        filesize = f.tell()
        nframe = filesize // framesize

        if idx < 0:
            idx += nframe
        f.seek(idx * framesize)

        buf = f.read(framesize)
        data = np.array(struct.unpack("%df" % (n * 3), buf))
        rho_m, vx_m, vy_m = data.reshape(3, ny, nx) / (para["dx"] * para["dx"])
        return rho_m, vx_m, vy_m


def get_prefix(para):
    if "dt" in para:
        if para["Lx"] == para["Ly"]:
            prefix = "%d_%.3f_%.3f_%.1f_%d_%d_%d_" % (
                para["Lx"], para["eta"], para["rho0"], para["v0"],
                para["seed"], para["dx"], para["dt"])
        else:
            prefix = "%d_%d_%.3f_%.3f_%.1f_%d_%d_%d_" % (
                para["Lx"], para["Ly"], para["eta"], para["rho0"], para["v0"],
                para["seed"], para["dx"], para["dt"])
    else:
        if para["Lx"] == para["Ly"]:
            prefix = "%d_%.3f_%.3f_%.1f_%d_%d_" % (para["Lx"], para["eta"],
                                                   para["rho0"], para["v0"],
                                                   para["seed"], para["dx"])
        else:
            prefix = "%d_%d_%.3f_%.3f_%.1f_%d_%d_" % (
                para["Lx"], para["Ly"], para["eta"], para["rho0"], para["v0"],
                para["seed"], para["dx"])
    return prefix


def get_fname(para):
    if "t_beg" not in para:
        print("Error, para has no key t_beg")
        sys.exit()
    prefix = get_prefix
    fname = "%s%d.bin" % (prefix, para["t_beg"])
    return fname


def get_tot_frames(fname0):
    para0 = get_para_field(fname0)
    prefix = get_prefix(para0)
    files = glob.glob("%s*.bin" % prefix)
    tot_frames = 0
    for f in files:
        tot_frames += get_nframe(f)
    return tot_frames


def get_matched_f0(pat):
    files = glob.glob(pat)
    f0_list = []
    for f in files:
        para = get_para_field(f)
        del para["dt"]
        prefix = get_prefix(para)
        f0 = f"{prefix}10000_0.bin"
        if f0 not in f0_list:
            f0_list.append(f0)
    return f0_list


def read_field_series(fname0, beg=0):
    para0 = get_para_field(fname0)
    dt = para0["dt"]
    # del para0["dt"]
    prefix = get_prefix(para0)
    # print(f"prefix={prefix}")
    files = glob.glob("%s*.bin" % prefix)
    t_beg_list = sorted([get_para_field(f)["t_beg"] for f in files])
    files = ["%s%d.bin" % (prefix, t_beg) for t_beg in t_beg_list]
    print("beg=", beg)
    # first_frame = 0
    # last_frame = 0
    # for i, f in enumerate(files):
    #     my_frames = get_nframe(f)
    #     last_frame += my_frames
    #     if beg < last_frame:
    #         if beg < first_frame:
    #             my_beg = 0
    #         else:
    #             my_beg = beg - first_frame
    #         yield from read_field(f, my_beg)
    #     first_frame += my_frames

    gl_beg_frame = 0
    gl_end_frame = 0
    for i, f in enumerate(files):
        nframe_fi = get_nframe(f)
        nframe = nframe_fi
        if i < len(files) - 1:
            next_beg_frame = t_beg_list[i+1] // dt
            if gl_beg_frame + nframe_fi > next_beg_frame:
                nframe = next_beg_frame - gl_beg_frame
        gl_end_frame += nframe
        if beg < gl_end_frame:
            if beg < gl_beg_frame:
                my_beg = 0
            else:
                my_beg = beg - gl_beg_frame
            if i < len(files) - 1:
                my_end = my_beg + nframe
            else:
                my_end = None
            yield from read_field(f, my_beg, my_end)
        else:
            print("skip", f)
        gl_beg_frame += nframe


def get_last_frame(fname0):
    def get_t_beg(elem):
        return get_para_field(elem)["t_beg"]

    para0 = get_para_field(fname0)
    del para0["dt"]
    prefix = get_prefix(para0)
    files = glob.glob("%s*.bin" % prefix)
    files.sort(key=get_t_beg)
    f_last = files[-1]
    para = get_para_field(f_last)

    with open(f_last, "rb") as f:
        nx, ny = para["Lx"] // para["dx"], para["Ly"] // para["dx"]
        n = nx * ny
        framesize = n * 12
        f.seek(0, 2)
        filesize = f.tell()
        nframe = filesize // framesize
        f.seek((nframe - 1) * framesize)
        buf = f.read(framesize)
        data = np.array(struct.unpack("%df" % (n * 3), buf))
        rho, vx, vy = data.reshape(3, ny, nx) / (para["dx"] * para["dx"])
        t = para["t_beg"] + nframe * para["dt"]

    return t, rho, vx, vy


if __name__ == "__main__":
    os.chdir("fields")
    f0 = "9600_2400_0.280_1.000_0.5_322_8_10000_0.bin"
    n_frames = get_tot_frames(f0)
    print("tot frames =", n_frames)
