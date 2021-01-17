import os
import shutil
import glob


def get_basename():
    basename_list = []
    files = glob.glob("snap/*.bin")
    for f in files:
        basename = os.path.basename(f)[:-13]
        if basename not in basename_list:
            basename_list.append(basename)
    return basename_list


def move_last(basename):
    files = glob.glob("snap/%s_*.bin" % basename)
    t_list = []
    for f in files:
        t = int(os.path.basename(f).rstrip(".bin").split("_")[-1])
        t_list.append(t)
    t_last = sorted(t_list)[-1]
    f_last = "snap/%s_%08d.bin" % (basename, t_last)
    f_dest = "last/%s_%08d.bin" % (basename, t_last)
    shutil.move(f_last, f_dest)
    print("%s-->%s" % (f_last, f_dest))


if __name__ == "__main__":
    if not os.path.exists("last"):
        os.mkdir("last")
    basename_list = get_basename()
    print(basename_list)
    for basename in basename_list:
        move_last(basename)
