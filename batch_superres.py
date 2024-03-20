import os
import sys
import time

from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

path = os.path.abspath(os.path.dirname(__file__))
target = "py -3.7 ./superres-video.py"

filelist = []
arg_scale = []
arg_cu = []
arg_denoise = []
extra_args = ""

if not os.path.normcase(os.getcwd()) == os.path.normcase(path):
    os.chdir(path)
    print("Changed cwd to: " + path)

path = r'"' + path + r'"'


def get_arg():
    scale = FloatPrompt.ask("Resolution scale", default=2.0)
    arg_scale.append(scale)

    cugan = Confirm.ask("Use cugan (or esrgan)", default=False)
    arg_cu.append(cugan)

    if arg_cu[-1]:
        denoise = Confirm.ask("Denoise", default=True)
        arg_denoise.append(denoise)
    else:
        arg_denoise.append(False)
    print()


i = 0
in_file = sys.argv[1:]
for f in in_file:
    i += 1
    if not f.startswith(r'"'):
        ff = r'"' + f + r'"'
    else:
        ff = f
    filelist.append(ff)
    print(f"Get file-{i:02d} path: ", f)

    get_arg()

while True:
    i += 1
    get = Prompt.ask(f"Input file-{i:02d} path").strip()
    if get == "":
        break
    if not get.startswith(r'"'):
        ff = r'"' + f + r'"'
    else:
        ff = get
    filelist.append(ff)
    get_arg()

print()

extra_args = ""
poweroff = Confirm.ask("Poweroff after superres", default=False)
crf = IntPrompt.ask("Quality", default=17)
extra_args += f"--quality {crf} "
if Confirm.ask("Use HEVC", default=False):
    extra_args += "--codec hevc_qsv "


def get_extra_args():
    get = Prompt.ask("> Extra args (? for help)").strip()
    if len(get) == 0:
        return ""
    if get[0] == "?":
        command = f"{target} --help"
        os.system(command)
        return get_extra_args()
    elif get != "":
        return " " + get
    return ""


extra_args += get_extra_args()

print()
i = 0
for file, scale, cu, denoise in zip(filelist, arg_scale, arg_cu, arg_denoise):
    i += 1
    print(
        f"File-{i:02d}: {file}\n> Args: Res-scale={scale} cugan={cu} denoise={denoise}"
    )
    print()
print(f"Global args: poweroff={poweroff} crf={crf} extra_args={extra_args}")
print()
input("Check above. Press Enter to continue...")
print("Starting superres")

error_files = []
t_start = time.time()
i = 0
for file, scale, cu, denoise in zip(filelist, arg_scale, arg_cu, arg_denoise):
    i += 1
    command = f"{target} --scale {scale} "
    if cu:
        command += "--model RealCUGAN "
        if denoise:
            command += "--denoise "
    else:
        command += "--model RealESR "
    command += f"{extra_args} "
    command += file
    print(f"[{i}/{len(filelist)}] command: {command}")
    ret = os.system(command)
    print(f"[{i}/{len(filelist)}] Done file: {file} ret: {ret}")
    if ret != 0:
        error_files.append(file)

print("Finished inference")
print(f"Total cost: {(time.time() - t_start)/60:.2f} mins")
print("Some files returned error:", error_files)
if poweroff:
    print("Poweroff in 60 seconds")
    os.system("shutdown -s -t 60")
input("Press Enter to exit")
