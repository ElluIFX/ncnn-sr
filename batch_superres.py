import os
import sys
import time

from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

path = os.path.abspath(os.path.dirname(__file__))
target = "py -3.7 ./superres-video.py"

filelist = []
arg_scale = []
arg_model = []
extra_args = ""

print = Console().print

if not os.path.normcase(os.getcwd()) == os.path.normcase(path):
    os.chdir(path)
    print(f"[blue]Changed cwd to: {path}")

path = r'"' + path + r'"'


def get_arg():
    scale = FloatPrompt.ask("[green]Resolution scale", default=2.0)
    arg_scale.append(scale)
    while True:
        model = Prompt.ask(
            "[green]Model [yellow](? for help)",
            choices=["janai3", "janai2", "cugan", "esr", "?"],
            default="janai2",
        )
        if model == "?":
            print("[blue]Sharpness: [green]esr > cugan > janai2 > janai3")
            print("[blue]Details:   [green]esr < cugan < janai2 < janai3")
            continue
        break
    cmd = (
        "--model "
        + ["AnimeJanaiV3", "AnimeJanaiV2", "RealCUGAN", "RealESR"][
            ["janai3", "janai2", "cugan", "esr"].index(model)
        ]
    )
    if model == "cugan":
        if Confirm.ask("[green]Denoise", default=False):
            cmd += " --denoise"
    elif "janai" in model:
        cmp = IntPrompt.ask("[green]Compact", default=0, choices=["0", "1", "2"])
        cmd += f" --compact {cmp}"
    arg_model.append(cmd)
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
    print(f"[green]Get file-{i:02d} path: [reset] {f}")

    get_arg()

while True:
    i += 1
    get = Prompt.ask(
        f"[green]Input file-{i:02d} path [yellow](return to finish)"
    ).strip()
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
poweroff = Confirm.ask("[green]Poweroff after superres", default=False)
crf = IntPrompt.ask("[green]Quality", default=17)
extra_args += f"--quality {crf} "
if Confirm.ask("[green]Use HEVC", default=False):
    extra_args += "--codec hevc_qsv "


def get_extra_args():
    get = Prompt.ask("[green]> Extra args [yellow](? for help)").strip()
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
for file, scale, model in zip(filelist, arg_scale, arg_model):
    i += 1
    print(f"[blue]File-{i:02d}: {file}\n[green]> Args: Res-scale={scale} Model={model}")
    print()
print(f"[green]Global args: poweroff={poweroff} crf={crf} extra_args={extra_args}")
print("\n[yellow]Check above. Press Enter to continue...")
input("")
print("[green]Starting process")

error_files = []
t_start = time.time()
i = 0
for file, scale, model in zip(filelist, arg_scale, arg_model):
    i += 1
    command = f"{target} --scale {scale} {model} "
    command += f"{extra_args} "
    command += file
    print(f"[yellow][{i}/{len(filelist)}] command: {command}")
    ret = os.system(command)
    print(f"[green][{i}/{len(filelist)}] Done file: {file} ret: {ret}")
    if ret != 0:
        error_files.append(file)

print("[green]Finished inference")
print(f"[yellow]Total cost: {(time.time() - t_start)/60:.2f} mins")
if error_files:
    print("[red]Some files returned error:\n" + "\n".join(error_files))
if poweroff:
    print("[yellow]Poweroff in 60 seconds")
    os.system("shutdown -s -t 60")
input("[yellow]Press Enter to exit")
