import os
import sys
import time

path = os.path.abspath(os.path.dirname(__file__))
excuteable = os.path.join(path, "venv", "Scripts", "python.exe")
if " " in excuteable:
    excuteable = '"' + excuteable + '"'
target = r"./superres-video.py"

filelist = []
arg_scale = []
arg_cu = []
arg_denoise = []

if not os.path.normcase(os.getcwd()) == os.path.normcase(path):
    os.chdir(path)
    print("Changed cwd to: " + path)

path = r'"' + path + r'"'


def get_arg():
    get = input("> Resolution scale (2): ").strip()
    if get == "":
        get = "2"
    get = int(get)
    arg_scale.append(get)

    get = input("> Use cugan (y/N): ").strip()
    if "y" in get.lower():
        arg_cu.append(True)
    else:
        arg_cu.append(False)

    if arg_cu[-1]:
        get = input("> Use denoise (y/N): ").strip()
        if "y" in get.lower():
            arg_denoise.append(True)
        else:
            arg_denoise.append(False)
    else:
        arg_denoise.append(False)
    print()


i = 0
in_file = sys.argv[1:]
for f in in_file:
    i += 1
    if not f.startswith(r'"'):
        f = r'"' + f + r'"'
    filelist.append(f)
    print(f"Get file-{i:02d} path: ", f)

    get_arg()

while True:
    i += 1
    get = input(f"Input file-{i:02d} path: ").strip()
    if get == "":
        break
    if not get.startswith(r'"'):
        get = r'"' + get + r'"'
    filelist.append(get)

    get_arg()
print()

poweroff = False
crf = 17
get = input("> Poweroff after superres (y/N): ").strip()
if "y" in get.lower():
    poweroff = True
get = input("> CRF (17): ").strip()
if get != "":
    crf = int(get)

print()
i = 0
for file, scale, cu, denoise in zip(filelist, arg_scale, arg_cu, arg_denoise):
    i += 1
    print(
        f"File-{i:02d}: {file}\n> Args: Res-scale={scale} cugan={cu} denoise={denoise}"
    )
    print()
print("Global args: poweroff={0} crf={1}".format(poweroff, crf))
print()
input("Check above. Press Enter to continue...")
print("Starting superres")

error_files = []
t_start = time.time()
i = 0
for file, scale, cu, denoise in zip(filelist, arg_scale, arg_cu, arg_denoise):
    i += 1
    command = f"{excuteable} {target} --scale {scale} "
    command += f"--crf {crf} "
    if cu:
        command += "--model RealCUGAN "
        if denoise:
            command += "--denoise "
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
