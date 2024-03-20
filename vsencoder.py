import argparse
import math
import os
import sys
import time

import cv2
from win32api import GetShortPathName as ShortName

PATH = os.path.dirname(os.path.abspath(__file__))
VSPATH = os.path.join(PATH, "VSET", "VSET_Main", "vapoursynth", "vspipe.exe")
FFMPEGPATH = "ffmpeg.exe"

parser = argparse.ArgumentParser(description="Use VapourSynth to encode a video.")

parser.add_argument(
    "video",
    metavar="VIDEO_PATH",
    type=str,
    default=None,
    help="Path to video file to be interpolated",
)

parser.add_argument(
    "--backend",
    default=f"trt",
    choices=["trt", "ort"],
    type=str,
    help="Backend for VapourSynth. [trt]",
)

parser.add_argument(
    "--fp16",
    default=1,
    choices=[0, 1],
    type=int,
    help="Use fp16 for inference. [1]",
)

parser.add_argument(
    "--scale",
    default=2,
    choices=[2, 3, 4],
    type=int,
    help="Scale factor. [2]",
)

parser.add_argument(
    "--tiles",
    default=2,
    type=int,
    help="Number of tiles to split the image into. [2]",
)

parser.add_argument(
    "--model",
    default="cugan",
    choices=["cugan", "esrgan"],
    type=str,
    help="Model for inference. [cugan]",
)

parser.add_argument(
    "--noise",
    default=0,
    choices=[-1, 0, 1, 2, 3],
    type=int,
    help="Denoise level for CUGAN. [-1]",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    type=str,
    help="Path to output file. [<video_path>_vsout.mp4]",
)

parser.add_argument(
    "--encoder",
    dest="encoder",
    type=str,
    default="h264_qsv",
    help="Encoder for ffmpeg [h264_qsv]",
)

parser.add_argument(
    "--crf",
    dest="crf",
    type=int,
    default=17,
    help="Compression factor for h264 encoder [17]",
)

args = parser.parse_args()

in_path = os.path.abspath(args.video)
if args.output is None:
    out_path = os.path.splitext(in_path)[0] + "_vsout_noaudio.mp4"
else:
    out_path = os.path.abspath(args.output)

script_path = os.path.join(PATH, "vs_script.vpy")

videoCapture = cv2.VideoCapture(in_path)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoCapture.release()
assert fps > 0 and tot_frame > 0 and width > 0 and height > 0, "Invalid video file."
print(f"fps: {fps}, width: {width}, height: {height}, tot_frame: {tot_frame}")

if os.path.exists(out_path):
    os.remove(out_path)
with open(out_path, "w") as f:
    pass
quality_option = "-crf" if "lib" in args.encoder else "-q:v"
command = [
    VSPATH,"-c","y4m", script_path, "-", "|",
    FFMPEGPATH, "-y",# "-hide_banner", "-loglevel", "error",
    "-i", "pipe:", "-pix_fmt", "yuv420p",
    "-c:v", args.encoder, quality_option, str(args.crf),
    ShortName(out_path),
]  # fmt: skip

print(" ".join(command))

os.environ["VS_INPUT_FILE_PATH"] = ShortName(in_path)
os.environ["VS_BACKEND"] = args.backend
os.environ["VS_FP16"] = str(args.fp16)
os.environ["VS_SCALE"] = str(args.scale)
os.environ["VS_MODEL"] = args.model
os.environ["VS_NOISE"] = str(args.noise)
os.environ["VS_TILES"] = str(args.tiles)

t0 = time.time()
ret = os.system(" ".join(command))
t1 = time.time()
assert ret == 0, "Process failed."
print(f"Process finished in {(t1-t0)//60}:{(t1-t0)%60:.2f}.")
print("Muxing audio...")
in_path_v = out_path
out_path = out_path.replace("_noaudio", "")
if os.path.exists(out_path):
    os.remove(out_path)
with open(out_path, "w") as f:
    pass
command = [
    FFMPEGPATH, "-y",
    "-i", ShortName(in_path_v),
    "-i", ShortName(in_path),
    "-c:v", "copy",
    "-c:a", "copy",
    "-map", "0:v:0",
    "-map", "1:a:0",
    ShortName(out_path),
]  # fmt: skip
print(" ".join(command))
ret = os.system(" ".join(command))
if ret == 0:
    os.remove(in_path_v)
print("Done.")
