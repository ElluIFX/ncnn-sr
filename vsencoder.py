import argparse
import math
import os
import sys

import cv2
from win32api import GetShortPathName as ShortName

parser = argparse.ArgumentParser(description="Use VapourSynth to encode a video.")

parser.add_argument(
    "script",
    metavar="SCRIPT_PATH",
    type=str,
    default=None,
    help="Path to VapourSynth script to be used for video processing",
)

parser.add_argument(
    "video",
    metavar="VIDEO_PATH",
    type=str,
    default=None,
    help="Path to video file to be interpolated",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    type=str,
    help="Path to output file.(default: <video_path>_vsout.mp4)",
)
parser.add_argument(
    "--encoder",
    dest="encoder",
    type=str,
    default="h264_qsv",
    help="Encoder for ffmpeg",
)
parser.add_argument(
    "--crf",
    dest="crf",
    type=int,
    default=17,
    help="Compression factor for h264 encoder",
)
args = parser.parse_args()

in_path = os.path.abspath(args.video)
if args.output is None:
    out_path = os.path.splitext(in_path)[0] + "_vsout_noaudio.mp4"
else:
    out_path = os.path.abspath(args.output)

script_path = os.path.abspath(args.script)

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
    "VSPipe.exe","-c","y4m", script_path, "-","|"
    "ffmpeg", "-y",# "-hide_banner", "-loglevel", "error",
    "-i", "pipe:", 
    "-c:v", args.encoder, quality_option, str(args.crf),
    ShortName(out_path),
]  # fmt: skip

print(" ".join(command))
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["VS_INPUT_FILE_PATH"] = ShortName(in_path)
ret = os.system(" ".join(command))
assert ret == 0, "Process failed."
print("Muxing audio...")
in_path_v = out_path
out_path = out_path.replace("_noaudio", "")
if os.path.exists(out_path):
    os.remove(out_path)
with open(out_path, "w") as f:
    pass
command = [
    "ffmpeg", "-y",
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
