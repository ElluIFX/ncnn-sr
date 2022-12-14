import _thread
import argparse
import os
import subprocess as sp
import sys
import time
from queue import Empty, Queue

import cv2
import numpy as np
import skvideo.io
from tqdm import tqdm
from win32api import GetShortPathName as ShortName

from logger import print

python_version = sys.version_info[:2]
if python_version != (3, 7):
    raise RuntimeError(f"This script only works with Python 3.7, not {python_version}")

dll_folder_path = os.path.join(os.path.dirname(__file__), "dlls")
sys.path.append(dll_folder_path)

from ncnnCugan import RealCUGAN
from ncnnRealESR import RealESRGAN

parser = argparse.ArgumentParser(description="Use RealESRGan enhance video")
parser.add_argument(
    "video",
    metavar="VIDEO_PATH",
    type=str,
    default=None,
    help="Path to video file to be interpolated",
)
parser.add_argument(
    "--model",
    dest="model",
    type=str,
    default="RealESR",
    help="default: RealESR, available: RealESR, RealCUGAN",
)
parser.add_argument(
    "--denoise",
    dest="denoise",
    action="store_true",
    help="Enable denoise for RealCUGAN",
)
parser.add_argument(
    "--scale",
    dest="scale",
    type=int,
    default=2,
    help="The final upsampling scale of the image (2/3/4)",
)
parser.add_argument(
    "--tilesize",
    dest="tilesize",
    type=int,
    default=0,
    help="Tile size, 0 for no tile during testing",
)
parser.add_argument(
    "--tile_pad", dest="tile_pad", type=int, default=0, help="Tile padding"
)
parser.add_argument(
    "--tta",
    dest="tta",
    action="store_true",
    help="TTA mode, 8 times slower with invisible improvement",
)
parser.add_argument("--gpu", dest="gpu", type=int, default=0, help="GPU device to use")
parser.add_argument(
    "--ext", dest="ext", type=str, default="mp4", help="Output video extension"
)
parser.add_argument(
    "--no_compression",
    dest="no_compression",
    action="store_true",
    help="Disable ffmpeg backend for video compression, causes output no audio",
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
parser.add_argument(
    "--stop_time",
    dest="stop_time",
    type=float,
    default=-1,
    help="Stop time in seconds, will disable audio copy",
)
parser.add_argument(
    "--start_point",
    dest="start_point",
    type=int,
    default=0,
    help="Set process start point if you want to skip some frame, will disable audio copy",
)
parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Show ffmpeg output for debugging",
)

args = parser.parse_args()

assert args.scale in [2, 3, 4], "Scale must be 2, 3 or 4"
assert args.model in ["RealESR", "RealCUGAN"], "Model must be RealESR or RealCUGAN"
if args.model != "RealCUGAN" and args.denoise:
    print("Warning: Denoise only works with RealCUGAN model, ignored")

print("Initializing ncnn model...\n")
if args.model == "RealESR":
    model_name = f"realesr-animevideov3-x{args.scale}"
    model = RealESRGAN(
        gpuid=args.gpu,
        model=model_name,
        tilesize=args.tilesize,
        tta_mode=args.tta,
    )
elif args.model == "RealCUGAN":
    model_name = f"up{args.scale}x-{'denoise_3' if args.denoise else 'conservative'}"
    model = RealCUGAN(
        gpuid=args.gpu,
        model=model_name,
        num_threads=4,
        tilesize=args.tilesize,
        tta_mode=args.tta,
    )
print(f"Loaded model: [{args.model}]{model_name} with device {args.gpu}")

videoCapture = cv2.VideoCapture(args.video)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoCapture.release()
print(
    f"Input info:{tot_frame} frames in total, {fps} fps, {width}x{height} to {width*args.scale}x{height*args.scale}"
)

assert args.start_point < tot_frame, "Start point should be smaller than total frame"
tot_frame -= args.start_point
if args.stop_time > 0:
    tot_frame = min(int(args.stop_time * fps), tot_frame)
tot_frame = int(tot_frame)
print(f"{tot_frame} frames to process")

videogen = skvideo.io.vreader(args.video)
frame_count = 0
if args.start_point > 0:
    print("Skipping frames...")
    for _ in tqdm(range(args.start_point - 1)):
        next(videogen)
    print("Continue processing...")
first_frame = next(videogen)
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
video_path_wo_ext, ext = os.path.splitext(args.video)
h, w, _ = first_frame.shape
vid_out_name = None
vid_out = None
proc = None

if args.start_point == 0:
    vid_out_name = f"{video_path_wo_ext}_{args.scale}X_{args.model}_noaudio.{args.ext}"
else:
    vid_out_name = f"{video_path_wo_ext}_{args.scale}X_{args.model}_noaudio_from{args.start_point}.{args.ext}"

if args.no_compression:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # avc1 / mp4v / I420(raw)
    vid_out = cv2.VideoWriter(
        vid_out_name,
        fourcc,
        fps,
        (w, h),
        (cv2.VIDEO_ACCELERATION_ANY | cv2.VIDEOWRITER_PROP_HW_ACCELERATION),
    )
    assert vid_out.isOpened(), "Cannot open video for writing"
    print("Output video without compression")
else:
    if h * w * args.scale * args.scale > 9437184 and "h264" in args.encoder:
        print(
            "Warning: frame size reached h264 encoder upper limit (4096x2304), switching to HEVC encoder"
        )
        args.encoder = "libx265"
    if os.path.exists(vid_out_name):
        os.remove(vid_out_name)
    with open(vid_out_name, "w") as f:
        pass
    origin_file = ShortName(os.path.abspath(args.video))
    quality_option = "-crf" if "lib" in args.encoder else "-q:v"
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{int(w*args.scale)}x{int(h*args.scale)}",
        "-pix_fmt", "bgr24", "-r", f"{fps}",
        "-i", "-",
        "-c:v", args.encoder, quality_option, str(args.crf),
        ShortName(vid_out_name),
    ]  # fmt: skip
    if args.debug:
        print(f"FFmpeg command: {' '.join(command)}")
    proc = sp.Popen(command, stdin=sp.PIPE, shell=True)
    print("FFmpeg backend initialized")

if (not args.no_compression) and args.stop_time <= 0 and args.start_point == 0:
    print("Audio will be copied from original video after processing is done")

running = True
stop_flag = False

empty_frame = np.zeros((180, 530, 3), dtype=np.uint8)
cv2.putText(
    empty_frame,
    "Preview disabled",
    (130, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    1,
)


show_empty = False
last_frame = empty_frame.copy()
buffer_size_write = 64
buffer_size_read = 32
target_short_side = 520
target_long_side = 960
write_buffer = Queue(maxsize=buffer_size_write)
read_buffer = Queue(maxsize=buffer_size_read)
read_buffer.put(first_frame)


def show_frame(
    frame, frame_id, scale, total_frame, in_queue_write, in_queue_read
) -> None:
    global show_empty
    global empty_frame
    global last_frame
    global target_short_side
    global target_long_side
    global stop_flag

    if frame is None:
        return
    o_height, o_width, _ = frame.shape
    if not show_empty:
        # short_side = min(frame.shape[:2])
        # ratio = target_short_side / short_side
        long_side = max(frame.shape[:2])
        ratio = target_long_side / long_side
        temp_frame = cv2.resize(
            frame,
            (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        temp_frame = empty_frame.copy()
    height = temp_frame.shape[0]
    width = temp_frame.shape[1]
    progress = frame_id / total_frame
    text1 = f"Res: {int(o_width/scale)}x{int(o_height/scale)} -> {o_width}x{o_height}"
    video_file_size = os.path.getsize(vid_out_name) / 1024 / 1024
    text2 = (
        f"FileSize={video_file_size:.2f}MB"
        if video_file_size < 1024
        else f"FileSize={video_file_size/1024:.2f}GB"
    )
    warning = False
    if in_queue_write > 6:
        text2 += (
            f" (WriteDelay +{in_queue_write:d}"
            + ("*" if in_queue_write >= buffer_size_write else "")
            + ")"
        )
        warning = True
    if in_queue_read < buffer_size_read - 6:
        text2 += f" (ReadDelay +{buffer_size_read - in_queue_read:d})"
        warning = True
    text3 = f"Frame={int(frame_id):d}/{int(total_frame):d} {progress:.2%}"
    cv2.putText(
        temp_frame,
        text1,
        (5, height - 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        temp_frame,
        text2,
        (5, height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255) if warning else (0, 0, 255),
        2,
    )
    cv2.putText(
        temp_frame,
        text3,
        (5, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.rectangle(
        temp_frame,
        (0, height - 10),
        (width, height),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )
    read_progress = (frame_id + in_queue_write + in_queue_read) / total_frame
    cv2.rectangle(
        temp_frame,
        (0, height - 10),
        (int(width * read_progress), height),
        (221, 202, 98),
        thickness=cv2.FILLED,
    )
    write_progress = (frame_id + in_queue_write) / total_frame
    cv2.rectangle(
        temp_frame,
        (0, height - 10),
        (int(width * write_progress), height),
        (0, 255, 255),
        thickness=cv2.FILLED,
    )
    cv2.rectangle(
        temp_frame,
        (0, height - 10),
        (int(width * progress), height),
        (0, 0, 255),
        thickness=cv2.FILLED,
    )
    cv2.imshow("Frame preview", temp_frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        show_empty = not show_empty
    elif key == ord("w"):
        target_long_side *= 1.1
        target_short_side *= 1.1
    elif key == ord("s"):
        target_long_side *= 0.9
        target_short_side *= 0.9
        target_long_side = max(40, target_long_side)
        target_short_side = max(40, target_short_side)
    elif key == ord("p"):
        stop_flag = True


def write_worker(user_args, write_buffer, read_buffer, total_frame):
    frame_id = 0
    no_compression = user_args.no_compression
    scale = user_args.scale
    global running
    cv2.namedWindow("Frame preview", cv2.WINDOW_AUTOSIZE)
    while True:
        frame = write_buffer.get()
        if frame is None:
            break
        try:
            if no_compression:
                vid_out.write(frame)
            else:
                proc.stdin.write(frame.tobytes())
        except:
            print("Error writing frame to video")
            running = False
            return
        in_queue_write = write_buffer.qsize()
        in_queue_read = read_buffer.qsize()
        show_frame(frame, frame_id, scale, total_frame, in_queue_write, in_queue_read)
        frame_id += 1


def read_worker(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)


_thread.start_new_thread(read_worker, (args, read_buffer, videogen))
_thread.start_new_thread(write_worker, (args, write_buffer, read_buffer, tot_frame))

pbar = tqdm(total=tot_frame)
end_point = None
try:
    while running:
        frame = read_buffer.get()
        if frame is None or frame_count >= tot_frame:
            break
        try:
            output = model.enhance(frame)
        except RuntimeError as error:
            print("Error", error)
            print(
                "If you encounter CUDA out of memory, try to set --tilesize with a smaller number."
            )
            break
        write_buffer.put(output)
        frame_count += 1
        pbar.update(1)
        if stop_flag:
            pbar.close()
            end_point = frame_count + args.start_point
            print(f"Manually stopped, ending point: {end_point}")
            break
except KeyboardInterrupt:
    print(f"Force stop")
    running = False
    write_buffer.put(None)
    sys.exit(1)
pbar.close()
write_buffer.put(None)
t0 = time.time()
try:
    if running:
        print("Waiting for write buffer to be empty")
        while not write_buffer.empty():
            if time.time() - t0 > 200:
                print("Timeout, force exit")
                break
            time.sleep(1)
except:
    print("Force release video writer")
    running = False

cv2.destroyAllWindows()

if args.no_compression:
    vid_out.release()
else:
    try:
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
    except:
        pass
    proc.wait()

if end_point is None and args.stop_time > 0:
    end_point = frame_count + args.start_point
if end_point is not None:
    target_name = (
        os.path.splitext(vid_out_name)[0]
        + f"_to{end_point}"
        + os.path.splitext(vid_out_name)[1]
    )
    if os.path.exists(target_name):
        os.remove(target_name)
    os.rename(vid_out_name, target_name)
    vid_out_name = target_name

if (not args.no_compression) and end_point is None and args.start_point == 0:
    # merge audio from original video
    print("Merging audio")
    target_name2 = vid_out_name.replace("_noaudio", "")
    if os.path.exists(target_name2):
        os.remove(target_name2)
    with open(target_name2, "w") as target:
        pass
    command = [
        "ffmpeg", "-y","-hide_banner", "-loglevel", "error",
        "-i", ShortName(vid_out_name), "-i", ShortName(origin_file),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "copy",
        ShortName(target_name2),
    ]  # fmt: skip
    if args.debug:
        print(f"FFmpeg command: {' '.join(command)}")
    failed = False
    try:
        sp.run(command, check=True)
    except:
        print("Failed to merge audio")
        print(f"FFmpeg command: {' '.join(command)}")
        failed = True
    if not failed:
        os.remove(vid_out_name)

print(
    "Process finished, waiting for model backend to be released, you can close the window now"
)
