import argparse
import os
import sys
import time
from math import ceil

import cv2
from loguru import logger
from tqdm import tqdm

from model import MultiProcessModel
from utils import (
    FramePreviewWindow,
    ThreadedVideoReader,
    ThreadedVideoWriter,
    check_ffmepg_available_codec,
    check_ffmpeg_installed,
    ffmpeg_merge_video_and_audio,
    ffmpeg_merge_videos,
    find_unfinished_last_file,
    find_unfinished_merge_list,
    get_video_info,
)


def main():
    check_ffmpeg_installed()
    codecs = check_ffmepg_available_codec()
    enc = []
    for codec in codecs.values():
        enc.extend(codec[0])
    enc_def = "h264_qsv" if "h264_qsv" in enc else "libx264"

    parser = argparse.ArgumentParser()
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
        default="AnimeJanaiV3",
        choices=["RealESR", "RealCUGAN", "AnimeJanaiV3", "AnimeJanaiV2"],
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
        type=float,
        default=2,
        help="The final upsampling scale of the video resolution (<4.0)",
    )
    parser.add_argument(
        "--compact",
        dest="compact",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Compact mode for AnimeJanaiVx, 2 is super-ultra-compact",
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
    parser.add_argument(
        "--gpu", dest="gpu", type=int, default=0, help="GPU device to use"
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=3,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--codec",
        dest="codec",
        type=str,
        default=enc_def,
        choices=enc,
        help="Codec for ffmpeg backend encoding",
    )
    parser.add_argument(
        "--ext", dest="ext", type=str, default="mp4", help="Output video extension"
    )
    parser.add_argument(
        "--quality",
        dest="quality",
        type=int,
        default=17,
        help="Compression factor for h264/hevc codec, smaller is better quality",
    )
    parser.add_argument(
        "--start_frame",
        dest="start_frame",
        type=int,
        default=0,
        help="Process from specific frame, will disable audio copy if > 1",
    )
    parser.add_argument(
        "--start_time",
        dest="start_time",
        type=float,
        default=0,
        help="Start time in seconds, will disable audio copy",
    )
    parser.add_argument(
        "--stop_time",
        dest="stop_time",
        type=float,
        default=-1,
        help="Stop time in seconds, will disable audio copy",
    )
    parser.add_argument(
        "--skip_frame",
        dest="skip_frame",
        type=int,
        default=0,
        help="Skip N frames per frame when reading original video (original_fps /= (1 + N))",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Disable preview window",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Write debug log to stdout",
    )

    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO" if not args.debug else "DEBUG")  # type: ignore
    logger.add("superres.log", level="DEBUG", encoding="utf-8", rotation="1 MB")

    true_scale = ceil(args.scale)
    if true_scale == 1:
        true_scale = 2
    assert true_scale in [2, 3, 4], "Scale out of range"
    target_scale = args.scale
    args.scale = true_scale
    if target_scale != true_scale:
        downscale = target_scale / true_scale
        logger.info(
            f"Video wiil be upscale to {true_scale}x then downscale to {downscale}x"
        )
    else:
        downscale = 1

    logger.info(f'Input video: "{args.video}"')
    fps, tot_frame, width, height = get_video_info(args.video)
    target_width = round(width * target_scale)
    target_height = round(height * target_scale)
    tot_frame = int(tot_frame / (1 + args.skip_frame))
    fps = fps / (1 + args.skip_frame)
    logger.info(
        f"Input format: {tot_frame} frames, {fps} fps, {width}x{height} => {target_width}x{target_height}"
    )

    if args.start_time > 0:
        args.start_frame = round(args.start_time * fps)
        assert (
            args.start_frame < tot_frame
        ), f"Start time should be smaller than total time ({tot_frame / fps} sec)"
    assert (
        args.start_frame < tot_frame
    ), f"Start frame should be smaller than total frame ({tot_frame})"

    video_path_wo_ext, ext = os.path.splitext(args.video)
    model_suffix = (
        "_dn"
        if (args.denoise and args.model == "RealCUGAN")
        else ("_c" + str(args.compact) if "AnimeJanai" in args.model else "")
    )
    video_path_prefix = (
        f"{video_path_wo_ext}_{target_scale}X_{args.model}{model_suffix}_noaudio"
    )
    video_path = f"{video_path_prefix}.{args.ext}"

    continue_process = False
    if args.start_frame == 0:
        last_file, last_num = find_unfinished_last_file(video_path)
        if last_file is not None:
            logger.success("Found unfinished file, continue processing...")
            continue_process = True
            args.start_frame = last_num + 1

    if args.start_frame != 0:
        video_path = f"{video_path_prefix}_from{args.start_frame}.{args.ext}"

    tot_frame -= args.start_frame
    if args.stop_time > 0:
        tot_frame = min(int(args.stop_time * fps), tot_frame)
    tot_frame = int(tot_frame)
    logger.info(f"{tot_frame} frames to process")

    model = MultiProcessModel(
        args.workers,
        scale=args.scale,
        model=args.model + "_NCNN",
        compact=args.compact,
        denoise=args.denoise,
        tilesize=args.tilesize,
        tta=args.tta,
        gpu=args.gpu,
    )

    def notify_callback(frame_id: int):
        info = [
            f"{width}x{height} => {target_width}x{target_height}",
            f"Workers: {model.pendings}",
        ]
        notify = None
        return info, notify

    writer = ThreadedVideoWriter(
        video_path,
        target_width,
        target_height,
        fps,
        codec=args.codec,
        quality=args.quality,
        convert_rgb=True,
        buffer_size=32,
    )
    reader = ThreadedVideoReader(
        args.video, start_time=args.start_frame / fps, skip_frame=args.skip_frame
    )  # inputdict={"-hwaccel": "d3d11va"} dxva2 / cuda / cuvid / d3d11va / qsv / opencl

    if not args.headless:
        preview = FramePreviewWindow(
            reader=reader,
            writer=writer,
            total_frame=tot_frame,
            fps=fps,
            file_name=video_path,
        )
        preview.reg_notify_callback(notify_callback)

    if args.stop_time <= 0 and args.start_frame == 0:
        logger.info("Audio will be copied from original video after processing is done")

    pbar = tqdm(total=tot_frame)
    end_point = None
    frame_count = 0
    end = False
    try:
        while True:
            if not args.headless and preview.is_stopped:
                pbar.close()
                end_point = frame_count + args.start_frame
                logger.warning(f"Manually stopped, ending point: {end_point}")
                break
            while not end and model.pending < args.workers * 3:
                frame = reader.get()
                if frame is None or frame_count >= tot_frame:
                    end = True
                    break
                model.put(frame)
            if model.pending > 0:
                output = model.get()
                if downscale != 1:
                    output = cv2.resize(
                        output,
                        (target_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    )
                writer.put(output)
                frame_count += 1
                pbar.update(1)
            elif end:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pbar.close()
        end_point = frame_count + args.start_frame
        logger.critical(f"[UNSAFE] Manually stopped, ending point: {end_point}")
    pbar.close()
    writer.close()
    reader.close()
    model.close()
    if not args.headless:
        preview.close()

    logger.debug(f"Processed: {frame_count} frames")

    if end_point is None and args.stop_time > 0:
        end_point = frame_count + args.start_frame
    if end_point is not None:
        target_name = (
            os.path.splitext(video_path)[0]
            + f"_to{end_point}"
            + os.path.splitext(video_path)[1]
        )
        if os.path.exists(target_name):
            os.remove(target_name)
        os.rename(video_path, target_name)
    else:
        if continue_process:
            logger.info("Merging videos")
            video_path = f"{video_path_prefix}.{args.ext}"
            merge_list = find_unfinished_merge_list(video_path)
            if merge_list is not None:
                if ffmpeg_merge_videos(video_path, merge_list):
                    args.start_frame = 0  # trick to trigger the next if

        if args.start_frame <= 1:
            logger.info("Merging audio")
            target_name = video_path.replace("_noaudio", "")
            if ffmpeg_merge_video_and_audio(
                target_name, video_path, os.path.abspath(args.video)
            ):
                os.remove(video_path)
    logger.success("Process finished")
    sys.exit(0)


if __name__ == "__main__":
    main()
