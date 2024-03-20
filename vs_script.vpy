import os

import vapoursynth as vs
from vapoursynth import core
from vsmlrt import *

video_path = os.environ["VS_INPUT_FILE_PATH"]
trt = os.environ["VS_BACKEND"] == "trt"  # [trt, ort]
fp16 = os.environ["VS_FP16"] == "1"  # [0, 1]
scale = int(os.environ["VS_SCALE"])  # [2, 3, 4]
tiles = int(os.environ["VS_TILES"])  # [1, 2, 4, 8, 16]
model = os.environ["VS_MODEL"]  # [cugan, esrgan]
noise = int(os.environ["VS_NOISE"])

assert model in ("cugan", "esrgan")
assert tiles == 1 or tiles % 2 == 0

core = vs.core
if trt:
    device = Backend.TRT()
else:
    device = Backend.ORT_CUDA()
device.device_id = 0
device.num_streams = 2
device.fp16 = fp16

clip = core.lsmas.LWLibavSource(video_path)
clip = core.resize.Bicubic(clip=clip, format=vs.YUV444P16)
clip = core.resize.Bicubic(clip=clip, range=1, matrix_in_s="709", format=vs.RGB48)
clip = core.fmtc.bitdepth(clip, bits=32)
if model == "cugan":
    assert scale in (2, 3, 4)
    assert noise in (-1, 0, 1, 2, 3)
    version = 2
    if scale not in (2, 3) or noise not in (-1, 0, 3):
        version = 1
    clip = CUGAN(
        clip,
        noise=noise,
        scale=scale,
        tiles=tiles,
        version=version,
        alpha=1.00,
        backend=device,
    )
elif model == "esrgan":
    assert scale in (2, 4)
    if scale == 2:
        model = RealESRGANModel.animevideo_xsx2
    else:
        model = RealESRGANModel.animevideov3
    clip = RealESRGAN(
        clip=clip,
        model=model,
        backend=device,
        scale=scale,
        tiles=tiles,
    )
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV444P16)
clip.set_output()
