import os

import vapoursynth as vs
from vapoursynth import core
from vsmlrt import *

video_path = os.environ["VS_INPUT_FILE_PATH"]
clip = core.ffms2.Source(video_path)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
# clip = RealESRGAN(
#     clip=clip,
#     model=RealESRGANModel.animevideo_xsx2,
#     backend=Backend.TRT(num_streams=2, fp16=True),
#     scale=2,
#     tilesize=400,
# )
clip = CUGAN(
    clip=clip,
    noise=0,
    scale=2,
    tilesize=400,
    backend=Backend.TRT(num_streams=2, fp16=True),
    version=2,
)
# clip = RIFE(
#     clip=clip,
#     multi=2,
#     scale=1,
#     model=44,
#     backend=Backend.TRT(num_streams=4, fp16=True),
#     tilesize=256,
# )
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

clip.set_output()
