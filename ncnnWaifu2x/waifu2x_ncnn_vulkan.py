#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: waifu2x ncnn Vulkan Python wrapper
Author: ArchieMeng
Date Created: February 4, 2021
Last Modified: May 13, 2021

Dev: K4YT3X
Last Modified: February 5, 2022
"""

# built-in imports
import importlib
import os
import pathlib
import sys

# third-party imports
import cv2
import numpy as np

# local imports
if __package__ is None:
    import waifu2x_ncnn_vulkan_wrapper as wrapped
else:
    wrapped = importlib.import_module(f"{__package__}.waifu2x_ncnn_vulkan_wrapper")


class Waifu2x:
    def __init__(
        self,
        gpuid: int = 0,
        denoise: int = 0,
        tilesize: int = 0,
        model: str = "cunet-noise0_model",
        tta_mode: bool = False,
        num_threads: int = 1,
    ):
        """
        Waifu2x class which can do image super resolution.

        :param gpuid: the id of the gpu device to use. -1 for cpu mode.
        :param model: the name or the path to the model
        :param tta_mode: whether to enable tta mode or not
        :param num_threads: the number of threads in upscaling
        :param denoise: denoise level, large value means strong denoise effect, -1 = no effect. value: -1/0/1/2/3. default: 0
        :param tilesize: tile size. 0 for automatically setting the size. default: 0
        """

        # check arguments' validity
        # assert scale in (1, 2), "scale must be 1 or 2" # TODO new class for Waifu2xRestore(1x model)
        assert denoise in range(4), "denoise must be 0-4"
        assert isinstance(tta_mode, bool), "tta_mode isn't a boolean value"
        assert num_threads >= 1, "num_threads must be an integer >= 1"

        self._waifu2x_object = wrapped.Waifu2xWrapped(gpuid, tta_mode, num_threads)
        self._model = model
        self._gpuid = gpuid

        # Parse scale
        self._waifu2x_object.scale = 2
        if "scale2x" in model:
            self._waifu2x_object.scale = 2
        else:
            self._waifu2x_object.scale = 1
        self._waifu2x_object.prepadding = self._get_prepadding()

        # Parse Denoise level
        self._waifu2x_object.noise = -1
        if "noise" not in model:
            self._waifu2x_object.noise = -1
        elif "noise0" in model:
            self._waifu2x_object.noise = 0
        elif "noise1" in model:
            self._waifu2x_object.noise = 1
        elif "noise2" in model:
            self._waifu2x_object.noise = 2
        elif "noise3" in model:
            self._waifu2x_object.noise = 3

        self._waifu2x_object.tilesize = (
            self._get_tilesize() if tilesize <= 0 else tilesize
        )
        self._waifu2x_object.prepadding = self._get_prepadding()
        self._load()

    def _load(
        self, param_path: pathlib.Path = None, model_path: pathlib.Path = None
    ) -> None:
        """
        Load models from given paths. Use self.model if one or all of the parameters are not given.

        :param param_path: the path to model params. usually ended with ".param"
        :param model_path: the path to model bin. usually ended with ".bin"
        :return: None
        """
        if param_path is None or model_path is None:
            model_dir = pathlib.Path(__file__).parent / "models" / self._model
            file_lists = os.listdir(str(model_dir))
            model_path = model_dir
            param_path = model_dir
            for f in file_lists:
                ext = os.path.splitext(f)[-1]
                if ext == ".param":
                    param_path = model_dir / f
                if ext == ".bin":
                    model_path = model_dir / f

        if param_path.exists() and model_path.exists():
            param_path_str, model_path_str = wrapped.StringType(), wrapped.StringType()
            if sys.platform in ("win32", "cygwin"):
                param_path_str.wstr = wrapped.new_wstr_p()
                wrapped.wstr_p_assign(param_path_str.wstr, str(param_path))
                model_path_str.wstr = wrapped.new_wstr_p()
                wrapped.wstr_p_assign(model_path_str.wstr, str(model_path))
            else:
                param_path_str.str = wrapped.new_str_p()
                wrapped.str_p_assign(param_path_str.str, str(param_path))
                model_path_str.str = wrapped.new_str_p()
                wrapped.str_p_assign(model_path_str.str, str(model_path))

            self._waifu2x_object.load(param_path_str, model_path_str)
        else:
            raise FileNotFoundError(f"{param_path} or {model_path} not found")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Process the incoming image

        :param image: np.ndarray
        :return: np.ndarray
        """
        in_bytes = bytearray(image.tobytes())
        h, w, channels = image.shape
        out_bytes = bytearray((self._waifu2x_object.scale**2) * len(in_bytes))

        raw_in_image = wrapped.Image(in_bytes, w, h, channels)
        raw_out_image = wrapped.Image(
            out_bytes,
            self._waifu2x_object.scale * w,
            self._waifu2x_object.scale * h,
            channels,
        )

        if self._gpuid != -1:
            self._waifu2x_object.process(raw_in_image, raw_out_image)
        else:
            self._waifu2x_object.tilesize = max(w, h)
            self._waifu2x_object.process_cpu(raw_in_image, raw_out_image)

        blen = (
            self._waifu2x_object.scale * h * self._waifu2x_object.scale * w * channels
        )

        return np.asarray(bytearray(out_bytes), dtype=np.uint8)[:blen].reshape(
            (self._waifu2x_object.scale * h, self._waifu2x_object.scale * w, channels)
        )

    def _get_prepadding(self) -> int:
        if "cunet" in self._model:
            if self._waifu2x_object.noise == -1:
                return 18
            elif self._waifu2x_object.scale == 1:
                return 28
            elif self._waifu2x_object.scale == 2:
                return 18
            else:
                return 18
        elif "anime" in self._model:
            return 7
        elif "photo" in self._model:
            return 7
        else:
            raise ValueError(f'model "{self._model}" is not supported')

    def _get_tilesize(self):
        if self._gpuid == -1:
            return 4000
        else:
            heap_budget = self._waifu2x_object.get_heap_budget()
            if "cunet" in self._model:
                if heap_budget > 2600:
                    return 400
                elif heap_budget > 740:
                    return 200
                elif heap_budget > 250:
                    return 100
                else:
                    return 32
            else:
                if heap_budget > 1900:
                    return 400
                elif heap_budget > 550:
                    return 200
                elif heap_budget > 190:
                    return 100
                else:
                    return 32


if __name__ == "__main__":
    from time import time

    t = time()
    im = cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\test\images\0001.png")
    w2x_obj = Waifu2x(0, denoise=3, scale=1)
    tt = 100
    # while tt:
    out_im = w2x_obj.enhance(im)
    cv2.imwrite("temp.png", out_im)
    # print(f"Elapsed time: {time() - t}s")
    # t = time()
    # tt-=1
