#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: RealCUGAN ncnn Vulkan Python wrapper
Author: K4YT3X
Date Created: March 19, 2022
Last Modified: March 19, 2022
"""

import importlib
import os
import pathlib
import sys

import numpy as np

if __package__ is None:
    import realcugan_ncnn_vulkan_wrapper as wrapped
else:
    wrapped = importlib.import_module(f"{__package__}.realcugan_ncnn_vulkan_wrapper")


class Realcugan:
    """
    Python FFI for RealCUGAN implemented with ncnn library

    :param gpuid int: gpu device to use (-1=cpu)
    :param tta_mode bool: enable test time argumentation
    :param num_threads int: processing thread count
    :param tilesize int: tile size
    :param syncgap int: sync gap mode
    :param model str: realcugan model name
    """

    def __init__(
        self,
        gpuid: int = 0,
        tta_mode: bool = False,
        num_threads: int = 1,
        tilesize: int = 0,
        syncgap: int = 3,
        model: str = "up2x-conservative",
        **_kwargs,
    ):
        # check arguments' validity
        assert gpuid >= -1, "gpuid must >= -1"
        # if scale not in range(1, 5):
        #     scale = 2
        assert tilesize == 0 or tilesize >= 32, "tilesize must >= 32 or be 0"
        assert syncgap in range(4), "syncgap must be 0-3"
        assert num_threads >= 1, "num_threads must be a positive integer"

        self._realcugan_object = wrapped.RealCUGANWrapped(gpuid, tta_mode, num_threads)
        self._model = model
        self._gpuid = gpuid

        # Parse scale
        self._realcugan_object.scale = 2
        if "up2x" in model:
            self._realcugan_object.scale = 2
        elif "up3x" in model:
            self._realcugan_object.scale = 3
        elif "up4x" in model:
            self._realcugan_object.scale = 4
        self._realcugan_object.prepadding = self._get_prepadding()

        # Parse Denoise level
        self._realcugan_object.noise = -1
        if "conservative" in model:
            self._realcugan_object.noise = -1
        elif "no-denoise" in model:
            self._realcugan_object.noise = 0
        elif "denoise1x" in model:
            self._realcugan_object.noise = 1
        elif "denoise2x" in model:
            self._realcugan_object.noise = 2
        elif "denoise3x" in model:
            self._realcugan_object.noise = 3

        self._realcugan_object.tilesize = (
            self._get_tilesize() if tilesize <= 0 else tilesize
        )
        self._realcugan_object.prepadding = self._get_prepadding()
        self._realcugan_object.syncgap = syncgap
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

            self._realcugan_object.load(param_path_str, model_path_str)
        else:
            if self._realcugan_object.scale != 2:
                self._realcugan_object.scale = 2
                self._load()
            raise FileNotFoundError(f"{param_path} or {model_path} not found")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Process the incoming image

        :param image: np.ndarray
        :return: np.ndarray
        """
        in_bytes = bytearray(image.tobytes())
        h, w, channels = image.shape
        out_bytes = bytearray((self._realcugan_object.scale**2) * len(in_bytes))

        raw_in_image = wrapped.Image(in_bytes, w, h, channels)
        raw_out_image = wrapped.Image(
            out_bytes,
            self._realcugan_object.scale * w,
            self._realcugan_object.scale * h,
            channels,
        )

        if self._gpuid != -1:
            self._realcugan_object.process(raw_in_image, raw_out_image)
        else:
            self._realcugan_object.tilesize = max(w, h)
            self._realcugan_object.process_cpu(raw_in_image, raw_out_image)

        blen = (
            self._realcugan_object.scale
            * h
            * self._realcugan_object.scale
            * w
            * channels
        )

        return np.asarray(bytearray(out_bytes), dtype=np.uint8)[:blen].reshape(
            (
                self._realcugan_object.scale * h,
                self._realcugan_object.scale * w,
                channels,
            )
        )

    def _get_prepadding(self) -> int:
        return {2: 18, 3: 14, 4: 19}.get(self._realcugan_object.scale, 0)

    def _get_tilesize(self):
        if self._gpuid == -1:
            return 400
        else:
            heap_budget = self._realcugan_object.get_heap_budget()
            if self._realcugan_object.scale == 2:
                if heap_budget > 1300:
                    return 400
                elif heap_budget > 800:
                    return 300
                elif heap_budget > 400:
                    return 200
                elif heap_budget > 200:
                    return 100
                else:
                    return 32
            elif self._realcugan_object.scale == 3:
                if heap_budget > 3300:
                    return 400
                elif heap_budget > 1900:
                    return 300
                elif heap_budget > 950:
                    return 200
                elif heap_budget > 320:
                    return 100
                else:
                    return 32
            elif self._realcugan_object.scale == 4:
                if heap_budget > 1690:
                    return 400
                elif heap_budget > 980:
                    return 300
                elif heap_budget > 530:
                    return 200
                elif heap_budget > 240:
                    return 100
                else:
                    return 32


class RealCUGAN(Realcugan):
    ...
