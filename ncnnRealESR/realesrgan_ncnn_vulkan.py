#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: RealESRGAN ncnn Vulkan Python wrapper
Author: Justin
Date Created: May 20, 2022
Last Modified: May 20, 2022
"""

import importlib
import pathlib
import sys

import cv2
import numpy as np

if __package__ is None:
    import realesrgan_ncnn_vulkan_wrapper as wrapped
else:
    wrapped = importlib.import_module(f"{__package__}.realesrgan_ncnn_vulkan_wrapper")


class Realesrgan:
    """
    Python FFI for RealESRGAN implemented with ncnn library

    :param gpuid int: gpu device to use (-1=cpu)
    :param tta_mode bool: enable test time argumentation
    :param tilesize int: tile size
    :param model str: realesrgan model name
    """

    def __init__(
        self,
        gpuid: int = 0,
        tta_mode: bool = False,
        tilesize: int = 0,
        model: str = "realesrgan-x4plus",
        **_kwargs,
    ):
        # check arguments' validity
        assert gpuid >= -1, "gpuid must >= -1"
        assert tilesize == 0 or tilesize >= 32, "tilesize must >= 32 or be 0"

        self._realesrgan_object = wrapped.RealESRGANWrapped(gpuid, tta_mode)
        self._model = model
        self._gpuid = gpuid

        # Parse scale
        self._realesrgan_object.scale = 2
        if "x2" in model:
            self._realesrgan_object.scale = 2
        elif "x3" in model:
            self._realesrgan_object.scale = 3
        elif "x4" in model:
            self._realesrgan_object.scale = 4
        self._realesrgan_object.prepadding = self._get_prepadding()

        self._realesrgan_object.tilesize = (
            self._get_tilesize() if tilesize <= 0 else tilesize
        )
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
            model_path = pathlib.Path(self._model)
            if not model_path.is_dir():
                model_path = pathlib.Path(__file__).parent / "models" / self._model
                param_path = model_path / f"{self._model}.param"
                model_path = model_path / f"{self._model}.bin"

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

            self._realesrgan_object.load(param_path_str, model_path_str)
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
        out_bytes = bytearray((self._realesrgan_object.scale**2) * len(in_bytes))

        raw_in_image = wrapped.Image(in_bytes, w, h, channels)
        raw_out_image = wrapped.Image(
            out_bytes,
            self._realesrgan_object.scale * w,
            self._realesrgan_object.scale * h,
            channels,
        )

        self._realesrgan_object.process(raw_in_image, raw_out_image)

        blen = (
            self._realesrgan_object.scale
            * h
            * self._realesrgan_object.scale
            * w
            * channels
        )

        return np.asarray(bytearray(out_bytes), dtype=np.uint8)[:blen].reshape(
            (
                self._realesrgan_object.scale * h,
                self._realesrgan_object.scale * w,
                channels,
            )
        )

    def _get_prepadding(self) -> int:
        return {2: 18, 3: 14, 4: 19}.get(self._realesrgan_object.scale, 0)

    def _get_tilesize(self):
        if self._gpuid == -1:
            return 400
        else:
            heap_budget = self._realesrgan_object.get_heap_budget()
            if self._realesrgan_object.scale == 2:
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
            elif self._realesrgan_object.scale == 3:
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
            elif self._realesrgan_object.scale == 4:
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


class RealESRGAN(Realesrgan):
    ...
