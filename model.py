import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Event, Queue, current_process
from typing import Any, Dict, List

import numpy as np
from loguru import logger

if current_process().name != "MainProcess":
    logger.add(sys.stderr, level="ERROR", enqueue=True)


def check_for_ncnn():
    python_version = sys.version_info[:2]
    if python_version != (3, 7):
        raise RuntimeError(
            f"This script only works with Python 3.7, not {python_version}"
        )

    dll_folder_path = os.path.join(os.path.dirname(__file__), "dlls")
    sys.path.append(dll_folder_path)


class Model:
    def __init__(self, args: Dict[str, Any]) -> None:
        model = args["model"]
        if model.endswith("NCNN"):
            check_for_ncnn()
        if model == "RealESR_NCNN":
            from ncnnRealESR import RealESRGAN

            denoise = args["denoise"]
            scale = args["scale"]
            if denoise:
                logger.warning("Denoise does not support for RealESR, ignored")
            model_name = f"realesr-animevideov3-x{scale}"
            self.model = RealESRGAN(
                gpuid=args["gpu"],
                model=model_name,
                tilesize=args["tilesize"],
                tta_mode=args["tta"],
            )
        elif model == "RealCUGAN_NCNN":
            from ncnnCugan import RealCUGAN

            denoise = args["denoise"]
            scale = args["scale"]
            model_name = f"up{scale}x-{'conservative' if denoise else 'no-denoise'}"
            if scale == 4 and model == "RealCUGAN":
                logger.warning(
                    "RealCUGAN pro model not support 4x scale yet, fallback to legacy model"
                )
            else:
                model_name = "pro-" + model_name
            self.model = RealCUGAN(
                gpuid=args["gpu"],
                model=model_name,
                num_threads=1,
                tilesize=args["tilesize"],
                tta_mode=args["tta"],
            )
        elif model == "AnimeJanaiV3_NCNN" or model == "AnimeJanaiV2_NCNN":
            from ncnnRealESR import RealESRGAN

            ver = model.split("_")[0][-1]
            model_name = [
                f"animejanai-v{ver}-compact-x2",
                f"animejanai-v{ver}-ultra-compact-x2",
                f"animejanai-v{ver}-super-ultra-compact-x2",
            ][args["compact"]]

            self.model = RealESRGAN(
                gpuid=args["gpu"],
                model=model_name,
                tilesize=args["tilesize"],
                tta_mode=args["tta"],
            )
        else:
            raise ValueError(f"Unknown model: {model}")

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        return self.model.enhance(frame)


class MultiProcessModelNode:
    def __init__(
        self,
        model_args: Dict[str, Any],
        in_queue: Queue,
        out_queue: Queue,
        ready_event: Event,  # type: ignore
        stop_event: Event,  # type: ignore
    ) -> None:
        self.model_args = model_args
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.ready_event = ready_event
        self.stop_event = stop_event

    @logger.catch()
    def run(self):
        model = Model(self.model_args)
        self.ready_event.set()
        while True:
            while self.in_queue.empty():
                if self.stop_event.is_set():
                    del model  # trigger swig destructor
                    return
                time.sleep(0.01)
            frame = self.in_queue.get()
            enhanced_frame = model.enhance(frame)
            self.out_queue.put(enhanced_frame)


def _run_node(node, *args):
    node = node(*args)
    node.run()


class MultiProcessModel:
    def __init__(
        self,
        worker_num: int,
        **kwargs,
    ) -> None:
        self.worker_num = worker_num
        self._model_args = kwargs
        self._procs: List[mp.Process] = []
        self._in_queues: List[mp.Queue] = []
        self._out_queues: List[mp.Queue] = []
        self._ready_events: List[mp.Event] = []
        self._stop_event = mp.Event()
        logger.info(f"Creating {worker_num} worker processes...")
        for i in range(worker_num):
            in_queue = mp.Queue(16)
            out_queue = mp.Queue(16)
            ready_event = mp.Event()
            process = mp.Process(
                target=_run_node,
                args=(
                    MultiProcessModelNode,
                    self._model_args,
                    in_queue,
                    out_queue,
                    ready_event,
                    self._stop_event,
                ),
                daemon=True,
            )
            process.start()
            self._procs.append(process)
            self._in_queues.append(in_queue)
            self._out_queues.append(out_queue)
            self._ready_events.append(ready_event)
        for i in range(worker_num):
            self._ready_events[i].wait()
        logger.success("All processes created")
        self.pendings = [0 for _ in range(worker_num)]
        self._target_in = 0
        self._target_out = 0

    def put(self, frame: np.ndarray):
        self._in_queues[self._target_in].put(frame)
        self.pendings[self._target_in] += 1
        self._target_in = (self._target_in + 1) % self.worker_num

    def get(self) -> np.ndarray:
        result = self._out_queues[self._target_out].get()
        self.pendings[self._target_out] -= 1
        self._target_out = (self._target_out + 1) % self.worker_num
        return result

    @property
    def pending(self):
        return sum(self.pendings)

    @property
    def available(self):
        return not self._out_queues[self._target_out].empty()

    def close(self):
        logger.info("Closing all worker processes...")
        self._stop_event.set()
        for i in range(self.worker_num):
            self._procs[i].terminate()
            self._procs[i].join(2)
            try:
                self._procs[i].close()
            except Exception as e:
                logger.error(f"Error when closing process {i}: {e}")
        for i in range(self.worker_num):
            self._in_queues[i].close()
            self._out_queues[i].close()
        logger.success("All processes terminated")
