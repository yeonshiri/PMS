# modules/trt_infer.py

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

class TRTInfer:
    def __init__(self, engine_path, input_shape=(1, 3, 640, 640)):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape

        self.input_binding_idx = self.engine.get_binding_index("images")  # 또는 0
        self.output_binding_idx = self.engine.get_binding_index("output")  # 또는 1

        self.inputs = cuda.mem_alloc(trt.volume(input_shape) * np.float32().itemsize)
        self.outputs = cuda.mem_alloc(trt.volume((1, 25200, 85)) * np.float32().itemsize)  # YOLOv5n 기준
        self.bindings = [int(self.inputs), int(self.outputs)]
        self.stream = cuda.Stream()

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # (1,3,640,640)
        return img

    def infer(self, frame):
        img = self.preprocess(frame)
        np.copyto(np.empty(self.input_shape, dtype=np.float32), img)

        cuda.memcpy_htod_async(self.inputs, img, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        output = np.empty((1, 25200, 85), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.outputs, self.stream)
        self.stream.synchronize()
        return output
