import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import pycuda.autoinit

TRT_LOGGER = trt.Logger()

class TRTInfer:
    def __init__(self, engine_path, class_names=['person', 'bottle', 'mat']):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        self.class_names = class_names

    def load_engine(self, path):
        with open(path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self, engine):
        inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, dw, dh

    def preprocess(self, img):
        img_letterboxed, ratio, dw, dh = self.letterbox(img, new_shape=(640, 640))
        img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_chw, axis=0)
        return input_tensor, ratio, dw, dh

    def postprocess(self, output, conf_thresh=0.4, iou_thresh=0.5):
        output = output.reshape(-1, 8)
        output = output[output[:, 4] > conf_thresh]
        if len(output) == 0:
            return np.empty((0, 8))
        return self.nms(output, iou_thresh)

    def nms(self, detections, iou_thresh=0.5):
        if len(detections) == 0:
            return detections
        detections = detections[detections[:, 4].argsort()[::-1]]
        keep = []
        while len(detections) > 0:
            best = detections[0]
            keep.append(best)
            if len(detections) == 1:
                break
            rest = detections[1:]
            ious = self.compute_iou(best, rest)
            detections = rest[ious < iou_thresh]
        return np.stack(keep)

    def compute_iou(self, box, boxes):
        x1 = np.maximum(box[0] - box[2]/2, boxes[:, 0] - boxes[:, 2]/2)
        y1 = np.maximum(box[1] - box[3]/2, boxes[:, 1] - boxes[:, 3]/2)
        x2 = np.minimum(box[0] + box[2]/2, boxes[:, 0] + boxes[:, 2]/2)
        y2 = np.minimum(box[1] + box[3]/2, boxes[:, 1] + boxes[:, 3]/2)
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = box[2] * box[3]
        boxes_area = boxes[:, 2] * boxes[:, 3]
        union_area = box_area + boxes_area - inter_area
        return inter_area / (union_area + 1e-6)

    def infer(self, frame):
        input_tensor, ratio, dw, dh = self.preprocess(frame)
        np.copyto(self.inputs[0][0], input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)
        self.stream.synchronize()
        detections = self.postprocess(self.outputs[0][0])
        
        # 복원 좌표 변환
        results = []
        for det in detections:
            x, y, w, h, conf, *cls_prob = det
            cls_id = int(np.argmax(cls_prob))
            x1 = int((x - w / 2 - dw) / ratio)
            y1 = int((y - h / 2 - dh) / ratio)
            x2 = int((x + w / 2 - dw) / ratio)
            y2 = int((y + h / 2 - dh) / ratio)
            label = self.class_names[cls_id]
            results.append([x1, y1, x2, y2, conf, label])
        return results
