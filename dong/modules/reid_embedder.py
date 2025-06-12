import cv2, numpy as np, onnxruntime as ort

class ReIDEmbedder:
    """
    OSNet-x0.25  (reid.onnx)  ->  512-D 임베딩 반환
    """
    def __init__(self, onnx_path: str, use_gpu: bool = False):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.H, self.W = 256, 128                    # 입력 해상도
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # --------------------- private helpers ----------------------------
    def _preprocess(self, img):
        img = cv2.resize(img, (self.W, self.H))      # (W,H) 주의!
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std           # Normalize
        img = img.transpose(2, 0, 1)              # (3,256,128) ← 배치축 제거
        return img.astype(np.float32)

    # ------------------------- public API -----------------------------
    def __call__(self, crops):
        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        # batch = np.concatenate([self._preprocess(c) for c in crops], axis=0)
        batch = np.stack([self._preprocess(c) for c in crops], axis=0)
        feats = self.session.run([self.output_name], {self.input_name: batch})[0]
        return feats.astype(np.float32)              # (N, 512)
