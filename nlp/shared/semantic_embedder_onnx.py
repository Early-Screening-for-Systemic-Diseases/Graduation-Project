import onnxruntime as ort
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

class ONNXEmbedder:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.session = ort.InferenceSession(str(self.model_dir / "model.onnx"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

    def encode(self, texts):
        """
        Accepts a single string or list of strings.
        Returns numpy array of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items()}
        outputs = self.session.run(None, ort_inputs)
        return np.array(outputs[0])

