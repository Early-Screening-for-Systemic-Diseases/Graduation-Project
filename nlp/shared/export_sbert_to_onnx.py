from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
onnx_dir = "nlp/shared/onnx_model"

model = SentenceTransformer(model_name, backend="onnx")
model.save_pretrained(onnx_dir)

print(f"ONNX model saved to {onnx_dir}")
