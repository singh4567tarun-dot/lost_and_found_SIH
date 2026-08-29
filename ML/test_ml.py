import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

print("1. Testing PyTorch CUDA/CPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Running on: {device}")

print("2. Loading SBERT text model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
text_emb = sbert_model.encode("Lost black wallet near library")
print(f"   SBERT text embedding shape: {text_emb.shape}")

print("3. Loading CLIP vision-language model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("   CLIP loaded successfully!")

print("\n--- ML Setup Verification Complete! ---")