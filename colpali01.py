import torch
from PIL import Image

from colpali_engine.models import ColQwen2, ColQwen2Processor

model_name = "vidore/colqwen2-v1.0"

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
).eval()
print(model)

processor = ColQwen2Processor.from_pretrained(model_name)

# Your inputs
images = [
    Image.new("RGB", (640, 640), color="white"),
    Image.new("RGB", (4, 4), color="black"),
]
queries = [
    # "Is attention really all you need?",
    "Are Benjamin, Antoine, Merve, and Jo best friends?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)
print(f"{batch_images=}")
print(f"{batch_queries=}")

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

print(f"{image_embeddings.size()=}")
print(f"{query_embeddings.size()=}")

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
print(f"{scores=}")
