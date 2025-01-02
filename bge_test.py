import torch
from transformers import AutoTokenizer, AutoModel

sentences = ["測試-1", "測試-2"]
# [CLS]: 101, [SEP]: 102
# 測試-1: [ 101, 3947, 6275,  118,  122,  102]
# => 101: [[ 0.6531, -0.2118, -0.6481,  ..., -0.4508, -0.4927,  0.0411],
#    3947: [ 0.5379,  0.3572, -0.7991,  ..., -0.9508, -0.9146, -0.9793],
#    6275: [ 0.5155,  0.2562, -0.4068,  ..., -0.8445, -0.9667, -0.7316],
#    118:  [ 0.4753, -0.1538, -0.7082,  ..., -0.2958, -0.7376, -0.9429],
#    122:  [ 0.1463, -0.0629, -0.6182,  ..., -0.3590, -0.2788, -0.0720],
#    102:  [ 0.6533, -0.2114, -0.6485,  ..., -0.4514, -0.4930,  0.0413]],
# 測試-2: [ 101, 3947, 6275,  118,  123,  102]
# => 101: [[ 0.8856, -0.5590, -0.5037,  ..., -0.5072, -0.4309,  0.1151],
#    3947: [ 0.8068,  0.1158, -0.6337,  ..., -1.0136, -1.0601, -1.0295],
#    6275: [ 0.6965, -0.0046, -0.2851,  ..., -0.9377, -1.0290, -0.7432],
#    118:  [ 0.9466, -0.4403, -0.6184,  ..., -0.1978, -0.5773, -0.6837],
#    122:  [ 1.3820, -0.0187, -1.2671,  ..., -0.3657, -0.4867, -0.7006],
#    102:  [ 0.8859, -0.5585, -0.5043,  ..., -0.5079, -0.4312,  0.1155]]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
print(f"{encoded_input=}")

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    print(f"{model_output=}")
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]

# normalize embeddings
print(f"{sentence_embeddings=}")
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print(f"{sentence_embeddings=}")
# print("Sentence embeddings:", sentence_embeddings)
