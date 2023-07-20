import torch
import torch.nn as nn

from experimental.streamk.utils import TritonMeasure, Measure, get_features

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = nn.Linear(4, 1, bias=True).to(device)
model.load_state_dict(torch.load("./experimental/streamk/model.pt"))
model.eval()

triton_measures = list()
for i in [82, 2*82]:
    t = TritonMeasure(
        two_tiles=True,
        sm=i,
    )
    triton_measures.append(t)

m, n, k = 768, 4864, 8192
blk_m, blk_n, blk_k = 128, 128, 32

features = get_features(triton_measures, total_tiles=(m // blk_m) * (n // blk_n), iters_per_tile=k // blk_k)
print(features)

X_train_min = 0.0
X_train_max = 791040.0
X = torch.tensor(features, dtype=torch.float, device=device)
X_normalized = (X - X_train_min) / (X_train_max - X_train_min)
print(X_normalized.shape)

with torch.inference_mode():
    y = model(X_normalized).squeeze()
    print(y)
