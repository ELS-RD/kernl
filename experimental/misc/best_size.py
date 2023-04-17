import json
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import List

torch.manual_seed(123)


@dataclass
class TritonMeasure:
    two_tiles: bool
    sm: int
    disc: float
    triton_ms: float


@dataclass
class Measure:
    m: int
    n: int
    k: int
    triton: List[TritonMeasure]
    pytorch_ms: float
    speedup: float

    def number_of_tiles(self, blk_m: int, blk_n: int) -> int:
        return (self.m // blk_m) * (self.n // blk_n)

    def iter_per_tile(self, blk_k: int) -> int:
        return self.k // blk_k

    def get_minimum_triton_measure(self) -> TritonMeasure:
        return min(self.triton, key=lambda x: x.triton_ms)


class LinearRegression(nn.Module):
    def __init__(self, input_dim: int):
        super(LinearRegression, self).__init__()
        # self.linear_1 = nn.Linear(input_dim, input_dim, bias=False)
        self.linear_2 = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # x = self.linear_1(x)
        return self.linear_2(x)


def from_dict_to_dataclass(data):
    return Measure(
        m=data['m'],
        n=data['n'],
        k=data['k'],
        triton=[TritonMeasure(
                    two_tiles=triton_data['2 tiles'],
                    sm=triton_data['sm'],
                    disc=triton_data['disc'],
                    triton_ms=triton_data['triton_ms']
                ) for triton_data in data['triton']],
        pytorch_ms=data['pytorch_ms'],
        speedup=data['speedup']
    )


data = list()
triton_timings = list()
blk_m, blk_n, blk_k = 128, 128, 32
with open("./experimental/misc/results.json") as f:
    measure_json = json.load(f)

to_skip = set()

# create features to be learned
for xp_measure in measure_json:
    m = from_dict_to_dataclass(xp_measure)
    total_tiles = m.number_of_tiles(blk_m, blk_n)
    iters_per_tile = m.iter_per_tile(blk_k)
    for triton in m.triton:
        total_programs_streamk = triton.sm
        total_tiles_streamk = total_tiles % total_programs_streamk
        # for two-tile Stream-K + data-parallel from original paper
        if triton.two_tiles and total_tiles - total_tiles_streamk > total_programs_streamk:
            total_tiles_streamk += total_programs_streamk
        # remaining tiles are computed using classical blocking
        total_blocking_tiles = total_tiles - total_tiles_streamk
        total_iters_streamk = total_tiles_streamk * iters_per_tile
        # iterations related to full waves
        total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
        # iterations related to last (partial) wave
        total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk

        # values used for prediction
        nb_synchronization = triton.sm  # there is 2 syncs per SM in stream-k
        nb_store = total_blocking_tiles  # there is 1 store per tile in blocking loop
        nb_iter_stream_k = total_iters_streamk  # includes loading
        nb_iter_blocking = total_blocking_tiles * iters_per_tile  # includes loading
        timing_triton = triton.triton_ms

        features = [nb_synchronization, nb_iter_stream_k, nb_iter_blocking, nb_store]
        if tuple(features) in to_skip:
            continue
        to_skip.add(tuple(features))
        data.append(features)
        triton_timings.append(timing_triton)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.tensor(data, dtype=torch.float, device=device)
y = torch.tensor(triton_timings, dtype=torch.float, device=device)

print(X[:5], y[:5])
print(X.shape, y.shape)

# Check for NaN or infinity values in the input and output data
assert not torch.isnan(X).any() and not torch.isnan(y).any(), "Input data contains NaN values."
assert not torch.isinf(X).any() and not torch.isinf(y).any(), "Input data contains infinity values."

# Train-eval split
split_ratio = 0.8
num_samples = X.shape[0]
split_idx = int(num_samples * split_ratio)
indices = torch.randperm(num_samples)
train_indices, eval_indices = indices[:split_idx], indices[split_idx:]
X_train, y_train = X[train_indices], y[train_indices]
X_eval, y_eval = X[eval_indices], y[eval_indices]

# Normalize or standardize train and eval datasets using train dataset statistics
X_train_min, X_train_max = X_train.min(), X_train.max()
X_train_normalized = (X_train - X_train_min) / (X_train_max - X_train_min)
X_eval_normalized = (X_eval - X_train_min) / (X_train_max - X_train_min)

# Model, loss, and optimizer
input_dim = X.shape[1]
model = LinearRegression(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with early stopping
num_epochs = 10000
batch_size = 64
patience = 500
num_train_samples = X_train_normalized.shape[0]

best_eval_loss = float('inf')
epochs_since_last_improvement = 0

for epoch in range(num_epochs):
    # Shuffle dataset
    indices = torch.randperm(num_train_samples)
    X_train_shuffled = X_train_normalized[indices]
    y_train_shuffled = y_train[indices]

    for i in range(0, num_train_samples, batch_size):
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on the evaluation set
    with torch.no_grad():
        eval_outputs = model(X_eval_normalized)
        eval_loss = criterion(eval_outputs.squeeze(), y_eval)

    if eval_loss.item() < best_eval_loss:
        best_eval_loss = eval_loss.item()
        epochs_since_last_improvement = 0
    else:
        epochs_since_last_improvement += 1

    if epochs_since_last_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Eval Loss: {eval_loss.item():.4f}')

with torch.inference_mode():
    y_pred = model(X_eval_normalized).squeeze()
    diff = y_pred - y_eval
    print(f"Mean {diff.mean().item():.2f} ms, std {diff.std().item():.2f} ms, max {diff.max().item():.2f} ms, min {diff.min().item():.2f} ms")
