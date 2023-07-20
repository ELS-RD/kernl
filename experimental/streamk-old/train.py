import json
import torch
import torch.nn as nn
import torch.optim as optim

from experimental.streamk.utils import TritonMeasure, Measure, get_timings, get_features

torch.manual_seed(123)


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


blk_m, blk_n, blk_k = 128, 128, 32
with open("./experimental/streamk/results.json") as f:
    measure_json = json.load(f)

to_skip = set()
data = list()
triton_timings = list()


for xp_measure in measure_json:
    m = from_dict_to_dataclass(xp_measure)
    total_tiles: int = m.number_of_tiles(blk_m, blk_n)
    iters_per_tile: int = m.iter_per_tile(blk_k)
    features = get_features(m.triton, total_tiles, iters_per_tile)
    to_pred = get_timings(m.triton)
    for f, t in zip(features, to_pred):
        if tuple(f) in to_skip:
            continue
        to_skip.add(tuple(f))
        data.append((f, m.m, m.n, m.k, t))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.tensor([d[0] for d in data], dtype=torch.float, device=device)
y = torch.tensor([d[4] for d in data], dtype=torch.float, device=device)

print(X[:5], y[:5])
print(X.shape, y.shape)

assert not torch.isnan(X).any() and not torch.isnan(y).any(), "Input data contains NaN values."
assert not torch.isinf(X).any() and not torch.isinf(y).any(), "Input data contains infinity values."


def custom_split(data, split_ratio=0.8):
    sorted_data = sorted(data, key=lambda x: (x[1], x[2], x[3]))
    num_samples = len(sorted_data)
    split_idx = int(num_samples * split_ratio)
    train_data = sorted_data[:split_idx]
    eval_data = sorted_data[split_idx:]
    return train_data, eval_data


train_data, eval_data = custom_split(data)

X_train = torch.tensor([d[0] for d in train_data], dtype=torch.float, device=device)
y_train = torch.tensor([d[4] for d in train_data], dtype=torch.float, device=device)
X_eval = torch.tensor([d[0] for d in eval_data], dtype=torch.float, device=device)
y_eval = torch.tensor([d[4] for d in eval_data], dtype=torch.float, device=device)


# Normalize or standardize train and eval datasets using train dataset statistics
X_train_min, X_train_max = X_train.min(dim=0).values, X_train.max(dim=0).values
X_train_normalized = (X_train - X_train_min) / (X_train_max - X_train_min)
X_eval_normalized = (X_eval - X_train_min) / (X_train_max - X_train_min)
print(f"X_train_min: {X_train_min}, X_train_max: {X_train_max}")
# Model, loss, and optimizer
input_dim = X.shape[1]
# model = LinearRegression(input_dim).to(device)
model = nn.Linear(input_dim, 1, bias=True).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training with early stopping
num_epochs = 200
batch_size = 50
patience = 200
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

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Eval Loss: {eval_loss.item():.4f}')

with torch.inference_mode():
    y_pred = model(X_eval_normalized).squeeze()
    diff = y_pred - y_eval
    print(f"Mean {diff.mean().item():.2f} ms, std {diff.std().item():.2f} ms, max {diff.max().item():.2f} ms, min {diff.min().item():.2f} ms")

torch.save(model.state_dict(), "./experimental/streamk/model.pt")
