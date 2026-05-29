import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

import time
import numpy as onp
import pandas as pd
import pennylane as qml
import jax
import jax.numpy as jnp
import optax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("JAX devices:", jax.devices())

# ========================== DATA ==========================
df = pd.read_csv("./creditcard_truncated_15000.csv")
X = df.drop(columns=["Class"]).values.astype(onp.float32)
y = df["Class"].values.astype(onp.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale first
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_features = 8
pca = PCA(n_components=n_features, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

# ========================== QUANTUM MODEL ==========================
n_qubits = n_features
n_layers = 5

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="jax", diff_method="backprop")
def circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

batched_circuit = jax.vmap(circuit, in_axes=(None, 0))

@jax.jit
def cost(weights, X_batch, y_batch, fraud_weight=2.0):
    raw = batched_circuit(weights, X_batch)
    preds = (raw + 1.0) / 2.0
    preds = jnp.clip(preds, 1e-7, 1.0 - 1e-7)

    bce = -(
        fraud_weight * y_batch * jnp.log(preds)
        + (1.0 - y_batch) * jnp.log(1.0 - preds)
    )
    return jnp.mean(bce)

cost_and_grad = jax.jit(jax.value_and_grad(cost))

# ========================== TRAINING ==========================
key = jax.random.PRNGKey(42)
weights = 0.1 * jax.random.normal(key, shape=(n_layers, n_qubits, 3))

batch_size = 256
epochs = 30

optimizer = optax.adam(learning_rate=0.005)
opt_state = optimizer.init(weights)

print("\nStarting training on", jax.devices()[0], "\n")
start = time.time()

for epoch in range(epochs):
    idx = onp.random.permutation(len(X_train))
    X_shuf = X_train[idx]
    y_shuf = y_train[idx]

    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, len(X_train), batch_size):
        Xb = X_shuf[i : i + batch_size]
        yb = y_shuf[i : i + batch_size]

        loss, grads = cost_and_grad(weights, Xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)

        epoch_loss += float(loss)
        n_batches += 1

    print(f"Epoch {epoch:2d} | Loss: {epoch_loss / max(n_batches, 1):.4f}")

print(f"\nTraining finished in {time.time() - start:.1f} seconds")

# ========================== EVALUATION ==========================
@jax.jit
def predict(X, weights, threshold):
    raw = batched_circuit(weights, X)
    probs = (raw + 1.0) / 2.0
    return (probs > threshold).astype(jnp.int32)

threshold = 0.4
y_pred = predict(X_test, weights, threshold=threshold)

accuracy = float(jnp.mean(y_pred == y_test))
tp = jnp.sum((y_pred == 1) & (y_test == 1))
fp = jnp.sum((y_pred == 1) & (y_test == 0))
fn = jnp.sum((y_pred == 0) & (y_test == 1))

recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Accuracy:     {accuracy:.4f}")
print(f"Fraud Recall: {recall:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"F1 Score:     {f1:.4f}")
print(f"Fraud in test: {int(jnp.sum(y_test))}/{len(y_test)}")
