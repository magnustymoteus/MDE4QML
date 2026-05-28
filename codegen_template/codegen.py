import pennylane as qml
import pennylane.numpy as np

# ============================================================
# GENERATED FROM QML METAMODEL
# Circuit  : {{#each elementsByClassName.QuantumCircuit}}{{name}}{{/each}}
# Qubits   : {{#each elementsByClassName.QuantumRegister}}{{numberOfQubits}}{{/each}}
# Bits     : {{#each elementsByClassName.ClassicRegister}}{{numberOfBits}}{{/each}}
# ============================================================

import os
import time
import numpy as onp
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
jax.config.update("jax_enable_x64", False)

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

dataset_name = "{{#each elementsByClassName.Dataset}}{{name}}{{/each}}"

features = [{{#each elementsByClassName.Dataset}}{{#each features}}"{{this}}"{{#unless @last}}, {{/unless}}{{/each}}{{/each}}]
labels = [{{#each elementsByClassName.Dataset}}{{#each labels}}"{{this}}"{{#unless @last}}, {{/unless}}{{/each}}{{/each}}]

# ------------------------------------------------------------
# QUANTUM DEVICE / EXECUTION CONFIG
# ------------------------------------------------------------

n_qubits = {{#each elementsByClassName.QuantumRegister}}{{numberOfQubits}}{{/each}}

{{#each elementsByClassName.QuantumExecutionConfig}}
execution_backend = "{{backendName}}"
execution_interface = "{{interface}}"
execution_diff_method = "{{diffMethod}}"
execution_seed = {{#if seed}}{{seed}}{{else}}42{{/if}}
{{/each}}

backend_name = os.environ.get("QML_DEVICE", execution_backend)
dev = qml.device(backend_name, wires=n_qubits)

# ------------------------------------------------------------
# PREPROCESSING
# Dataset -> PreprocessingPipeline -> DataEmbedding
# ------------------------------------------------------------

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA

def apply_preprocessing(X_train, X_test):
    """Fit preprocessing on train and transform both train and test."""

    {{#each elementsByClassName.PreprocessingPipeline}}
    # PreprocessingPipeline: {{name}}

    # --- StandardScalerStep ---
    {{#each ../elementsByClassName.StandardScalerStep}}
    scaler_{{@index}} = StandardScaler(
        with_mean={{#if withMean}}{{withMean}}{{else}}True{{/if}},
        with_std={{#if withStd}}{{withStd}}{{else}}True{{/if}}
    )
    X_train = scaler_{{@index}}.fit_transform(X_train)
    X_test = scaler_{{@index}}.transform(X_test)

    {{/each}}

    # --- MinMaxScalerStep ---
    {{#each ../elementsByClassName.MinMaxScalerStep}}
    scaler_{{@index}} = MinMaxScaler(
        feature_range={{#if featureRange}}{{featureRange}}{{else}}(0,1){{/if}}
    )
    X_train = scaler_{{@index}}.fit_transform(X_train)
    X_test = scaler_{{@index}}.transform(X_test)

    {{/each}}

    # --- RobustScalerStep ---
    {{#each ../elementsByClassName.RobustScalerStep}}
    scaler_{{@index}} = RobustScaler()
    X_train = scaler_{{@index}}.fit_transform(X_train)
    X_test = scaler_{{@index}}.transform(X_test)

    {{/each}}

    # --- NormalizerStep ---
    {{#each ../elementsByClassName.NormalizerStep}}
    normalizer_{{@index}} = Normalizer()
    X_train = normalizer_{{@index}}.fit_transform(X_train)
    X_test = normalizer_{{@index}}.transform(X_test)

    {{/each}}

    # --- PCA Step ---
    {{#each ../elementsByClassName.PCAStep}}
    pca_{{@index}} = PCA(
        n_components={{#if targetDimension}}{{targetDimension}}{{else}}{{#if nComponents}}{{nComponents}}{{else}}8{{/if}}{{/if}},
        random_state=42
    )

    X_train = pca_{{@index}}.fit_transform(X_train)
    X_test = pca_{{@index}}.transform(X_test)

    {{/each}}

    {{/each}}

    return X_train, X_test


# ------------------------------------------------------------
# DATA LOADING / SPLIT
# ------------------------------------------------------------

def load_data(csv_path=None):
    if csv_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, dataset_name + ".csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Dataset file not found: " + str(csv_path) +
            "\nCreate a CSV with columns: " + str(features + labels)
        )

    df = pd.read_csv(csv_path)

    missing = [c for c in features + labels if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing columns: " + str(missing) +
            "\nFound columns: " + str(list(df.columns))
        )

    X = df[features].values.astype(onp.float32)
    y = df[labels].values.squeeze().astype(onp.float32)
    return X, y


def prepare_data(test_size=0.25, random_state=42):
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(onp.unique(y)) > 1 else None,
    )

    X_train, X_test = apply_preprocessing(X_train, X_test)

    X_train = jnp.asarray(X_train, dtype=jnp.float32)
    X_test = jnp.asarray(X_test, dtype=jnp.float32)
    y_train = jnp.asarray(y_train, dtype=jnp.float32)
    y_test = jnp.asarray(y_test, dtype=jnp.float32)
    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------
# TRAINABLE PARAMETERS
# params[i] corresponds to the i-th TrainableParameter element
# ParameterizedGates reference these by name via param_index
# ------------------------------------------------------------

init_params = jnp.array([
{{#each elementsByClassName.TrainableParameter}}    float({{currentValue}}),  # params[{{@index}}] = {{name}}
{{/each}}], dtype=jnp.float32)

key = jax.random.PRNGKey(42)
# FIX 2: Use 0.1 scale (standard for QML ansatze) instead of 0.01.
# 0.01 caused near-zero initial rotations and vanishing gradients.
params = init_params + 0.1 * jax.random.normal(key, shape=init_params.shape, dtype=jnp.float32)

_param_list = [
{{#each elementsByClassName.TrainableParameter}}    ("{{name}}", {{@index}}){{#unless @last}},{{/unless}}
{{/each}}
]
param_index = dict(_param_list)

# ------------------------------------------------------------
# DATA EMBEDDING
# ------------------------------------------------------------

def embedding(x):
{{#each elementsByClassName.DataEmbedding}}    # {{name}}
{{#if (eq embeddingType "AngleEmbedding")}}    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
{{/if}}{{#if (eq embeddingType "AmplitudeEmbedding")}}    qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
{{/if}}{{#if (eq embeddingType "BasisEmbedding")}}    qml.BasisEmbedding(x, wires=range(n_qubits))
{{/if}}{{#if (eq embeddingType "IQPEmbedding")}}    qml.IQPEmbedding(x, wires=range(n_qubits))
{{/if}}{{/each}}

# ------------------------------------------------------------
# ANSATZ
# Strategy: iterate Layer elements in model-declared order (by index).
# For each layer, filter each operation class by layerIndex == layer.index.
# ------------------------------------------------------------

def ansatz(params):
{{#each elementsByClassName.Layer}}    # --- layer {{index}}: {{name}} ({{layerType}}) ---
{{#each ../elementsByClassName.StatePreparation}}{{#if (eq layerIndex ../index)}}{{#if (eq preparationType "ZeroState")}}    # state prep: {{name}} — |0> initialization is implicit in PennyLane
{{/if}}{{#if (eq preparationType "PlusState")}}    qml.Hadamard(wires={{targetQubits.0.index}})  # |+> state prep: {{name}}
{{/if}}{{/if}}{{/each}}{{#each ../elementsByClassName.ElementaryQuantumGate}}{{#if (eq layerIndex ../index)}}{{#if (eq gateType "Hadamard")}}    qml.Hadamard(wires={{targetQubits.0.index}})
{{/if}}{{#if (eq gateType "PauliX")}}    qml.PauliX(wires={{targetQubits.0.index}})
{{/if}}{{#if (eq gateType "PauliY")}}    qml.PauliY(wires={{targetQubits.0.index}})
{{/if}}{{#if (eq gateType "PauliZ")}}    qml.PauliZ(wires={{targetQubits.0.index}})
{{/if}}{{#if (eq gateType "S")}}    qml.S(wires={{targetQubits.0.index}})
{{/if}}{{#if (eq gateType "T")}}    qml.T(wires={{targetQubits.0.index}})
{{/if}}{{#if (eq gateType "PhaseShift")}}    qml.PhaseShift({{phi}}, wires={{targetQubits.0.index}})
{{/if}}{{#if (eq gateType "U3")}}    qml.Rot({{theta}}, {{phi}}, {{lambda}}, wires={{targetQubits.0.index}})
{{/if}}{{#if (eq gateType "CNOT")}}    qml.CNOT(wires=[{{controlQubits.0.index}}, {{targetQubits.0.index}}])
{{/if}}{{#if (eq gateType "CZ")}}    qml.CZ(wires=[{{controlQubits.0.index}}, {{targetQubits.0.index}}])
{{/if}}{{#if (eq gateType "SWAP")}}    qml.SWAP(wires=[{{targetQubits.0.index}}, {{targetQubits.1.index}}])
{{/if}}{{#if (eq gateType "Toffoli")}}    qml.Toffoli(wires=[{{controlQubits.0.index}}, {{controlQubits.1.index}}, {{targetQubits.0.index}}])
{{/if}}{{/if}}{{/each}}{{#each ../elementsByClassName.ParameterizedGate}}{{#if (eq layerIndex ../index)}}    # {{name}} — refs param: {{parameterRef.0.name}}
{{#if (eq name "RY")}}    qml.RY(params[param_index["{{parameterRef.0.name}}"]], wires={{targetQubits.0.index}})
{{/if}}{{#if (eq name "RX")}}    qml.RX(params[param_index["{{parameterRef.0.name}}"]], wires={{targetQubits.0.index}})
{{/if}}{{#if (eq name "RZ")}}    qml.RZ(params[param_index["{{parameterRef.0.name}}"]], wires={{targetQubits.0.index}})
{{/if}}{{#if (eq name "Rot")}}    qml.Rot(params[param_index["{{parameterRef.0.name}}"]], params[param_index["{{parameterRef.1.name}}"]], params[param_index["{{parameterRef.2.name}}"]], wires={{targetQubits.0.index}})
{{/if}}{{/if}}{{/each}}
{{/each}}

# ------------------------------------------------------------
# OBSERVABLES
# ------------------------------------------------------------

{{#each elementsByClassName.PauliZ}}obs_{{@index}} = qml.PauliZ(wires={{#each qubits}}{{#if @first}}{{this}}{{/if}}{{/each}})
{{/each}}{{#each elementsByClassName.PauliX}}obs_x_{{@index}} = qml.PauliX(wires={{#each qubits}}{{#if @first}}{{this}}{{/if}}{{/each}})
{{/each}}{{#each elementsByClassName.PauliY}}obs_y_{{@index}} = qml.PauliY(wires={{#each qubits}}{{#if @first}}{{this}}{{/if}}{{/each}})
{{/each}}{{#each elementsByClassName.Hermitian}}obs_herm_{{@index}} = qml.Hermitian(np.array({{matrix}}), wires={{#each qubits}}{{#if @first}}{{this}}{{/if}}{{/each}})
{{/each}}

# ------------------------------------------------------------
# HAMILTONIAN / MEASUREMENT
# ------------------------------------------------------------

{{#each elementsByClassName.Hamiltonian}}# Hamiltonian: {{name}}  coefficients: {{#each coefficients}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}
{{/each}}

# FIX 3: circuit signature is (params, x) so it is consistent with
# loss(params, x, y) and batched_circuit vmap(in_axes=(None, 0)).
# The old (x, params) order caused loss() to pass arguments in the
# wrong slots when called via jax.vmap(loss, in_axes=(None, 0, 0)).
@qml.qnode(dev, interface=execution_interface, diff_method=execution_diff_method)
def circuit(params, x):
    embedding(x)
    ansatz(params)
    measurements = []
{{#each elementsByClassName.PauliZ}}
    measurements.append(qml.expval(qml.PauliZ(wires={{#each qubits}}{{#if @first}}{{this}}{{/if}}{{/each}})))
{{/each}}
{{#each elementsByClassName.PauliX}}
    measurements.append(qml.expval(qml.PauliX(wires={{#each qubits}}{{#if @first}}{{this}}{{/if}}{{/each}})))
{{/each}}
{{#each elementsByClassName.PauliY}}
    measurements.append(qml.expval(qml.PauliY(wires={{#each qubits}}{{#if @first}}{{this}}{{/if}}{{/each}})))
{{/each}}
{{#each elementsByClassName.Hermitian}}
    measurements.append(qml.expval(obs_herm_{{@index}}))
{{/each}}
    return qml.math.stack(measurements)

# params is constant across the batch; x is batched over axis 0
batched_circuit = jax.vmap(circuit, in_axes=(None, 0))

# ------------------------------------------------------------
# OUTPUT MAPPING / POST-PROCESSING
# ------------------------------------------------------------

def raw_to_probability(raw):
    prob = raw
{{#each elementsByClassName.Prediction}}
{{#if outputScale}}    prob = prob * {{outputScale}}
{{/if}}{{#if outputShift}}    prob = prob + {{outputShift}}
{{/if}}{{#if (eq postprocessingMethod "Sigmoid")}}    prob = 1 / (1 + jnp.exp(-prob))
{{/if}}{{#if (eq postprocessingMethod "Softmax")}}    prob = jax.nn.softmax(prob)
{{/if}}{{/each}}
    return prob

def postprocess(prob):
{{#each elementsByClassName.Prediction}}
{{#if (eq postprocessingMethod "Threshold")}}    return (prob > {{threshold}}).astype(jnp.int32)
{{/if}}{{#if (eq postprocessingMethod "Identity")}}    return prob
{{/if}}{{#if (eq postprocessingMethod "Sigmoid")}}    return prob
{{/if}}{{#if (eq postprocessingMethod "Softmax")}}    return prob
{{/if}}{{/each}}
    return prob

# ------------------------------------------------------------
# LOSS FUNCTION
# ------------------------------------------------------------

def loss(params, x, y):
    raw = circuit(params, x)
    prob = raw_to_probability(raw)
    prob = jnp.clip(prob, 1e-8, 1.0 - 1e-8)
{{#each elementsByClassName.LossFunction}}    # {{name}}
{{#if (eq function "MSE")}}    return jnp.mean((prob - y) ** 2)
{{/if}}{{#if (eq function "MAE")}}    return jnp.mean(jnp.abs(prob - y))
{{/if}}{{#if (eq function "BinaryCrossEntropy")}}{{#if positiveClassWeight}}    pos_weight = {{positiveClassWeight}}
{{else}}    pos_weight = 1.0
{{/if}}    return -jnp.mean(
        pos_weight * y * jnp.log(prob)
        + (1.0 - y) * jnp.log(1.0 - prob)
    )
{{/if}}{{#if (eq function "HingeLoss")}}    return jnp.mean(jnp.maximum(0, 1.0 - y * prob))
{{/if}}{{/each}}

# ------------------------------------------------------------
# BATCH LOSS
# ------------------------------------------------------------

def batch_loss(params, X, y):
    per_example_losses = jax.vmap(loss, in_axes=(None, 0, 0))(params, X, y)
    return jnp.mean(per_example_losses)

# ------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------

{{#each elementsByClassName.Optimizer}}# Optimizer: {{name}}
{{#if (eq optimizerType "GradientDescent")}}optimizer = optax.sgd(learning_rate={{learningRate}})
{{/if}}{{#if (eq optimizerType "Adam")}}optimizer = optax.adam(learning_rate={{learningRate}})
{{/if}}{{#if (eq optimizerType "Adagrad")}}optimizer = optax.adagrad(learning_rate={{learningRate}})
{{/if}}{{#if (eq optimizerType "RMSProp")}}optimizer = optax.rmsprop(learning_rate={{learningRate}})
{{/if}}{{#if (eq optimizerType "Momentum")}}optimizer = optax.sgd(learning_rate={{learningRate}}, momentum={{momentum}})
{{/if}}
max_iterations = {{maxIterations}}
batch_size = {{#if batchSize}}{{batchSize}}{{else}}32{{/if}}
{{/each}}

opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, X_batch, y_batch):
    loss_value, grads = jax.value_and_grad(batch_loss)(params, X_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

# ------------------------------------------------------------
# TRAINING LOOP
# FIX 1: max_iterations is treated as epochs (full passes over the
# training data), not as raw gradient steps. The old code drew a
# single random mini-batch per iteration, giving ~44x fewer gradient
# updates for the same max_iterations value and causing the loss to
# plateau around 1.5. An epoch loop with batch_size-sized windows is
# the correct general semantic for any QML training configuration.
# ------------------------------------------------------------

def train(X_train, y_train):
    global params, opt_state

    rng = onp.random.default_rng(execution_seed)
    n_samples = len(X_train)

    print("\\nStarting training on", backend_name, "\\n")
    start = time.time()

    for epoch in range(max_iterations):
        # Shuffle all training data at the start of each epoch
        idx = rng.permutation(n_samples)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuf[i : i + batch_size]
            y_batch = y_shuf[i : i + batch_size]

            params, opt_state, current_loss = train_step(params, opt_state, X_batch, y_batch)

            epoch_loss += float(current_loss)
            n_batches += 1

        print(
            "Epoch "
            + str(epoch + 1)
            + "/"
            + str(max_iterations)
            + "  loss="
            + str(round(epoch_loss / max(n_batches, 1), 6))
        )

    print(f"\\nTraining finished in {time.time() - start:.1f} seconds")
    return params

# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

def predict(X, trained_params):
    X = jnp.asarray(X, dtype=jnp.float32)

    if X.ndim == 1:
        raw = circuit(trained_params, X)
        prob = raw_to_probability(raw)
        return onp.asarray(postprocess(prob))

    raw = batched_circuit(trained_params, X)
    prob = raw_to_probability(raw)
    return onp.asarray(postprocess(prob))

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=== QML Workflow ===")
    print("Dataset : " + str(dataset_name))
    print("Features: " + str(features))
    print("Labels  : " + str(labels))
    print("Qubits  : " + str(n_qubits))
    print("Params  : " + str(params))

    X_train, X_test, y_train, y_test = prepare_data()
    print(
        "Samples : "
        + str(len(X_train) + len(X_test))
        + " total  ("
        + str(len(X_train))
        + " train / "
        + str(len(X_test))
        + " test)"
    )

    trained_params = train(X_train, y_train)

    print("\\nTrained parameters: " + str(trained_params))

    results = predict(X_test, trained_params)
    print("\\nTest inference results (first 10 samples):")
    preview_n = min(10, len(results))
    for i in range(preview_n):
        res = onp.asarray(results[i])
        if res.shape == ():
            pred_str = str(round(float(res), 4))
        else:
            pred_str = onp.array2string(res, precision=4, separator=", ")
        true_lbl = y_test[i]
        true_lbl = int(true_lbl) if onp.asarray(true_lbl).shape == () else true_lbl
        print("  sample " + str(i) + ": prediction=" + str(pred_str) + "  true_label=" + str(true_lbl))
    if len(results) > preview_n:
        print("  ... " + str(len(results) - preview_n) + " more samples omitted")
