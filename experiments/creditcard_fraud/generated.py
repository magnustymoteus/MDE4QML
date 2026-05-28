import pennylane as qml
import pennylane.numpy as np

# ============================================================
# GENERATED FROM QML METAMODEL
# Circuit  : creditcard_fraud_classifier_qc
# Qubits   : 8
# Bits     :
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

dataset_name = "creditcard_truncated_15000"

features = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
labels = ["Class"]

# ------------------------------------------------------------
# QUANTUM DEVICE / EXECUTION CONFIG
# ------------------------------------------------------------

n_qubits = 8

execution_backend = "default.qubit"
execution_interface = "jax"
execution_diff_method = "backprop"
execution_seed = 42

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

    # PreprocessingPipeline: preprocessing_pipeline

    # --- StandardScalerStep ---
    scaler_0 = StandardScaler(
        with_mean=True,
        with_std=True
    )
    X_train = scaler_0.fit_transform(X_train)
    X_test = scaler_0.transform(X_test)


    # --- MinMaxScalerStep ---

    # --- RobustScalerStep ---

    # --- NormalizerStep ---

    # --- PCA Step ---
    pca_0 = PCA(
        n_components=8,
        random_state=42
    )

    X_train = pca_0.fit_transform(X_train)
    X_test = pca_0.transform(X_test)



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
    float(0),  # params[0] = theta_l0_q0_0
    float(0),  # params[1] = theta_l0_q0_1
    float(0),  # params[2] = theta_l0_q0_2
    float(0),  # params[3] = theta_l0_q1_0
    float(0),  # params[4] = theta_l0_q1_1
    float(0),  # params[5] = theta_l0_q1_2
    float(0),  # params[6] = theta_l0_q2_0
    float(0),  # params[7] = theta_l0_q2_1
    float(0),  # params[8] = theta_l0_q2_2
    float(0),  # params[9] = theta_l0_q3_0
    float(0),  # params[10] = theta_l0_q3_1
    float(0),  # params[11] = theta_l0_q3_2
    float(0),  # params[12] = theta_l0_q4_0
    float(0),  # params[13] = theta_l0_q4_1
    float(0),  # params[14] = theta_l0_q4_2
    float(0),  # params[15] = theta_l0_q5_0
    float(0),  # params[16] = theta_l0_q5_1
    float(0),  # params[17] = theta_l0_q5_2
    float(0),  # params[18] = theta_l0_q6_0
    float(0),  # params[19] = theta_l0_q6_1
    float(0),  # params[20] = theta_l0_q6_2
    float(0),  # params[21] = theta_l0_q7_0
    float(0),  # params[22] = theta_l0_q7_1
    float(0),  # params[23] = theta_l0_q7_2
    float(0),  # params[24] = theta_l1_q0_0
    float(0),  # params[25] = theta_l1_q0_1
    float(0),  # params[26] = theta_l1_q0_2
    float(0),  # params[27] = theta_l1_q1_0
    float(0),  # params[28] = theta_l1_q1_1
    float(0),  # params[29] = theta_l1_q1_2
    float(0),  # params[30] = theta_l1_q2_0
    float(0),  # params[31] = theta_l1_q2_1
    float(0),  # params[32] = theta_l1_q2_2
    float(0),  # params[33] = theta_l1_q3_0
    float(0),  # params[34] = theta_l1_q3_1
    float(0),  # params[35] = theta_l1_q3_2
    float(0),  # params[36] = theta_l1_q4_0
    float(0),  # params[37] = theta_l1_q4_1
    float(0),  # params[38] = theta_l1_q4_2
    float(0),  # params[39] = theta_l1_q5_0
    float(0),  # params[40] = theta_l1_q5_1
    float(0),  # params[41] = theta_l1_q5_2
    float(0),  # params[42] = theta_l1_q6_0
    float(0),  # params[43] = theta_l1_q6_1
    float(0),  # params[44] = theta_l1_q6_2
    float(0),  # params[45] = theta_l1_q7_0
    float(0),  # params[46] = theta_l1_q7_1
    float(0),  # params[47] = theta_l1_q7_2
    float(0),  # params[48] = theta_l2_q0_0
    float(0),  # params[49] = theta_l2_q0_1
    float(0),  # params[50] = theta_l2_q0_2
    float(0),  # params[51] = theta_l2_q1_0
    float(0),  # params[52] = theta_l2_q1_1
    float(0),  # params[53] = theta_l2_q1_2
    float(0),  # params[54] = theta_l2_q2_0
    float(0),  # params[55] = theta_l2_q2_1
    float(0),  # params[56] = theta_l2_q2_2
    float(0),  # params[57] = theta_l2_q3_0
    float(0),  # params[58] = theta_l2_q3_1
    float(0),  # params[59] = theta_l2_q3_2
    float(0),  # params[60] = theta_l2_q4_0
    float(0),  # params[61] = theta_l2_q4_1
    float(0),  # params[62] = theta_l2_q4_2
    float(0),  # params[63] = theta_l2_q5_0
    float(0),  # params[64] = theta_l2_q5_1
    float(0),  # params[65] = theta_l2_q5_2
    float(0),  # params[66] = theta_l2_q6_0
    float(0),  # params[67] = theta_l2_q6_1
    float(0),  # params[68] = theta_l2_q6_2
    float(0),  # params[69] = theta_l2_q7_0
    float(0),  # params[70] = theta_l2_q7_1
    float(0),  # params[71] = theta_l2_q7_2
    float(0),  # params[72] = theta_l3_q0_0
    float(0),  # params[73] = theta_l3_q0_1
    float(0),  # params[74] = theta_l3_q0_2
    float(0),  # params[75] = theta_l3_q1_0
    float(0),  # params[76] = theta_l3_q1_1
    float(0),  # params[77] = theta_l3_q1_2
    float(0),  # params[78] = theta_l3_q2_0
    float(0),  # params[79] = theta_l3_q2_1
    float(0),  # params[80] = theta_l3_q2_2
    float(0),  # params[81] = theta_l3_q3_0
    float(0),  # params[82] = theta_l3_q3_1
    float(0),  # params[83] = theta_l3_q3_2
    float(0),  # params[84] = theta_l3_q4_0
    float(0),  # params[85] = theta_l3_q4_1
    float(0),  # params[86] = theta_l3_q4_2
    float(0),  # params[87] = theta_l3_q5_0
    float(0),  # params[88] = theta_l3_q5_1
    float(0),  # params[89] = theta_l3_q5_2
    float(0),  # params[90] = theta_l3_q6_0
    float(0),  # params[91] = theta_l3_q6_1
    float(0),  # params[92] = theta_l3_q6_2
    float(0),  # params[93] = theta_l3_q7_0
    float(0),  # params[94] = theta_l3_q7_1
    float(0),  # params[95] = theta_l3_q7_2
    float(0),  # params[96] = theta_l4_q0_0
    float(0),  # params[97] = theta_l4_q0_1
    float(0),  # params[98] = theta_l4_q0_2
    float(0),  # params[99] = theta_l4_q1_0
    float(0),  # params[100] = theta_l4_q1_1
    float(0),  # params[101] = theta_l4_q1_2
    float(0),  # params[102] = theta_l4_q2_0
    float(0),  # params[103] = theta_l4_q2_1
    float(0),  # params[104] = theta_l4_q2_2
    float(0),  # params[105] = theta_l4_q3_0
    float(0),  # params[106] = theta_l4_q3_1
    float(0),  # params[107] = theta_l4_q3_2
    float(0),  # params[108] = theta_l4_q4_0
    float(0),  # params[109] = theta_l4_q4_1
    float(0),  # params[110] = theta_l4_q4_2
    float(0),  # params[111] = theta_l4_q5_0
    float(0),  # params[112] = theta_l4_q5_1
    float(0),  # params[113] = theta_l4_q5_2
    float(0),  # params[114] = theta_l4_q6_0
    float(0),  # params[115] = theta_l4_q6_1
    float(0),  # params[116] = theta_l4_q6_2
    float(0),  # params[117] = theta_l4_q7_0
    float(0),  # params[118] = theta_l4_q7_1
    float(0),  # params[119] = theta_l4_q7_2
], dtype=jnp.float32)

key = jax.random.PRNGKey(42)
# FIX 2: Use 0.1 scale (standard for QML ansatze) instead of 0.01.
# 0.01 caused near-zero initial rotations and vanishing gradients.
params = init_params + 0.1 * jax.random.normal(key, shape=init_params.shape, dtype=jnp.float32)

_param_list = [
    ("theta_l0_q0_0", 0),
    ("theta_l0_q0_1", 1),
    ("theta_l0_q0_2", 2),
    ("theta_l0_q1_0", 3),
    ("theta_l0_q1_1", 4),
    ("theta_l0_q1_2", 5),
    ("theta_l0_q2_0", 6),
    ("theta_l0_q2_1", 7),
    ("theta_l0_q2_2", 8),
    ("theta_l0_q3_0", 9),
    ("theta_l0_q3_1", 10),
    ("theta_l0_q3_2", 11),
    ("theta_l0_q4_0", 12),
    ("theta_l0_q4_1", 13),
    ("theta_l0_q4_2", 14),
    ("theta_l0_q5_0", 15),
    ("theta_l0_q5_1", 16),
    ("theta_l0_q5_2", 17),
    ("theta_l0_q6_0", 18),
    ("theta_l0_q6_1", 19),
    ("theta_l0_q6_2", 20),
    ("theta_l0_q7_0", 21),
    ("theta_l0_q7_1", 22),
    ("theta_l0_q7_2", 23),
    ("theta_l1_q0_0", 24),
    ("theta_l1_q0_1", 25),
    ("theta_l1_q0_2", 26),
    ("theta_l1_q1_0", 27),
    ("theta_l1_q1_1", 28),
    ("theta_l1_q1_2", 29),
    ("theta_l1_q2_0", 30),
    ("theta_l1_q2_1", 31),
    ("theta_l1_q2_2", 32),
    ("theta_l1_q3_0", 33),
    ("theta_l1_q3_1", 34),
    ("theta_l1_q3_2", 35),
    ("theta_l1_q4_0", 36),
    ("theta_l1_q4_1", 37),
    ("theta_l1_q4_2", 38),
    ("theta_l1_q5_0", 39),
    ("theta_l1_q5_1", 40),
    ("theta_l1_q5_2", 41),
    ("theta_l1_q6_0", 42),
    ("theta_l1_q6_1", 43),
    ("theta_l1_q6_2", 44),
    ("theta_l1_q7_0", 45),
    ("theta_l1_q7_1", 46),
    ("theta_l1_q7_2", 47),
    ("theta_l2_q0_0", 48),
    ("theta_l2_q0_1", 49),
    ("theta_l2_q0_2", 50),
    ("theta_l2_q1_0", 51),
    ("theta_l2_q1_1", 52),
    ("theta_l2_q1_2", 53),
    ("theta_l2_q2_0", 54),
    ("theta_l2_q2_1", 55),
    ("theta_l2_q2_2", 56),
    ("theta_l2_q3_0", 57),
    ("theta_l2_q3_1", 58),
    ("theta_l2_q3_2", 59),
    ("theta_l2_q4_0", 60),
    ("theta_l2_q4_1", 61),
    ("theta_l2_q4_2", 62),
    ("theta_l2_q5_0", 63),
    ("theta_l2_q5_1", 64),
    ("theta_l2_q5_2", 65),
    ("theta_l2_q6_0", 66),
    ("theta_l2_q6_1", 67),
    ("theta_l2_q6_2", 68),
    ("theta_l2_q7_0", 69),
    ("theta_l2_q7_1", 70),
    ("theta_l2_q7_2", 71),
    ("theta_l3_q0_0", 72),
    ("theta_l3_q0_1", 73),
    ("theta_l3_q0_2", 74),
    ("theta_l3_q1_0", 75),
    ("theta_l3_q1_1", 76),
    ("theta_l3_q1_2", 77),
    ("theta_l3_q2_0", 78),
    ("theta_l3_q2_1", 79),
    ("theta_l3_q2_2", 80),
    ("theta_l3_q3_0", 81),
    ("theta_l3_q3_1", 82),
    ("theta_l3_q3_2", 83),
    ("theta_l3_q4_0", 84),
    ("theta_l3_q4_1", 85),
    ("theta_l3_q4_2", 86),
    ("theta_l3_q5_0", 87),
    ("theta_l3_q5_1", 88),
    ("theta_l3_q5_2", 89),
    ("theta_l3_q6_0", 90),
    ("theta_l3_q6_1", 91),
    ("theta_l3_q6_2", 92),
    ("theta_l3_q7_0", 93),
    ("theta_l3_q7_1", 94),
    ("theta_l3_q7_2", 95),
    ("theta_l4_q0_0", 96),
    ("theta_l4_q0_1", 97),
    ("theta_l4_q0_2", 98),
    ("theta_l4_q1_0", 99),
    ("theta_l4_q1_1", 100),
    ("theta_l4_q1_2", 101),
    ("theta_l4_q2_0", 102),
    ("theta_l4_q2_1", 103),
    ("theta_l4_q2_2", 104),
    ("theta_l4_q3_0", 105),
    ("theta_l4_q3_1", 106),
    ("theta_l4_q3_2", 107),
    ("theta_l4_q4_0", 108),
    ("theta_l4_q4_1", 109),
    ("theta_l4_q4_2", 110),
    ("theta_l4_q5_0", 111),
    ("theta_l4_q5_1", 112),
    ("theta_l4_q5_2", 113),
    ("theta_l4_q6_0", 114),
    ("theta_l4_q6_1", 115),
    ("theta_l4_q6_2", 116),
    ("theta_l4_q7_0", 117),
    ("theta_l4_q7_1", 118),
    ("theta_l4_q7_2", 119)
]
param_index = dict(_param_list)

# ------------------------------------------------------------
# DATA EMBEDDING
# ------------------------------------------------------------

def embedding(x):
    # angle_embedding
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")


# ------------------------------------------------------------
# ANSATZ
# Strategy: iterate Layer elements in model-declared order (by index).
# For each layer, filter each operation class by layerIndex == layer.index.
# ------------------------------------------------------------

def ansatz(params):
    # --- layer 0: trainable_layer_0 (Trainable) ---
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 5])
    qml.CNOT(wires=[5, 6])
    qml.CNOT(wires=[6, 7])
    qml.CNOT(wires=[7, 0])
    # Rot — refs param: theta_l0_q0_0
    qml.Rot(params[param_index["theta_l0_q0_0"]], params[param_index["theta_l0_q0_1"]], params[param_index["theta_l0_q0_2"]], wires=0)
    # Rot — refs param: theta_l0_q1_0
    qml.Rot(params[param_index["theta_l0_q1_0"]], params[param_index["theta_l0_q1_1"]], params[param_index["theta_l0_q1_2"]], wires=1)
    # Rot — refs param: theta_l0_q2_0
    qml.Rot(params[param_index["theta_l0_q2_0"]], params[param_index["theta_l0_q2_1"]], params[param_index["theta_l0_q2_2"]], wires=2)
    # Rot — refs param: theta_l0_q3_0
    qml.Rot(params[param_index["theta_l0_q3_0"]], params[param_index["theta_l0_q3_1"]], params[param_index["theta_l0_q3_2"]], wires=3)
    # Rot — refs param: theta_l0_q4_0
    qml.Rot(params[param_index["theta_l0_q4_0"]], params[param_index["theta_l0_q4_1"]], params[param_index["theta_l0_q4_2"]], wires=4)
    # Rot — refs param: theta_l0_q5_0
    qml.Rot(params[param_index["theta_l0_q5_0"]], params[param_index["theta_l0_q5_1"]], params[param_index["theta_l0_q5_2"]], wires=5)
    # Rot — refs param: theta_l0_q6_0
    qml.Rot(params[param_index["theta_l0_q6_0"]], params[param_index["theta_l0_q6_1"]], params[param_index["theta_l0_q6_2"]], wires=6)
    # Rot — refs param: theta_l0_q7_0
    qml.Rot(params[param_index["theta_l0_q7_0"]], params[param_index["theta_l0_q7_1"]], params[param_index["theta_l0_q7_2"]], wires=7)

    # --- layer 1: trainable_layer_1 (Trainable) ---
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 5])
    qml.CNOT(wires=[5, 6])
    qml.CNOT(wires=[6, 7])
    qml.CNOT(wires=[7, 0])
    # Rot — refs param: theta_l1_q0_0
    qml.Rot(params[param_index["theta_l1_q0_0"]], params[param_index["theta_l1_q0_1"]], params[param_index["theta_l1_q0_2"]], wires=0)
    # Rot — refs param: theta_l1_q1_0
    qml.Rot(params[param_index["theta_l1_q1_0"]], params[param_index["theta_l1_q1_1"]], params[param_index["theta_l1_q1_2"]], wires=1)
    # Rot — refs param: theta_l1_q2_0
    qml.Rot(params[param_index["theta_l1_q2_0"]], params[param_index["theta_l1_q2_1"]], params[param_index["theta_l1_q2_2"]], wires=2)
    # Rot — refs param: theta_l1_q3_0
    qml.Rot(params[param_index["theta_l1_q3_0"]], params[param_index["theta_l1_q3_1"]], params[param_index["theta_l1_q3_2"]], wires=3)
    # Rot — refs param: theta_l1_q4_0
    qml.Rot(params[param_index["theta_l1_q4_0"]], params[param_index["theta_l1_q4_1"]], params[param_index["theta_l1_q4_2"]], wires=4)
    # Rot — refs param: theta_l1_q5_0
    qml.Rot(params[param_index["theta_l1_q5_0"]], params[param_index["theta_l1_q5_1"]], params[param_index["theta_l1_q5_2"]], wires=5)
    # Rot — refs param: theta_l1_q6_0
    qml.Rot(params[param_index["theta_l1_q6_0"]], params[param_index["theta_l1_q6_1"]], params[param_index["theta_l1_q6_2"]], wires=6)
    # Rot — refs param: theta_l1_q7_0
    qml.Rot(params[param_index["theta_l1_q7_0"]], params[param_index["theta_l1_q7_1"]], params[param_index["theta_l1_q7_2"]], wires=7)

    # --- layer 2: trainable_layer_2 (Trainable) ---
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 5])
    qml.CNOT(wires=[5, 6])
    qml.CNOT(wires=[6, 7])
    qml.CNOT(wires=[7, 0])
    # Rot — refs param: theta_l2_q0_0
    qml.Rot(params[param_index["theta_l2_q0_0"]], params[param_index["theta_l2_q0_1"]], params[param_index["theta_l2_q0_2"]], wires=0)
    # Rot — refs param: theta_l2_q1_0
    qml.Rot(params[param_index["theta_l2_q1_0"]], params[param_index["theta_l2_q1_1"]], params[param_index["theta_l2_q1_2"]], wires=1)
    # Rot — refs param: theta_l2_q2_0
    qml.Rot(params[param_index["theta_l2_q2_0"]], params[param_index["theta_l2_q2_1"]], params[param_index["theta_l2_q2_2"]], wires=2)
    # Rot — refs param: theta_l2_q3_0
    qml.Rot(params[param_index["theta_l2_q3_0"]], params[param_index["theta_l2_q3_1"]], params[param_index["theta_l2_q3_2"]], wires=3)
    # Rot — refs param: theta_l2_q4_0
    qml.Rot(params[param_index["theta_l2_q4_0"]], params[param_index["theta_l2_q4_1"]], params[param_index["theta_l2_q4_2"]], wires=4)
    # Rot — refs param: theta_l2_q5_0
    qml.Rot(params[param_index["theta_l2_q5_0"]], params[param_index["theta_l2_q5_1"]], params[param_index["theta_l2_q5_2"]], wires=5)
    # Rot — refs param: theta_l2_q6_0
    qml.Rot(params[param_index["theta_l2_q6_0"]], params[param_index["theta_l2_q6_1"]], params[param_index["theta_l2_q6_2"]], wires=6)
    # Rot — refs param: theta_l2_q7_0
    qml.Rot(params[param_index["theta_l2_q7_0"]], params[param_index["theta_l2_q7_1"]], params[param_index["theta_l2_q7_2"]], wires=7)

    # --- layer 3: trainable_layer_3 (Trainable) ---
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 5])
    qml.CNOT(wires=[5, 6])
    qml.CNOT(wires=[6, 7])
    qml.CNOT(wires=[7, 0])
    # Rot — refs param: theta_l3_q0_0
    qml.Rot(params[param_index["theta_l3_q0_0"]], params[param_index["theta_l3_q0_1"]], params[param_index["theta_l3_q0_2"]], wires=0)
    # Rot — refs param: theta_l3_q1_0
    qml.Rot(params[param_index["theta_l3_q1_0"]], params[param_index["theta_l3_q1_1"]], params[param_index["theta_l3_q1_2"]], wires=1)
    # Rot — refs param: theta_l3_q2_0
    qml.Rot(params[param_index["theta_l3_q2_0"]], params[param_index["theta_l3_q2_1"]], params[param_index["theta_l3_q2_2"]], wires=2)
    # Rot — refs param: theta_l3_q3_0
    qml.Rot(params[param_index["theta_l3_q3_0"]], params[param_index["theta_l3_q3_1"]], params[param_index["theta_l3_q3_2"]], wires=3)
    # Rot — refs param: theta_l3_q4_0
    qml.Rot(params[param_index["theta_l3_q4_0"]], params[param_index["theta_l3_q4_1"]], params[param_index["theta_l3_q4_2"]], wires=4)
    # Rot — refs param: theta_l3_q5_0
    qml.Rot(params[param_index["theta_l3_q5_0"]], params[param_index["theta_l3_q5_1"]], params[param_index["theta_l3_q5_2"]], wires=5)
    # Rot — refs param: theta_l3_q6_0
    qml.Rot(params[param_index["theta_l3_q6_0"]], params[param_index["theta_l3_q6_1"]], params[param_index["theta_l3_q6_2"]], wires=6)
    # Rot — refs param: theta_l3_q7_0
    qml.Rot(params[param_index["theta_l3_q7_0"]], params[param_index["theta_l3_q7_1"]], params[param_index["theta_l3_q7_2"]], wires=7)

    # --- layer 4: trainable_layer_4 (Trainable) ---
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 5])
    qml.CNOT(wires=[5, 6])
    qml.CNOT(wires=[6, 7])
    qml.CNOT(wires=[7, 0])
    # Rot — refs param: theta_l4_q0_0
    qml.Rot(params[param_index["theta_l4_q0_0"]], params[param_index["theta_l4_q0_1"]], params[param_index["theta_l4_q0_2"]], wires=0)
    # Rot — refs param: theta_l4_q1_0
    qml.Rot(params[param_index["theta_l4_q1_0"]], params[param_index["theta_l4_q1_1"]], params[param_index["theta_l4_q1_2"]], wires=1)
    # Rot — refs param: theta_l4_q2_0
    qml.Rot(params[param_index["theta_l4_q2_0"]], params[param_index["theta_l4_q2_1"]], params[param_index["theta_l4_q2_2"]], wires=2)
    # Rot — refs param: theta_l4_q3_0
    qml.Rot(params[param_index["theta_l4_q3_0"]], params[param_index["theta_l4_q3_1"]], params[param_index["theta_l4_q3_2"]], wires=3)
    # Rot — refs param: theta_l4_q4_0
    qml.Rot(params[param_index["theta_l4_q4_0"]], params[param_index["theta_l4_q4_1"]], params[param_index["theta_l4_q4_2"]], wires=4)
    # Rot — refs param: theta_l4_q5_0
    qml.Rot(params[param_index["theta_l4_q5_0"]], params[param_index["theta_l4_q5_1"]], params[param_index["theta_l4_q5_2"]], wires=5)
    # Rot — refs param: theta_l4_q6_0
    qml.Rot(params[param_index["theta_l4_q6_0"]], params[param_index["theta_l4_q6_1"]], params[param_index["theta_l4_q6_2"]], wires=6)
    # Rot — refs param: theta_l4_q7_0
    qml.Rot(params[param_index["theta_l4_q7_0"]], params[param_index["theta_l4_q7_1"]], params[param_index["theta_l4_q7_2"]], wires=7)


# ------------------------------------------------------------
# OBSERVABLES
# ------------------------------------------------------------

obs_0 = qml.PauliZ(wires=0)

# ------------------------------------------------------------
# HAMILTONIAN / MEASUREMENT
# ------------------------------------------------------------

# Hamiltonian: z0_hamiltonian  coefficients: 1

# FIX 3: circuit signature is (params, x) so it is consistent with
# loss(params, x, y) and batched_circuit vmap(in_axes=(None, 0)).
# The old (x, params) order caused loss() to pass arguments in the
# wrong slots when called via jax.vmap(loss, in_axes=(None, 0, 0)).
@qml.qnode(dev, interface=execution_interface, diff_method=execution_diff_method)
def circuit(params, x):
    embedding(x)
    ansatz(params)
    measurements = []
    measurements.append(qml.expval(qml.PauliZ(wires=0)))
    return qml.math.stack(measurements)

# params is constant across the batch; x is batched over axis 0
batched_circuit = jax.vmap(circuit, in_axes=(None, 0))

# ------------------------------------------------------------
# OUTPUT MAPPING / POST-PROCESSING
# ------------------------------------------------------------

def raw_to_probability(raw):
    prob = raw
    prob = prob * 0.5
    prob = prob + 0.5

    return prob

def postprocess(prob):
    return (prob > 0.4).astype(jnp.int32)

    return prob

# ------------------------------------------------------------
# LOSS FUNCTION
# ------------------------------------------------------------

def loss(params, x, y):
    raw = circuit(params, x)
    prob = raw_to_probability(raw)
    prob = jnp.clip(prob, 1e-8, 1.0 - 1e-8)
    # weighted_bce
    pos_weight = 2
    return -jnp.mean(
        pos_weight * y * jnp.log(prob)
        + (1.0 - y) * jnp.log(1.0 - prob)
    )


# ------------------------------------------------------------
# BATCH LOSS
# ------------------------------------------------------------

def batch_loss(params, X, y):
    per_example_losses = jax.vmap(loss, in_axes=(None, 0, 0))(params, X, y)
    return jnp.mean(per_example_losses)

# ------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------

# Optimizer: adam
optimizer = optax.adam(learning_rate=0.005)
max_iterations = 30
batch_size = 256

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
