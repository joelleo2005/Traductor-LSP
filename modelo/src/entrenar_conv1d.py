
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from comun import ejecutar


def modelo_conv1d(n_clases):
    return keras.Sequential([
        keras.Input(shape=(30, 258)),
        layers.Conv1D(64, 3, activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_clases, activation="softmax"),
    ], name="conv1d")


def preparar_entrada(X):
    """La Conv1D usa la secuencia tal cual: (N, 30, 258)."""
    return X.astype("float32")


if __name__ == "__main__":
    ejecutar("conv1d", modelo_conv1d, preparar_entrada)