"""
Modelo Conv1D (línea base del artículo) para reconocer 10 señas de la LSP.

Ejecutar desde la carpeta modelo/:
    python src/entrenar_conv1d.py                      # una partición (semilla 42)
    python src/entrenar_conv1d.py --semillas 1 2 3 4 5 # varias particiones -> media ± desviación

Entrada: cada seña es una secuencia de 30 fotogramas x 258 valores
  (pose 33 x (x,y,z,visibilidad) + mano izq 21 x (x,y,z) + mano der 21 x (x,y,z)).
La convolución 1D recorre solo el TIEMPO; los 258 valores entran como canales.
"""
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