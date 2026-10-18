"""
Modelo BASE: red convolucional 1D (Conv1D).

Trata cada seña como una serie de tiempo: 30 fotogramas x 258 valores por fotograma.
Las convoluciones se deslizan solo en el eje del tiempo y en cada paso miran los 258 valores a la vez.

Ejecutar desde la carpeta modelo/:
    python src/entrenar_conv1d.py --semillas 1 2 3 4 5

La carga de datos, la partición, el aumento, el entrenamiento, la evaluación y los CSV de resultados
están en comun.py (son los mismos para Conv1D y Conv2D).
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from comun import ejecutar


def modelo_conv1d(n_clases):
    """Construye la Conv1D (≈ 84 mil parámetros). Entrada (30, 258) -> probabilidad de cada una de las n_clases."""
    return keras.Sequential([
        keras.Input(shape=(30, 258)),
        # Bloque 1: 64 filtros que miran 3 fotogramas seguidos
        layers.Conv1D(64, 3, activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),                     # 30 -> 15 fotogramas
        # Bloque 2: 128 filtros sobre patrones de movimiento más largos
        layers.Conv1D(128, 3, activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),            # promedio en el tiempo -> vector de 128
        # Clasificador
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
