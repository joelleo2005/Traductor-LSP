
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from comun import ejecutar

POSE, MANO_IZQ, MANO_DER = 0, 33, 54        # índices dentro de los 75 puntos


def a_puntos(X):
    """(N, 30, 258) -> (N, 30, 75, 3). Se usan x,y,z (la visibilidad no existe para las manos)."""
    N, T, _ = X.shape
    pose = X[:, :, :132].reshape(N, T, 33, 4)[..., :3]
    izq = X[:, :, 132:195].reshape(N, T, 21, 3)
    der = X[:, :, 195:258].reshape(N, T, 21, 3)
    return np.concatenate([pose, izq, der], axis=2).astype("float32")


def orden_tssi():
    hijos = {}

    def unir(a, b):
        hijos.setdefault(a, []).append(b)

    for p in (2, 5, 9, 10):                 # nariz -> ojos y boca (rasgos no manuales)
        unir(0, p)
    unir(0, 11); unir(11, 13); unir(13, 15); unir(15, MANO_IZQ); unir(11, 23)   # brazo izq
    unir(0, 12); unir(12, 14); unir(14, 16); unir(16, MANO_DER); unir(12, 24)   # brazo der
    for base in (MANO_IZQ, MANO_DER):        # 5 dedos desde la muñeca
        for dedo in ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            prev = base
            for j in dedo:
                unir(prev, base + j)
                prev = base + j

    def recorrer(n):
        seq = [n]
        for h in hijos.get(n, []):
            seq += recorrer(h) + [n]
        return seq

    return recorrer(0)


ORDEN = orden_tssi()


def a_imagen(X):
    P = a_puntos(X)
    V = np.zeros_like(P)
    V[:, 1:] = P[:, 1:] - P[:, :-1]                       # movimiento entre fotogramas
    H = np.zeros_like(P)
    for off in (MANO_IZQ, MANO_DER):                      # dedos respecto a la muñeca, escala de la mano
        rel = P[:, :, off:off + 21] - P[:, :, off:off + 1]
        tam = np.linalg.norm(rel[..., :2], axis=-1).max(-1, keepdims=True)[..., None] + 1e-6
        hay = (np.abs(P[:, :, off:off + 21]).sum((-1, -2)) > 1e-3)[..., None, None]
        H[:, :, off:off + 21] = np.where(hay, rel / tam, 0)
    return np.concatenate([P, V * 5, H], axis=-1)[:, :, ORDEN, :].astype("float32")


def modelo_conv2d(n_clases):
    return keras.Sequential([
        keras.Input(shape=(30, len(ORDEN), 9)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(7, 1)),      # promedio temporal (30 -> 15 -> 7 -> 1)
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_clases, activation="softmax"),
    ], name="conv2d")


if __name__ == "__main__":
    print(f"Conv2D: {len(ORDEN)} columnas TSSI (de 75 puntos) x 9 canales")
    ejecutar("conv2d", modelo_conv2d, a_imagen)