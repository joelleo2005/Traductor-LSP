"""
Modelo PROPUESTO: red convolucional 2D (Conv2D) sobre una imagen esqueleto-temporal (TSSI).

Cada seña se convierte en una "imagen" de 30 x 109 x 9 (Laines et al., 2023):
    - filas    (30) : fotogramas (tiempo)
    - columnas (109): articulaciones ordenadas recorriendo el árbol del esqueleto (TSSI), de modo que
                      puntos vecinos en el cuerpo (ej. dedos de una misma mano) quedan en columnas vecinas
    - canales  (9)  : posición (x, y, z) + velocidad (x, y, z) + forma de la mano respecto a la muñeca (x, y, z)
Así la Conv2D aprende patrones en el tiempo Y entre articulaciones cercanas a la vez.

Ejecutar desde la carpeta modelo/:
    python src/entrenar_conv2d.py --semillas 1 2 3 4 5

La carga de datos, la partición, el aumento, el entrenamiento, la evaluación y los CSV de resultados
están en comun.py (son los mismos para Conv1D y Conv2D).
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from comun import ejecutar

POSE, MANO_IZQ, MANO_DER = 0, 33, 54        # índices dentro de los 75 puntos


def a_puntos(X):
    """(N, 30, 258) -> (N, 30, 75, 3). Se usan x,y,z (la visibilidad no existe para las manos)."""
    N, T, _ = X.shape
    pose = X[:, :, :132].reshape(N, T, 33, 4)[..., :3]     # 33 puntos, se descarta la visibilidad
    izq = X[:, :, 132:195].reshape(N, T, 21, 3)             # 21 puntos de la mano izquierda
    der = X[:, :, 195:258].reshape(N, T, 21, 3)             # 21 puntos de la mano derecha
    return np.concatenate([pose, izq, der], axis=2).astype("float32")


def orden_tssi():
    """Devuelve el orden de las columnas de la imagen TSSI (lista de 109 índices de los 75 puntos).

    Se arma el esqueleto como un árbol con raíz en la nariz (punto 0) y se recorre en profundidad (DFS).
    Cada vez que se vuelve de una rama se repite el punto padre, por eso hay 109 columnas y no 75.
    """
    hijos = {}                               # árbol: {punto: [puntos hijos]}

    def unir(a, b):
        """Agrega la conexión padre a -> hijo b."""
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
        """Recorrido en profundidad: visita n, cada uno de sus hijos y regresa a n."""
        seq = [n]
        for h in hijos.get(n, []):
            seq += recorrer(h) + [n]
        return seq

    return recorrer(0)


ORDEN = orden_tssi()                         # se calcula una sola vez (109 columnas)


def a_imagen(X):
    """(N, 30, 258) -> (N, 30, 109, 9): la imagen esqueleto-temporal que recibe la Conv2D."""
    P = a_puntos(X)                                       # canales 0-2: posición
    V = np.zeros_like(P)
    V[:, 1:] = P[:, 1:] - P[:, :-1]                       # movimiento entre fotogramas
    H = np.zeros_like(P)
    for off in (MANO_IZQ, MANO_DER):                      # dedos respecto a la muñeca, escala de la mano
        rel = P[:, :, off:off + 21] - P[:, :, off:off + 1]                     # cada punto - muñeca
        tam = np.linalg.norm(rel[..., :2], axis=-1).max(-1, keepdims=True)[..., None] + 1e-6   # tamaño de la mano
        hay = (np.abs(P[:, :, off:off + 21]).sum((-1, -2)) > 1e-3)[..., None, None]           # ¿se detectó la mano?
        H[:, :, off:off + 21] = np.where(hay, rel / tam, 0)
    # Se juntan los 9 canales (velocidad x5 para que tenga una escala parecida a la posición)
    # y se reordenan las columnas según el recorrido TSSI
    return np.concatenate([P, V * 5, H], axis=-1)[:, :, ORDEN, :].astype("float32")


def modelo_conv2d(n_clases):
    """Construye la Conv2D (≈ 170 mil parámetros). Entrada (30, 109, 9) -> probabilidad de cada una de las n_clases."""
    return keras.Sequential([
        keras.Input(shape=(30, len(ORDEN), 9)),
        layers.BatchNormalization(),                    # iguala la escala de los 9 canales
        # Bloque 1: filtros de 3 fotogramas x 3 articulaciones vecinas
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),                    # 30 x 109 -> 15 x 54
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),                    # 15 x 54 -> 7 x 27
        # Bloque 3
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(7, 1)),      # promedio temporal (30 -> 15 -> 7 -> 1)
        layers.Flatten(),                               # se conserva la información de cada zona del cuerpo
        # Clasificador
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_clases, activation="softmax"),
    ], name="conv2d")


if __name__ == "__main__":
    print(f"Conv2D: {len(ORDEN)} columnas TSSI (de 75 puntos) x 9 canales")
    ejecutar("conv2d", modelo_conv2d, a_imagen)
