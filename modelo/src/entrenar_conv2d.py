"""
Modelo PROPUESTO Conv2D (PyTorch) sobre la imagen esqueleto-temporal TSSI.

Cada seña se convierte en una imagen de fotogramas x 109 columnas x 9 canales:
posición (x, y, z) + velocidad (x, y, z) + forma de la mano respecto a la muñeca (x, y, z).
La configuración de entrenamiento está en comun.CONFIG (la misma para los dos modelos).

Ejecutar desde la carpeta modelo/ (evaluación final con 5 semillas):
    python src/entrenar_conv2d.py --semillas 1 2 3 4 5

Los resultados se guardan en data/ (resultados_conv2d.csv, historial_conv2d.csv, modelo_conv2d.pt, ...).
"""
import numpy as np
import torch.nn as nn

from comun import ejecutar

POSE, MANO_IZQ, MANO_DER = 0, 33, 54        # índices dentro de los 75 puntos


def a_puntos(X):
    """(N, T, 258) -> (N, T, 75, 3). Se usan x, y, z (la visibilidad no existe para las manos)."""
    N, T, _ = X.shape
    pose = X[:, :, :132].reshape(N, T, 33, 4)[..., :3]
    izq = X[:, :, 132:195].reshape(N, T, 21, 3)
    der = X[:, :, 195:258].reshape(N, T, 21, 3)
    return np.concatenate([pose, izq, der], axis=2).astype("float32")


def orden_tssi():
    """Orden de las columnas TSSI: recorrido en profundidad del árbol del esqueleto (109 columnas, 55 puntos)."""
    hijos = {}

    def unir(a, b):
        hijos.setdefault(a, []).append(b)

    for p in (2, 5, 9, 10):                 # nariz -> ojos y boca
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
    """(N, T, 258) -> (N, T, 109, 9): la imagen esqueleto-temporal ."""
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


class Conv2DModelo(nn.Module):
    """Conv2D (≈ 170 mil parámetros). Entrada (B, fotogramas, 109, 9) -> puntaje de cada una de las n_clases."""
    def __init__(self, n_clases, forma_entrada, dropout=0.5):
        super().__init__()
        T, columnas, canales = forma_entrada                       # (48, 109, 9)
        bn = dict(eps=1e-3, momentum=0.01)                         # mismos valores que la versión inicial en Keras
        self.red = nn.Sequential(
            nn.BatchNorm2d(canales, **bn),                         # iguala la escala de los 9 canales
            # Bloque 1: filtros de 3 fotogramas x 3 articulaciones vecinas
            nn.Conv2d(canales, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32, **bn),
            nn.MaxPool2d(2),                                       # 48 x 109 -> 24 x 54
            # Bloque 2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64, **bn),
            nn.MaxPool2d(2),                                       # 24 x 54 -> 12 x 27
            # Bloque 3
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64, **bn),
            nn.AvgPool2d((T // 4, 1)),                             # promedio temporal (48 -> 24 -> 12 -> 1)
            nn.Flatten(),                                          # se conserva cada zona del cuerpo
            # Clasificador
            nn.Dropout(dropout), nn.Linear(64 * (columnas // 4), 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_clases),           # softmax va dentro de la pérdida
        )

    def forward(self, x):
        return self.red(x.permute(0, 3, 1, 2))                     # Conv2d espera (B, canales, alto, ancho)


if __name__ == "__main__":
    print(f"Conv2D: {len(ORDEN)} columnas TSSI (de 55 puntos) x 9 canales")
    ejecutar("conv2d", Conv2DModelo, a_imagen)
