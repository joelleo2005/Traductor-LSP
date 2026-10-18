"""
Aumento de datos (data augmentation) sobre secuencias de landmarks.

Cada seña es una matriz (30, 258): 30 fotogramas x 258 valores
(pose: 33 puntos x [x, y, z, visibilidad] = 132; mano izquierda y derecha: 21 puntos x [x, y, z] = 63 cada una).

Se generan copias "nuevas" de cada seña aplicando pequeñas variaciones realistas:
    - jitter          : ruido gaussiano (temblor leve de la detección)
    - escalar         : persona más cerca o más lejos de la cámara
    - rotar           : cámara o persona ligeramente inclinada
    - warp_temporal   : la seña se hace más rápido o más lento

IMPORTANTE: este aumento se aplica SOLO al conjunto de entrenamiento y DESPUÉS de la partición
(ver comun.particion), para que no haya fuga de datos hacia validación ni prueba.
"""
import numpy as np


def jitter(seq, sigma=0.02):
    """Suma ruido gaussiano (media 0, desviación sigma) a todos los valores de la secuencia."""
    return seq + np.random.normal(0, sigma, seq.shape)


def escalar(seq, rango=(0.85, 1.15)):
    """Multiplica toda la secuencia por un factor aleatorio (simula cambiar la distancia a la cámara)."""
    return seq * np.random.uniform(*rango)


def rotar(seq, max_ang=0.2):
    """Rota en el plano (x, y) todos los puntos un ángulo aleatorio entre -max_ang y +max_ang radianes (≈ ±11°)."""
    ang = np.random.uniform(-max_ang, max_ang)
    c, s = np.cos(ang), np.sin(ang)
    out = seq.copy()
    # Pares de columnas (x, y) de cada punto:
    #   pose  -> 33 puntos cada 4 valores (x, y, z, visibilidad)
    #   manos -> 42 puntos cada 3 valores (x, y, z), desde la columna 132
    idx_xy = [(4*i, 4*i+1) for i in range(33)] + [(b, b+1) for b in range(132, 258, 3)]
    for xi, yi in idx_xy:
        x, y = out[:, xi].copy(), out[:, yi].copy()
        out[:, xi] = c*x - s*y          # matriz de rotación 2D
        out[:, yi] = s*x + c*y
    return out


def warp_temporal(seq, rango=(0.7, 1.3)):
    """Cambia la velocidad de la seña: la remuestrea a T*factor fotogramas y luego la devuelve a T fotogramas."""
    T = seq.shape[0]
    factor = np.random.uniform(*rango)                  # < 1: más rápida, > 1: más lenta
    n = max(2, int(round(T * factor)))
    idx = np.linspace(0, T - 1, n)
    res = seq[np.round(idx).astype(int)]                # secuencia estirada o comprimida
    idx2 = np.linspace(0, len(res) - 1, T).astype(int)
    return res[idx2]                                    # vuelve a tener T fotogramas


def aumentar(X, y, n_copias=8):
    """Devuelve las muestras originales + n_copias aumentadas de cada una (el conjunto crece x(n_copias + 1)).

    X: (N, 30, 258), y: (N,)  ->  X_agg: (N * (n_copias + 1), 30, 258), y_agg con las etiquetas repetidas.
    """
    X_agg, y_agg = list(X), list(y)                     # se conservan las originales
    for seq, etiqueta in zip(X, y):
        for _ in range(n_copias):
            # Las 4 transformaciones encadenadas, cada una con valores aleatorios distintos
            X_agg.append(warp_temporal(rotar(jitter(escalar(seq)))))
            y_agg.append(etiqueta)
    return np.array(X_agg), np.array(y_agg)
