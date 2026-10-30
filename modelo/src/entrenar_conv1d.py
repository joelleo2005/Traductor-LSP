"""
Modelo BASE Conv1D (PyTorch).

Trata cada seña como una serie de tiempo: fotogramas x 258 valores por fotograma.
La configuración de entrenamiento está en comun.CONFIG (la misma para los dos modelos).

Ejecutar desde la carpeta modelo/ (evaluación final con 5 semillas):
    python src/entrenar_conv1d.py --semillas 1 2 3 4 5

Los resultados se guardan en data/ (resultados_conv1d.csv, historial_conv1d.csv, modelo_conv1d.pt, ...).
"""
import torch.nn as nn

from comun import ejecutar


class Conv1DModelo(nn.Module):
    """Conv1D (≈ 84 mil parámetros). Entrada (B, fotogramas, 258) -> puntaje de cada una de las n_clases."""
    def __init__(self, n_clases, forma_entrada, dropout=0.5):
        super().__init__()
        _, n_valores = forma_entrada                               # (fotogramas, 258)
        bn = dict(eps=1e-3, momentum=0.01)                         # mismos valores que la versión inicial en Keras
        self.red = nn.Sequential(
            # Bloque 1: 64 filtros que miran 3 fotogramas seguidos
            nn.Conv1d(n_valores, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64, **bn),
            nn.MaxPool1d(2),                                       # 48 -> 24 fotogramas
            # Bloque 2: 128 filtros sobre patrones de movimiento más largos
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128, **bn),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),                 # promedio en el tiempo -> vector de 128
            # Clasificador
            nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_clases),           # softmax va dentro de la pérdida
        )

    def forward(self, x):
        return self.red(x.transpose(1, 2))                         # Conv1d espera (B, canales, tiempo)


def preparar_entrada(X):
    """La Conv1D usa la secuencia tal cual: (N, fotogramas, 258)."""
    return X.astype("float32")


if __name__ == "__main__":
    ejecutar("conv1d", Conv1DModelo, preparar_entrada)
