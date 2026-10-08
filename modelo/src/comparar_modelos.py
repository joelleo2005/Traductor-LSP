"""
Comparación Conv1D vs Conv2D para el reconocimiento de 10 señas de la LSP.

Ejecutar desde la carpeta modelo/ (terminal de Android Studio):
    python src/comparar_modelos.py                     # una partición (semilla 42)
    python src/comparar_modelos.py --semillas 1 2 3    # varias particiones -> media ± desviación

Ambos modelos usan EXACTAMENTE los mismos datos, la misma partición y el mismo
aumento de datos; solo cambia la arquitectura.
  - 80 % entrenamiento / 20 % prueba, estratificado por clase (artículo, sección 5.2)
  - la validación para Early Stopping sale del 80 % de entrenamiento (15 %), NO de la prueba
  - el 20 % de prueba solo se usa una vez, al final
"""
import argparse
import os
import time

import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from normalizar import normalizar
from augmentation import aumentar

# ===== CONFIGURACIÓN =====
SEÑAS_USAR = {"YO", "PENSAR", "MUJER", "MAMA", "QUE", "VER", "COMER", "BIEN", "CAMINAR", "CASA"}
N_COPIAS_AUG = 8
EPOCHS = 200
BATCH = 16
PACIENCIA = 30
VAL_SIZE = 0.15     # fracción del 80 % de entrenamiento usada como validación


# =====================================================================
# Entrada de la Conv2D: el vector de 258 valores se reorganiza como una
# "imagen" (30 fotogramas x articulaciones x 3 coordenadas).
#   258 = pose 33 x (x,y,z,visibilidad) + mano izq 21 x (x,y,z) + mano der 21 x (x,y,z)
# Se usan x,y,z de los 75 puntos (la visibilidad no existe para las manos).
# =====================================================================
POSE, MANO_IZQ, MANO_DER = 0, 33, 54        # índices dentro de los 75 puntos


def a_puntos(X):
    """(N, 30, 258) -> (N, 30, 75, 3)"""
    N, T, _ = X.shape
    pose = X[:, :, :132].reshape(N, T, 33, 4)[..., :3]
    izq = X[:, :, 132:195].reshape(N, T, 21, 3)
    der = X[:, :, 195:258].reshape(N, T, 21, 3)
    return np.concatenate([pose, izq, der], axis=2).astype("float32")


def orden_tssi():
    """Orden de columnas tipo árbol esquelético (TSSI, Laines et al., 2023).
    Se recorre el esqueleto en profundidad volviendo al padre, de modo que dos
    columnas vecinas son siempre articulaciones conectadas: así el filtro 3x3 de
    la Conv2D mira movimientos de articulaciones que realmente están unidas."""
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
    """(N, 30, 258) -> (N, 30, columnas_TSSI, 9)
    Canales: posición (x,y,z) + velocidad (x,y,z) + forma de la mano relativa a la muñeca (x,y,z)."""
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


# =====================================================================
# Modelos
# =====================================================================
def modelo_conv1d(n_clases):
    """El mismo Conv1D de entrenar.py (línea base del artículo)."""
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


def modelo_conv2d(n_clases, n_cols):
    """Conv2D sobre (tiempo x articulaciones). Al final se promedia SOLO en el tiempo,
    para no perder qué articulación hizo cada movimiento."""
    return keras.Sequential([
        keras.Input(shape=(30, n_cols, 9)),
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


# =====================================================================
# Utilidades
# =====================================================================
def imprimir_matriz(cm, clases):
    w = max(len(c) for c in clases)
    print("\n=== Matriz de confusión (fila = real, columna = predicho) ===")
    print(" " * (w + 2) + " ".join(f"{c[:7]:>7s}" for c in clases))
    for c, fila in zip(clases, cm):
        print(f"{c:>{w}s}  " + " ".join(f"{v:7d}" for v in fila))


def a_tflite(model):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    return conv.convert()


def latencia_tflite(tfl, x, repeticiones=200):
    """Tiempo medio de inferencia de UNA seña en CPU (ms)."""
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    i = it.get_input_details()[0]
    muestra = x[:1].astype("float32")
    for _ in range(10):
        it.set_tensor(i["index"], muestra); it.invoke()
    t = time.perf_counter()
    for _ in range(repeticiones):
        it.set_tensor(i["index"], muestra); it.invoke()
    return (time.perf_counter() - t) / repeticiones * 1000


def entrenar_y_evaluar(nombre, model, Xtr, ytr, Xva, yva, Xte, yte, clases, guardar):
    n = len(clases)
    print("\n" + "=" * 70)
    print(f" MODELO {nombre.upper()}  |  entrada {Xtr.shape[1:]}  |  parámetros {model.count_params():,}")
    print("=" * 70)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]
    t0 = time.time()
    hist = model.fit(Xtr, keras.utils.to_categorical(ytr, n),
                     validation_data=(Xva, keras.utils.to_categorical(yva, n)),
                     epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks, verbose=1)
    minutos = (time.time() - t0) / 60
    epocas = len(hist.history["loss"])

    # ---- evaluación en PRUEBA (20 %), una sola vez
    pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
    acc = float((pred == yte).mean())
    f1 = float(f1_score(yte, pred, average="macro"))
    print(f"\n>>> {nombre}: precisión en test = {acc * 100:.1f}%  |  F1 macro = {f1 * 100:.1f}%  "
          f"|  {epocas} épocas en {minutos:.1f} min")
    print(classification_report(yte, pred, target_names=clases, digits=2, zero_division=0))
    imprimir_matriz(confusion_matrix(yte, pred), clases)

    tfl = a_tflite(model)
    ms = latencia_tflite(tfl, Xte)
    print(f"\nTFLite: {len(tfl) / 1024:.0f} KB  |  inferencia ≈ {ms:.2f} ms por seña (CPU)")
    if guardar:
        model.save(f"data/modelo_{nombre}.keras")
        with open(f"data/modelo_{nombre}.tflite", "wb") as f:
            f.write(tfl)
    return {"modelo": nombre, "acc": acc, "f1": f1, "epocas": epocas, "min": minutos,
            "params": model.count_params(), "kb": len(tfl) / 1024, "ms": ms}


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    # 1. Cargar los 3 datasets (X.npy) + tesistas (si ya fue procesado), normalizar y filtrar
    X_crudo = np.load("data/X.npy")
    y = np.load("data/y.npy")
    if os.path.exists("data/X_tesistas.npy"):
        X_crudo = np.concatenate([X_crudo, np.load("data/X_tesistas.npy")])
        y = np.concatenate([y, np.load("data/y_tesistas.npy")])
        print("Incluye dataset PARA TESISTAS:", len(np.load("data/y_tesistas.npy")), "muestras")
    X = normalizar(X_crudo)
    for ini, fin in ((132, 195), (195, 258)):          # manos no detectadas -> 0 (no una "mano falsa")
        falta = (X_crudo[:, :, ini:fin] == 0).all(-1)
        X[:, :, ini:fin][falta] = 0.0
    mask = np.array([et in SEÑAS_USAR for et in y])
    X, y = X[mask], y[mask]
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    clases = list(le.classes_)
    print("Dataset:", X.shape, "-", len(clases), "clases")
    print("Por seña:", dict(Counter(y)))
    print(f"Conv2D: {len(ORDEN)} columnas TSSI (de 75 puntos) x 3 coordenadas")

    resultados = []
    for k, seed in enumerate(args.semillas):
        print("\n" + "#" * 70 + f"\n# PARTICIÓN CON SEMILLA {seed}\n" + "#" * 70)

        # 2. Partición 80/20 estratificada; validación desde el 80 %
        X_trv, X_te, y_trv, y_te = train_test_split(X, y_num, test_size=0.2, random_state=seed, stratify=y_num)
        X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=VAL_SIZE, random_state=seed,
                                                  stratify=y_trv)
        # 3. Aumento SOLO en entrenamiento (las mismas copias para ambos modelos)
        np.random.seed(seed)
        X_tr, y_tr = aumentar(X_tr, y_tr, n_copias=N_COPIAS_AUG)
        print(f"Entrenamiento: {len(X_tr)} (con aumento x{N_COPIAS_AUG + 1}) | "
              f"validación: {len(X_va)} | prueba: {len(X_te)}")

        # 4. Conv1D
        keras.utils.set_random_seed(seed)
        resultados.append(entrenar_y_evaluar("conv1d", modelo_conv1d(len(clases)),
                                             X_tr, y_tr, X_va, y_va, X_te, y_te, clases, guardar=(k == 0)))
        # 5. Conv2D (mismos datos, reorganizados como imagen)
        keras.utils.set_random_seed(seed)
        resultados.append(entrenar_y_evaluar("conv2d", modelo_conv2d(len(clases), len(ORDEN)),
                                             a_imagen(X_tr), y_tr, a_imagen(X_va), y_va, a_imagen(X_te), y_te,
                                             clases, guardar=(k == 0)))

    # 6. Resumen comparativo
    print("\n" + "=" * 70 + "\n RESUMEN COMPARATIVO (conjunto de prueba)\n" + "=" * 70)
    print(f"{'Modelo':8s} {'Accuracy':>16s} {'F1 macro':>16s} {'Parámetros':>11s} {'TFLite':>8s} {'ms/seña':>8s}")
    lineas = ["modelo,semilla,accuracy,f1_macro,epocas,minutos,parametros,tflite_kb,ms_por_sena"]
    for nombre in ("conv1d", "conv2d"):
        r = [x for x in resultados if x["modelo"] == nombre]
        a = np.array([x["acc"] for x in r]) * 100
        f = np.array([x["f1"] for x in r]) * 100
        print(f"{nombre:8s} {a.mean():9.1f} ± {a.std():4.1f}% {f.mean():9.1f} ± {f.std():4.1f}% "
              f"{r[0]['params']:>11,} {r[0]['kb']:6.0f}KB {np.mean([x['ms'] for x in r]):8.2f}")
        for s, x in zip(args.semillas, r):
            lineas.append(f"{nombre},{s},{x['acc']:.4f},{x['f1']:.4f},{x['epocas']},{x['min']:.2f},"
                          f"{x['params']},{x['kb']:.1f},{x['ms']:.3f}")
    with open("data/comparacion.csv", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    print("\nGuardado: data/comparacion.csv, data/modelo_conv1d.(keras|tflite), data/modelo_conv2d.(keras|tflite)")


if __name__ == "__main__":
    main()