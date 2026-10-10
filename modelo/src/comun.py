
import argparse
import time

import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from normalizar import normalizar
from augmentation import aumentar

# ===== CONFIGURACIÓN =====
SEÑAS_USAR = {"YO", "PENSAR", "MUJER", "MAMA", "QUE", "VER", "COMER", "BIEN", "CAMINAR", "CASA"}
N_COPIAS_AUG = 8
EPOCHS = 200
BATCH = 16
VAL_SIZE = 0.15     # fracción del 80 % de entrenamiento usada como validación


def cargar_datos():
    """Carga X.npy, normaliza, pone en 0 las manos no detectadas y filtra las 10 señas."""
    X_crudo = np.load("data/X.npy")
    y = np.load("data/y.npy")
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
    return X, y_num, clases


def particion(X, y_num, seed):
    """80/20 estratificado; validación desde el 80 %; aumento solo en entrenamiento."""
    X_trv, X_te, y_trv, y_te = train_test_split(X, y_num, test_size=0.2, random_state=seed, stratify=y_num)
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=VAL_SIZE, random_state=seed,
                                              stratify=y_trv)
    np.random.seed(seed)
    X_tr, y_tr = aumentar(X_tr, y_tr, n_copias=N_COPIAS_AUG)
    print(f"Entrenamiento: {len(X_tr)} (con aumento x{N_COPIAS_AUG + 1}) | "
          f"validación: {len(X_va)} | prueba: {len(X_te)}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def imprimir_matriz(cm, clases):
    w = max(len(c) for c in clases)
    print("\n=== Matriz de confusión (fila = real, columna = predicho) ===")
    print(" " * (w + 2) + " ".join(f"{c[:7]:>7s}" for c in clases))
    for c, fila in zip(clases, cm):
        print(f"{c:>{w}s}  " + " ".join(f"{v:7d}" for v in fila))


def a_tflite(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


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
    cm = confusion_matrix(yte, pred, labels=list(range(n)))
    imprimir_matriz(cm, clases)
    rep = classification_report(yte, pred, target_names=clases, zero_division=0, output_dict=True)

    tfl = a_tflite(model)
    ms = latencia_tflite(tfl, Xte)
    print(f"\nTFLite: {len(tfl) / 1024:.0f} KB  |  inferencia ≈ {ms:.2f} ms por seña (CPU)")
    if guardar:
        model.save(f"data/modelo_{nombre}.keras")
        with open(f"data/modelo_{nombre}.tflite", "wb") as f:
            f.write(tfl)
    return {"acc": acc, "f1": f1, "epocas": epocas, "min": minutos,
            "params": model.count_params(), "kb": len(tfl) / 1024, "ms": ms,
            "por_sena": {c: rep[c] for c in clases}, "cm": cm, "hist": hist.history}


def ejecutar(nombre, construir_modelo, preparar_entrada):
    """Bucle completo para UN modelo.
    construir_modelo(n_clases) -> modelo Keras sin entrenar
    preparar_entrada(X)        -> X con la forma que espera ese modelo"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    X, y_num, clases = cargar_datos()
    resultados = []
    for k, seed in enumerate(args.semillas):
        print("\n" + "#" * 70 + f"\n# {nombre.upper()} - PARTICIÓN CON SEMILLA {seed}\n" + "#" * 70)
        X_tr, y_tr, X_va, y_va, X_te, y_te = particion(X, y_num, seed)
        keras.utils.set_random_seed(seed)
        resultados.append(entrenar_y_evaluar(
            nombre, construir_modelo(len(clases)),
            preparar_entrada(X_tr), y_tr, preparar_entrada(X_va), y_va, preparar_entrada(X_te), y_te,
            clases, guardar=(k == 0)))

    # ---- resumen de este modelo + CSV (los lee comparar.py)
    a = np.array([r["acc"] for r in resultados]) * 100
    f = np.array([r["f1"] for r in resultados]) * 100
    print("\n" + "=" * 70 + f"\n RESUMEN {nombre.upper()} ({len(resultados)} semilla(s), conjunto de prueba)\n" + "=" * 70)
    print(f"Accuracy: {a.mean():.1f} ± {a.std():.1f}%  |  F1 macro: {f.mean():.1f} ± {f.std():.1f}%  |  "
          f"parámetros: {resultados[0]['params']:,}  |  TFLite: {resultados[0]['kb']:.0f} KB  |  "
          f"{np.mean([r['ms'] for r in resultados]):.2f} ms/seña")
    lineas = ["modelo,semilla,accuracy,f1_macro,epocas,minutos,parametros,tflite_kb,ms_por_sena"]
    for s, r in zip(args.semillas, resultados):
        lineas.append(f"{nombre},{s},{r['acc']:.4f},{r['f1']:.4f},{r['epocas']},{r['min']:.2f},"
                      f"{r['params']},{r['kb']:.1f},{r['ms']:.3f}")
    with open(f"data/resultados_{nombre}.csv", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    with open(f"data/por_sena_{nombre}.csv", "w", encoding="utf-8") as fh:
        fh.write("modelo,semilla,sena,precision,recall,f1,soporte\n")
        for s, r in zip(args.semillas, resultados):
            for c, m in r["por_sena"].items():
                fh.write(f"{nombre},{s},{c},{m['precision']:.4f},{m['recall']:.4f},"
                         f"{m['f1-score']:.4f},{int(m['support'])}\n")
    with open(f"data/matriz_{nombre}.csv", "w", encoding="utf-8") as fh:      # suma de todas las semillas
        fh.write("real," + ",".join(clases) + "\n")
        total = sum(r["cm"] for r in resultados)
        for c, fila in zip(clases, total):
            fh.write(c + "," + ",".join(str(int(v)) for v in fila) + "\n")
    with open(f"data/historial_{nombre}.csv", "w", encoding="utf-8") as fh:   # curvas de aprendizaje
        fh.write("modelo,semilla,epoca,loss,accuracy,val_loss,val_accuracy\n")
        for s, r in zip(args.semillas, resultados):
            h = r["hist"]
            for e in range(len(h["loss"])):
                fh.write(f"{nombre},{s},{e + 1},{h['loss'][e]:.4f},{h['accuracy'][e]:.4f},"
                         f"{h['val_loss'][e]:.4f},{h['val_accuracy'][e]:.4f}\n")
    print(f"\nGuardado en data/: resultados_{nombre}.csv, por_sena_{nombre}.csv, matriz_{nombre}.csv, "
          f"historial_{nombre}.csv, modelo_{nombre}.(keras|tflite)")