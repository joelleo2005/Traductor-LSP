"""
Base común de entrenamiento y evaluación (PyTorch), compartida por los dos modelos.

Contiene la configuración final (CONFIG) y todo lo que no depende de la arquitectura:
    cargar_datos       carga data/X.npy y data/y.npy, normaliza, marca las manos no detectadas,
                       interpola los huecos de las manos y filtra las 10 señas de SEÑAS_USAR
    particion          partición estratificada 80/20 (+15 % de validación) y aumento SOLO en entrenamiento
    correr             entrena un modelo con una configuración y una semilla y devuelve sus métricas
    ejecutar           evaluación final en varias semillas con opciones de línea de comandos
                       (la usan entrenar_conv1d.py y entrenar_conv2d.py)

Con la misma semilla y la misma configuración los resultados son reproducibles (ver fijar_semilla).
"""
import argparse
import gc
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from normalizar import normalizar
from augmentation import aumentar

MODELO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(MODELO, "data")
DATOS_POR_FOTOGRAMAS = {48: DATA, 30: os.path.join(MODELO, "data_base_30f")}

# CONFIGURACIÓN
SEÑAS_USAR = {"YO", "PENSAR", "MUJER", "VER", "COMER", "BIEN", "CAMINAR", "CASA", "MAMA", "QUE"}
BATCH = 16
VAL_SIZE = 0.15     # fracción del 80 % de entrenamiento usada como validación
LR = 1e-3           # tasa de aprendizaje inicial de Adam

CONFIG = {
    "fotogramas": 48,         # fotogramas por seña (48 o 30)
    "epocas": 100,            # número fijo de épocas (sin parada temprana)
    "suavizado": 0.05,        # label smoothing: 0 = desactivado; 0.05 = la seña correcta vale 0.955
    "pesos_clase": False,     # True = las señas con menos muestras pesan más en la pérdida
    "copias_aug": 8,          # copias aumentadas por cada seña de entrenamiento (entrenamiento x9)
    "ruido_en_ceros": False,  # True = el ruido del aumento también mueve los puntos no detectados (versión inicial)
    "reg_l2": 1e-4,           # regularización L2 de las capas convolucionales
    "dropout": 0.5,           # dropout del clasificador
    "interpolar_manos": True,   # True = las manos no detectadas se rellenan interpolando entre fotogramas vecinos
    "espejo": 0.5,            # probabilidad de reflejar cada copia aumentada (0 = sin espejo); ajuste 7
}


def dispositivo():
    """GPU si está disponible; si no, CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(seed):
    """Hace reproducibles el aumento de datos, los pesos iniciales y el orden de los lotes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # La GPU usa por defecto convoluciones más rápidas pero no reproducibles (cambian un poco en cada
    # ejecución); con esto, la misma semilla y la misma configuración dan siempre el mismo resultado.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def interpolar_manos(X, falta):
    """Rellena las manos no detectadas interpolando linealmente entre los fotogramas vecinos donde sí se vio.

    Solo se rellenan los huecos interiores (con la mano detectada antes y después); al inicio y al final de
    la seña la mano se deja en 0, porque puede estar realmente fuera del cuadro. Si la mano no aparece en
    ningún fotograma, también se deja en 0.
    falta[k]: (N, T) True donde falta la mano k (0 = izquierda, 1 = derecha).
    """
    X = X.copy()
    t = np.arange(X.shape[1])
    rellenados = 0
    for k, ini in enumerate((132, 195)):
        for n in range(len(X)):
            vista = ~falta[k][n]
            if vista.sum() < 2 or vista.all():
                continue
            primero, ultimo = np.argmax(vista), len(vista) - 1 - np.argmax(vista[::-1])
            hueco = ~vista & (t > primero) & (t < ultimo)
            if not hueco.any():
                continue
            for c in range(ini, ini + 63):
                X[n, hueco, c] = np.interp(t[hueco], t[vista], X[n, vista, c])
            rellenados += int(hueco.sum())
    return X, rellenados


def cargar_datos(fotogramas=CONFIG["fotogramas"], interpolar=CONFIG["interpolar_manos"]):
    """Carga X.npy, normaliza, pone en 0 las manos no detectadas y filtra las señas de SEÑAS_USAR.

    Devuelve X (N, fotogramas, 258), y_num (etiquetas como números) y clases (nombres en orden alfabético).
    """
    carpeta = DATOS_POR_FOTOGRAMAS[fotogramas]
    X_crudo = np.load(os.path.join(carpeta, "X.npy"))
    y = np.load(os.path.join(carpeta, "y.npy"))
    X = normalizar(X_crudo)
    faltas = []
    for ini, fin in ((132, 195), (195, 258)):                      # manos no detectadas -> 0 (no una "mano falsa")
        falta = (X_crudo[:, :, ini:fin] == 0).all(-1)
        X[:, :, ini:fin][falta] = 0.0
        faltas.append(falta)
    if interpolar:
        X, rellenados = interpolar_manos(X, faltas)
        print(f"Manos interpoladas: {rellenados} de {int(sum(f.sum() for f in faltas))} faltantes")
    mask = np.array([et in SEÑAS_USAR for et in y])
    X, y = X[mask], y[mask]
    le = LabelEncoder()                                            # "BIEN" -> 0, "CAMINAR" -> 1, ...
    y_num = le.fit_transform(y)
    clases = list(le.classes_)
    print("Dataset:", X.shape, "-", len(clases), "clases")
    print("Por seña:", dict(Counter(y)))
    return X, y_num, clases


def particion(X, y_num, seed, copias_aug=CONFIG["copias_aug"], ruido_en_ceros=CONFIG["ruido_en_ceros"],
              p_espejo=CONFIG["espejo"]):
    """80/20 estratificado; validación desde el 80 %; aumento solo en entrenamiento.

    La semilla fija qué muestras caen en cada conjunto, así ambos modelos usan la misma partición.
    """
    X_trv, X_te, y_trv, y_te = train_test_split(X, y_num, test_size=0.2, random_state=seed, stratify=y_num)
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=VAL_SIZE, random_state=seed,
                                              stratify=y_trv)
    np.random.seed(seed)
    X_tr, y_tr = aumentar(X_tr, y_tr, n_copias=copias_aug, ruido_en_ceros=ruido_en_ceros,
                          p_espejo=p_espejo)                        # SOLO entrenamiento
    print(f"Entrenamiento: {len(X_tr)} (con aumento x{copias_aug + 1}) | "
          f"validación: {len(X_va)} | prueba: {len(X_te)}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def imprimir_matriz(cm, clases):
    """Muestra la matriz de confusión en la terminal (fila = seña real, columna = seña predicha)."""
    w = max(len(c) for c in clases)
    print("\n=== Matriz de confusión (fila = real, columna = predicho) ===")
    print(" " * (w + 2) + " ".join(f"{c[:7]:>7s}" for c in clases))
    for c, fila in zip(clases, cm):
        print(f"{c:>{w}s}  " + " ".join(f"{v:7d}" for v in fila))


def penalizacion_l2(modelo):
    """Suma de los pesos al cuadrado de las capas convolucionales (regularización L2)."""
    return sum((m.weight ** 2).sum() for m in modelo.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d)))


def contar_parametros(modelo):
    """Pesos entrenables + medias y varianzas de BatchNormalization (todo lo que se guarda del modelo)."""
    return sum(p.numel() for p in modelo.parameters()) + \
        sum(b.numel() for nom, b in modelo.named_buffers() if "running" in nom)


def predecir(modelo, X, disp, lote=256):
    """Probabilidades del modelo para un array X (modo evaluación, sin aumento)."""
    modelo.eval()
    salida = []
    with torch.no_grad():
        for i in range(0, len(X), lote):
            x = torch.from_numpy(X[i:i + lote]).to(disp)
            salida.append(torch.softmax(modelo(x), dim=1).cpu().numpy())
    return np.concatenate(salida)


def latencia_cpu(modelo, x, repeticiones=200):
    """Tiempo medio de inferencia de UNA seña en CPU (ms), más cercano a lo que hará el celular que la GPU."""
    modelo = modelo.to("cpu").eval()
    muestra = torch.from_numpy(x[:1])
    with torch.no_grad():
        for _ in range(10):                                         # calentamiento (no se mide)
            modelo(muestra)
        t = time.perf_counter()
        for _ in range(repeticiones):
            modelo(muestra)
    return (time.perf_counter() - t) / repeticiones * 1000


def correr(construir_modelo, preparar_entrada, X, y_num, clases, seed, cfg=None, mostrar=True, guardar_en=None):
    """Entrena UN modelo con la configuración cfg en la partición de la semilla y lo evalúa.

    construir_modelo(n_clases, forma_entrada, dropout) -> modelo PyTorch sin entrenar
    preparar_entrada(X)                                -> X con la forma que espera ese modelo
    Devuelve un diccionario con las métricas de validación y de prueba, el historial y la matriz de confusión.
    """
    cfg = {**CONFIG, **(cfg or {})}
    disp = dispositivo()
    n = len(clases)
    X_tr, y_tr, X_va, y_va, X_te, y_te = particion(X, y_num, seed, cfg["copias_aug"], cfg["ruido_en_ceros"],
                                                   cfg["espejo"])
    Xtr, Xva, Xte = preparar_entrada(X_tr), preparar_entrada(X_va), preparar_entrada(X_te)
    del X_tr, X_va, X_te

    fijar_semilla(seed)                                             # pesos iniciales reproducibles
    modelo = construir_modelo(n, Xtr.shape[1:], dropout=cfg["dropout"]).to(disp)
    n_params = contar_parametros(modelo)
    if mostrar:
        print(f"Modelo en {disp.type.upper()} | entrada {Xtr.shape[1:]} | parámetros {n_params:,} | "
              + " ".join(f"{k}={v}" for k, v in cfg.items()))

    pesos = None
    if cfg["pesos_clase"]:
        w = compute_class_weight("balanced", classes=np.arange(n), y=y_tr)
        pesos = torch.tensor(w, dtype=torch.float32, device=disp)
    criterio = nn.CrossEntropyLoss(weight=pesos, label_smoothing=cfg["suavizado"])
    criterio_val = nn.CrossEntropyLoss(label_smoothing=cfg["suavizado"])
    optim = torch.optim.Adam(modelo.parameters(), lr=LR, eps=1e-7)
    # Si la pérdida de validación no mejora en 10 épocas, reduce la tasa de aprendizaje a la mitad
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, min_lr=1e-5)

    Xtr_t = torch.from_numpy(Xtr).to(disp)                           # todo el entrenamiento cabe en la GPU
    ytr_t = torch.from_numpy(y_tr.astype(np.int64)).to(disp)
    Xva_t = torch.from_numpy(Xva).to(disp)
    yva_t = torch.from_numpy(y_va.astype(np.int64)).to(disp)
    hist = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    epocas = cfg["epocas"]
    t0 = time.time()
    for ep in range(1, epocas + 1):
        modelo.train()
        orden = torch.randperm(len(Xtr_t), device=disp)
        total, ok = 0.0, 0
        for i in range(0, len(orden), BATCH):
            b = orden[i:i + BATCH]
            logits = modelo(Xtr_t[b])
            loss = criterio(logits, ytr_t[b])
            optim.zero_grad()
            (loss + cfg["reg_l2"] * penalizacion_l2(modelo)).backward()
            optim.step()
            total += loss.item() * len(b)
            ok += (logits.argmax(1) == ytr_t[b]).sum().item()
        modelo.eval()
        with torch.no_grad():
            lv = modelo(Xva_t)
            val_loss = criterio_val(lv, yva_t).item()
            val_acc = (lv.argmax(1) == yva_t).float().mean().item()
        plan.step(val_loss)
        hist["loss"].append(total / len(Xtr_t))
        hist["accuracy"].append(ok / len(Xtr_t))
        hist["val_loss"].append(val_loss)
        hist["val_accuracy"].append(val_acc)
        if mostrar and (ep == 1 or ep % 10 == 0):
            print(f"Época {ep:3d}/{epocas} | loss {hist['loss'][-1]:.4f} | accuracy {hist['accuracy'][-1]:.4f} | "
                  f"val_loss {val_loss:.4f} | val_accuracy {val_acc:.4f} | lr {optim.param_groups[0]['lr']:.1e}")
    minutos = (time.time() - t0) / 60
    del Xtr_t, ytr_t, Xva_t, yva_t                                   # libera la memoria de la GPU

    # ---- validación (para elegir la configuración) y prueba (20 %, para reportar)
    prob_va = predecir(modelo, Xva, disp)
    pred_va = prob_va.argmax(1)
    pred = predecir(modelo, Xte, disp).argmax(1)
    rep = classification_report(y_te, pred, labels=list(range(n)), target_names=clases, zero_division=0,
                                output_dict=True)
    vl = hist["val_loss"]
    r = {"acc_val": float((pred_va == y_va).mean()),
         "f1_val": float(f1_score(y_va, pred_va, average="macro")),
         "perdida_real_val": float(log_loss(y_va, prob_va, labels=list(range(n)))),   # sin suavizado
         "val_loss_min": float(min(vl)), "val_loss_final": float(vl[-1]),
         "subida": float(vl[-1] - min(vl)),                         # cuánto sube la curva desde su mínimo
         "acc": float((pred == y_te).mean()),
         "f1": float(f1_score(y_te, pred, average="macro")),
         "epocas": epocas, "min": minutos, "params": n_params, "kb": n_params * 4 / 1024,   # float32
         "por_sena": {c: rep[c] for c in clases},
         "cm": confusion_matrix(y_te, pred, labels=list(range(n))), "hist": hist, "pred": pred, "y_te": y_te}
    if guardar_en:
        torch.save(modelo.state_dict(), guardar_en)
    r["ms"] = latencia_cpu(modelo, Xte)
    del modelo, Xtr, Xva, Xte
    gc.collect()
    if disp.type == "cuda":
        torch.cuda.empty_cache()
    return r


def guardar_historial(ruta, filas):
    """filas: lista de (etiqueta, semilla, historial). Escribe un CSV con una fila por época."""
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("modelo,semilla,epoca,loss,accuracy,val_loss,val_accuracy\n")
        for nombre, s, h in filas:
            for e in range(len(h["loss"])):
                fh.write(f"{nombre},{s},{e + 1},{h['loss'][e]:.4f},{h['accuracy'][e]:.4f},"
                         f"{h['val_loss'][e]:.4f},{h['val_accuracy'][e]:.4f}\n")


def ejecutar(nombre, construir_modelo, preparar_entrada):
    """Evaluación final de UN modelo con CONFIG en varias semillas; guarda los CSV en data/.

    Opciones para pruebas adicionales sin tocar CONFIG (los resultados se guardan con --sufijo para no
    reemplazar los de la configuración final):
        --interpolar      rellena las manos no detectadas
        --espejo 0.5      refleja la mitad de las copias aumentadas
        --l2 0.001        cambia la regularización L2
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--interpolar", action="store_true")
    ap.add_argument("--espejo", type=float)
    ap.add_argument("--l2", type=float)
    ap.add_argument("--sufijo", default="")
    args = ap.parse_args()
    cfg = {}
    if args.interpolar:
        cfg["interpolar_manos"] = True
    if args.espejo is not None:
        cfg["espejo"] = args.espejo
    if args.l2 is not None:
        cfg["reg_l2"] = args.l2
    if cfg and not args.sufijo:
        ap.error("con --interpolar, --espejo o --l2 hay que indicar --sufijo (p. ej. --sufijo _interp) "
                 "para no reemplazar los resultados de la configuración final")
    cfg_total = {**CONFIG, **cfg}
    nombre_salida = nombre + args.sufijo
    disp = dispositivo()
    print("Dispositivo:", torch.cuda.get_device_name(0) if disp.type == "cuda" else "CPU (no se encontró GPU)")
    if cfg:
        print("Cambios respecto a CONFIG:", cfg, "| resultados con el nombre", nombre_salida)

    X, y_num, clases = cargar_datos(cfg_total["fotogramas"], cfg_total["interpolar_manos"])
    resultados = []
    for k, seed in enumerate(args.semillas):
        print("\n" + "#" * 70 + f"\n# {nombre.upper()} - PARTICIÓN CON SEMILLA {seed}\n" + "#" * 70)
        r = correr(construir_modelo, preparar_entrada, X, y_num, clases, seed, cfg,
                   guardar_en=os.path.join(DATA, f"modelo_{nombre_salida}.pt") if k == 0 else None)
        print(f"\n>>> {nombre}: accuracy en prueba = {r['acc'] * 100:.1f}%  |  F1 macro = {r['f1'] * 100:.1f}%  "
              f"|  {r['epocas']} épocas en {r['min']:.1f} min")
        print(classification_report(r["y_te"], r["pred"], labels=list(range(len(clases))), target_names=clases,
                                    digits=2, zero_division=0))
        imprimir_matriz(r["cm"], clases)
        print(f"\nTamaño: {r['kb']:.0f} KB (float32)  |  inferencia ≈ {r['ms']:.2f} ms por seña (CPU)")
        resultados.append(r)

    # ---- resumen + CSV (los lee comparar_modelos.py)
    a = np.array([r["acc"] for r in resultados]) * 100
    f = np.array([r["f1"] for r in resultados]) * 100
    print("\n" + "=" * 70 + f"\n RESUMEN {nombre.upper()} ({len(resultados)} semilla(s), prueba)\n" + "=" * 70)
    print(f"Accuracy: {a.mean():.1f} ± {a.std():.1f}%  |  F1 macro: {f.mean():.1f} ± {f.std():.1f}%  |  "
          f"parámetros: {resultados[0]['params']:,}  |  tamaño: {resultados[0]['kb']:.0f} KB  |  "
          f"{np.mean([r['ms'] for r in resultados]):.2f} ms/seña")
    lineas = ["modelo,semilla,accuracy,f1_macro,epocas,minutos,parametros,tamano_kb,ms_por_sena"]
    for s, r in zip(args.semillas, resultados):
        lineas.append(f"{nombre_salida},{s},{r['acc']:.4f},{r['f1']:.4f},{r['epocas']},{r['min']:.2f},"
                      f"{r['params']},{r['kb']:.1f},{r['ms']:.3f}")
    with open(os.path.join(DATA, f"resultados_{nombre_salida}.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    with open(os.path.join(DATA, f"por_sena_{nombre_salida}.csv"), "w", encoding="utf-8") as fh:
        fh.write("modelo,semilla,sena,precision,recall,f1,soporte\n")
        for s, r in zip(args.semillas, resultados):
            for c, m in r["por_sena"].items():
                fh.write(f"{nombre_salida},{s},{c},{m['precision']:.4f},{m['recall']:.4f},"
                         f"{m['f1-score']:.4f},{int(m['support'])}\n")
    with open(os.path.join(DATA, f"matriz_{nombre_salida}.csv"), "w", encoding="utf-8") as fh:   # suma de semillas
        fh.write("real," + ",".join(clases) + "\n")
        total = sum(r["cm"] for r in resultados)
        for c, fila in zip(clases, total):
            fh.write(c + "," + ",".join(str(int(v)) for v in fila) + "\n")
    guardar_historial(os.path.join(DATA, f"historial_{nombre_salida}.csv"),
                      [(nombre_salida, s, r["hist"]) for s, r in zip(args.semillas, resultados)])
    print(f"\nGuardado en data/: resultados_{nombre_salida}.csv, por_sena_{nombre_salida}.csv, "
          f"matriz_{nombre_salida}.csv, historial_{nombre_salida}.csv, modelo_{nombre_salida}.pt")
