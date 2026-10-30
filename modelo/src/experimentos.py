"""
Pruebas de la fase de ajuste (sección de experimentación de la tesis).

Entrena Conv1D y Conv2D con una sola partición (semilla 1) cambiando una cosa a la vez, desde la
línea base hasta la configuración final de comun.CONFIG:

    ajustes  -> línea base y ajustes 1 a 5 (cada uno parte del anterior); se reportan en PRUEBA
    grilla1  -> suavizado x pesos por clase x copias de aumento (100 épocas); se elige con VALIDACIÓN
    grilla2  -> regularización L2 x dropout con suavizado 0.05 (100 épocas); se elige con VALIDACIÓN
    ajuste7  -> sobre la configuración final: interpolar las manos no detectadas y/o aumento con espejo

Ejecutar desde la carpeta modelo/:
    python src/experimentos.py                        # todo (ajustes, grilla1, grilla2 y ajuste7), ambos modelos
    python src/experimentos.py --grupo ajustes        # solo un grupo
    python src/experimentos.py --modelos conv2d       # solo un modelo
    python src/experimentos.py --grupo ajuste7        # solo las pruebas de interpolación y espejo

Guarda en data/experimentos/:
    experimentos.csv                 una fila por prueba y modelo (configuración + métricas de validación y prueba)
    historial_<prueba>_<modelo>.csv  curvas de pérdida y accuracy por época (para las figuras)

Si se detiene, al volver a ejecutarlo se salta las pruebas que ya están en experimentos.csv.
Para empezar de cero, borrar la carpeta data/experimentos/.
"""
import argparse
import csv
import itertools
import os

import torch

import comun
from comun import cargar_datos, correr, guardar_historial
from entrenar_conv1d import Conv1DModelo, preparar_entrada
from entrenar_conv2d import Conv2DModelo, a_imagen

MODELOS = {"conv1d": (Conv1DModelo, preparar_entrada), "conv2d": (Conv2DModelo, a_imagen)}
CARPETA = os.path.join(comun.DATA, "experimentos")
RUTA = os.path.join(CARPETA, "experimentos.csv")

# ===== Línea base: la primera versión del entrenamiento =====
BASE = {"fotogramas": 30, "epocas": 200, "suavizado": 0.0, "pesos_clase": False, "copias_aug": 8,
        "ruido_en_ceros": True, "reg_l2": 1e-4, "dropout": 0.5, "interpolar_manos": False, "espejo": 0.0}


def con(cfg, **cambios):
    return {**cfg, **cambios}


# ===== Ajustes acumulativos (cada uno parte del anterior) =====
AJ1 = con(BASE, fotogramas=48, pesos_clase=True)           # 1. más fotogramas + pesos por clase
AJ2 = con(AJ1, suavizado=0.1)                              # 2. suavizado de etiquetas
AJ3 = con(AJ2, ruido_en_ceros=False)                       # 3. corrección del aumento de datos
AJ4 = con(AJ3, suavizado=0.0, pesos_clase=False)           # 4. sin suavizado ni pesos (resultado de la grilla 1)
AJ5A = con(AJ4, suavizado=0.1, epocas=100)                 # 5a. suavizado 0.1 con 100 épocas
AJ5B = con(AJ4, suavizado=0.05, epocas=100)                # 5b. suavizado 0.05 con 100 épocas
GRUPOS = {
    "ajustes": [("base", BASE), ("aj1_48f_pesos", AJ1), ("aj2_suavizado", AJ2), ("aj3_aumento", AJ3),
                ("aj4_sin_suav_ni_pesos", AJ4), ("aj5a_suav0.1_100ep", AJ5A), ("aj5b_suav0.05_100ep", AJ5B)],
    "grilla1": [(f"g1_suav{s}_pesos{int(p)}_copias{c}", con(AJ3, epocas=100, suavizado=s, pesos_clase=p, copias_aug=c))
                for s, p, c in itertools.product([0.0, 0.1], [False, True], [4, 8])],
    "grilla2": [(f"g2_l2{l2:g}_drop{d}", con(AJ5B, reg_l2=l2, dropout=d))
                for l2, d in itertools.product([1e-4, 1e-3], [0.5, 0.7])],
    "ajuste7": [("aj7a_interp_manos", con(AJ5B, interpolar_manos=True)),
                ("aj7b_espejo", con(AJ5B, espejo=0.5)),
                ("aj7c_interp_espejo", con(AJ5B, interpolar_manos=True, espejo=0.5))],
}
COLUMNAS = (["grupo", "prueba", "modelo", "semilla"] + list(comun.CONFIG) +
            ["acc_val", "f1_val", "perdida_real_val", "val_loss_min", "val_loss_final", "subida",
             "acc_test", "f1_test", "parametros", "minutos"])


def actualizar_columnas():
    """Si experimentos.csv es de una versión anterior (sin las columnas nuevas de CONFIG), las agrega
    con su valor por defecto, que es el que tenían esas pruebas."""
    if not os.path.exists(RUTA):
        return
    with open(RUTA, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
        fh.seek(0)
        cabecera = fh.readline().strip().split(",")
    if cabecera == COLUMNAS:
        return
    with open(RUTA, "w", encoding="utf-8") as fh:
        fh.write(",".join(COLUMNAS) + "\n")
        for f in filas:
            fh.write(",".join(str(f.get(c, comun.CONFIG.get(c, ""))) for c in COLUMNAS) + "\n")
    print("experimentos.csv actualizado con las columnas nuevas:", [c for c in COLUMNAS if c not in cabecera])


def hechas():
    """Pruebas que ya están en experimentos.csv (para no repetirlas)."""
    if not os.path.exists(RUTA):
        return set()
    with open(RUTA, encoding="utf-8") as fh:
        return {(f["prueba"], f["modelo"], f["semilla"]) for f in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupo", choices=list(GRUPOS) + ["todo"], default="todo")
    ap.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS))
    ap.add_argument("--semilla", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(CARPETA, exist_ok=True)
    actualizar_columnas()
    if not os.path.exists(RUTA):
        with open(RUTA, "w", encoding="utf-8") as fh:
            fh.write(",".join(COLUMNAS) + "\n")
    print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no se encontró GPU)")

    grupos = list(GRUPOS) if args.grupo == "todo" else [args.grupo]
    pendientes = [(g, p, cfg) for g in grupos for p, cfg in GRUPOS[g]]
    ya = hechas()
    datos = {}                                                      # datos cargados (fotogramas, interpolación)
    for k, (grupo, prueba, cfg) in enumerate(pendientes, 1):
        for nombre in args.modelos:
            if (prueba, nombre, str(args.semilla)) in ya:
                print(f"[{k}/{len(pendientes)}] {prueba} {nombre}: ya está, se salta")
                continue
            print("\n" + "#" * 80 + f"\n# [{k}/{len(pendientes)}] {grupo} - {prueba} - {nombre.upper()} "
                  f"- semilla {args.semilla}\n" + "#" * 80)
            clave = (cfg["fotogramas"], cfg["interpolar_manos"])
            if clave not in datos:
                datos[clave] = cargar_datos(*clave)
            X, y_num, clases = datos[clave]
            construir, preparar = MODELOS[nombre]
            r = correr(construir, preparar, X, y_num, clases, args.semilla, cfg)
            print(f">>> validación: acc {r['acc_val'] * 100:.1f}% | F1 {r['f1_val'] * 100:.1f}% | "
                  f"pérdida real {r['perdida_real_val']:.2f} | val_loss mín {r['val_loss_min']:.2f} -> "
                  f"final {r['val_loss_final']:.2f} (subida {r['subida']:.2f})")
            print(f">>> prueba:     acc {r['acc'] * 100:.1f}% | F1 {r['f1'] * 100:.1f}% | {r['min']:.1f} min")
            fila = ([grupo, prueba, nombre, args.semilla] + [cfg[c] for c in comun.CONFIG] +
                    [f"{r['acc_val']:.4f}", f"{r['f1_val']:.4f}", f"{r['perdida_real_val']:.4f}",
                     f"{r['val_loss_min']:.4f}", f"{r['val_loss_final']:.4f}", f"{r['subida']:.4f}",
                     f"{r['acc']:.4f}", f"{r['f1']:.4f}", r["params"], f"{r['min']:.2f}"])
            with open(RUTA, "a", encoding="utf-8") as fh:              # se guarda al terminar cada prueba
                fh.write(",".join(str(v) for v in fila) + "\n")
            guardar_historial(os.path.join(CARPETA, f"historial_{prueba}_{nombre}.csv"),
                              [(nombre, args.semilla, r["hist"])])

    # ---- resumen
    print("\n" + "=" * 100 + f"\n RESUMEN (semilla {args.semilla})\n" + "=" * 100)
    print(f"{'prueba':<32}{'modelo':<8}{'acc val':>8}{'F1 val':>8}{'subida':>8}{'acc test':>10}{'F1 test':>9}")
    with open(RUTA, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if f["grupo"] in grupos and f["modelo"] in args.modelos and f["semilla"] == str(args.semilla):
                print(f"{f['prueba']:<32}{f['modelo']:<8}{float(f['acc_val']) * 100:>7.1f}%"
                      f"{float(f['f1_val']) * 100:>7.1f}%{float(f['subida']):>8.2f}"
                      f"{float(f['acc_test']) * 100:>9.1f}%{float(f['f1_test']) * 100:>8.1f}%")
    print(f"\nGuardado en {RUTA}")


if __name__ == "__main__":
    main()
