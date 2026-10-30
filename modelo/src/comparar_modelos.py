"""
Paso 3 del pipeline: tabla comparativa de los modelos (Conv1D y Conv2D) en Excel.

Lee los CSV que generan entrenar_conv1d.py y entrenar_conv2d.py (data/resultados_*,
por_sena_*, matriz_*, historial_*) y crea data/resultados.xlsx con las hojas:
    Resumen, Por semilla, Por seña, Detalle señas, Matriz <modelo> (una por modelo), Historial
Solo incluye los modelos que ya fueron entrenados (los que tienen su resultados_<modelo>.csv).

Ejecutar desde la carpeta modelo/, después de entrenar los modelos:
    python src/comparar_modelos.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

NOMBRE = {"conv1d": "Conv1D", "conv2d": "Conv2D"}                        # nombre para mostrar
MODELOS = tuple(m for m in NOMBRE if os.path.exists(f"data/resultados_{m}.csv"))   # solo los entrenados
SALIDA = "data/resultados.xlsx"


def leer(ruta):
    """Lee un CSV y devuelve una lista de diccionarios (uno por fila)."""
    with open(ruta, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pct(v):
    """Convierte una proporción (0.7234) a porcentaje con un decimal (72.3)."""
    return round(float(v) * 100, 1)


# ---- Lectura de los CSV de cada modelo
res = {m: leer(f"data/resultados_{m}.csv") for m in MODELOS}
sen = {m: leer(f"data/por_sena_{m}.csv") for m in MODELOS}

# Las semillas deben ser las mismas en todos los modelos (misma partición de datos)
semillas = [r["semilla"] for r in res[MODELOS[0]]]
if any(semillas != [r["semilla"] for r in res[m]] for m in MODELOS):
    print("OJO: los modelos se corrieron con semillas distintas -> la comparación no es justa")

# ---- Hoja 1: Resumen (media y desviación estándar de las semillas)
resumen = [["Modelo", "Accuracy media (%)", "Accuracy desv. (%)", "F1 macro media (%)", "F1 macro desv. (%)",
            "Parámetros", "Tamaño (KB)", "Inferencia (ms/seña)", "Épocas (media)", "N° semillas"]]
for m in MODELOS:
    r = res[m]
    a = np.array([pct(x["accuracy"]) for x in r])
    f = np.array([pct(x["f1_macro"]) for x in r])
    resumen.append([m.upper(), round(a.mean(), 1), round(a.std(), 1), round(f.mean(), 1), round(f.std(), 1),
                    int(r[0]["parametros"]), round(float(r[0]["tamano_kb"])),
                    round(np.mean([float(x["ms_por_sena"]) for x in r]), 2),
                    round(np.mean([int(x["epocas"]) for x in r])), len(r)])

# ---- Hoja 2: Por semilla
por_semilla = [["Modelo", "Semilla", "Accuracy (%)", "F1 macro (%)", "Épocas", "Minutos"]]
for m in MODELOS:
    for x in res[m]:
        por_semilla.append([m.upper(), int(x["semilla"]), pct(x["accuracy"]), pct(x["f1_macro"]),
                            int(x["epocas"]), float(x["minutos"])])

# ---- Hoja 3: Por seña (promedio de las semillas)
prom = defaultdict(lambda: defaultdict(list))          # prom[seña][(modelo, métrica)] = [valores]
for m in MODELOS:
    for x in sen[m]:
        for met in ("precision", "recall", "f1"):
            prom[x["sena"]][(m, met)].append(float(x[met]))
hay_1d_2d = "conv1d" in MODELOS and "conv2d" in MODELOS
por_sena = [["Seña"] + [f"{col} {NOMBRE[m]} (%)" for m in MODELOS for col in ("Precisión", "Recall", "F1")]
            + (["Diferencia F1 (2D - 1D)"] if hay_1d_2d else [])]
for s in sorted(prom):
    v = {k: pct(np.mean(lst)) for k, lst in prom[s].items()}
    fila = [s] + [v[(m, met)] for m in MODELOS for met in ("precision", "recall", "f1")]
    if hay_1d_2d:
        fila.append(round(v[("conv2d", "f1")] - v[("conv1d", "f1")], 1))
    por_sena.append(fila)

# ---- Hoja 4: Detalle (crudo, cada seña en cada semilla)
detalle = [["Modelo", "Semilla", "Seña", "Precisión (%)", "Recall (%)", "F1 (%)", "N° muestras de prueba"]]
for m in MODELOS:
    for x in sen[m]:
        detalle.append([m.upper(), int(x["semilla"]), x["sena"], pct(x["precision"]), pct(x["recall"]),
                        pct(x["f1"]), int(x["soporte"])])

# ---- Hojas de matrices de confusión, una por modelo (suma de las semillas)
matrices = {}
for m in MODELOS:
    with open(f"data/matriz_{m}.csv", encoding="utf-8") as fh:
        filas_m = list(csv.reader(fh))
    tabla = [["Real / Predicho"] + filas_m[0][1:] + ["Total", "Aciertos (%)"]]
    for f in filas_m[1:]:
        v = [int(x) for x in f[1:]]
        i = filas_m[0][1:].index(f[0])                 # posición de la diagonal (aciertos)
        tabla.append([f[0]] + v + [sum(v), round(100 * v[i] / max(1, sum(v)), 1)])
    matrices[m] = tabla

# ---- Última hoja: historial de entrenamiento
historial = [["Modelo", "Semilla", "Época", "Loss entrenamiento", "Accuracy entrenamiento (%)",
              "Loss validación", "Accuracy validación (%)"]]
for m in MODELOS:
    for x in leer(f"data/historial_{m}.csv"):
        historial.append([m.upper(), int(x["semilla"]), int(x["epoca"]), float(x["loss"]), pct(x["accuracy"]),
                          float(x["val_loss"]), pct(x["val_accuracy"])])

# ---- Escribir el Excel
wb = Workbook()
wb.remove(wb.active)                                     # quita la hoja vacía por defecto
hojas = ([("Resumen", resumen), ("Por semilla", por_semilla), ("Por seña", por_sena), ("Detalle señas", detalle)]
         + [(f"Matriz {NOMBRE[m]}", matrices[m]) for m in MODELOS] + [("Historial", historial)])
for nombre, tabla in hojas:
    ws = wb.create_sheet(nombre)
    for fila in tabla:
        ws.append(fila)
    for c in ws[1]:                                      # encabezado en negrita con fondo
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for col in ws.columns:                               # ancho de columnas
        ws.column_dimensions[col[0].column_letter].width = max(10, min(22, len(str(col[0].value)) + 2))
    if nombre.startswith("Matriz"):                     # diagonal (aciertos) en verde y negrita
        for k in range(2, ws.max_row + 1):
            c = ws.cell(row=k, column=k)
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="C6EFCE")
            ws.cell(row=k, column=1).font = Font(bold=True)
    # Inmoviliza el encabezado (y la primera columna en las matrices)
    ws.freeze_panes = "B2" if nombre.startswith("Matriz") else "A2"
wb.save(SALIDA)

# ---- Mostrar en pantalla
print("=" * 78 + "\n RESUMEN COMPARATIVO (conjunto de prueba)\n" + "=" * 78)
for fila in resumen[1:]:
    print(f"{fila[0]:7s} accuracy {fila[1]:5.1f} ± {fila[2]:4.1f}%  |  F1 {fila[3]:5.1f} ± {fila[4]:4.1f}%  |  "
          f"{fila[5]:,} parámetros  |  {fila[6]} KB  |  {fila[7]} ms/seña")
print(f"Semillas: {', '.join(semillas)}")
print(f"\nGuardado: {SALIDA}  ({len(hojas)} hojas, incluidas las matrices de confusión)")