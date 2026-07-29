import os, glob
from eaf import leer_anotaciones

DATASET_DIR = r"C:\Users\Joel\Downloads\2015-2023_LSP_peru1235_PUCP305_glosas (1)\5. Segundo avance (corregido)"
clases = set(d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)))

carpeta = os.path.join(DATASET_DIR, "AMIGO")
oraciones = [v for v in glob.glob(os.path.join(carpeta, "*.mp4"))
             if "ORACION" in os.path.basename(v).upper()]
print("Videos de oración en AMIGO:", len(oraciones))
for v in oraciones:
    eaf = v.replace(".mp4", ".eaf")
    print("¿existe el eaf?", os.path.exists(eaf))
    anots = leer_anotaciones(eaf)
    print("anotaciones encontradas:", len(anots))
    for etiqueta, ini, fin in anots:
        base = etiqueta.rsplit("_", 1)[0]
        print(f"   {etiqueta} -> base='{base}'  ¿en clases? {base in clases}")