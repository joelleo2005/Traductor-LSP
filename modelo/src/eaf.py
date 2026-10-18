"""
Lectura de anotaciones ELAN (.eaf).

Los videos de oraciones del dataset PUCP 305 vienen con un archivo .eaf (XML de ELAN) que indica,
en la capa "GLOSA_IA", qué seña aparece y en qué intervalo de tiempo (milisegundos).
procesar_todo.py usa estas anotaciones para recortar cada seña dentro de la oración.

Prueba rápida (desde modelo/):  python src/eaf.py
"""
import xml.etree.ElementTree as ET


def leer_anotaciones(ruta_eaf, tier_id="GLOSA_IA"):
    """Lee un .eaf y devuelve [(etiqueta, inicio_ms, fin_ms), ...] de la capa indicada."""
    raiz = ET.parse(ruta_eaf).getroot()

    # 1. Marcas de tiempo: {id_de_la_marca: milisegundos}
    tiempos = {s.get("TIME_SLOT_ID"): int(s.get("TIME_VALUE"))
               for s in raiz.iter("TIME_SLOT")}

    # 2. Recorrer las capas y quedarnos con la que nos interesa (GLOSA_IA)
    anotaciones = []
    for tier in raiz.iter("TIER"):
        if tier.get("TIER_ID") != tier_id:
            continue
        for ann in tier.iter("ALIGNABLE_ANNOTATION"):
            etiqueta = ann.find("ANNOTATION_VALUE").text
            inicio = tiempos[ann.get("TIME_SLOT_REF1")]   # id de inicio → ms
            fin = tiempos[ann.get("TIME_SLOT_REF2")]       # id de fin → ms
            anotaciones.append((etiqueta, inicio, fin))

    return anotaciones


if __name__ == "__main__":
    # Ejemplo: imprime las señas anotadas en un video del PUCP 305 (cambiar la ruta según la PC)
    ruta = r"C:\Users\Joel\Downloads\2015-2023_LSP_peru1235_PUCP305_glosas (1)\5. Segundo avance (corregido)\ABRIR\ABRIR_1.eaf"
    print(leer_anotaciones(ruta))
