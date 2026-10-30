# Datasets

Los videos utilizados **no se incluyen** en el repositorio ni en el archivo `.zip` porque superan 1 GB y se distribuyen con
licencias propias. Son públicos y se descargan del **Archivo Digital de Lenguas Peruanas** del Portal de Datos Abiertos de la
Pontificia Universidad Católica del Perú (PUCP): https://datos.pucp.edu.pe/dataverse/adlp

| Dataset | Enlace público | Licencia | Variable en `procesar_todo.py` | Carpeta que se usa |
|---|---|---|---|---|
| LSP - PUCP 305 (glosas) | https://hdl.handle.net/20.500.12534/JU4OLG | CC BY-NC 4.0 (uso no comercial) | `PUCP` | `5. Segundo avance (corregido)` |
| LSP - Aprendo en Casa 2020 | https://hdl.handle.net/20.500.12534/HDOAGH | CC0 | `APRENDO` | `Videos/SEGMENTED_SIGN` |
| LSP - PUCP-DGI156 | https://hdl.handle.net/20.500.12534/OJYYYS | CC0 | `PERUSIL` | `Videos/SEGMENTED_SIGN_ADJUSTED` |

## Pasos

1. Descargar y descomprimir los tres datasets desde los enlaces anteriores.
2. Editar las rutas `PERUSIL`, `APRENDO` y `PUCP` al inicio de `modelo/src/procesar_todo.py`.
3. Desde la carpeta `modelo/`, ejecutar `python src/procesar_todo.py`. Se generan `data/X.npy` (landmarks de 1331 señas de
   18 glosas, 48 fotogramas × 258 valores) y `data/y.npy` (nombre de cada seña).

Para el entrenamiento se usan 10 señas (YO, PENSAR, MUJER, MAMA, QUE, VER, COMER, BIEN, CAMINAR y CASA), con un total de
946 muestras. Las instrucciones completas están en [`README.md`](README.md).
