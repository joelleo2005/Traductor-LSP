import tempfile, os
from fastapi import FastAPI, UploadFile, File
from inferencia import predecir

app = FastAPI(title="Traductor LSP")

@app.get("/")
def inicio():
    return {"estado": "ok", "mensaje": "Backend Traductor LSP activo"}

@app.post("/predecir")
async def predecir_endpoint(video: UploadFile = File(...)):
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        ruta = tmp.name
    try:
        resultado = predecir(ruta)
        print(">>> DIAGNÓSTICO:", resultado)   # aparece en esta terminal
    finally:
        os.remove(ruta)
    return {"sena": resultado["sena"], "confianza": round(resultado["confianza"], 4)}