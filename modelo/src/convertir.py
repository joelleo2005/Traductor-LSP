import numpy as np
import tensorflow as tf

# 1. Cargar el modelo entrenado
model = tf.keras.models.load_model("data/modelo_lsp.keras")

# 2. Convertir a TensorFlow Lite (Conv1D convierte limpio, sin Flex)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Guardar el .tflite (esto es lo que va a la app)
with open("data/modelo_lsp.tflite", "wb") as f:
    f.write(tflite_model)

# 4. Guardar las clases como texto plano (para que la app las lea)
clases = np.load("data/clases.npy", allow_pickle=True)
with open("data/clases.txt", "w", encoding="utf-8") as f:
    for c in clases:
        f.write(str(c) + "\n")

print("Modelo convertido:", round(len(tflite_model)/1024, 1), "KB")
print("Clases:", list(clases))