import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from normalizar import normalizar
from augmentation import aumentar

# ===== CONFIGURACIÓN =====
MIN_EJEMPLOS = 8
N_COPIAS_AUG = 8
EPOCHS       = 150

# 1. Cargar y normalizar
X = np.load("data/X.npy")
y = np.load("data/y.npy")
X = normalizar(X)
print("Dataset original:", X.shape, "-", len(set(y)), "clases")

# 2. Filtrar señas con pocos ejemplos
conteo = Counter(y)
clases_ok = {c for c, n in conteo.items() if n >= MIN_EJEMPLOS}
mask = np.array([et in clases_ok for et in y])
X, y = X[mask], y[mask]
print(f"Tras filtrar (>= {MIN_EJEMPLOS} ejemplos):", X.shape, "-", len(clases_ok), "clases")

# 3. Etiquetas texto -> números
le = LabelEncoder()
y_num = le.fit_transform(y)
n_clases = len(le.classes_)

# 4. Split estratificado
X_train, X_test, ytr, yte = train_test_split(
    X, y_num, test_size=0.2, random_state=42, stratify=y_num)
print("Train:", X_train.shape, " Test:", X_test.shape)

# 5. Augmentation SOLO en train
X_train, ytr = aumentar(X_train, ytr, n_copias=N_COPIAS_AUG)
print("Train tras augmentation:", X_train.shape)

y_train = to_categorical(ytr, num_classes=n_clases)
y_test  = to_categorical(yte, num_classes=n_clases)

# 6. Modelo Conv1D (convierte limpio a TFLite, sin Flex)
model = Sequential([
    Conv1D(64, 3, activation="relu", padding="same",
           kernel_regularizer=l2(1e-4), input_shape=(30, 258)),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(128, 3, activation="relu", padding="same",
           kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(n_clases, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 7. Entrenar
early = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=EPOCHS, batch_size=16,
          callbacks=[early])

# 8. Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"\n>>> Precisión en test: {acc*100:.1f}%")

# 9. Guardar
model.save("data/modelo_lsp.keras")
np.save("data/clases.npy", le.classes_)
print("Modelo y clases guardados en data/")