import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf

from normalizar import normalizar

SEÑAS_USAR = {"YO","PENSAR","MUJER","HOMBRE","CASA","VER","ESPERAR","CAMINAR","QUÉ","MAMÁ"}

X = np.load("data/X.npy")
y = np.load("data/y.npy")
X = normalizar(X)

mask = np.array([et in SEÑAS_USAR for et in y])
X, y = X[mask], y[mask]

le = LabelEncoder()
y_num = le.fit_transform(y)

X_train, X_test, ytr, yte = train_test_split(
    X, y_num, test_size=0.2, random_state=42, stratify=y_num)

model = tf.keras.models.load_model("data/modelo_lsp.keras")
pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(yte, pred, target_names=le.classes_, zero_division=0))