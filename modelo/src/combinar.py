import numpy as np
from collections import Counter

# Cargar los tres datasets ya procesados
Xa = np.load("data/X_perusil.npy"); ya = np.load("data/y_perusil.npy")
Xu = np.load("data/X_pucp.npy");    yu = np.load("data/y_pucp.npy")
Xp = np.load("data/X_aprendo.npy"); yp = np.load("data/y_aprendo.npy")

# Unirlos en uno solo
X = np.concatenate([Xa, Xu, Xp], axis=0)
y = np.concatenate([ya, yu, yp], axis=0)

# Guardar como X.npy / y.npy (lo que lee el entrenamiento)
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("Dataset combinado (PUCP + PeruSIL + Aprendo):", X.shape, y.shape)
print("Clases:", len(set(y)))
print("Por seña:")
for s, n in sorted(Counter(y).items(), key=lambda x: -x[1]):
    print(f"  {s}: {n}")