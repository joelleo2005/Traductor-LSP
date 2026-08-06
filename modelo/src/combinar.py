import numpy as np
from collections import Counter

# Cargar los dos datasets ya procesados
Xp = np.load("data/X_perusil.npy")
yp = np.load("data/y_perusil.npy")
Xu = np.load("data/X_pucp.npy")
yu = np.load("data/y_pucp.npy")

# Unirlos en uno solo
X = np.concatenate([Xp, Xu], axis=0)
y = np.concatenate([yp, yu], axis=0)

# Guardar como X.npy / y.npy (lo que lee el entrenamiento)
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("Dataset combinado:", X.shape, y.shape)
print("Clases:", len(set(y)))
print("Por seña:")
for s, n in sorted(Counter(y).items(), key=lambda x: -x[1]):
    print(f"  {s}: {n}")