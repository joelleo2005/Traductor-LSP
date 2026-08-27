import numpy as np

def jitter(seq, sigma=0.02):
    return seq + np.random.normal(0, sigma, seq.shape)

def escalar(seq, rango=(0.85, 1.15)):
    return seq * np.random.uniform(*rango)

def rotar(seq, max_ang=0.2):
    ang = np.random.uniform(-max_ang, max_ang)
    c, s = np.cos(ang), np.sin(ang)
    out = seq.copy()
    idx_xy = [(4*i, 4*i+1) for i in range(33)] + [(b, b+1) for b in range(132, 258, 3)]
    for xi, yi in idx_xy:
        x, y = out[:, xi].copy(), out[:, yi].copy()
        out[:, xi] = c*x - s*y
        out[:, yi] = s*x + c*y
    return out

def warp_temporal(seq, rango=(0.7, 1.3)):
    T = seq.shape[0]
    factor = np.random.uniform(*rango)
    n = max(2, int(round(T * factor)))
    idx = np.linspace(0, T - 1, n)
    res = seq[np.round(idx).astype(int)]
    idx2 = np.linspace(0, len(res) - 1, T).astype(int)
    return res[idx2]

def aumentar(X, y, n_copias=8):
    X_agg, y_agg = list(X), list(y)
    for seq, etiqueta in zip(X, y):
        for _ in range(n_copias):
            X_agg.append(warp_temporal(rotar(jitter(escalar(seq)))))
            y_agg.append(etiqueta)
    return np.array(X_agg), np.array(y_agg)