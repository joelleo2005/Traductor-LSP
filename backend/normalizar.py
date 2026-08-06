import numpy as np

def normalizar(X):
    """Centra los keypoints en los hombros y los escala por el ancho de hombros.
    Hace la seña invariante a DÓNDE y a qué TAMAÑO está la persona en el cuadro."""
    Xn = X.copy().astype("float32")
    N, T, D = Xn.shape
    for n in range(N):
        for t in range(T):
            f = Xn[n, t]
            lx, ly = f[11*4], f[11*4+1]   # hombro izquierdo (pose 11)
            rx, ry = f[12*4], f[12*4+1]   # hombro derecho  (pose 12)
            if lx == 0 and rx == 0:        # sin pose detectada -> dejar igual
                continue
            cx, cy = (lx+rx)/2, (ly+ry)/2
            escala = np.hypot(lx-rx, ly-ry) or 1.0
            for i in range(33):
                f[4*i]   = (f[4*i]   - cx) / escala
                f[4*i+1] = (f[4*i+1] - cy) / escala
            for base in range(132, 258, 3):
                f[base]   = (f[base]   - cx) / escala
                f[base+1] = (f[base+1] - cy) / escala
    return Xn