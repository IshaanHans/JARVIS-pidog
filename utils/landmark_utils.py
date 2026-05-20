import numpy as np

def normalise_landmarks(vector):
    pts = vector.reshape(-1, 3)
    pts[:, :2] = pts[:, :2] - pts[0, :2]
    scale = np.max(np.abs(pts[:, :2]))
    if scale > 0:
        pts[:, :2] = pts[:, :2] / scale
    return pts.flatten().astype(np.float32)
