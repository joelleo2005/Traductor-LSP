import numpy as np

def extraer_keypoints(results):
    pose = np.array([[p.x, p.y, p.z, p.visibility]
                     for p in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)
    mano_izq = np.array([[p.x, p.y, p.z]
                         for p in results.left_hand_landmarks.landmark]).flatten() \
               if results.left_hand_landmarks else np.zeros(21 * 3)
    mano_der = np.array([[p.x, p.y, p.z]
                         for p in results.right_hand_landmarks.landmark]).flatten() \
               if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, mano_izq, mano_der])   # 258