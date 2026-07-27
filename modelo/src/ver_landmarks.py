import cv2
import mediapipe as mp

# MediaPipe Holistic detecta cuerpo + manos + cara en un solo modelo
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Ruta a UN video de prueba (ajústala a donde tengas el dataset)
video_path = r"C:\Users\Joel\Downloads\2015-2023_LSP_peru1235_PUCP305_glosas (1)\5. Segundo avance (corregido)\ABRIR\ABRIR_1.mp4"

cap = cv2.VideoCapture(video_path)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # se acabó el video

        # MediaPipe trabaja en RGB; OpenCV lee en BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)      # <-- aquí ocurre la magia
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Dibujar los puntos detectados sobre el video
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("LSP - landmarks", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):  # presiona 'q' para salir
            break

cap.release()
cv2.destroyAllWindows()
#Qué hace cada parte:

##Holistic es el modelo de MediaPipe que detecta pose (33 puntos), mano izquierda (21), mano derecha (21) y cara. Para señas, las manos y la pose son lo que importa.
##El bucle while lee el video frame por frame.
##holistic.process(image) es donde MediaPipe analiza cada frame y devuelve las coordenadas de los puntos.
##draw_landmarks los dibuja para que veas con tus propios ojos que funciona.##