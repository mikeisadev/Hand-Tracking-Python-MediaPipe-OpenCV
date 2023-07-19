# Importiamo i moduli
import cv2
import mediapipe as mp
import time

# Selezioniamo la webcam principale
camera = cv2.VideoCapture(0)

# Utilizziamo il modulo di riconoscimento delle mani
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Prepariamo delle variabili per il calcolo degli FPS
pTime = 0
cTime = 0

while True:
    # Leggi l'immagine della webcam
    success, img = camera.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Riconosci se ci sono mani nell'immagine ed esegui il loop per ogni mano prendendo i landmarks (x, y)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calcola gli FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Mostra gli FPS su schermo
    cv2.putText(img, 'FPS ' + str(int(fps)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Riconoscimento mani - Progetto Computer Vision", img)
    cv2.waitKey(1)
