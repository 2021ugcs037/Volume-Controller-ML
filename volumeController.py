import cv2
import mediapipe as mp
import time
import math
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get the volume range
minVol, maxVol, _ = volume.GetVolumeRange()

# Main loop
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = list(enumerate(hand_landmarks.landmark))
            h, w, c = img.shape

            px1, py1 = int(landmarks[4][1].x * w), int(landmarks[4][1].y * h)
            px2, py2 = int(landmarks[8][1].x * w), int(landmarks[8][1].y * h)

            cv2.circle(img, (px1, py1), 20, (255, 0, 0), -1)
            cv2.circle(img, (px2, py2), 20, (255, 0, 0), -1)

            if all([px1, py1, px2, py2]):
                dist = math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
                cv2.line(img, (px1, py1), (px2, py2), (27, 228, 20), 3)

                # Convert the distance to volume level
                vol = np.interp(dist, [10, 270], [minVol, maxVol])
                vol = np.clip(vol, minVol, maxVol)  # Ensure vol is within the range
                volume.SetMasterVolumeLevel(vol, None)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
