import cv2
import math
import mediapipe as mp
import pyautogui
from datetime import datetime as dt


def resize(image):
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    # cv2.imshow("frame", img)
    return img


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = BaseOptions(
    model_asset_path="./gesture_recognizer.task", delegate=BaseOptions.Delegate.CPU
)
options = GestureRecognizerOptions(
    base_options=base_options, running_mode=VisionRunningMode.VIDEO, num_hands=1
)
recognizer = GestureRecognizer.create_from_options(options)
cap = cv2.VideoCapture(0)
timestamp = 0

while True:
    ret, frame = cap.read()
    # frame = resize(frame)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow("Frame", cv2.flip(frame, 1))

    # cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    timestamp += 33
    recognition_result = recognizer.recognize_for_video(image, timestamp)

    if recognition_result.gestures:
        segno = recognition_result.gestures[0][0]
        mano = recognition_result.handedness[0][0]

        print(f"{dt.now()}: {segno.category_name}: {round(segno.score * 100, 2)} - {mano.display_name}: {round(mano.score * 100, 2)}")

        if segno.category_name == "Thumb_Up":
            print("Moving cursor up")
            pyautogui.move(0, -5)
        elif segno.category_name == "Thumb_Down":
            print("Moving cursor down")
            pyautogui.move(0, 5)
        elif segno.category_name == "Victory":
            print("Moving cursor left")
            pyautogui.move(-5, 0)
        elif segno.category_name == "Open_Palm":
            print("Moving cursor right")
            pyautogui.move(5, 0)
        elif segno.category_name == "Closed_Fist":
            print("Double clicking on the current cursor position")
            pyautogui.doubleClick(interval=0.25)
            # pyautogui.click(interval=0.25)
            # pyautogui.hotkey('command', 'o')

        # multi_hand_landmarks = recognition_result.hand_landmarks[0]
        # print(multi_hand_landmarks)
        # print(recognition_result)

recognizer.close()
cap.release()
cv2.destroyAllWindows()
