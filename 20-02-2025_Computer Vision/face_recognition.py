from typing import Union, Tuple
import cv2
import mediapipe as mp
import numpy as np
import math
import datetime


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bounding_box = detection.bounding_box
        start_point = bounding_box.origin_x, bounding_box.origin_y
        end_point = (
            bounding_box.origin_x + bounding_box.width,
            bounding_box.origin_y + bounding_box.height,
        )
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(
                keypoint.x, keypoint.y, width, height
            )
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = "" if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        text_location = (
            MARGIN + bounding_box.origin_x,
            MARGIN + ROW_SIZE + bounding_box.origin_y,
        )
        cv2.putText(
            annotated_image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return annotated_image


# Abbreviazioni dei nomi delle classi
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Crea un oggetto FaceDetector
base_options = BaseOptions(
    model_asset_path="blaze_face_short_range.tflite", delegate=BaseOptions.Delegate.CPU
)

options = FaceDetectorOptions(
    base_options=base_options, running_mode=VisionRunningMode.VIDEO
)

detector = FaceDetector.create_from_options(options)
cap = cv2.VideoCapture(0)
timestamp = 0

while True:
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Carica il frame
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Analizza il frame per riconoscere un viso(dopo aver aggiornato il timestamp)
    timestamp += 33
    detection_result = detector.detect_for_video(image, timestamp)

    # Crea una copia del frame e visualizzalo annotato con i bounding boxes
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", annotated_image)

    # Se riconoscimento andato a buon fine
    print(
        f"{datetime.datetime.now()}: Rilevati {len(detection_result.detections)} visi"
    )

detector.close()
cap.release()
cv2.destroyAllWindows()
