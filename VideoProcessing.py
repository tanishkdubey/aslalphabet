import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from tensorflow.keras.applications.resnet50 import preprocess_input

# Configuration
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']
CONFIDENCE_THRESHOLD = 0.8
HISTORY_LENGTH = 15

# Load model
model = load_model("asl_ResNet50_improved.h5")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

# Prediction history for temporal smoothing
prediction_history = deque(maxlen=HISTORY_LENGTH)


def preprocess_frame(frame):
    """Preprocess frame matching training pipeline"""
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_frame


def get_hand_roi(results, frame):
    """Extract hand region with padding"""
    h, w = frame.shape[:2]
    x_coords = [lm.x * w for lm in results.multi_hand_landmarks[0].landmark]
    y_coords = [lm.y * h for lm in results.multi_hand_landmarks[0].landmark]

    padding = 30
    xmin = max(0, int(min(x_coords)) - padding)
    xmax = min(w, int(max(x_coords)) + padding)
    ymin = max(0, int(min(y_coords)) - padding)
    ymax = min(h, int(max(y_coords)) + padding)

    return frame[ymin:ymax, xmin:xmax], (xmin, ymin, xmax, ymax)


def predict_sign(hand_img):
    """Make prediction with confidence check"""
    hand_resized = cv2.resize(hand_img, (224, 224))
    hand_processed = preprocess_input(hand_resized.copy())
    hand_expanded = np.expand_dims(hand_processed, axis=0)

    pred = model.predict(hand_expanded, verbose=0)
    confidence = np.max(pred)
    class_id = np.argmax(pred)

    return class_id, confidence, hand_resized


# Video processing loop
cap = cv2.VideoCapture("your_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess and detect hands
    rgb_frame = preprocess_frame(frame)
    results = hands.process(rgb_frame)
    output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        try:
            # Get hand ROI
            hand_roi, (xmin, ymin, xmax, ymax) = get_hand_roi(results, output_frame)

            if hand_roi.size != 0:
                # Make prediction
                class_id, confidence, processed_hand = predict_sign(hand_roi)

                # Update prediction history if confident
                if confidence > CONFIDENCE_THRESHOLD:
                    prediction_history.append(class_id)

                # Get most frequent recent prediction
                current_label = "Unknown"
                if prediction_history:
                    final_pred = max(set(prediction_history),
                                     key=prediction_history.count)
                    current_label = LABELS[final_pred]

                # Display results
                cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(output_frame, f"{current_label} ({confidence:.2f})",
                            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                # Debug view
                cv2.imshow("Hand ROI", processed_hand)

        except Exception as e:
            print(f"Processing error: {e}")

    # Display output
    cv2.imshow("ASL Recognition", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()