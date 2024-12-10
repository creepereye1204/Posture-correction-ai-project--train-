import os
import sys
import mediapipe as mp
import numpy as np
import cv2
import joblib
import pandas as pd
import time
import winsound
import threading

NOSE = 0
LEFT_EYE = 7
RIGHT_EYE = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class PostureCorrection:
    def __init__(self, model_path):
        self.pose = mp_pose.Pose()
        self.model = joblib.load(model_path)
        self.abnormal_duration = 0
        self.alarm_threshold = 2
        self.last_prediction = 0
        self.alarm_active = False

    def process_image(self, image):
        pose_results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        black_image = np.zeros_like(image)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            results = self._preprocess_landmarks(landmarks)
            self._draw_landmarks(black_image, landmarks)
            black_image = cv2.flip(black_image, 1)

            features = np.array(results).reshape(1, -1)
            features_df = pd.DataFrame(features, columns=["Nose And Shoulder Angle", "Eye Angle"])
            prediction = self.model.predict(features_df)[0]

            status_text = "Normal" if prediction == 0 else "Abnormal"
            cv2.putText(black_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            self._display_angles(black_image, results)

            if prediction == 1:
                self.abnormal_duration += 1 / 30
                if self.abnormal_duration >= self.alarm_threshold and not self.alarm_active:
                    self.alarm_active = True
                    threading.Thread(target=self.sound_alarm).start()
            else:
                self.abnormal_duration = 0
                self.alarm_active = False

        return black_image

    def sound_alarm(self):
        frequency = 1000
        duration = 1000
        while self.alarm_active:
            winsound.Beep(frequency, duration)
            time.sleep(1)

    def _draw_landmarks(self, image, landmarks):
        h, w, _ = image.shape
        for landmark in landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    def _preprocess_landmarks(self, landmarks):
        points = [self._change_to_np(landmark) for landmark in landmarks]
        nose, left_eye, right_eye, left_shoulder, right_shoulder = (
            points[NOSE],
            points[LEFT_EYE],
            points[RIGHT_EYE],
            points[LEFT_SHOULDER],
            points[RIGHT_SHOULDER],
        )

        return self._preprocess(nose, left_eye, right_eye, left_shoulder, right_shoulder)

    def _preprocess(self, nose, left_eye, right_eye, left_shoulder, right_shoulder):
        parameters_combination = [
            [(nose, left_shoulder, right_shoulder), self._calculate_angle],
            [(nose, left_eye, right_eye), self._calculate_angle],
        ]
        return [func(*params) for params, func in parameters_combination]

    def _display_angles(self, image, results):
        angle_labels = [
            "Nose And Shoulder Angle",
            "Eye Angle",
        ]
        for i, angle in enumerate(results):
            cv2.putText(
                image,
                f"{angle_labels[i]}: {angle:.2f}",
                (10, 60 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def _change_to_np(self, joint):
        return np.array([joint.x, joint.y])

    def change_to_vector(func):
        def wrapper(self, A, B, C=None):
            AB = B - A
            AC = C - A if C is not None else None
            return func(self, AB, AC) if AC is not None else func(self, AB)

        return wrapper

    @change_to_vector
    def _calculate_ratio(self, AB, AC):
        return np.linalg.norm(AB) / np.linalg.norm(AC)

    @change_to_vector
    def _calculate_angle(self, AB, AC=None):
        if AC is not None:
            cos_theta = np.dot(AB, AC) / (np.linalg.norm(AB) * np.linalg.norm(AC))
            return np.degrees(np.arccos(cos_theta))
        return np.degrees(np.arctan2(AB[1], AB[0]))

    def release_resources(self):
        self.pose.close()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if getattr(sys, "frozen", False):
        model_path = os.path.join(sys._MEIPASS, "random_forest_model12.pkl")
    else:
        model_path = "./random_forest_model12.pkl"

    posture_model = PostureCorrection(model_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = posture_model.process_image(frame)
        cv2.imshow("Video Stream", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    posture_model.release_resources()
