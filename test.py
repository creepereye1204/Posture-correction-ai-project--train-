import mediapipe as mp
import numpy as np
import cv2


NOSE = 0
LEFT_EYE = 7
RIGHT_EYE = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class Recording:
    def __init__(self):
        self.pose = mp_pose.Pose()

    def process_image(self, image):
        black_image = np.ones_like(image)
        pose_results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if pose_results.pose_landmarks:

            nose_landmark = self._change_to_np(pose_results.pose_landmarks.landmark[NOSE])
            left_eye_landmark = self._change_to_np(pose_results.pose_landmarks.landmark[LEFT_EYE])
            right_eye_landmark = self._change_to_np(pose_results.pose_landmarks.landmark[RIGHT_EYE])
            left_shoulder_landmark = self._change_to_np(pose_results.pose_landmarks.landmark[LEFT_SHOULDER])
            right_shoulder_landmark = self._change_to_np(pose_results.pose_landmarks.landmark[RIGHT_SHOULDER])

            middle_shoulder_landmark = (left_shoulder_landmark + right_shoulder_landmark) / 2

            results = self._preprocess(
                nose_landmark,
                left_eye_landmark,
                right_eye_landmark,
                left_shoulder_landmark,
                right_shoulder_landmark,
                middle_shoulder_landmark,
            )

            self._draw_landmarks(black_image, pose_results.pose_landmarks)
            black_image = cv2.flip(black_image, 1)
            self._display_angles(black_image, results)
        else:
            print("포즈 또는 얼굴 감지 실패.")
        return black_image

    def _draw_landmarks(self, image, landmarks):

        for landmark in landmarks.landmark:
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    def _preprocess(self, nose, left_eye, right_eye, left_shoulder, right_shoulder, middle_shoulder):
        parameters_combination = [
            [(nose, left_eye, right_eye), self._calculate_ratio],
            [(nose, left_shoulder, right_shoulder), self._calculate_angle],
            [(nose, middle_shoulder), self._calculate_angle],
            [(left_shoulder, right_shoulder), self._calculate_angle],
            [(nose, left_eye, right_eye), self._calculate_angle],
        ]
        results = []
        for parameters, func in parameters_combination:
            result = func(*parameters)
            results.append(result)

        return results

    def _display_angles(self, image, results):

        angle_labels = [
            "Eye Ratio",
            "Nose And Shoulder Angle",
            "Y Angle",
            "X Angle",
            "Eye Angle",
        ]

        for i, angle in enumerate(results):
            cv2.putText(
                image,
                f"{angle_labels[i]}: {angle:.2f}",
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def _change_to_np(self, joint):
        return np.array([joint.x, joint.y])

    def change_to_vector(func):
        def wrapper(self, A, B, C=None):
            if C is not None:
                AB = B - A
                AC = C - A
                return func(self, AB, AC)
            else:
                AB = B - A
                return func(self, AB)

        return wrapper

    @change_to_vector
    def _calculate_ratio(self, AB, AC):
        ratio = np.linalg.norm(AB) / np.linalg.norm(AC)
        return ratio

    @change_to_vector
    def _calculate_angle(self, AB, AC=None):
        if AC is not None:
            cos_theta = np.dot(AB, AC) / (np.linalg.norm(AB) * np.linalg.norm(AC))
            theta = np.arccos(cos_theta)
        else:
            tan_theta = AB[1] / AB[0]
            theta = np.arctan(tan_theta)

        angle = np.degrees(theta)
        return angle

    def release_resources(self):
        self.pose.close()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    recorder = Recording()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = recorder.process_image(frame)

        cv2.imshow("Video Stream", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    recorder.release_resources()
