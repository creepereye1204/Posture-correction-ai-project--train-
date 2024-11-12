import numpy as np
import cv2
import mediapipe as mp
import logging
import csv
import os

# 랜드마크 인덱스 정의
NOSE = 0
LEFT_EYE = 7
RIGHT_EYE = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
CHIN = 152

# Mediapipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# 로깅 설정
logging.basicConfig(level=logging.INFO)


class Recording:
    def __init__(self):
        self.pose = mp_pose.Pose()
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.csv_file = "landmarks.csv"
        self._initialize_csv()

    def _initialize_csv(self):
        # CSV 파일 초기화
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                # CSV 헤더 작성
                writer.writerow(["one", "two", "three", "four", "five"])

    def process_image(self, image):
        # 이미지 처리
        pose_results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        face_results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if pose_results.pose_landmarks and face_results.multi_face_landmarks:
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            face_landmarks = face_results.multi_face_landmarks[0]

            # 랜드마크 좌표 가져오기
            nose_landmark = self._change_to_np(pose_results.pose_landmarks.landmark[NOSE])
            left_eye_landmark = self._change_to_np(face_landmarks.landmark[LEFT_EYE])
            right_eye_landmark = self._change_to_np(face_landmarks.landmark[RIGHT_EYE])
            chin_landmark = self._change_to_np(face_landmarks.landmark[CHIN])
            left_shoulder_landmark = self._change_to_np(
                pose_results.pose_landmarks.landmark[LEFT_SHOULDER]
            )
            right_shoulder_landmark = self._change_to_np(
                pose_results.pose_landmarks.landmark[RIGHT_SHOULDER]
            )

            # 중간 어깨 랜드마크 계산
            middle_shoulder_landmark = (left_shoulder_landmark + right_shoulder_landmark) / 2
            left_frame_landmark = np.array([0, chin_landmark[1]])
            h, w, _ = image.shape
            cx, cy = int(chin_landmark[0] * w), int(chin_landmark[1] * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

            # 사전 처리 함수 호출
            self._preprocess(
                nose_landmark,
                left_eye_landmark,
                right_eye_landmark,
                left_shoulder_landmark,
                right_shoulder_landmark,
                chin_landmark,
                middle_shoulder_landmark,
                left_frame_landmark,
            )
        else:
            logging.info("포즈 또는 얼굴 감지 실패.")

    def _preprocess(
        self,
        nose,
        left_eye,
        right_eye,
        left_shoulder,
        right_shoulder,
        chin,
        middle_shoulder,
        left_frame,
    ):
        parameters_combination = [
            [(nose, left_eye, right_eye), self._calculate_ratio],
            [(nose, left_shoulder, right_shoulder), self._calculate_angle],
            [(nose, middle_shoulder), self._calculate_angle],
            [(left_shoulder, right_shoulder), self._calculate_angle],
            [(left_frame, nose, chin), self._calculate_angle],
        ]
        results = []  # 결과를 저장할 리스트
        for parameters, func in parameters_combination:
            result = func(*parameters)
            results.append(result)  # 결과 리스트에 추가
            logging.info(result)

        # 결과를 CSV에 저장 (5개씩)
        if len(results) >= 5:
            self._save_parameters_to_csv(results)

    def _save_parameters_to_csv(self, results):
        with open(self.csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(results)  # 결과 리스트를 CSV에 저장

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
        self.face_mesh.close()


# 사용 예시
if __name__ == "__main__":
    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(0)
    recorder = Recording()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        recorder.process_image(frame)

        cv2.imshow("Video Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    recorder.release_resources()
