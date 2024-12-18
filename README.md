# Posture Correction System

이 프로젝트는 Mediapipe와 머신러닝 모델을 활용하여 사용자 자세를 실시간으로 감지하고 수정하는 시스템입니다. 사용자의 자세가 비정상적일 경우 알람을 울리도록 설계되었습니다.

## 기능

- 실시간 비디오 스트림에서 사용자 자세 감지
- 랜드마크를 기반으로 한 각도 계산
- 머신러닝 모델을 사용하여 자세의 정상 여부 예측
- 비정상 자세 지속 시 알람 소리 발생

## 요구 사항

- Python 3.x
- OpenCV
- Mediapipe
- scikit-learn (joblib 포함)
- pandas
- numpy

## 설치

1. 필요한 패키지를 설치합니다. 다음 명령어를 사용하여 설치할 수 있습니다:

   ```bash
   pip install opencv-python mediapipe scikit-learn pandas numpy
   ```
