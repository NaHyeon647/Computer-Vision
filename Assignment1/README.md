# Computer Vision — Assignment 1

Fourier 변환으로 얻은 Magnitude에서 특징을 추출하고, 이를 KNN으로 분류하는 이미지 분류 프로젝트입니다.

## 실험 구성

처음에는 정방향 S, T, V의 **Fourier Magnitude에서 에너지가 집중되는 방향**으로 문자를 구별하려 했습니다. 하지만 이미지가 회전하면 Magnitude도 함께 회전하므로, **회전에도 비교적 유지되는 특징**을 고민했습니다.

최종적으로 다음 두 특징을 선택해 **KNN(k=3)**으로 분류했습니다.

- **Entropy:** 에너지가 여러 방향에 얼마나 고르게 퍼져 있는지 → S와 T·V 구별
- **Angle:** 에너지가 가장 강한 두 방향 사이의 상대 각도 → T와 V 구별

**15° 간격**을 기본으로 구현한 뒤, 오분류 원인을 분석하며 **5°·10°·20°**와 비교했습니다. 특징 추출의 안정성과 추가 회전 테스트 결과를 고려해 **15° 모델을 최종 선택**했습니다.

## 실행 방법

VS Code 등 에디터의 터미널에서 `stv_classifier.py`가 있는 폴더로 이동한 뒤, 아래 명령어로 실행합니다.

```bash
python stv_classifier.py test.csv
```

저장소를 clone한 뒤 **`Assignment1(CV)_s5283639_검토용/`** 폴더에서 실행하면, **15° 모델 관련 파일만 모아 정리한 구성**으로 확인할 수 있습니다.

## 파일 구성

```text
Assignment1/
├── Assignment1(CV)_s5283639_검토용/               # 15° 모델 관련 파일을 모아 정리한 검토용 폴더 ⭐
├── Assignment1(CV)_s5283639_제출.zip              # 실제 제출한 ZIP 파일
│
├── stv_classifier.py                              # 기본 분류 코드 (15°) ⭐
├── stv_classifier_knn_debug_5deg.py                # 5° 실험 및 시각화
├── stv_classifier_knn_debug_10deg.py               # 10° 실험 및 시각화
├── stv_classifier_knn_debug_15deg.py               # 15° 실험 및 시각화 ⭐
├── stv_classifier_knn_debug_15deg_with_rotations.py # 15° 회전 실험 ⭐
├── stv_classifier_knn_debug_20deg.py               # 20° 실험 및 시각화
│
├── train_data/                                   # 학습용 이미지
├── test_data/                                    # 테스트용 이미지
├── train.csv                                     # 학습 이미지 정답 라벨
├── test.csv                                      # 테스트 이미지 정답 라벨
│
├── test_predictions_5deg.csv                      # 5° 예측 결과
├── test_predictions_10deg.csv                     # 10° 예측 결과
├── test_predictions_15deg.csv                     # 15° 예측 결과 ⭐
├── test_predictions_20deg.csv                     # 20° 예측 결과
│
├── fft_debug_plots_5deg/                          # 5° 간격 Fourier Magnitude 및 상위 피크 2개 그래프
├── fft_debug_plots_10deg/                         # 10° 간격 Fourier Magnitude 및 상위 피크 2개 그래프
├── fft_debug_plots_15deg/                         # 15° 간격 Fourier Magnitude 및 상위 피크 2개 그래프 ⭐
└── fft_debug_plots_20deg/                         # 20° 간격 Fourier Magnitude 및 상위 피크 2개 그래프
```

