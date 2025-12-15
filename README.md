# Unveiling the Invisible: The Critical Role of Frequency-Aware Data Preprocessing in Diffusion Image Detection

본 연구는 최근 급격히 발전한 확산 모델(Diffusion Model, 예: Midjourney v6, DALL-E 3) 기반의 생성형 이미지를 효과적으로 탐지하기 위한 최적의 방법론을 제안합니다.

우리는 모델의 아키텍처를 복잡하게 만드는 대신, 데이터 전처리(Preprocessing)에 집중하는 **Data-Centric AI** 접근 방식을 통해 과적합 문제를 해결하고 탐지 성능을 비약적으로 향상시켰습니다.

## Experiments & Results

### 1. Model-Centric vs Data-Centric Approach
우리는 세 가지 모델(ResNet-50, EfficientNet-B4, XceptionNet)에 대해 기본 전처리(Basic Pipeline)와 주파수 강건성 증강(Frequency-Aware Pipeline)을 각각 적용하여 비교 실험을 수행했습니다.

실험 결과, 단순 모델 고도화보다는 **데이터 전처리 전략의 개선**이 성능 향상에 결정적인 역할을 함을 확인했습니다. 특히 과적합이 심했던 모델들에서 비약적인 성능 향상이 관찰되었습니다.

| Model | Basic Pipeline (Test Acc) | **Frequency-Aware (Test Acc)** | Improvement |
| :--- | :---: | :---: | :---: |
| **ResNet-50** | 72.6% | **96.3%** | **+23.8%p** |
| **EfficientNet-B4** | 60.7% | **88.9%** | **+28.2%p** |
| **XceptionNet** | 78.6% | **92.4%** | **+13.6%p** |

[Data source: Table 2 & Table 3 in Paper]
