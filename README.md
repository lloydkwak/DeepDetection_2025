# Unveiling the Invisible: The Critical Role of Frequency-Aware Data Preprocessing in Diffusion Image Detection

본 연구는 최근 급격히 발전한 확산 모델(Diffusion Model, 예: Midjourney v6, DALL-E 3) 기반의 생성형 이미지를 효과적으로 탐지하기 위한 최적의 방법론을 제안합니다.

우리는 모델의 아키텍처를 복잡하게 만드는 대신, 데이터 전처리(Preprocessing)에 집중하는 **Data-Centric AI** 접근 방식을 통해 과적합 문제를 해결하고 탐지 성능을 비약적으로 향상시켰습니다.

## Dataset Overview: DeepDetect-2025

본 연구는 Kaggle의 **[DeepDetect-2025](https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025)** 데이터셋을 활용하여 모델을 학습 및 검증했습니다. 이 데이터셋은 생성형 AI 탐지 연구를 위해 구축된 대규모 고해상도 이미지 데이터셋입니다.

### 1. Data Composition
전체 데이터는 **총 115,000장**으로 구성되어 있으며, 특정 도메인에 편향되지 않도록 다양한 소스에서 수집되었습니다.

* **Real Images (~60,000장):**
    * **Source:** Unsplash, Pixabay, Pexels 등 고해상도 스톡 이미지 사이트.
    * **Feature:** 실제 사진의 자연스러운 노이즈와 다양한 조명 조건을 포함하여, 기존 저해상도 학술 데이터셋(COCO 등) 대비 실용적인 탐지 환경을 제공합니다.
* **Fake Images (~55,000장):**
    * **Source:** Midjourney, Stable Diffusion, DALL-E 등 최신 SOTA 생성 모델.
    * **Feature:** 육안으로 구분이 힘든 고품질(High-Fidelity) 이미지를 포함합니다.

### 2. Data Split Strategy
데이터의 클래스 불균형을 방지하기 위해 **70:15:15** 비율로 계층적 분할(Stratified Split)을 수행했습니다.

| Split | Real (Count) | Fake (Count) | Total | Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 42,000 | 38,500 | 80,500 | 70% |
| **Validation** | 9,000 | 8,250 | 17,250 | 15% |
| **Test** | 9,000 | 8,250 | 17,250 | 15% |
| **Total** | **60,000** | **55,000** | **115,000** | **100%** |


### 3. Statistical Analysis (EDA)
우리는 데이터셋의 픽셀 강도(Pixel Intensity) 분포를 분석하여 모델 설계의 단서를 얻었습니다.

* **Real Images:** 넓은 분산(High Variance)을 가지며 0~255 범위에 픽셀이 고르게 분포합니다.
* **Fake Images:** 생성 모델의 정규화 특성으로 인해 픽셀 값이 중간값(128) 부근에 밀집(Low Variance) 되는 경향을 보입니다.
* **Insight:** 이러한 통계적 차이는 생성형 이미지가 가지는 '오버 스무딩' 및 '주파수 이상' 특징과 연결되며, 본 연구의 전처리 전략 수립의 근거가 되었습니다.

## Experiments & Results

### 1. Model-Centric vs Data-Centric Approach
우리는 세 가지 모델(ResNet-50, EfficientNet-B4, XceptionNet)에 대해 기본 전처리(Basic Pipeline)와 주파수 강건성 증강(Frequency-Aware Pipeline)을 각각 적용하여 비교 실험을 수행했습니다.

실험 결과, 단순 모델 고도화보다는 **데이터 전처리 전략의 개선**이 성능 향상에 결정적인 역할을 함을 확인했습니다. 특히 과적합이 심했던 모델들에서 비약적인 성능 향상이 관찰되었습니다.

| Model | Basic Pipeline (Test Acc) | **Frequency-Aware (Test Acc)** | Improvement |
| :--- | :---: | :---: | :---: |
| **ResNet-50** | 72.6% | **96.3%** | **+23.8%p** |
| **EfficientNet-B4** | 60.7% | **88.9%** | **+28.2%p** |
| **XceptionNet** | 78.6% | **92.4%** | **+13.6%p** |

