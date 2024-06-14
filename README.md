
### STT
<details>
  <summary>whispher</summary>

  - weakly large-scale supervised pretrained model
  - multi-task, multi-lingual -> robustness, generalization
  - (R) transcript-ese
    - 인터넷 transcript, audio 들을 모아 방대한 양의 데이터지만 기존 ASR 시스템들의 output도 포함하여 품질이 보장되지 않음
    - we developed many heuristics to detect and remove machine-generated transcripts from the training dataset.
  - Voxilingual107
    - spoken language detection dataset
    - youtube- title, description
    - 107 language
  - mel spectrum
    - 인간은 고주파일수록 민감하게 반응하고 쉽게 구분. 이런 특성을 spectrum에 반영
    - (B) 기계가 encoding하는건데 반영할 필요 있나
  - multitask training data
    - english transcription, any-to-english translation, non-english transcription, no speech
  - text normalization
    - 실제로 같은 의미, 문법이 다르게 처리되는 경우 방지 위해 eval 에서 standardization 처리해줌
    - whispher랑 같이 text normalization 로 개발. whispher에 특화돼 generalization 능력 잃었는지 확인위해 외부 text normalization 툴과 비교
  - superhuman performance in-distribution, subsuman performancec out-of-distribution
    - 기존 speech recongnition task에서 machine으로 측정할 때 인간이 측정하는 것보다 성능 높은 현상
    - speech recognition 분야에서 훈련된 모델들이 다른 세팅(ex. machine)에서 human eval보다 높게 나오는 현상
    - 인간은 out-of-distribution  generalization의 view로 평가하기 때문 (훈련 데이터에 대한 정도 없음)
  - librispeech
    - 1000시간의 audiobook data, clean/other version 있음
    - text, audio
  - (B) shifting the window
</details>
