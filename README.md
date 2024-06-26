
### DATA
<details>
  <summary>DataComp-LM</summary>

  - DataComp-LM: In search of the next generation of training sets for language models
  - controlled comparison LM을 위한 데이터 방법론들을 위한 벤치마크를 제공 (DCLM benchmark)
  - raw인 DCLM-pool부터 data 전처리 전략을 적용한 DCLM-BaseLine
  - 각 데이터 전처리 전략의 BEST 결과를 report (ex. For ECLM-Pool and remaining experiments, we use resiliparse to extract text)
</details>

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

### reasoning
<details>
  <summary>TransNAR</summary>
  
  - 딥마인드 Transformers meet Neural Algorithmic Reasoners 
  - Transformer는 정확한 정보와 추론을 요구하는 algorithmic reasoning에 상대적으로 약한 성능을 보임
  - 이를 완화하기 위해 Transformer와 GNN을 결합한 architecture를 제안
  - GNN Node cross attention
</details>

### agent
<details>
  <summary>Mixture of Agent</summary>
  
  - 각 agent들의 답변을 aggregator가 취합/통일 하여 다름 layer로 넘김.이 과정을 몇 layer 반복
  - https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-mixture-of-agents/README.md
  ```
  from llama_index.core.llama_pack import download_llama_pack 
  # download and install dependencies
  MixtureOfAgentsPack = download_llama_pack(
      "MixtureOfAgentsPack", "./mixture_of_agents_pack"
  )
  ```
</details>
