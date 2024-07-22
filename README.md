
### multi-modal
<details>
  <summary>LLAVA</summary>

  - microsoft research. first instruct-multi modal data + baseline 제공
  - https://huggingface.co/docs/transformers/model_doc/llava , LlavaForConditionalGeneration
  - CLIP
      - Llava 에서 활용한 visual encoder
      - image와 짧은 description을 contrastive learning
      - clip image encoder로는 resnet,ViT가 고려됐었고 ViT가 더 높은 성능
  - vicuna
      - llava에서 활용한 language decoder
      - 오픈소스 챗봇으로 sharedGPT(사용자와 gpt 대화) 로부터 수집된 사용자들의 대화로 llama를 finetuning한 모델
  - symbolic representation
    - 저자는 multi-modal instruct data 를 text-only GPT만을 활용해 만들기 위해 symbolic representation을 활용
    - captions, bounding box(객체, 좌표)
  - visual linear layer
    - image encoder가 LM모델에 입력전 거치는 layer로 훈련
  - systematic understanding
    - 저자는 approximate theorical upperbound를 제공하기 위해 text-only GPT 4의 결과와 실험모델의 답변을 GPT4 가 judge
  - LLAVA-Bench *
    - coco version (coco-val-2014. 3 type question), in-the wild version (novel domain)
  - scienceQA
    - https://paperswithcode.com/dataset/scienceqa
  - ablation 결과 CC3M 으로 pretraining이 성능에 큰 영향
</details>

### DATA
<details>
  <summary>DataComp-LM</summary>

  - DataComp-LM: In search of the next generation of training sets for language models
  - controlled comparison LM을 위한 데이터 방법론들을 위한 벤치마크를 제공 (DCLM benchmark)
  - raw인 DCLM-pool부터 data 전처리 전략을 적용한 DCLM-BaseLine
  - 각 데이터 전처리 전략의 BEST 결과를 report (ex. For ECLM-Pool and remaining experiments, we use resiliparse to extract text)
</details>
<details>
  <summary>KLUE (Korean Language Understanding Evaluation)</summary>

  - annotator: selectStars
  - 8 task. TC, STS, NLI, NER, RE, DP, MRC, DST
  - noise filtering. hashtag, spaces, copyright tags etc remove, 20 char 이상 중국어, 일본어 filter, toxic content removal (hate speech detection), PII removal (regex)
  - STS
    - AIRBNB, Formal news 등에서 random으로 sentence 선택 후 rouge 높은 sentence pair 준비. annotator가 sentence-level 유사성 labeling
    - 0~5 bins의 분포 차이 있지만 eval/test는 uniform하게 준비
  - NLI
    - 한명의 annotator가 주어진 문장으로부터 NLI 관계 문장 생성하면 나머지 annotator의 majority 일치시 set에 포함
</details>
<details>
  <summary>Kor-NLU (kor-sts, kor-nli)</summary>

  - kor-sti. sts-b가 train/dev/eval 모두
  - kor-nli. train(snli, mnli), dev/test(xnli)
  - train은 기계번역(어디껀지 안나와있는듯) test/dev는 기계번역 -> 번역 전문가가 post editing
  - cross/bi encoding. simcse 논문 방식으로 훈련시 xlm이 ko-pretrained 보다 성능 높음
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

### fast inference
<details>
  <summary>quantization</summary>

  - Quantization infernece
    - AI 모델 bit 압축하여 메모리/속도 향상
    - 변경된 bit 따라 value(numeric) scale이 달라짐 (scale 상수) -> 양자화된 값, 역양자화된 값 모두 scale방식으로 구할 수 있음. 이때, scale로 나눠주면 또 float과 같은 bit이 필요하다고 생각할 수있지만 반올림하기에 quant bit으로 표현가능 (단 역양자화때 오차있음)
    - ex. FP32 -> INT8
  - PTQ (Post-Training Quant) 훈련후 양자화 (GPTQ, GGUF/GGML, AWQ)
  - QAT (Quant-aware Traing)
    - 훈련하면서 양자화
    - base model -> quant Forward -> loss -> no Quant gradient -> base model update
    - low bit 으로 처음부터 하는것보다 우수한가..
</details>
<details>
  <summary>
    MoE
  </summary>

  - https://huggingface.co/blog/moe 
  - LLM에서 next module (FFN) 을 통과하게 route(each token)을 통해 부분 expert만 통과하도록 하는 기법
  - k개의 exxpert를 활용하고 싶다면 top k 의 logit으로 다시 softmax 후 FFN 결과들을 weighted sum
  - Load balancing: 각 expert에 할당된 batch  연산 정해져있음(scale factor로 조정가능). 할당량 넘으면 residual connection으로 넘기기도함.
  - Fine-tuning시 불안정하기도 한데, second는 random으로보내서 overfitting 방지
  - 분산훈련으로 expert parallelism도 가능
  - Noisy Top-k Gating. softmax(H) 에서 H에 x널때 noise weight 같이 수행
  - MoE communication cost → 물리적 (ex device) 간의 통신 비용 
</details>

### time-series prediction
<details>
  <summary>patchTST</summary>

  - multivatriate timme series forecasting
    - 여러개의 종속변수
    - 논문에서 다루는 data 들이 multivariate 이지만 model은 하나의 모델로 univariate 형태로 독립적으로 다양한 variable을 학습 및 추론함
  - channel-independence
    - transformer 입력될 때 여러 변수들이 mixing(ex. TST 처럼) 되어 입력되지 않고 각각 따로 입력 및 출력됨
    - cross variable 학습이 안된다고 생각할 수 있지만 저자말로는 하나의 모델로 여러 변수를 학습해기에 각 변수 간 영향 끼친다고 함
  - subseries-level patch
    - timet 로 stride 기간 별로 묶어 embedding 하여 모델에 입력. local semantic을 학습하게 되고 attention 이 channel 독립적으로 되기에 computation and memory save
    - look back window를 늘려 long-history 봄
  - representation learning. ~= patch MLM
</details>
<details>
  <summary>
    TST (Time Series Transformer)
  </summary>

  - IBM KDD 2021
  - multivariate time series. 종속변수가 여러개인 시계열
    - 이 논문에서는 embedding 시 변수간 mixing 하는걸로 보임
  - learnable positional embedding, batch norm (outlier value mitigate), final representation X class n or 1 (regression)
  - unsupervised (self-suspervised) training. ~= MLM  but masked sequence.
</details>
<details>
  <summary>
    How is AI being applied to time series forecasting?
  </summary>

  - https://research.ibm.com/blog/AI-time-series-forecasting
  - models: TST(Time Series Transformer), PatchTST, PatchTSMixer, Tiny Time mixers
  - data: Monash Time Series Forecasting Repository (https://arxiv.org/pdf/2105.06643). open time-series repository. paper에 여러 시계열 데이터에 대한 baseline 성능 있음
</details>
<details>
  <summary>
    gradient-boosting
  </summary>

  - 의사결정 트리에서 잔차학습을 통해 regression 예측. 앙상블과 다르게 여러 모델이 이전 모델의 부족한 부분을 sequential하게 학습 (노드는 훈련시 loss, Information 여부따라 추가되는듯)
  - LightBGM의 경우 훈련시 의사결정 노드를 수직으로만 확장하여 영향이 큰 노드만 남아있음 -> 속도 향상
  - 아마존 수요예측에서 top-rank 모델들이 활용한 모델 https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview
</details>
<details>
  <summary>
    Rolled DeepAR
  </summary>

  - Robust recurrent network model for intermittent time-series forecasting
  - teach forcing 없는 DeepAR. DeepAR 은 RNN 기반의 time-series forecasting 모델로 공변량을 입력한다. (teach forshing 없으면 error가 다음 inference에 전달 된다고 볼 수있으나, 본 논문에서는 high zero, intermittent data + 낮은 level(day) data sum -> high level (week) 이기에 augmentation 이자 robust 역할도 한다고 함
  - (B) tweedie distribution. EDMs의 일종으로 p param에 따라 포아송 분포가 되기도 함. p따라 zero 범위가 달라지는데, zero 값이 많은 m5 데이터셋을 훈련하기 위해 target distribution으로 적합하다고 함
  - sequential feature. 모델에 입력되는 순차정보들은 sales, price 등이 있고 raw값과 moving average, norm value by store.. 등이 입력됨
  - categorial feature. 카테고리 정보들은 nlp의 vocab과 같이 embedding table통해 입력됨
</details>

### basic
<details>
  <summary>
    entropy
  </summary>

  - entropy
    - 사건이 일어날 때의 정보량 기댓값으로, 모든 사건의 확률이 같을 때 가장 높은 엔트로피 가짐
    - I(x) = log 1/p(x) = -logp(x). 섀넌의 정보이론에서는 기본단위가 bit이기에 정보량을 필요 비트로 나타냄 -> log_2로 얻을 수있음 . -log2_2 = 1bit 필요
  - cross entropy
    - 정답 분포 p를 통해 학습하는 분포 Q의 정보량을 줄이고자 한다.
    - sum p(x) log Q(x)
  - perplexity
    - 언어모델의 성능 평가 metric중 하나로 문장 길이로 norm된 확률값.
    - root_n(1/p(i) + p(i-1)...p)
    - 사건이 발생할 확률(정보량)을 sequential하게 측정한다는 점에서 entropy 개념과 관계있음

  - BM25
    - 검색하는 Query와 다른 문서들의 연관성을 평가하는 알고리즘
    - TF-IDF (Term-frequency(문성 D에서 q의 frequency) Inverse-document frequency(전체문서에서 word count)
    - score (document,query) = n sum(q) IDF(q) * TF norm
</details>
