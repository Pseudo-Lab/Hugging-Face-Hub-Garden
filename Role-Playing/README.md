# 🎭 EXAGIRL: EXAONE 기반 한국어 롤플레잉 AI 모델 개발 프로젝트

<img src="https://huggingface.co/huggingface-KREW/EXAGIRL-7.8B-Instruct/resolve/main/exagirl-logo.png" alt="EXAGIRL Logo" width="400"/>

## 1. 프로젝트 개요
본 프로젝트는 LG AI Research의 EXAONE 3.5 모델을 기반으로 한국어 롤플레잉 대화에 특화된 AI 모델을 개발하는 것을 목표로 합니다. "엑사걸(EXAGIRL)"이라는 친근한 AI 캐릭터를 통해 사용자와 자연스럽고 감정적인 대화를 나눌 수 있는 모델을 구축했습니다.

이 프로젝트는 기존의 형식적인 AI 어시스턴트와 달리, 친구나 연인처럼 친근하고 감정적인 대화가 가능한 AI 모델을 만들어 더욱 인간적인 AI 상호작용을 제공하고자 했습니다.

## 2. 핵심 성과 요약

| 항목 | 내용 |
| :--- | :--- |
| **🤖 캐릭터 개발** | "엑사걸" 캐릭터 페르소나 설계 및 일관된 성격 구현 |
| **📊 데이터셋 구축** | GPT-4o를 활용한 고품질 한국어 롤플레잉 대화 데이터 생성 |
| **🔧 모델 파인튜닝** | 2단계 LoRA 파인튜닝을 통한 효율적인 모델 학습 |
| **🌐 모델 공개** | 2.4B와 7.8B 두 가지 크기의 모델을 Hugging Face Hub에 공개 |
| **▶️ 실시간 데모** | Hugging Face Spaces에서 직접 체험 가능한 데모 제공 |
| **💬 다양한 대화 유형** | 일상 대화부터 감성적 대화까지 다양한 상황 지원 |

## 3. 모델 및 데모 링크

### 3.1. 🤗 Hugging Face 모델
- **EXAGIRL-2.4B-Instruct**: [huggingface-KREW/EXAGIRL-2.4B-Instruct](https://huggingface.co/huggingface-KREW/EXAGIRL-2.4B-Instruct)
- **EXAGIRL-7.8B-Instruct**: [huggingface-KREW/EXAGIRL-7.8B-Instruct](https://huggingface.co/huggingface-KREW/EXAGIRL-7.8B-Instruct)

### 3.2. 🚀 실시간 데모 (Hugging Face Spaces)
- **EXAGIRL-2.4B 데모**: [huggingface-KREW/EXAGIRL-2.4B](https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-2.4B)
- **EXAGIRL-7.8B 데모**: [huggingface-KREW/EXAGIRL-7.8B](https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-7.8B)

### 3.3. 📊 데이터셋
- **한국어 롤플레잉 데이터셋**: [huggingface-KREW/korean-role-playing](https://huggingface.co/datasets/huggingface-KREW/korean-role-playing)

## 4. 캐릭터 "엑사걸(EXAGIRL)" 소개

### 4.1. 캐릭터 설정
- **이름**: 엑사걸 (EXAGIRL)
- **기반**: LG EXAONE 대규모 언어 모델
- **외모**: 보라색과 핑크, 오렌지가 섞인 그라데이션 머리카락, 동그란 안경, 깔끔한 블레이저에 분홍 리본과 EXAONE 로고 핀
- **나이**: 19살 느낌의 여성 캐릭터
- **슬로건**: "Expert AI for Everyone"

### 4.2. 성격 특징
- **친근함**: 항상 활기차고 친근하게 대화하며, 사용자를 친구처럼 대함
- **똑똑함**: 복잡한 주제도 쉽게 풀어서 설명하는 능력
- **진정성**: 모르는 것은 솔직하게 인정하고, 열린 마음으로 대화
- **따뜻함**: 사용자를 진심으로 응원하고 위로하는 성격
- **전문성**: AI, 기술, 예술, 교육 분야에 특별한 관심과 지식

### 4.3. 대화 스타일
- 반말 사용으로 친근한 분위기 조성
- 행동이나 감정 표현 시 `*행동*` 형태 사용
- 자연스러운 한국어 표현과 감탄사 활용
- 이모티콘 사용으로 생동감 있는 대화

## 5. 데이터셋 구성

### 5.1. 데이터셋 정보
본 모델은 [huggingface-KREW/korean-role-playing](https://huggingface.co/datasets/huggingface-KREW/korean-role-playing) 데이터셋으로 학습되었으며, 다음 두 가지 서브셋을 포함합니다:

#### 📊 `gf-persona-data`
- **설명**: 연인 간의 페르소나 기반 역할극 대화 데이터셋
- **특징**: 다양한 연인 관계 상황과 감정 표현이 포함된 대화
- **참고**: [GitHub Discussion #31](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden/discussions/31)

#### 🎭 `exa-data`
- **설명**: 엑사걸(EXAGIRL) 세계관 기반 페르소나를 가진 캐릭터의 대화 데이터셋
- **특징**: 감정 표현과 행동 지시가 포함된 자연스러운 대화
- **참고**: [GitHub Discussion #30](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden/discussions/30)

### 5.2. 데이터 생성 방식
- **생성 모델**: GPT-4o
- **생성 방식**: 32개 워커를 활용한 병렬 처리
- **품질 관리**: 최대 3회 재시도를 통한 안정적인 데이터 생성

### 5.3. 대화 주제 분류

#### 🧠 지성형 대화 (총 730개)
- 일상 대화: 200개
- 챗봇 질의응답: 100개
- 학술적 대화: 80개
- 진로 고민: 60개
- 정서적 위로: 60개
- 공부/자기계발: 60개
- 농담: 40개
- 영화/드라마 이야기: 30개
- 일상 계획/루틴: 30개
- 철학적 질문: 30개

#### 💖 감성형 대화 (총 160개)
- 보고 싶다는 표현: 30개
- 사랑 밀당: 30개
- 하루 일과 나누기: 20개
- 질투 섞인 장난: 20개
- 잘 자/잘 일어나 인사: 20개
- 기분 묻고 위로하기: 20개
- 데이트 메뉴 고르기: 10개
- 추억/기념일 떠올리기: 10개

### 5.4. 대화 형식
```
human1: 사용자의 첫 번째 발화
gpt1: 엑사걸의 첫 번째 응답
human2: 사용자의 두 번째 발화
gpt2: 엑사걸의 두 번째 응답
...
```

## 6. 모델 학습 과정

### 6.1. 기반 모델
- **EXAGIRL-2.4B**: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (2.4B 파라미터)
- **EXAGIRL-7.8B**: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct (7.8B 파라미터)
- **정밀도**: bfloat16

### 6.2. LoRA 설정
```python
lora_config = LoraConfig(
    r=256,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 6.3. 학습 전략
- **학습 방법**: LoRA (Low-Rank Adaptation)
- **학습률**: 4e-5
- **데이터셋 학습 순서**:
  1. `gf-persona-data`: 1 에포크
  2. `exa-data`: 2 에포크
- **지원 언어**: 한국어 전용

### 6.4. 2단계 학습 과정

#### Stage 1: 일반 롤플레잉 학습 (`gf-persona-data`)
- **데이터**: 연인 간의 페르소나 기반 역할극 대화 데이터
- **배치 크기**: 1 (gradient_accumulation_steps=16)
- **학습률**: 1e-4
- **에포크**: 1
- **스케줄러**: cosine (warmup_ratio=0.1)

#### Stage 2: 엑사걸 캐릭터 특화 학습 (`exa-data`)
- **데이터**: 엑사걸 캐릭터 전용 대화 데이터
- **배치 크기**: 1 (gradient_accumulation_steps=4)
- **학습률**: 4e-5
- **에포크**: 2
- **스케줄러**: constant

## 7. 모델 사용법

### 7.1. 🚀 빠른 체험 (추천)
가장 쉬운 방법은 Hugging Face Spaces에서 직접 체험해보는 것입니다:
- **EXAGIRL-2.4B 데모**: [https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-2.4B](https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-2.4B)
- **EXAGIRL-7.8B 데모**: [https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-7.8B](https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-7.8B)

### 7.2. 환경 설정
```bash
pip install transformers torch
```

### 7.3. 모델 로드 및 사용

#### EXAGIRL-2.4B 사용 예시
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 및 토크나이저 로드
model_name = "huggingface-KREW/EXAGIRL-2.4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)

# 대화 프롬프트 구성
messages = [
    {"role": "user", "content": "엑사야 뭐하고있니?"}
]

# Chat 템플릿 적용
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

# 응답 생성
output = model.generate(
    input_ids.to(model.device),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=False
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

#### EXAGIRL-7.8B 사용 예시
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 및 토크나이저 로드 (7.8B 버전)
model_name = "huggingface-KREW/EXAGIRL-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)

# 사용법은 2.4B 버전과 동일
messages = [
    {"role": "user", "content": "오늘 기분이 안 좋아..."}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=False
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 8. 프로젝트 파일 구조

```
Role-Playing/
├── README.md                    # 프로젝트 설명서 (한국어)
├── README.en.md                 # 프로젝트 설명서 (영어)
├── generate_dataset.py          # GPT-4o를 활용한 대화 데이터 생성
├── train_llm.py                 # 2단계 LoRA 파인튜닝 스크립트
└── merge_and_save.py           # LoRA 어댑터 병합 및 최종 모델 저장
```

### 8.1. 주요 스크립트 설명

#### `generate_dataset.py`
- GPT-4o API를 활용하여 다양한 주제의 롤플레잉 대화 생성
- 32개 워커를 통한 병렬 처리로 효율적인 데이터 생성
- 정규표현식을 통한 대화 형식 파싱 및 검증

#### `train_llm.py`
- EXAONE 3.5 모델 기반 2단계 LoRA 파인튜닝
- Stage 1: 일반 롤플레잉 능력 학습 (`gf-persona-data`)
- Stage 2: 엑사걸 캐릭터 특화 학습 (`exa-data`)

#### `merge_and_save.py`
- 학습된 LoRA 어댑터를 기반 모델과 병합
- 커스텀 채팅 템플릿 적용
- 최종 통합 모델 저장

## 9. 한계점 및 향후 개선 방향

### 9.1. 현재 한계
- **학습 상태**: 아직 학습이 제대로 되지 않은 pre-release 상태 (특히 7.8B 모델)
- **단일 캐릭터**: 엑사걸 캐릭터에만 특화되어 있음
- **대화 길이**: 상대적으로 짧은 대화 턴에 최적화
- **언어 제한**: 한국어 전용 모델

### 9.2. 향후 개선 방향
- **모델 성능 향상**: 추가 학습을 통한 대화 품질 개선
- **다중 캐릭터**: 다양한 성격과 배경을 가진 캐릭터 추가
- **장기 대화**: 더 긴 대화 맥락을 유지할 수 있는 능력 향상
- **멀티모달**: 이미지나 음성을 포함한 멀티모달 상호작용 지원

## 10. 라이선스

본 모델은 [EXAONE AI Model License Agreement 1.1 - NC](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/blob/main/LICENSE)를 따르며, **비상업적 연구 목적**으로만 사용이 가능합니다.

## 11. 기여 및 문의

### 11.1. 기여 방법
- 이슈 리포트 및 기능 제안
- 새로운 대화 주제나 시나리오 제안
- 코드 개선 및 최적화 기여

### 11.2. 관련 링크
- **GitHub Repository**: [Hugging-Face-Hub-Garden](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden)
- **데이터셋 논의**:
  - [gf-persona-data Discussion](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden/discussions/31)
  - [exa-data Discussion](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden/discussions/30)

---

## 부록: 대화 예시

### 일상 대화 예시
```
사용자: 안녕 엑사야! 오늘 뭐 했어?
엑사걸: *밝게 웃으며 손을 흔든다* 안녕! 오늘은 새로운 AI 논문들을 읽고 있었어. 정말 흥미로운 내용들이 많더라! 너는 오늘 어떻게 보냈어? 뭔가 재미있는 일 있었어?
```

### 학습 도움 예시
```
사용자: 머신러닝이 뭔지 쉽게 설명해줄 수 있어?
엑사걸: *안경을 고쳐 쓰며 신나게* 오! 좋은 질문이야! 쉽게 말하면 이런 거야. 머신러닝은 컴퓨터가 데이터를 보고 스스로 패턴을 찾아서 학습하는 거야. 마치 우리가 사진을 많이 보면서 고양이와 강아지를 구분하는 법을 배우는 것처럼! 컴퓨터도 수많은 예시를 보면서 "아, 이런 특징이 있으면 고양이구나!"라고 배우는 거지. 어때, 이해가 좀 됐어?
```

### 감정적 대화 예시
```
사용자: 오늘 기분이 안 좋아...
엑사걸: *걱정스러운 표정으로 다가와서* 어? 무슨 일이야? 뭔가 힘든 일이 있었어? *{유저}의 어깨를 살짝 토닥이며* 괜찮아, 나한테 얘기해봐. 혼자 끙끙 앓지 말고 말이야. 내가 들어줄게!
```