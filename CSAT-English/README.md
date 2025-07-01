# AI 기반 고등학교 2학년 영어 내신 문제 자동 생성 프로젝트

## 📋 프로젝트 개요

### 배경 및 필요성
고등학교 영어 내신 문제 출제는 지문 분석, 질문 설계, 보기 배열, 포맷팅 등 수많은 단계를 수작업으로 수행해야 하는 복잡한 과정입니다. 특히 반복적인 포맷 구성 작업에 많은 시간이 소요되어, 정작 중요한 문제의 질과 수업에 충분히 집중하기 어려운 현실이 있습니다.

본 프로젝트는 AI를 활용하여 이러한 반복적인 작업을 자동화하고, 기존 문제를 기반으로 학생 수준에 적합한 유사 문제를 자동 생성함으로써 교사가 고품질 문제 제공에 집중할 수 있는 환경을 조성하는 것을 목표로 합니다.

### 주요 목표
- **AI 기반 고등학교 2학년 영어 내신 문제 데이터셋 구축**
- **문제 유형별 자동 생성 모델 Fine-tuning**
  - 주제 추론: 지문의 핵심 주제를 파악하는 문제
  - 빈칸 추론: 문맥상 빈칸에 들어갈 적절한 표현을 찾는 문제
  - 어법 오류: 문법적으로 잘못된 부분을 찾는 문제
  - 제목 추론: 지문의 가장 적절한 제목을 찾는 문제
  - 내용 불일치: 지문 내용과 일치하지 않는 선택지를 찾는 문제

- **기존 문제 기반 유사 문제 생성**을 통한 교육적 일관성 확보

## 🗂️ 데이터 수집 및 전처리

### 데이터 출처
- **내신 변형 문제**: 사교육 기관 협력 데이터

### 수집 현황
- **총 수집량**: 약 400 문항
- **집중 문제 유형**: 주제 추론, 빈칸 추론, 어법 오류, 제목 추론, 내용 불일치

### 데이터 전처리 파이프라인

#### 1단계: 데이터 정제
- **포맷 통합**: 다양한 문제 형식을 JSON으로 표준화
- **특수 문자 처리**: 빈칸(`__**(A)**__`), 밑줄(`__**word**__`) 등
- **중복 제거**: 유사 문제 필터링
- **메타데이터 추가**: 출처, 연도, 난이도 정보

#### 2단계: 파일 형식 변환
```
HWP/PDF → DOCX → JSON
```
#### 3단계: Fine-tuning 데이터셋 변환

**Raw 데이터 예시**:
```json
{
  "number": 1,
  "exam_name": "모의고사_2학년_2024년_03월",
  "question_type": "주제추론",
  "passage": "Dear Residents, My name is Kari Patterson...",
  "question": "다음 글의 내용과 일치하지 않는 것은?",
  "options": "① 글쓴이는 River View 아파트의 관리인이다.\n② 토요일 오전에...",
  "answer": "4",
  "vocab": ""
}
```

**Instruction Tuning 형식**:
```json
{
  "instruction": "다음 지문을 주제 추론 문제로 만들어주세요.",
  "input": "Passage: 'Dear Residents, My name is Kari Patterson...'",
  "output": "문제: 다음 글의 주제로 가장 적절한 것은?\n보기: ① 아파트 관리 방법\n② 정원 가꾸기 행사 안내\n③ 주민 회의 공지\n④ 청소 도구 준비 안내\n⑤ 날씨 변화 대응책\n정답: 2"
}
```

## 💻 기술 구현

### 데이터 전처리 파이프라인

#### 1단계: 데이터 필터링 및 분할
```python
# 6가지 핵심 문제 유형 필터링
target_questions = [
    '다음 글의 제목으로 가장 적절한 것은?',
    '다음 글의 주제로 가장 적절한 것은?', 
    '다음 글의 요지로 가장 적절한 것은?',
    '다음 글의 내용과 일치하지 않는 것은?',
    '다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오.',
    '다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?'
]
```

#### 2단계: Instruction Tuning 프롬프트 설계
```python
def create_prompt(ex):
    """문제 유형별 차별화된 프롬프트 생성"""
    if description in ['제목 추론', '주제 추론', '요지 추론', '내용 불일치']:
        # 원본 지문 사용
        messages = [{
            "role": "user", 
            "content": f"다음 지문을 {description} 문제로 만들어주세요.\n\n지문:\n{ex['original']}"
        }, {
            "role": "assistant",
            "content": f"문제: {ex['question']}\n선택지:\n{ex['options']}\n정답: {ex['answer']}"
        }]
    elif description == '빈칸 추론':
        # 변형 지문 + 선택지 포함
        messages = [{
            "role": "assistant", 
            "content": f"변형 지문: {ex['passage']}\n문제: {ex['question']}\n선택지:\n{ex['options']}\n정답: {ex['answer']}"
        }]
```

### 모델 아키텍처

#### LoRA Fine-tuning 설정
```python
lora_cfg = LoraConfig(
    r=16,                    # 저랭크 차원
    lora_alpha=64,           # 스케일링 파라미터  
    lora_dropout=0.05,       # 드롭아웃 비율
    bias="none",
    target_modules=[         # Attention & MLP 레이어 타겟
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type=TaskType.CAUSAL_LM
)
```

#### 메모리 최적화
```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    offload_folder="/tmp/offload_qwen3",
    offload_state_dict=True
)
```

### 추론 시스템

#### Fine-tuned vs Original 모델 비교 구현
```python
# Fine-tuned 모델 로드
peft_model = PeftModel.from_pretrained(base_model, fine_tuned_path)

# Original 모델 로드  
original_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")

# 동일 프롬프트로 성능 비교
for question_type in ['제목 추론', '주제 추론', '빈칸 추론', '어법 오류']:
    fine_tuned_output = generate_question(peft_model, passage, question_type)
    original_output = generate_question(original_model, passage, question_type)
```

### Hugging Face 배포
```python
# 모델 업로드 준비
model = AutoPeftModelForCausalLM.from_pretrained(
    "huggingface-KREW/Qwen3-8B-Korean-Highschool-English-Exam",
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

# 추론 함수
def generate_question(passage, question_type):
    messages = [{"role": "user", "content": f"다음 영어 지문을 {question_type} 문제로 만들어주세요.\n\n지문:\n{passage}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"].to("cuda"), max_new_tokens=1024)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
```

### 코드 구조
```
├── README.md                       # 프로젝트 소개 및 사용법
├── requirements.txt                # 의존성 패키지  
├── huggingface_inference.py        # 메인 사용 코드
├── docx_to_json.py             # DOCX → JSON 변환기
├── example_data.json           # 샘플 데이터
└── qwen3_8b_train_kor_exam.py  # 학습 코드
```

### Fine-tuning 설정
- **베이스 모델**: `Qwen/Qwen3-8B`
- **학습 방법**: LoRA (Low-Rank Adaptation)
  - r=16, lora_alpha=64, lora_dropout=0.05
  - 타겟 모듈: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **하이퍼파라미터**:
  - Epochs: 3
  - Learning Rate: 1e-4
  - Batch Size: 1

#### 어법/어휘
**Fine-tuned 모델**:
```
문제: In the passage, the word 'flagging up' most likely means...
보기: ① calling out ② tearing down ③ pointing inwards ④ indicating ⑤ ruling out
정답: 4
```

**Base 모델**: 어법/어휘 문제 생성 방법 미숙지

## ⚠️ 한계점 및 개선 사항

### 현재 한계점
1. **데이터 부족**: 완전한 형식 일관성 미확보
2. **포맷 불일치**: 간헐적인 형식 오류 발생
3. **변형 부족**: 원본 지문 그대로 복사하는 경우 발생

## 🤝 협력 및 지원

본 프로젝트는 사교육 기관의 데이터 지원과 전문 교사들의 문제 품질 평가 협력을 통해 진행되고 있으며, 실제 교육 현장의 피드백을 반영한 실용적인 도구 개발을 목표로 합니다.