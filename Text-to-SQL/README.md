# 📊 LLaMA 3.1 기반 한국어 Text-to-SQL 모델 개발 프로젝트

## 1. 프로젝트 개요
본 프로젝트는 한국어 자연어 질의를 SQL 쿼리로 변환하는 한국어 특화 Text-to-SQL 모델 개발을 목표로 진행되었습니다. 기존 영어 중심의 Text-to-SQL 연구 환경에서 한국어 데이터셋과 특화 모델의 부재를 해결하고, 누구나 쉽게 활용할 수 있는 오픈소스 모델 및 데이터셋을 구축하여 국내 AI 생태계에 기여하고자 했습니다.

이를 위해 표준 벤치마크 데이터셋인 'Spider'를 고품질 한국어로 번역 및 정제하고, 4-bit 양자화와 LoRA 기법을 적용하여 효율적인 파인튜닝을 수행했습니다. 최종적으로 개발된 모델과 데이터셋, 실시간 데모는 Hugging Face를 통해 전체 공개되었습니다.

## 2. 핵심 성과 요약

| 항목 | 내용 |
| :--- | :--- |
| **🗂️ 데이터셋 구축** | 번역 오류 분석 및 품질 개선을 통해 고품질의 한국어 학습 데이터셋 구축 |
| **🤖 모델 개발** | 한국어 데이터셋을 활용하여 Text-to-SQL 태스크에 특화된 모델 개발 |
| **🚀 성능 향상** | Spider 검증 데이터셋 기준, **정확 일치율(EM) 42.65%, 실행 정확도(EX) 65.47%** 달성 |
| **🌐 모델 및 데이터셋 공개** | 모델과 한국어 데이터셋을 Hugging Face Hub에 공개하여 누구나 활용 가능 |
| **▶️ 실시간 데모 배포** | 생성된 모델을 직접 테스트해볼 수 있는 데모 애플리케이션을 Hugging Face Spaces에 배포 |

## 3. 성능 평가 결과

### 3.1. 평가 지표 해설: 정확 일치율(EM) vs. 실행 정확도(EX)
Text-to-SQL 모델의 성능은 주로 두 가지 지표로 평가합니다.

* **정확 일치율 (Exact Match, EM)**: 모델이 생성한 SQL 쿼리가 정답 SQL 쿼리와 문자열 수준에서 완전히 일치하는지 평가하는 엄격한 지표입니다. (※ 본 평가에서는 공정한 비교를 위해 정규화 과정을 거침)
* **실행 정확도 (Execution Accuracy, EX)**: 모델이 생성한 SQL 쿼리를 실제 데이터베이스에서 실행했을 때, 그 결과가 정답 쿼리의 실행 결과와 일치하는지 평가합니다.

**실행 정확도(EX)가 더 중요한 이유:**
실행 정확도가 정확 일치율보다 높은 이유는, SQL 쿼리가 의미적으로는 동일하더라도 표현 방식이 다를 수 있기 때문입니다. 예를 들어, 테이블에 부여하는 별칭(alias)이 다르거나(`... AS T1` vs. `... AS T2`), JOIN 하는 테이블의 순서가 달라져도 실행 결과는 동일할 수 있습니다.

이러한 미세한 구문 차이는 문자열 기반의 정확 일치율(EM)을 낮추지만, 실제 쿼리 실행 결과에는 영향을 주지 않으므로 실행 정확도(EX)는 더 높게 측정됩니다. 따라서 실제 비즈니스 환경에서는 실행 정확도가 모델의 실질적인 성능을 더 잘 나타냅니다.

> **참고**: `SELECT` 절의 컬럼 순서가 달라지는 경우는 실행 결과 테이블의 구조가 바뀌므로 EM뿐만 아니라 EX에서도 불일치로 처리됩니다.

### 3.2. 최종 성능
Spider 검증 데이터셋(1,034개)을 활용하여 파인튜닝된 모델의 성능을 측정한 결과입니다.

| 평가 지표 | 성능 | 비고 |
| :--- | :--- | :--- |
| **정확 일치율 (EM)** | **42.65%** (441/1034) | SQL 쿼리가 정규화 후 정확히 같은 경우 |
| **실행 정확도 (EX)** | **65.47%** (677/1034) | SQL 쿼리 실행 결과가 같은 경우 |

### 3.3. 원본 모델과의 성능 비교 분석
파인튜닝의 효과를 입증하기 위해, 원본 LLaMA 3.1 모델의 성능과 파인튜닝된 모델의 성능을 종합적으로 비교했습니다.

| 모델 | 처리 언어 | 정확 일치율 (EM) | 실행 정확도 (EX) | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **원본 LLaMA 3.1** | 영어 | 2.22% | 7.83% | 영어 원문 데이터셋 기준 |
| **원본 LLaMA 3.1** | 한국어 | 0.39% | 0.87% | 프로젝트 시작점 (Baseline) |
| **파인튜닝 모델** | **한국어** | **42.65%** | **65.47%** | **최종 결과물** |

**분석 결과:**

1.  **한국어 성능의 극적인 향상**: 파인튜닝 후 모델의 한국어 질문 처리 성능은 실행 정확도(EX) 기준 **0.87%에서 65.47%로 약 75배 수직 상승**했습니다. 이는 한국어 특화 데이터셋을 활용한 파인튜닝이 해당 언어 및 태스크에 대한 모델의 이해도를 결정적으로 높였음을 의미합니다.
2.  **태스크 특화 튜닝 효과**: 가장 주목할 점은 태스크 특화 튜닝의 효과입니다. 파인튜닝된 모델의 한국어 실행 정확도(65.47%)는 원본 모델이 영어로 기록한 성능(7.83%)마저 크게 뛰어넘었습니다. 이는 단순히 다른 언어로의 지식 전이를 넘어, 특정 태스크(Text-to-SQL)에 고도로 특화된 데이터로 파인튜닝하는 과정이 모델의 근본적인 문제 해결 능력을 강화시켰음을 시사합니다.

### 3.4. 정확 일치율(EM) 평가를 위한 SQL 정규화
정확 일치율(EM)은 문자열을 직접 비교하므로, SQL의 의미와 관계없는 공백, 대소문자, 따옴표 종류 등의 차이만으로도 실패할 수 있습니다. 공정한 비교를 위해, 모델이 생성한 쿼리와 정답 쿼리를 모두 아래의 정규화(Normalization) 코드를 거친 후 비교를 수행했습니다.

```python
def normalize_sql(sql):
    import re
    if not isinstance(sql, str):
        return sql
    # 1. 따옴표로 묶인 값들을 추출하여 임시 저장
    quoted_values = []
    def replace_quotes(match):
        content = match.group(1) or match.group(2)
        placeholder = f"__QUOTED_VALUE_{len(quoted_values)}__"
        quoted_values.append(content)
        return f"'{placeholder}'"
    pattern = r"'([^']*)'|\"([^\"]*)\""
    sql_with_placeholders = re.sub(pattern, replace_quotes, sql)

    # 2. 나머지 SQL 정규화 (식별자, 키워드 등)
    normalized_sql = sql_with_placeholders.lower().strip()
    normalized_sql = re.sub(r'\s+', ' ', normalized_sql)
    normalized_sql = re.sub(r'\s*,\s*', ', ', normalized_sql)
    normalized_sql = re.sub(r';+\s*$', '', normalized_sql)

    comparison_ops = ['!=', '<>', '>=', '<=', '=', '>', '<']
    for op in comparison_ops:
        pattern = r'\s*' + re.escape(op) + r'\s*'
        normalized_sql = re.sub(pattern, f' {op} ', normalized_sql)

    arithmetic_ops = ['+', '-', '*', '/']
    for op in arithmetic_ops:
        pattern = r'\s*' + re.escape(op) + r'\s*'
        normalized_sql = re.sub(pattern, f' {op} ', normalized_sql)

    keyword_ops = ['and', 'or', 'not', 'in', 'like', 'between']
    for op in keyword_ops:
        pattern = r'\s+' + op + r'\s+'
        normalized_sql = re.sub(pattern, f' {op} ', normalized_sql, flags=re.IGNORECASE)

    normalized_sql = re.sub(r'\s+', ' ', normalized_sql)

    # 3. 저장했던 원래 값들 다시 넣기
    for i, value in enumerate(quoted_values):
        placeholder = f"__quoted_value_{i}__"
        normalized_sql = normalized_sql.replace(f"'{placeholder}'", f"'{value}'")

    return normalized_sql.strip()
```

## 4. 모델 및 데이터셋 상세 정보

### 4.1. 모델 정보
- **Hugging Face 모델**: [huggingface-KREW/Llama-3.1-8B-Spider-SQL-Ko](https://huggingface.co/huggingface-KREW/Llama-3.1-8B-Spider-SQL-Ko)
- **Hugging Face 데이터셋**: [huggingface-KREW/spider-ko](https://huggingface.co/datasets/huggingface-KREW/spider-ko) (영어 Spider 데이터셋 한국어 번역 및 검수 버전)
- **Hugging Face Spaces 데모**: [Hugging Face Spaces](https://huggingface.co/spaces/huggingface-KREW/Llama-3.1-8B-Spider-SQL-Ko)
- **기반 모델**: `unsloth/Meta-Llama-3.1-8B-Instruct` (4bit 양자화 적용)


### 4.2. 데이터셋 구축 및 검수
초기 번역 과정에서 LLM이 불필요한 설명을 덧붙이거나, 비교급 표현("이상" vs "초과"), 서술어 호응, 도메인 어휘 등을 잘못 번역하는 문제가 발견되었습니다. 이를 해결하기 위해 번역 프롬프트를 개선하고, 수동 검수 및 오류 유형 분석을 통해 데이터 품질을 대폭 향상시켰습니다. (상세 내용은 Appendix 참고)

### 4.3. 파인튜닝 세부 정보
- **GPU**: NVIDIA Tesla T4 (16GB)
- **라이브러리**: Unsloth
- **VRAM 사용량**: 최대 7.6GB
- **학습 시간**: 약 4시간
- **학습 코드**: [Google Colab](https://drive.google.com/file/d/1tAlGr7t8r60j2jxafyhP0MGQU7s0j1XH/view?usp=sharing)
- **주요 하이퍼파라미터**:
  ```python
  training_args = {
      "per_device_train_batch_size": 2,
      "gradient_accumulation_steps": 4,
      "learning_rate": 5e-4,
      "num_train_epochs": 1,
      "optimizer": "adamw_8bit",
      "lr_scheduler_type": "cosine",
      "warmup_ratio": 0.05
  }

  lora_config = {
      "r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0,
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
  }
  ```

## 5. 모델 사용법 및 데모

### 5.1. Hugging Face 데모 사용법
1.  **[Hugging Face Spaces](https://huggingface.co/spaces/huggingface-KREW/Llama-3.1-8B-Spider-SQL-Ko) 접속**
2.  데이터베이스 스키마 정보 입력 (테이블명, 컬럼명 등)
3.  한국어로 질문 작성 후 제출
4.  생성된 SQL 쿼리 확인

### 5.2. 로컬 환경에서 사용하기

```python
# 1. 필요 라이브러리 설치
!pip install "unsloth[colab-new]" -q
!pip install --no-deps trl peft accelerate bitsandbytes

# 2. 모델 및 토크나이저 로딩
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="huggingface-KREW/Llama-3.1-8B-Spider-SQL-Ko",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 3. SQL 생성 함수 정의 및 실행
def generate_sql(question, schema_info):
    prompt = f"""다음 데이터베이스 스키마를 참고하여 질문에 대한 SQL 쿼리를 생성하세요.

### 데이터베이스 스키마:
{schema_info}

### 질문: {question}

### SQL 쿼리:"""

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    outputs = model.generate(inputs, max_new_tokens=150, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split("### SQL 쿼리:")[-1].strip()

# 4. 사용 예시
schema = """테이블: singer
컬럼: singer_id, name, country, age"""

query = generate_sql("가수는 몇 명이 있나요?", schema)
print(query)
# 출력: SELECT count(*) FROM singer
```

## 6. 한계점 및 향후 개선 방향

### 6.1. 현재 한계
- **영어 스키마 제한**: 테이블/컬럼명은 현재 영어로만 사용 가능합니다.
- **도메인 제약**: Spider 데이터셋에 포함된 도메인에 최적화되어 있습니다.
- **복잡한 쿼리 제한**: 매우 복잡한 중첩 쿼리의 경우 정확도가 하락할 수 있습니다.

### 6.2. 향후 개선 방향
- **데이터셋 품질 개선**: 테이블 및 컬럼명의 대소문자, 약어 사용 등을 일관되게 표준화한 버전을 추가 제공하여 개발자 편의성을 높일 계획입니다.
- **데이터 증강**: 다양한 질문 스타일과 복잡도를 포함하도록 데이터를 추가하여 모델의 강건성을 높입니다.
- **모델 확장**: 더 큰 규모의 모델(예: 13B 이상)을 활용하여 정확도 향상을 시도합니다.

---
## Appendix: 데이터셋 번역 및 검수 과정에서 발견된 주요 오류 유형

번역 데이터의 품질을 높이기 위해 수동 검수를 진행했으며, 발견된 주요 오류는 다음과 같습니다.

| 오류 유형 | 주요 내용 | 수정 방향 |
| :--- | :--- | :--- |
| 1. **시간/수 비교 표현 오류** | `more than`을 '이상'으로, `before`/`after`를 '이전'/'이후'로 번역 | '초과', '미만', '~보다 전/후' 등 경계값을 포함하지 않는 명확한 표현으로 수정 |
| 2. **어색한 한국어 표현** | 불필요한 주어(`우리는`)나 수식어(`모든 다른`), 어색한 단어(`역알파벳순`) 사용 | 한국어 문맥에 맞게 자연스럽게 생략하거나 '국가들은', '알파벳 역순' 등으로 수정 |
| 3. **서술어 호응 오류** | 개수나 나이를 묻는데 '무엇인가요?'로 답하거나, 여러 주어에 하나의 서술어만 사용 | 주어에 맞는 서술어('몇 명인가요?', '얼마인가요?')를 사용하고, 각 주어에 맞는 서술어를 분리하여 질문 |
| 4. **도메인 어휘 오번역** | `production time`을 '생산 시간'으로, `accelerate`를 '가속도'로 번역 | 도메인 용어에 맞게 '생산일자', '가속 성능' 등으로 수정 |
| 5. **의미 누락 및 변형** | `average`를 '평균'으로만 번역하여 '평균 수용 인원'의 의미가 누락됨 | 원문의 의도를 정확히 반영하도록 누락된 정보를 보충하여 수정 |
