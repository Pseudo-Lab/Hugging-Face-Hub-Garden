
![KoCulture-Dialogues_4](https://github.com/user-attachments/assets/50a1e7d8-b8a4-493d-9ff3-e6680bf82d1c)

## 목차
- [학습모델](#학습모델)
- [데이터](#데이터)
- [평가 결과](#평가-결과)
- [파일 구조 및 설명](#파일-구조-및-설명)
- [라이선스](#라이선스)
- [기여](#기여)

## 학습모델 
| 모델 | UNDERSTAND(full/unsloth) | USAGE(full/unsloth) |
|---|:---:|:---:|
| EXAONE-3.5-7.8B-Instruct | [link](https://huggingface.co/huggingface-KREW/EXAONE-3.5-7.8B-Instruct-KoCulture)/- | -/- |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | -/- | -/- |
| kanana-1.5-8b-instruct-2505 | -/- | -/- |
| Qwen3-8B | -/- | -/- |

## 데이터
* [밈 설명(KoCultre-Descriptions)](https://huggingface.co/datasets/huggingface-KREW/KoCultre-Descriptions)
* [밈 사용 대화쌍(KoCulture-Dialogues)](https://huggingface.co/datasets/huggingface-KREW/KoCulture-Dialogues)

## 파일 구조 및 설명
이 저장소는 밈(meme) 데이터셋 증강, 모델 훈련, 그리고 평가를 위한 코드와 데이터를 포함하고 있습니다. 주요 파일과 디렉토리는 다음과 같습니다.
```
.
├── generate_augmented_pairs_gemini.py
├── merged_final.json
└── src
    ├── eval
    │   ├── exp.sh
    │   ├── infer_module.py
    │   ├── infer.py
    │   ├── prompt.py
    │   ├── understand_questions.json
    │   ├── usage_questions.json
    │   ├── meme_sample_with_questions.jsonl
    │   ├── understanding_evaluation_augmenter.py
    │   ├── score.py
    │   └── results
    │       ├── understand
    │       │   ├── infer_results
    │       │   └── score_results       
    │       └── usage
    │           ├── infer_results
    │           └── score_results
    └── train
        ├── train_all.sh
        ├── train_transformers_full_fine_tuning.py
        ├── train_transformers_full_fine_tuning.sh
        ├── train_unsloth_fine_tuning.py
        └── train_unsloth_fine_tuning.sh
```
최상위 디렉토리
---
>`generate_augmented_pairs_gemini.py`: 크롤링한 한국어 신조어 데이터에 Gemini API를 사용하여 상황에 따른 밈 질문-답변 대화쌍을 증강하는 스크립트입니다. 데이터 증강을 통해 모델 학습에 필요한 데이터 다양성을 확보합니다.

>`merged_final.json`: 위 대화쌍 증강 코드를 통해 생성한 최종 데이터셋입니다. 이 파일은 모델 학습에 사용되는 핵심 데이터이며 [이 허깅페이스 링크](https://huggingface.co/datasets/huggingface-KREW/KoCulture-Dialogues)에 업로드 되어있습니다.

src 디렉토리
---
src 디렉토리는 크게 **평가(eval)**와 훈련(train) 두 부분으로 나뉩니다.

src/eval 디렉토리
--
모델의 성능을 평가하기 위한 스크립트와 결과물이 포함되어 있습니다.

>`exp.sh`: 평가 실험을 실행하기 위한 셸 스크립트입니다. 다양한 평가 시나리오를 자동화하는 데 사용됩니다.

>`infer_module.py`: 모델 추론을 위한 핵심 모듈입니다. 실제 질문에 대한 모델의 답변을 생성하는 기능을 담당합니다.

>`infer.py`: 추론 프로세스를 실행하는 스크립트입니다. infer_module.py를 사용하여 데이터를 처리하고 결과를 생성합니다.

>`prompt.py`: 이 도구에서 사용하는 프롬프트들을 모아둔 스크립트입니다. 프롬프트 엔지니어링은 모델의 응답 품질에 큰 영향을 미칩니다.

>`understand_questions.json`: 밈의 이해(understand) 능력을 평가하기 위한 각 밈에 대한 설명이 포함된 JSON 파일입니다.

>`usage_questions.json`: 평가용 질문 목록 100건이 포함된 JSON 파일입니다.

>`meme_sample_with_questions.jsonl`: 밈의 이해(understand) 능력을 평가하기 위해 understand_questions.json과 usage_questions.json을 함께 활용하여 만들었습니다.

>`understanding_evaluation_augmenter.py`: 밈의 이해(understand) 능력을 평가하기 위해서 meme_sample_with_questions.jsonl의 데이터를 증강시키는 스크립트입니다.

>`score.py`: 밈의 이해(understand) 또는 밈의 사용(usage) 능력을 채점하고 평가 점수를 계산하는 스크립트입니다.

>`results`: 밈의 이해(understand) 또는 밈의 사용(usage)에 대해 각각 추론 결과와 평가 결과가 저장되는 디렉토리입니다.

src/train 디렉토리
--
모델 훈련을 위한 스크립트들이 포함되어 있습니다.

>`train_all.sh`: 여러 모델에 대해 전체 훈련 과정을 제어하는 통합 셸 스크립트입니다.

>`train_transformers_full_fine_tuning.py`: Transformers 라이브러리를 사용하여 모델을 **전체 파인 튜닝(Full Fine-Tuning)**하는 Python 스크립트입니다. 모델의 모든 레이어를 훈련시켜 특정 작업에 최적화합니다.

>`train_transformers_full_fine_tuning.sh`: 위 train_transformers_full_fine_tuning.py 스크립트를 실행하기 위한 셸 스크립트입니다.

>`train_unsloth_fine_tuning.py`: Unsloth 라이브러리를 사용하여 모델을 파인 튜닝하는 Python 스크립트입니다. Unsloth는 더 빠르고 효율적인 파인 튜닝을 가능하게 합니다.

>`train_unsloth_fine_tuning.sh`: 위 train_unsloth_fine_tuning.py 스크립트를 실행하기 위한 셸 스크립트입니다.

## 개발자
Hugging Face KREW(유용상, 김하림, 오성민)

## 라이선스

>데이터 셋의 경우 사용자는 본 데이터셋을 CC BY-NC-SA 4.0 라이선스 조건에 따라 비영리 목적으로만 사용하고, 
>출처(Hugging Face KREW 및 원본 데이터 제공처)를 명확히 밝혀야 합니다.

## 기여

- 문제나 제안이 있다면 Issue에 기록하십시오.
- 수정 또는 추가가 있으면 Pull Request를 보내주십시오.
    - `dev` 에서 각각의 분기를 만들고,`dev`를 향해 Pull Request를 보내세요.
    - Pull Request의 병합은 검토 후에 수행됩니다. `dev`에서`main`으로의 병합은 타이밍을 보고합니다.
