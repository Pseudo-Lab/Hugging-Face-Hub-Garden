from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from huggingface_hub import login
import torch
from datasets import load_dataset, Dataset
import json
import pandas as pd
from trl import SFTTrainer, SFTConfig
# from peft import get_peft_model, LoraConfig, TaskType  # PEFT 관련 import 제거
import argparse
import copy

# 대화 시작을 위한 고정된 접두사 정의.
# 이 접두사는 모델이 사용자 질문에 대해 특정 페르소나로 응답하도록 유도합니다.
PREFIX = "친구와 채팅을 하고 있다고 가정하고 다음 질문에 밈과 유행어를 활용하여 대답하세요."

def process_data(dataset_name) -> Dataset:
    """
    Hugging Face 데이터셋을 로드하고, 각 질문에 정의된 PREFIX를 추가하여 가공합니다.
    이 전처리 과정은 모델이 특정 형식의 질문에 응답하도록 훈련시키는 데 중요합니다.

    Args:
        dataset_name (str): Hugging Face Hub에서 로드할 데이터셋의 이름 (예: 'huggingface-KREW/KoCulture-Dialogues').

    Returns:
        Dataset: PREFIX가 추가된 질문과 원본 답변을 포함하는 Hugging Face Dataset 객체.
                 Dataset의 각 entry는 'question'과 'answer' 키를 가집니다.
    """
    def process_conversations(examples):
        """
        배치 단위로 질문과 답변을 처리하여 PREFIX를 질문에 추가합니다.
        """
        questions = []
        answers = []
        
        # 각 질문-답변 쌍에 대해 PREFIX를 추가하고 리스트에 저장
        for q, a in zip(examples['question'], examples['answer']):
            processed_q = f"{PREFIX}: {q}"
            questions.append(processed_q)
            answers.append(a)
        
        return {
            'question': questions,
            'answer': answers
        }
    
    # Hugging Face Hub에서 'train' 스플릿 데이터셋 로드
    try:
        dataset = load_dataset(
            dataset_name, 
            split='train', 
            download_mode="force_redownload", # 캐시를 무시하고 강제로 다시 다운로드
            verification_mode="no_checks" # 데이터셋 검증 절차 건너뛰기
        )
    except Exception as e:
        # 'train' 스플릿이 없는 경우를 대비하여 스플릿 없이 재시도
        dataset = load_dataset(
            dataset_name, 
            download_mode="force_redownload",
            verification_mode="no_checks"
        )
    
    # process_conversations 함수를 데이터셋에 적용.
    # batched=True를 통해 효율적인 배치 처리를 수행합니다.
    processed_dataset = dataset.map(
        process_conversations,
        batched=True
    )
    
    return processed_dataset

def formatting_function(examples, tokenizer):
    """
    훈련을 위해 질문과 답변 쌍을 모델이 이해할 수 있는 대화 형식 텍스트로 변환합니다.
    이는 `tokenizer.apply_chat_template`을 사용하여 진행되며, 
    모델의 입력 형식을 맞추는 데 필수적입니다.

    Args:
        examples (dict): 'question'과 'answer' 키를 포함하는 딕셔너리.
        tokenizer (AutoTokenizer): 대화 템플릿을 적용할 토크나이저 인스턴스.

    Returns:
        dict: 모델의 'text' 필드로 사용될 변환된 대화 텍스트를 포함하는 딕셔너리.
    """
    q = examples['question']
    a = examples['answer']
    
    # 사용자 역할과 어시스턴트 역할의 메시지 형식 정의
    messages = [{'role': 'user', 'content': q},
               {'role': 'assistant', 'content': a}]
    
    # 토크나이저의 `apply_chat_template`을 사용하여 대화 형식을 적용
    # add_generation_prompt=False: 생성 프롬프트 추가 안 함 (훈련 시에는 보통 False)
    # tokenize=False: 토큰화는 하지 않고 텍스트만 반환
    # enable_thinking=False: Llama3의 사고 과정 출력 비활성화
    tokenized_conv = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, 
        tokenize=False,
        enable_thinking=False
    )
    
    return {'text':tokenized_conv}

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_seq_length)

def main():
    """
    주요 학습 및 평가 흐름을 정의하는 함수입니다.
    명령줄 인자를 파싱하고, 모델을 로드하며, 데이터를 전처리하고, 모델을 훈련시킨 후,
    최종적으로 훈련된 모델을 평가하고 Hugging Face Hub에 업로드합니다.
    """
    global args # tokenize_function에서 args를 사용할 수 있도록 global로 선언
    parser = argparse.ArgumentParser(description='Transformers 라이브러리를 사용하여 언어 모델을 Full Fine-tuning합니다.')
    parser.add_argument('--model_name', type=str, required=True, 
                        help='사전 훈련된 모델의 Hugging Face 이름.')
    parser.add_argument('--max_seq_length', type=int, default=2048, 
                        help='모델의 최대 시퀀스 길이 (컨텍스트 길이). 이 길이에 맞춰 입력이 잘리거나 패딩됩니다.')
    parser.add_argument('--option_4bit', action='store_true', 
                        help='모델을 4비트 양자화로 로드하여 메모리 사용량을 최적화합니다.')
    parser.add_argument('--option_8bit', action='store_true', 
                        help='모델을 8비트 양자화로 로드하여 메모리 사용량을 최적화합니다.')
    parser.add_argument('--data_path', type=str, default='huggingface-KREW/KoCulture-Dialogues', 
                        help='학습 데이터셋의 Hugging Face 경로 또는 로컬 경로.')
    parser.add_argument('--output_dir', type=str, default='./results_transformers', 
                        help='훈련 결과 (체크포인트, 로그 등)가 저장될 디렉토리.')
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='각 GPU 장치당 훈련 배치 크기.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                        help='경사 누적 스텝 수. 이를 통해 GPU 메모리 제약 하에 더 큰 배치 크기를 시뮬레이션할 수 있습니다.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                        help='훈련 중 사용될 학습률.')
    parser.add_argument('--num_epochs', type=int, default=3, 
                        help='훈련 에포크 수.')
    parser.add_argument('--private', action='store_true', 
                        help='Hugging Face Hub에 모델을 비공개로 푸시합니다.')

    args = parser.parse_args()

    # Hugging Face Hub에 로그인.
    # TODO: "hf_yourtoken"을 실제 Hugging Face 토큰으로 변경하거나,
    # 터미널에서 `huggingface-cli login` 명령어를 사용하여 로그인하세요.
    login(token="hf_yourtoken")

    # Full fine-tuning에서는 양자화 사용 시 경고
    # 양자화는 일반적으로 PEFT (LoRA)와 함께 사용됩니다. 만약 Full fine-tuning 시에도 양자화를 사용한다면 예상치 못한 문제나 성능 저하가 발생할 수 있습니다.
    if args.option_4bit or args.option_8bit:
        print("Warning: Full fine-tuning with quantization may cause issues. Consider using smaller batch sizes or disabling quantization.")
    
    # 1. 토크나이저 로드 및 패딩 토큰 설정
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 사전 훈련된 모델 로드 (Full Fine-tuning 목적)
    # AutoModelForCausalLM은 텍스트 생성을 위한 인과적 언어 모델을 로드합니다.
    # trust_remote_code=True: 사용자 정의 모델 아키텍처를 로드할 때 필요할 수 있습니다.
    # torch_dtype=torch.bfloat16: 모델을 BFloat16 정밀도로 로드하여 메모리 사용량을 줄이고 학습 속도를 향상시킵니다.
    # device_map="auto": 모델을 사용 가능한 GPU에 자동으로 분산하여 로드합니다.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True, # 일부 모델은 이 옵션이 필요할 수 있음
        torch_dtype=torch.bfloat16, # 모델을 16-bit로 로드
        device_map="auto",  # 자동으로 GPU에 할당
    )

    # PEFT 설정 제거 - Full fine-tuning 사용
    print("Using full fine-tuning (all parameters will be updated)")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 3. 데이터셋 로드 및 포맷팅
    # 훈련 데이터셋을 로드하고, 모델 훈련에 적합한 대화 형식으로 변환합니다.
    dataset = process_data(args.data_path)
    # formatting_function을 먼저 적용하여 'text' 필드 생성
    formatted_dataset = dataset.map(
        formatting_function, 
        fn_kwargs={'tokenizer': tokenizer}, 
        batched=False # 각 예제별로 함수 적용
    ) 
    # 그 다음 'text' 필드를 토큰화하여 Trainer가 사용할 수 있는 형태로 변환
    
    print(formatted_dataset[0]) # 포맷팅된 데이터 예시 출력

    # 4. SFTTrainer 설정 및 훈련 시작
    # SFTTrainer는 Supervised Fine-Tuning을 위한 TRL 라이브러리의 클래스입니다.
    # SFTConfig는 훈련에 필요한 다양한 하이퍼파라미터를 정의합니다.
    # dataset_text_field: 훈련 데이터셋에서 모델 입력으로 사용할 텍스트 필드 지정 ('text')
    # optim: 옵티마이저 (예: "adamw_8bit"는 메모리 효율적인 AdamW 구현)
    # lr_scheduler_type: 학습률 스케줄러 유형 (예: "linear", "cosine")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset,
        eval_dataset = None, # 본 코드에서는 별도의 평가 데이터셋을 사용하지 않습니다.
        args = SFTConfig(
            dataset_text_field = "text", # 훈련에 사용될 텍스트 필드
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_ratio=0.05, # 워밍업 스텝 비율
            num_train_epochs= args.num_epochs,
            learning_rate = args.learning_rate, 
            logging_steps = 5, # 훈련 로그를 기록하는 빈도
            optim = "adamw_8bit", # 8비트 AdamW 옵티마이저 (메모리 효율적)
            weight_decay = 0.01, # 가중치 감소 (L2 정규화)
            lr_scheduler_type = "linear", # 선형 학습률 스케줄러
            seed = 3407, # 훈련 재현성을 위한 시드 고정
            report_to = "none", # W&B 등 보고 도구 비활성화
            output_dir = args.output_dir, # 훈련 로그 및 체크포인트 저장 디렉토리
            use_liger= True #TRL의 Liger kernel 사용 (옵션, 성능 향상 기대 가능)
        ),
    )
    trainer_stats = trainer.train()

    # 5. 모델 업로드 (Full fine-tuned 모델)
    # Hugging Face Hub에 푸시될 최종 모델 이름 정의
    true_name = args.model_name.split("/")[-1]
    train_option_str = "4bit" if args.option_4bit else "8bit" if args.option_8bit else "full"
    hub_model_name = f"{true_name}-KoCulture-{train_option_str}train-transformers"
    
    try:
        # 전체 모델과 토크나이저를 Hugging Face Hub에 업로드
        model.push_to_hub(hub_model_name, private=args.private) 
        tokenizer.push_to_hub(hub_model_name, private=args.private)
        print(f"Model pushed to hub: {hub_model_name}")
    except Exception as e:
        print(f"Failed to push model to hub: {e}")
        print("Make sure you are logged in via `huggingface-cli login` or pass a token.")

if __name__ == "__main__":
    main()