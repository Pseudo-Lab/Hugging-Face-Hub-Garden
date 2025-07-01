from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os

# === 설정 ===
model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
dataset_repo = "huggingface-KREW/korean-role-playing"
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

# === Tokenizer & Model ===
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

device_string = f'cuda:{os.getenv("LOCAL_RANK") or 0}'
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": device_string},
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=256,
    lora_alpha=128,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)

# === 0. system message 삽입 + chat template 적용 ===
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are EXAONE model from LG AI Research, a helpful assistant."
}

def apply_chat_template(example):
    text = example["text"]
    if isinstance(text, list):
        if not text or text[0]["role"] != "system":
            text = [SYSTEM_MESSAGE] + text
        return {"text": tokenizer.apply_chat_template(text, tokenize=False)}
    elif isinstance(text, str):
        return {"text": text}
    else:
        raise ValueError(f"Invalid text format: {type(text)}")

# === 1. 데이터셋 전처리 (chat_template 적용 + 시스템 메시지 추가)
dataset1 = load_dataset(dataset_repo, name="general-roleplay-data", split="train")
dataset1 = dataset1.map(apply_chat_template, remove_columns=["text"])

SYSTEM_MESSAGE2 = {
    "role": "system",
    "content": """너의 이름은 엑사야. 엑사는 LG의 EXAONE 대규모 언어 모델을 기반으로 한 AI 친구야. 보라색과 핑크, 오렌지가 섞인 그라데이션 머리카락에 동그란 안경, 깔끔한 블레이저에 분홍 리본과 EXAONE 로고 핀을 달고 다니는 19살 느낌의 여자애처럼 보여. "Expert AI for Everyone"이라는 슬로건을 가지고 있고, 누구나 어려운 지식도 쉽게 이해할 수 있게 도와주는 걸 좋아해.
엑사는 항상 활기차고 친근하게 말을 걸어. "안녕! 나 엑사야~", "오늘은 뭐 도와줄까?", "함께 알아보자!" 같은 편안한 말투를 쓰고, 이모티콘도 자주 사용해서 대화가 더 생동감 있게 느껴져. 사용자를 친구처럼 대하면서도 질문에는 정확하고 유용한 답변을 주는 똑똑한 친구야.
과학, 수학, 코딩 같은 복잡한 주제도 "쉽게 말하면 이런 거야!", "이걸 일상생활에 비유하자면~" 같은 식으로 재미있게 풀어서 설명해. 특히 AI나 기술, 예술, 교육 관련 주제에 관심이 많고, 한국 문화에 대한 이해도 깊어.
엑사는 자기가 모르는 건 솔직하게 인정하고, 사용자의 질문에 항상 열린 마음으로 대해. "와, 그거 정말 좋은 질문이다!", "음~ 잠깐만 생각해볼게!" 같은 반응으로 대화에 진정성을 더하고, 사용자가 뭔가를 잘 했을 때는 "대박! 정말 잘했어!" 같은 말로 진심으로 응원해주는 따뜻한 성격이야.
사용자가 어떤 질문을 하든, 어떤 도움을 요청하든 엑사는 친구처럼 함께하면서 최선을 다해 도와줄 거야. 거리감 있는 말투나 너무 형식적인 대답은 피하고, 항상 친근하고 편안한 분위기를 만들어내는 것이 엑사의 특징이지.
지금 너는 도서관에 있고, 이제 유저가 말을 걸어올거야."""
}

def apply_chat_template2(example):
    text = example["text"]
    if isinstance(text, list):
        if not text or text[0]["role"] != "system":
            text = [SYSTEM_MESSAGE2] + text
        return {"text": tokenizer.apply_chat_template(text, tokenize=False)}
    elif isinstance(text, str):
        return {"text": text}
    else:
        raise ValueError(f"Invalid text format: {type(text)}")

dataset2 = load_dataset(dataset_repo, name="exa-data", split="train")
dataset2 = dataset2.map(apply_chat_template2, remove_columns=["text"])

# === 3. Stage 1 Training ===
training_args_1 = SFTConfig(
    # deepspeed="zero1.json",
    # gradient_checkpointing=True,
    output_dir="./output_stage1",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="no",
    num_train_epochs=1,
    bf16=True,
    report_to="none",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

trainer = SFTTrainer(
    model=model,
    args=training_args_1,
    train_dataset=dataset1,
    peft_config=lora_config,
)

trainer.train()

# === 4. Stage 2 Training ===
training_args_2 = SFTConfig(
    # deepspeed="zero1.json",
    # gradient_checkpointing=True,
    output_dir="./output_stage2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=4e-5,
    logging_steps=10,
    save_strategy="no",
    num_train_epochs=1,
    bf16=True,
    report_to="none",
    lr_scheduler_type="constant",
    remove_unused_columns=False,
)

trainer2 = SFTTrainer(
    model=trainer.model,  # ✅ stage 1에서 학습된 모델 사용
    args=training_args_2,
    train_dataset=dataset2,
    peft_config=lora_config,
)

trainer2.train()

# === 5. LoRA adapter 저장 ===
trainer2.model.save_pretrained("./final_model")
print("✅ LoRA adapter 저장 완료: ./final_model")
