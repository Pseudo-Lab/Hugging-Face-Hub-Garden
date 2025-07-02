import os, json, random, datetime, numpy as np, torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import TrainerCallback

# ─────────────── 환경 세팅 ───────────────
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all()

BASE = os.path.join(os.getcwd(), "research")
TRAIN_JSON = f"{BASE}/example.json"
# VALID_JSON = f"{BASE}/valid.json"

OUT_DIR = f"{BASE}/kor-english-exam"
LOG_DIR = f"{OUT_DIR}/logs"; VIS_DIR = f"{OUT_DIR}/visualizations"
for d in (OUT_DIR, LOG_DIR, VIS_DIR): os.makedirs(d, exist_ok=True)
EXP_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_ID = "Qwen/Qwen3-8B"

# ─────────────── 데이터 로드 ───────────────
def load_ds(p): return Dataset.from_list(json.load(open(p, encoding="utf-8")))
train_raw = load_ds(TRAIN_JSON)
#valid_raw = load_ds(VALID_JSON)


print(f"훈련 데이터: {len(train_raw)}개")
# print(f"검증 데이터: {len(valid_raw)}개")

# ─────────────── 토크나이저 ───────────────
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token; tok.padding_side = "right"

def create_prompt(ex):
    """영어 문제 변형을 위한 프롬프트 생성"""
    question = ex.get("question")

    # 질문 타입별 설명 추가
    question_descriptions = {
        '다음 글의 제목으로 가장 적절한 것은?': '제목 추론',
        '다음 글의 주제로 가장 적절한 것은?': '주제 추론',
        '다음 글의 요지로 가장 적절한 것은?': '요지 추론',
        '다음 글의 내용과 일치하지 않는 것은?': '내용 불일치',
        '다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오.': "빈칸 추론",
        '다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?': "어법 오류"
    }
    description = question_descriptions.get(question)

    if description in ['제목 추론', '주제 추론', '요지 추론', '내용 불일치']:
        # 기본 형태 (원본 지문 사용)
        messages = [
            {
                "role": "user", 
                "content": (
                    f"다음 지문을 {description} 문제로 만들어주세요.\n\n"
                    f"지문:\n{ex['original']}\n\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    f"문제: {ex['question']}\n"
                    f"선택지:\n{ex['options']}\n"
                    f"정답: {ex['answer']}"
                )
            }
        ]
    
    elif description == '빈칸 추론':
        # 빈칸 추론 (변형 지문 + 선택지 포함)
        messages = [
            {
                "role": "user", 
                "content": (
                    f"다음 영어 지문을 {description} 문제로 만들어주세요.\n\n"
                    f"지문:\n{ex['original']}\n\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    f"변형 지문: {ex['passage']}\n"
                    f"문제: {ex['question']}\n"
                    f"선택지:\n{ex['options']}\n"
                    f"정답: {ex['answer']}"
                )
            }
        ]
    
    elif description == '어법 오류':
        # 어법 오류 (변형 지문 + 선택지 없음)
        messages = [
            {
                "role": "user", 
                "content": (
                    f"다음 영어 지문을 {description} 문제로 만들어주세요.\n\n"
                    f"지문:\n{ex['original']}\n\n"
                )
            },
            {
                "role": "assistant",
                "content": (
                    f"변형 지문: {ex['passage']}\n"
                    f"문제: {ex['question']}\n"
                    f"정답: {ex['answer']}"
                )
            }
        ]
    else:
        print("ERROR!!!\n\n\n")
    
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True
    )


print("프롬프트 생성 중...")
train_ds = Dataset.from_dict({"text": [create_prompt(e) for e in tqdm(train_raw, desc="Train")]})
# valid_ds = Dataset.from_dict({"text": [create_prompt(e) for e in tqdm(valid_raw, desc="Valid")]})

def tok_fn(batch):
    out = tok(batch["text"], padding=False, truncation=True, max_length=1024)
    out["labels"] = out["input_ids"].copy()
    return out

tok_train = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
# tok_valid = valid_ds.map(tok_fn, batched=True, remove_columns=["text"])

# ─────────────── 모델 & LoRA ───────────────
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=bnb, torch_dtype=torch.bfloat16, trust_remote_code=True
)
base = prepare_model_for_kbit_training(base)

lora_cfg = LoraConfig(
    r=16, lora_alpha=64, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base, lora_cfg)
print(model.print_trainable_parameters())

collator = DataCollatorForSeq2Seq(tok, model=model, padding="longest", return_tensors="pt")

# ─────────────── 콜백 ───────────────
class HistCB(TrainerCallback):
    def __init__(self, d): self.d = d; self.h = {}
    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            self.h.setdefault("log", []).append(logs)
            json.dump(self.h, open(f"{self.d}/log_{EXP_ID}.json", "w"), indent=2)

# ─────────────── Trainer ───────────────
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    # per_device_eval_batch_size=2,
    learning_rate=1e-4,
    num_train_epochs=3,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    save_strategy="epoch",
    eval_strategy="no",
    # load_best_model_at_end=False,
    # metric_for_best_model="eval_loss",
    save_total_limit=3,
    logging_steps=25,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    # eval_dataset=tok_valid,
    data_collator=collator,
    callbacks=[HistCB(LOG_DIR)],
)

# ─────────────── 학습 시작 ───────────────
print("="*50)
print("영어 문제 변형 모델 훈련 시작")
print("="*50)

try:
    trainer.train()
    print("훈련 완료!")
except Exception as e:
    print(f"훈련 중 오류 발생: {e}")
    raise

# ─────────────── 저장 ───────────────
BEST = f"{OUT_DIR}/best_model_final_{EXP_ID}"
model.save_pretrained(BEST); tok.save_pretrained(BEST)
print(f"✓ best model saved to {BEST}")
