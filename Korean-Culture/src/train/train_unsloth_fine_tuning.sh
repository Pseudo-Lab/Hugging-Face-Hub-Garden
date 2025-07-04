# 기본 설정
MODEL_NAME="unsloth/Qwen3-32B-unsloth-bnb-4bit"  
# MODEL_NAME="kakaocorp/kanana-1.5-8b-instruct-2505"
DATA_PATH="huggingface-KREW/KoCulture-Dialogues"
MAX_SEQ_LENGTH=1024
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=32
LEARNING_RATE=2e-4
NUM_EPOCHS=3

# 학습 실행
uv run train_unsloth_fine_tuning.py \
  --model_name $MODEL_NAME \
  --data_path $DATA_PATH \
  --max_seq_length $MAX_SEQ_LENGTH \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --option_4bit  # 4비트 정밀도로 모델 로드 (메모리 효율적)
  # --option_8bit  # 대신 8비트를 원하면 이 줄의 주석을 해제하고 위 줄을 주석 처리
  # --option_full  # 전체 미세 조정을 원하면 이 줄의 주석을 해제하고 위 줄을 주석 처리