# 기본 설정
# MODEL_NAME="unsloth/Qwen3-32B-unsloth-bnb-4bit"  
# MODEL_NAME="kakaocorp/kanana-1.5-8b-instruct-2505"
# MODEL_NAME="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
# MODEL_NAME="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
MODEL_NAME="unsloth/Qwen3-8B"
DATA_PATH="huggingface-KREW/KoCulture-Dialogues"
MAX_SEQ_LENGTH=512
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=64
LEARNING_RATE=6e-5
NUM_EPOCHS=1

# 학습 실행
uv run train_transformers_full_fine_tuning.py \
  --model_name $MODEL_NAME \
  --data_path $DATA_PATH \
  --max_seq_length $MAX_SEQ_LENGTH \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --private
  # --option_8bit  # 대신 8비트를 원하면 이 줄의 주석을 해제하고 위 줄을 주석 처리
  # --option_full  # 전체 미세 조정을 원하면 이 줄의 주석을 해제하고 위 줄을 주석 처리