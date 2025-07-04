#!/bin/bash

# 공통 설정
DATA_PATH="huggingface-KREW/KoCulture-Dialogues-v2"
MAX_SEQ_LENGTH=512
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=64
LEARNING_RATE=6e-5
NUM_EPOCHS=3

# 학습할 모델 목록
# MODELS="kakaocorp/kanana-1.5-8b-instruct-2505 LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
# MODELS="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
MODELS="unsloth/Qwen3-8B"

# 각 모델에 대해 순차적으로 학습 실행
for MODEL_NAME in $MODELS; do
    echo "========================================="
    echo "Starting training for: $MODEL_NAME"
    echo "========================================="
    
    # 현재 시간 기록
    START_TIME=$(date)
    echo "Start time: $START_TIME"
    
    
    # 학습 실행
    CUDA_VISIBLE_DEVICES=1 uv run train_transformers_full_fine_tuning.py \
        --model_name "$MODEL_NAME" \
        --data_path "$DATA_PATH" \
        --max_seq_length $MAX_SEQ_LENGTH \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --private
    
    # 학습 결과 확인
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully for: $MODEL_NAME"
    else
        echo "❌ Training failed for: $MODEL_NAME"
        echo "Continuing with next model..."
    fi
    
    # 종료 시간 기록
    END_TIME=$(date)
    echo "End time: $END_TIME"
    echo "========================================="
    echo ""
    
    # GPU 메모리 정리를 위한 잠시 대기
    sleep 10
done

echo "🎉 All model training completed!"