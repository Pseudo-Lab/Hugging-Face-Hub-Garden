#!/bin/bash

# 일반화된 모델 실험 스크립트
# 사용법: ./exp.sh [model_name] [model_type] [task_types...]
# 예시: ./exp.sh "gpt-4o" "api" "understand usage"
#       ./exp.sh "llama-3.1-8b" "local" "understand"

# 기본값 설정
DEFAULT_MODEL_NAME="gpt-4o"
DEFAULT_MODEL_TYPE="api"
DEFAULT_TASK_TYPES="understand usage"

# 인자 파싱
MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}
MODEL_TYPE=${2:-$DEFAULT_MODEL_TYPE}
TASK_TYPES=${3:-$DEFAULT_TASK_TYPES}

echo "🚀 모델 실험 시작..."
echo "📝 실험 설정:"
echo "   - 모델 이름: $MODEL_NAME"
echo "   - 모델 타입: $MODEL_TYPE"
echo "   - 실행 태스크: $TASK_TYPES"
echo ""


# 실험 결과 추적
SUCCESS_COUNT=0
TOTAL_COUNT=0
FAILED_TASKS=()

# 각 태스크 실행
for task in $TASK_TYPES; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo "=== 태스크 실행: $task (모델: $MODEL_NAME) ==="
    
    # 시작 시간 기록
    start_time=$(date +%s)
    
    # Python 스크립트 실행
    uv run infer.py "$MODEL_NAME" "$MODEL_TYPE" "$task"
    exit_code=$?
    
    # 종료 시간 기록
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $task 태스크 완료 (소요시간: ${duration}초)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ $task 태스크 실패 (종료 코드: $exit_code)"
        FAILED_TASKS+=("$task")
    fi
    echo ""
done

# 실험 결과 요약
echo "📊 실험 결과 요약"
echo "=================="
echo "총 실행 태스크: $TOTAL_COUNT"
echo "성공한 태스크: $SUCCESS_COUNT"
echo "실패한 태스크: $((TOTAL_COUNT - SUCCESS_COUNT))"

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "실패한 태스크 목록: ${FAILED_TASKS[*]}"
fi

echo ""
if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo "🎉 모든 실험이 성공적으로 완료되었습니다!"
    exit 0
else
    echo "⚠️  일부 실험이 실패했습니다."
    exit 1
fi