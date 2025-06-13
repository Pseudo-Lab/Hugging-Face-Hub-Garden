import pandas as pd
import boto3
import json
import time
import os
from datasets import Dataset, DatasetDict

# 1. Parquet 파일 읽기
def load_parquet_dataset(train_path, val_path):
    print("Parquet 파일 로드 중...")
    try:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        
        print(f"훈련 데이터 크기: {train_df.shape}")
        print(f"검증 데이터 크기: {val_df.shape}")
        
        # HuggingFace datasets 객체로 변환
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # DatasetDict로 결합
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        return dataset
    except Exception as e:
        print(f"Parquet 파일 로드 실패: {e}")
        return None

# 2. Amazon Bedrock 클라이언트 설정
def init_bedrock_client(region_name="us-east-1"):
    print("Amazon Bedrock 클라이언트 초기화 중...")
    try:
        session = boto3.Session(region_name=region_name)
        bedrock_client = session.client(service_name="bedrock-runtime")
        return bedrock_client
    except Exception as e:
        print(f"Bedrock 클라이언트 초기화 실패: {e}")
        return None

# 3. Bedrock Claude 모델을 사용한 번역 함수
def translate_with_bedrock(client, texts, batch_size=10, max_retries=3):
    """
    Amazon Bedrock의 Claude 모델을 사용하여 텍스트를 한국어로 번역
    """
    if not client:
        print("Bedrock 클라이언트가 초기화되지 않았습니다.")
        return None
    
    translations = []
    
    # 배치 단위로 처리
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i+batch_size, len(texts))]
        print(f"번역 중: {i}/{len(texts)} - {min(i+batch_size, len(texts))}")
        
        batch_translations = []
        for text in batch:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Claude 모델에 번역 요청
                    prompt = f"""Human: 다음 영어 텍스트를 한국어로 번역하세요. 자연스러운 한국어 문장 하나만 작성하고 다른 설명이나 선택지는 제공하지 마세요.
                    번역 결과에 쌍따옴표를 포함하지 마세요.

                    영어 문장: "{text}"
                    
                    한국어 번역:

                    Assistant:"""
                    
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1000,
                        "temperature": 0.1,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                    
                    response = client.invoke_model(
                        modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                        body=body
                    )
                    
                    response_body = json.loads(response.get('body').read())
                    translation = response_body.get('content')[0].get('text').strip()
                    
                    batch_translations.append(translation)
                    # 요청 한도를 넘지 않도록 잠시 대기
                    break
                    
                except Exception as e:
                    print(f"번역 오류 (재시도 {retry_count+1}/{max_retries}): {e}")
                    retry_count += 1
                    time.sleep(1)  # 재시도 전 대기
            
            if retry_count == max_retries:
                print(f"최대 재시도 횟수 초과, 원본 텍스트 유지: {text[:30]}...")
                batch_translations.append("번역 실패")
        
        translations.extend(batch_translations)
    
    return translations

# 4. 체크포인트 기능이 있는 데이터셋 번역 함수
def translate_dataset_with_checkpoint(dataset, bedrock_client, output_dir="spider_data", checkpoint_interval=50):
    """
    Spider 데이터셋의 question 필드를 번역하고 주기적으로 체크포인트 저장
    """
    print("데이터셋 번역 시작...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 이전 체크포인트 확인
    checkpoint_file = os.path.join(output_dir, "translation_checkpoint.json")
    
    start_index = 0
    processed_results = []
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            start_index = checkpoint_data["next_index"]
            processed_results = checkpoint_data["results"]
            print(f"체크포인트 발견: {start_index}번째 샘플부터 계속 진행합니다.")
    
    # 번역 대상 선택
    total_samples = len(dataset["train"])
    print(f"전체 샘플 수: {total_samples}, 시작 인덱스: {start_index}")
    
    # 이미 모두 처리된 경우
    if start_index >= total_samples:
        print("이미 모든 샘플이 처리되었습니다.")
        return pd.DataFrame(processed_results)
    
    # 남은 샘플 처리
    for i in range(start_index, total_samples, checkpoint_interval):
        end_idx = min(i + checkpoint_interval, total_samples)
        print(f"=== 처리 중: {i}~{end_idx-1} / {total_samples} ===")
        
        # 현재 배치 선택
        batch_samples = dataset["train"].select(range(i, end_idx))
        questions = batch_samples["question"]
        
        # 번역 실행
        translated_questions = translate_with_bedrock(bedrock_client, questions)
        
        # 결과 저장
        for j, (original, translated) in enumerate(zip(questions, translated_questions)):
            idx = i + j
            item = batch_samples[j]
            item_dict = dict(item)
            item_dict["question_ko"] = translated
            processed_results.append(item_dict)
        
        # 체크포인트 저장
        checkpoint_data = {
            "next_index": end_idx,
            "results": processed_results
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        print(f"체크포인트 저장됨 ({end_idx}/{total_samples})")
        
        # 중간 결과 CSV로도 저장
        temp_df = pd.DataFrame(processed_results)
        temp_df.to_csv(os.path.join(output_dir, f"translated_spider_partial_{end_idx}.csv"), index=False)
    
    # 최종 결과
    final_df = pd.DataFrame(processed_results)
    return final_df

# 5. 메인 실행 함수
def main():
    # 파일 경로 설정
    train_path = "spider/spider/train-00000-of-00001.parquet"
    val_path = "spider/spider/validation-00000-of-00001.parquet"
    
    # 출력 디렉토리
    output_dir = "spider_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터셋 로드
    spider_dataset = load_parquet_dataset(train_path, val_path)
    if spider_dataset is None:
        print("데이터셋 로드에 실패했습니다.")
        return
    
    # Bedrock 클라이언트 초기화
    bedrock_client = init_bedrock_client()
    if bedrock_client is None:
        print("Bedrock 클라이언트 초기화에 실패했습니다.")
        return
    
    # 전체 데이터셋 번역 (체크포인트 기능 포함)
    translated_df = translate_dataset_with_checkpoint(spider_dataset, bedrock_client, output_dir)
    
    # 결과 저장
    final_output_path = os.path.join(output_dir, "translated_spider_full.csv")
    translated_df.to_csv(final_output_path, index=False)
    print(f"전체 번역 결과가 {final_output_path}에 저장되었습니다.")
    
    # 결과 요약
    print("\n번역 완료 통계:")
    print(f"총 번역된 샘플 수: {len(translated_df)}")
    success_count = len(translated_df) - translated_df["question_ko"].value_counts().get("번역 실패", 0)
    print(f"성공적으로 번역된 샘플 수: {success_count}")
    print(f"번역 실패 샘플 수: {len(translated_df) - success_count}")

# 실행
if __name__ == "__main__":
    main()
