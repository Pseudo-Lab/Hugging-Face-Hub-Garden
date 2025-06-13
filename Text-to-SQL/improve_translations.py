import pandas as pd
import boto3
import json
import time
import os
import signal
import sys
from datasets import Dataset, DatasetDict
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import google.generativeai as genai  # Gemini API

# 중간 저장 및 상태 관리를 위한 글로벌 변수
checkpoint_file = "spider_data/translation_checkpoint.json"
processed_indices = set()  # 이미 처리된 인덱스 추적

# 1. Parquet 파일 읽기 함수 (기존과 동일)
def load_validation_dataset(val_path, train_path=None):
    print("Validation 데이터셋 로드 중...")
    try:
        val_df = pd.read_parquet(val_path)
        print(f"검증 데이터 크기: {val_df.shape}")
        
        # HuggingFace datasets 객체로 변환
        val_dataset = Dataset.from_pandas(val_df)
        
        # DatasetDict로 변환 (API 일관성 유지)
        dataset = DatasetDict({
            "validation": val_dataset
        })
        
        return dataset
    except Exception as e:
        print(f"Parquet 파일 로드 실패: {e}")
        return None

# 2. Amazon Bedrock 및 Gemini 클라이언트 설정
def init_bedrock_client(region_name="ap-northeast-2"):
    print("Amazon Bedrock 클라이언트 초기화 중...")
    try:
        session = boto3.Session(region_name=region_name)
        bedrock_client = session.client(service_name="bedrock-runtime")
        return bedrock_client
    except Exception as e:
        print(f"Bedrock 클라이언트 초기화 실패: {e}")
        return None

def init_gemini_client(api_key):
    """API 키를 통한 Gemini 클라이언트 초기화"""
    print("Gemini 클라이언트 초기화 중...")
    try:
        genai.configure(api_key=api_key)
        return genai
    except Exception as e:
        print(f"Gemini 클라이언트 초기화 실패: {e}")
        return None

# 체크포인트 저장 및 로드 함수
def save_checkpoint(awkward_indices, fixed_translations=None):
    """현재 작업 상태를 체크포인트로 저장"""
    global processed_indices
    
    data = {
        'awkward_indices': list(awkward_indices),
        'processed_indices': list(processed_indices),
        'fixed_translations': fixed_translations if fixed_translations else []
    }
    
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"체크포인트 저장 완료: {checkpoint_file}")
    except Exception as e:
        print(f"체크포인트 저장 실패: {e}")

def load_checkpoint():
    """저장된 체크포인트 로드"""
    if not os.path.exists(checkpoint_file):
        return None, None
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        awkward_indices = set(data.get('awkward_indices', []))
        fixed_translations = data.get('fixed_translations', [])
        
        # 처리된 인덱스 업데이트
        global processed_indices
        processed_indices = set(data.get('processed_indices', []))
        
        print(f"체크포인트 로드 완료: {len(awkward_indices)}개 어색한 번역, {len(fixed_translations)}개 수정된 번역")
        return awkward_indices, fixed_translations
    except Exception as e:
        print(f"체크포인트 로드 실패: {e}")
        return None, None

# 신호 처리 설정 (Ctrl+C 등의 중단 시그널 처리)
def setup_signal_handler(awkward_indices, fixed_translations):
    def signal_handler(sig, frame):
        print("\n\n프로그램 중단! 현재까지의 결과 저장 중...")
        save_checkpoint(awkward_indices, fixed_translations)
        print("작업을 중단하고 현재까지의 결과를 저장했습니다.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# 3. Gemini를 사용한 어색한 번역 탐지 함수
def detect_awkward_translations_with_gemini(translated_df, gemini_module, batch_size=100, max_chunks=None):
    """Gemini를 사용하여 어색한 번역 탐지"""
    global processed_indices
    awkward_indices = set()  # 중복 방지를 위해 set 사용
    
    # 이전 체크포인트에서 awkward_indices 로드
    loaded_indices, _ = load_checkpoint()
    if loaded_indices:
        continue_previous = input("이전에 저장된 어색한 번역 목록이 있습니다. 계속하시겠습니까? (y/n): ")
        if continue_previous.lower() == 'y':
            awkward_indices = loaded_indices
            print(f"이전에 식별된 {len(awkward_indices)}개 어색한 번역을 로드했습니다.")
            return list(awkward_indices)
    
    # 체크포인트 간격 설정 (배치 수 기준)
    checkpoint_interval = 5  # 5개 배치마다 체크포인트 저장
    
    # 성능을 위해 배치 처리
    total_rows = len(translated_df)
    chunks = list(range(0, total_rows, batch_size))
    
    if max_chunks and len(chunks) > max_chunks:
        print(f"분석할 청크를 {max_chunks}개로 제한합니다.")
        chunks = chunks[:max_chunks]
    
    # 신호 핸들러 설정
    setup_signal_handler(awkward_indices, [])
    
    # Gemini 모델 초기화 (2.0 Pro 모델 사용)
    model = gemini_module.GenerativeModel('gemini-2.0-flash')
    
    for chunk_idx, start_idx in enumerate(chunks):
        end_idx = min(start_idx + batch_size, total_rows)
        print(f"번역 분석 중: {start_idx}~{end_idx-1} / {total_rows} (청크 {chunk_idx+1}/{len(chunks)})")
        
        # 현재 청크 추출
        chunk_df = translated_df.iloc[start_idx:end_idx]
        
        # Gemini에 보낼 데이터 준비
        # 데이터를 표 형식으로 변환
        data_for_analysis = "다음은 영어에서 한국어로 번역된 텍스트 목록입니다. 각 행에는 인덱스, 원문, 번역이 포함되어 있습니다.\n\n"
        data_for_analysis += "인덱스 | 원문(영어) | 번역(한국어)\n"
        data_for_analysis += "--- | --- | ---\n"
        
        for i, row in chunk_df.iterrows():
            idx = start_idx + (i - chunk_df.index[0])
            data_for_analysis += f"{idx} | {row['question']} | {row.get('question_ko', '')}\n"
        
        # Gemini 프롬프트
        prompt = f"""당신은 영어-한국어 번역 전문가입니다. 다음 CSV 데이터에서 어색한 한국어 번역을 찾아주세요:

{data_for_analysis}

구체적으로:
1. "~를 표시합니다", "~를 출력합니다", "~를 보여줍니다", "~를 나열합니다"와 같은 설명형/명령형 종결로 된 번역은 어색합니다.
2. 대신 "~를 알려주세요", "~를 보여주세요", "~는 무엇인가요?"와 같은 요청/질문 형식이어야 합니다.
3. 번역이 부자연스럽거나 문법적으로 잘못된 경우도 포함해주세요.

결과는 JSON 형식으로 다음과 같이 반환해주세요:
```json
{{
  "awkward_translations": [
    {{
      "index": 행 인덱스,
      "original": "원본 영어 텍스트",
      "translation": "어색한 한국어 번역",
      "reason": "어색한 이유 설명"
    }},
    ...
  ]
}}
```

번역이 자연스럽고 어색함이 없다면 빈 배열을 반환해주세요. 반드시 JSON 형식으로만 응답해주세요.
"""
        
        try:
            # 새로운 API 구조에 맞게 매개변수 전달
            generation_config = {
                "temperature": 0.1,
                "max_output_tokens": 8192,
                "top_p": 1.0
            }
            
            # Gemini 모델 호출 - 매개변수 설정
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_text = response.text
            
            # JSON 파싱
            try:
                # 응답에서 JSON 부분 추출 (코드 블록 안에 있을 수 있음)
                json_start = response_text.find("{")
                json_end = response_text.rfind("}")
                
                if json_start >= 0 and json_end >= 0:
                    json_str = response_text[json_start:json_end+1]
                    result = json.loads(json_str)
                    
                    if "awkward_translations" in result:
                        # 어색한 번역 인덱스 추가
                        for item in result["awkward_translations"]:
                            idx = item.get("index")
                            if isinstance(idx, int) and 0 <= idx < total_rows:
                                awkward_indices.add(idx)
                                print(f"어색한 번역 감지 (인덱스 {idx}): {item.get('translation')}")
                                print(f"이유: {item.get('reason')}")
                else:
                    print("Gemini 응답에서 JSON을 찾을 수 없습니다:", response_text[:200])
            
            except json.JSONDecodeError:
                print("Gemini 응답을 JSON으로 파싱할 수 없습니다:", response_text[:200])
        
        except Exception as e:
            print(f"Gemini API 오류: {e}")
            # 오류 발생 시 현재까지의 결과 저장
            save_checkpoint(awkward_indices)
            print("오류 발생으로 인해 지금까지의 결과를 저장했습니다.")
            # 잠시 대기 후 계속 진행
            time.sleep(5)
            continue
        
        # API 한도를 넘지 않도록 잠시 대기
        time.sleep(1)
        
        # 일정 주기로 체크포인트 저장
        if (chunk_idx + 1) % checkpoint_interval == 0 or chunk_idx == len(chunks) - 1:
            save_checkpoint(awkward_indices)
            print(f"체크포인트 저장 완료 (청크 {chunk_idx+1}/{len(chunks)})")
    
    print(f"총 {len(awkward_indices)}개의 어색한 번역 감지")
    return list(awkward_indices)

# 4. Bedrock Claude 모델을 사용한 번역 함수 - 어색한 번역 수정용
def retranslate_with_bedrock(client, samples, batch_size=10, max_retries=3):
    """
    어색한 번역이 식별된 샘플만 재번역
    """
    global processed_indices
    
    if not client:
        print("Bedrock 클라이언트가 초기화되지 않았습니다.")
        return None
    
    translations = []
    
    # 체크포인트에서 기존 번역 결과 로드
    _, saved_translations = load_checkpoint()
    if saved_translations:
        continue_previous = input("이전에 저장된 번역 결과가 있습니다. 계속하시겠습니까? (y/n): ")
        if continue_previous.lower() == 'y':
            translations = saved_translations
            print(f"이전에 완료된 {len(translations)}개 번역을 로드했습니다.")
            
            # 이미 처리된 인덱스 업데이트
            processed_indices = {item['original_idx'] for item in translations}
            
            # 처리할 필요 없는 샘플 필터링
            samples = [s for s in samples if s.get('original_idx') not in processed_indices]
            print(f"남은 처리 대상: {len(samples)}개 샘플")
    
    # 신호 핸들러 설정 (중간에 중단 시 저장)
    setup_signal_handler([], translations)
    
    # 체크포인트 간격 설정
    checkpoint_interval = 5  # 5개 배치마다 저장
    
    # 배치 단위로 처리
    for i in range(0, len(samples), batch_size):
        batch = samples[i:min(i+batch_size, len(samples))]
        print(f"재번역 중: {i}/{len(samples)} - {min(i+batch_size, len(samples))}")
        
        batch_translations = []
        for sample in batch:
            # 이미 처리된 인덱스는 건너뛰기
            if sample.get('original_idx') in processed_indices:
                continue
                
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # 컨텍스트 정보를 포함한 번역 요청 준비
                    db_id = sample.get("db_id", "")
                    query = sample.get("query", "")
                    question = sample.get("question", "")
                    
                    # Claude 모델에 번역 요청 (컨텍스트 포함 + 개선된 프롬프트)
                    prompt = f"""Human: 다음 영어 질문을 한국어로 번역해주세요. 다음 규칙을 엄격히 따라주세요:

1. 항상 질문이나 요청 형식으로 번역하세요.
2. "~를 표시합니다", "~를 출력합니다", "~를 보여줍니다" 같은 설명형/명령형 종결로 번역하지 마세요.
3. 대신 "~를 알려주세요", "~를 보여주세요", "~는 무엇인가요?" 같은 요청/질문 형식으로 번역하세요.
4. 번역문만 작성하고 쌍따옴표나 다른 설명 부호는 사용하지 마세요.
5. SQL 쿼리와 영어 질문이 일치하지 않을 수 있으니, SQL 쿼리의 내용을 우선적으로 참고하세요.
6. 데이터베이스 컨텍스트에 맞는 적절한 용어를 사용하세요.

데이터베이스 ID: {db_id}
SQL 쿼리: {query}
영어 질문: {question}

잘못된 예시:
- ❌ "수용량이 5,000명에서 10,000명 사이인 모든 경기장의 위치와 이름을 표시합니다."
- ⭕ "수용량이 5,000명에서 10,000명 사이인 모든 경기장의 위치와 이름을 보여주세요."

한국어 번역:
"""

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
                        modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
                        body=body
                    )
                    
                    response_body = json.loads(response.get('body').read())
                    translation = response_body.get('content')[0].get('text').strip()
                    
                    # 수정된 번역 및 원본 인덱스 저장
                    original_idx = sample.get('original_idx', 0)
                    batch_translations.append({
                        'original_idx': original_idx,
                        'new_translation': translation
                    })
                    
                    # 처리된 인덱스 추가
                    processed_indices.add(original_idx)
                    
                    # 요청 한도를 넘지 않도록 잠시 대기
                    time.sleep(0.5)
                    break
                    
                except Exception as e:
                    print(f"번역 오류 (재시도 {retry_count+1}/{max_retries}): {e}")
                    retry_count += 1
                    time.sleep(2)
            
            if retry_count == max_retries:
                print(f"최대 재시도 횟수 초과: {question[:30]}...")
                original_idx = sample.get('original_idx', 0)
                batch_translations.append({
                    'original_idx': original_idx,
                    'new_translation': "번역 실패"
                })
                processed_indices.add(original_idx)
        
        translations.extend(batch_translations)
        
        # 일정 주기로 체크포인트 저장
        batch_number = i // batch_size
        if (batch_number + 1) % checkpoint_interval == 0 or i + batch_size >= len(samples):
            save_checkpoint([], translations)
            print(f"번역 체크포인트 저장 완료 ({len(translations)}/{len(samples) + len(translations)})")
    
    return translations

# 5. 전체 워크플로우 함수 - Gemini로 어색한 번역을 찾고 Claude로 재번역
def improve_translations_with_gemini(input_file, api_key, output_dir="spider_data", batch_size=100, max_chunks=None):
    """
    1. 번역된 파일 로드
    2. Gemini로 어색한 번역 식별
    3. Claude로 어색한 번역만 재번역
    4. 결과 저장
    """
    print(f"입력 파일 {input_file} 로드 중...")
    
    try:
        # 중간 결과 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 번역된 파일 로드
        translated_df = pd.read_csv(input_file)
        
        # 초기 통계
        total_rows = len(translated_df)
        print(f"총 {total_rows}개의 번역된 행을 로드했습니다.")
        
        # Gemini 초기화
        gemini_module = init_gemini_client(api_key)
        if not gemini_module:
            return None
        
        # Gemini로 어색한 번역 감지
        print("\nGemini로 어색한 번역 분석 중...")
        awkward_indices = detect_awkward_translations_with_gemini(
            translated_df,
            gemini_module,
            batch_size=batch_size,
            max_chunks=max_chunks
        )
        
        if not awkward_indices:
            print("어색한 번역이 감지되지 않았습니다. 프로그램을 종료합니다.")
            return translated_df
        
        # 어색한 번역의 비율 계산
        awkward_percentage = len(awkward_indices) / total_rows * 100
        print(f"총 {len(awkward_indices)}개의 번역이 어색합니다 ({awkward_percentage:.2f}%).")
        
        # 어색한 번역 목록 저장 (CSV 형식으로)
        awkward_df = pd.DataFrame({'awkward_index': awkward_indices})
        awkward_file = os.path.join(output_dir, "awkward_indices.csv")
        awkward_df.to_csv(awkward_file, index=False)
        print(f"어색한 번역 인덱스 목록 저장 완료: {awkward_file}")
        
        # 계속 진행할지 확인
        proceed = input("어색한 번역을 수정하시겠습니까? (y/n): ")
        if proceed.lower() != 'y':
            print("프로그램을 종료합니다.")
            return translated_df
        
        # Bedrock 클라이언트 초기화
        bedrock_client = init_bedrock_client()
        if bedrock_client is None:
            print("Bedrock 클라이언트 초기화에 실패했습니다.")
            return translated_df
        
        # 어색한 번역 재번역을 위한 샘플 준비
        samples_to_fix = []
        for idx in awkward_indices:
            if 0 <= idx < total_rows:
                row = translated_df.iloc[idx]
                sample = {
                    'db_id': row.get('db_id', ''),
                    'query': row.get('query', ''),
                    'question': row.get('question', ''),
                    'original_idx': idx
                }
                samples_to_fix.append(sample)
        
        print(f"\n{len(samples_to_fix)}개의 번역 개선 예정...")
        
        # 부분 결과를 저장할 임시 파일 경로
        temp_output_path = os.path.join(output_dir, "improved_train_translations_partial.csv")
        
        # Claude로 어색한 번역만 재번역
        if samples_to_fix:
            fixed_translations = retranslate_with_bedrock(bedrock_client, samples_to_fix)
            
            # 번역 결과를 원본 데이터프레임에 반영하고 부분 결과 저장
            for batch_idx, item in enumerate(fixed_translations):
                idx = item['original_idx']
                new_translation = item['new_translation']
                if idx < total_rows:
                    print(f"인덱스 {idx} 번역 변경:")
                    print(f"  이전: {translated_df.iloc[idx].get('question_ko', '')}")
                    print(f"  이후: {new_translation}")
                    translated_df.loc[idx, 'question_ko'] = new_translation
                
                # 10개 항목마다 부분 결과 저장
                if (batch_idx + 1) % 10 == 0:
                    translated_df.to_csv(temp_output_path, index=False)
                    print(f"부분 결과 저장 완료 ({batch_idx + 1}/{len(fixed_translations)})")
        
        # 최종 결과 저장
        output_path = os.path.join(output_dir, "improved_train_translations.csv")
        translated_df.to_csv(output_path, index=False)
        print(f"수정된 번역 결과 저장 완료: {output_path}")
        
        # 임시 파일 삭제
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except:
                pass
        
        # 체크포인트 파일 삭제
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                print(f"체크포인트 파일 삭제 완료: {checkpoint_file}")
            except:
                pass
        
        # 결과 통계
        print("\n=== 번역 개선 통계 ===")
        print(f"총 행 수: {total_rows}")
        print(f"수정된 행 수: {len(awkward_indices)} ({awkward_percentage:.2f}%)")
        
        return translated_df
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 중단되었습니다.")
        save_checkpoint(awkward_indices if 'awkward_indices' in locals() else [], 
                       fixed_translations if 'fixed_translations' in locals() else [])
        print("현재까지의 결과가 저장되었습니다.")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        if 'awkward_indices' in locals() and 'fixed_translations' in locals():
            save_checkpoint(awkward_indices, fixed_translations)
            print("오류 발생 시점까지의 결과가 저장되었습니다.")
        return None

# 6. 메인 실행 함수
def main():
    # 설정
    input_file = "translated_spider_train_full.csv"
    output_dir = "spider_data"
    
    # Gemini API 키 (환경 변수 또는 입력으로 받을 수 있음)
    gemini_api_key = os.environ.get("GEMINI_API_KEY") 
    if not gemini_api_key:
        gemini_api_key = input("Gemini API 키를 입력하세요: ")
    
    # 이미 번역된 파일이 있는지 확인
    if not os.path.exists(input_file):
        print(f"번역 파일 {input_file}이 존재하지 않습니다.")
        # 필요하다면 여기서 전체 번역 프로세스 실행
        return
    
    # 체크포인트 파일이 있는지 확인
    if os.path.exists(checkpoint_file):
        print(f"이전 작업의 체크포인트 파일이 발견되었습니다: {checkpoint_file}")
        continue_from_checkpoint = input("이전 작업을 계속하시겠습니까? (y/n): ")
        if continue_from_checkpoint.lower() != 'y':
            print("체크포인트 파일을 삭제하고 처음부터 시작합니다.")
            try:
                os.remove(checkpoint_file)
            except:
                print("체크포인트 파일 삭제 실패")
    
    # Gemini와 Bedrock을 사용한 번역 개선 워크플로우 실행
    improve_translations_with_gemini(input_file, gemini_api_key, output_dir)

# 실행
if __name__ == "__main__":
    main()