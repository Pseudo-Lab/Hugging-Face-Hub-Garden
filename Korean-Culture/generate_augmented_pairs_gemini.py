import os
import time
import pandas as pd
from google import genai
from google.genai import types

# Gemini API 설정
# 시스템 환경 변수에 GEMINI_API_KEY를 설정해야 합니다.
# 예: export GEMINI_API_KEY='your-api-key'
api_key = ''

client = genai.Client(api_key=api_key)
model_name = "gemini-2.5-flash"

# CSV 파일에서 신조어 데이터 읽기
csv_path = "ko_culture_samples.csv"
df = pd.read_csv(csv_path)

# 결과 저장 폴더
output_dir = "augmented_jsons_gemini"
os.makedirs(output_dir, exist_ok=True)

for idx, row in df.iterrows():
    title = row["title"]
    content = row["content"]
    example_q = row["example_q"]
    example_a = row["example_a"]

    # 이미 생성된 신조어는 건너뛰기
    already_exists = any(fname.startswith(f"{title}_") for fname in os.listdir(output_dir))
    if already_exists:
        print(f"이미 생성됨: {title} → 건너뜀")
        continue

    prompt = f"""
    ## 작업 요구사항:
    1. **카톡 대화 스타일 반영**:
       - 동의/반대, 놀림/칭찬, 공감/위로 등 다양한 반응을 표현하세요
       - 친구, 동기, 연인 등 친한 사람들끼리의 일상 대화 느낌을 살리세요
       - 끊어서 보내는 메시지, 맞춤법 무시 등을 자연스럽게 포함하세요
       - "ㅋㅋㅋ", "ㅠㅠ", "ㄹㅇ", "ㅇㅇ" 등의 온라인 축약어를 적절히 활용하세요
       - 이모티콘/이모지 활용은 하지 않습니다

    2. **MZ세대 언어 특성 강화**:
       - 문장 끝을 흐리거나 생략하는 경향 (예: "그래서 난 그냥...")
       - 영어 단어를 한글로 음차하는 표현 (예: "완전 레전드", "처음 봤을 때 빡쳤음")
       - 과장된 표현과 강조 (예: "진짜 미쳤다...", "대박 실화냐")
       - 기존 표현의 변형

    3. **다양한 대화 상황 포함**:
       - 친구들과의 단톡방
       - 일대일 카톡 대화
       - 온라인 게임 채팅
       - 학과/직장 단체 채팅

    4. **신조어 활용 방식 다양화**:
       - 주어진 신조어를 문장의 다양한 위치(문두, 문중, 문미)에서 활용하세요.
       - 신조어의 다양한 사용 맥락과 뉘앙스가 드러나도록 하세요.
       - 모든 질문이나 답변이 같은 패턴으로 시작하거나 끝나지 않도록 하세요.

    ## 신조어 정보
    신조어 제목: {title}
    신조어 설명: {content}
    예문Q: {example_q}
    예문A: {example_a}

    ## 출력 형식:

    반드시 아래의 JSON 형식으로 출력해주세요. 다른 설명이나 마크다운 형식은 포함하지 마세요.

    {{
      "title": "신조어 제목",
      "description": "신조어에 대한 간략한 설명",
      "conversations": [
        {{
          "context": "대화 상황 (예: 대학생 단톡방, 온라인 게임 채팅, 인스타 댓글)",
          "q": "첫 번째 발화",
          "a": "두 번째 발화"
        }},
        ...
      ]
    }}

    ## 주의사항:
    1. 진짜 SNS나 메신저에서 볼 법한 자연스러운 대화를 생성하세요. 너무 정제된 문장은 피하세요.
    2. 동일한 패턴이 반복되지 않도록 다양한 길이와 형식의 대화를 포함하세요.
    3. 신조어를 억지로 사용하기보다 해당 단어가 자연스럽게 등장할 법한 상황을 설정하세요.
    4. 신조어의 다양한 활용 방식과 뉘앙스가 드러나도록 하되 너무 반복적이지 않게 하세요.
    5. 반드시 싱글턴 대화만 생성해주세요.

    실제 인터넷 커뮤니티나 카톡에서 볼 법한 톤과 분위기가 담긴 대화를 만들어주세요.
    """

    system_prompt = "당신은 한국어 온라인 커뮤니티와 카톡 대화에 자주 등장하는 신조어와 유행어에 정통한 언어 전문가입니다. 주어진 신조어를 활용해 실제 인터넷 커뮤니티와 친구 간 카톡에서 볼 법한 자연스러운 대화 데이터 10개를 생성해주세요."
    full_prompt = f"{system_prompt}\n\n{prompt}"

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )

    # 응답에서 JSON 부분만 추출
    output_text = response.text

    # 파일명: 신조어_타임스탬프.json (중복 방지)
    filename = f"{title}_{int(time.time())}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"Saved: {filepath}")
