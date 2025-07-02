import re
import json
from docx import Document

def parse_docx_to_json(doc_path, output_path):
    """
    DOCX 파일을 파싱하여 JSON 형식으로 변환
    
    Args:
        doc_path (str): 입력 DOCX 파일 경로
        output_path (str): 출력 JSON 파일 경로
    """
    
    # 1. 문서 로드
    doc = Document(doc_path)
    
    # 2. 문단 읽기 및 스타일 처리
    paragraphs = []
    for para in doc.paragraphs:
        run_texts = []
        for run in para.runs:
            text = run.text.replace('\xad', '-')
            if run.bold:
                text = f"**{text}**"
            if run.underline:
                text = f"__{text}__"
            run_texts.append(text)
        
        styled_text = ''.join(run_texts).strip()
        plain_text = ''.join(r.text for r in para.runs).strip()
        
        if plain_text:  # 빈 문단 제외
            paragraphs.append({
                "text": styled_text,
                "plain": plain_text
            })
    
    # 3. 문제별 파싱
    problems = []
    current_problem = None
    state = "idle"
    
    for para in paragraphs:
        plain = para['plain']
        styled = para['text']
        
        # 문제 번호 감지 (예: 1. 2. 3.)
        if re.match(r"^\d+\.", plain):
            if current_problem:
                problems.append(current_problem)
            
            current_problem = {
                "number": int(plain.split('.')[0]),
                "exam_name": "모의고사_2학년_2024년_변형문제",
                "source": "EBSI",
                "year": "2024",
                "month": "3",
                "grade": "고2",
                "question_type": "",
                "question": plain.split('.', 1)[1].strip(),
                "passage": "",
                "options": "",
                "answer": "",
                "vocab": "",
            }
            state = "await_meta"
            
        elif state == "await_meta":
            # 메타 정보 추출 [문제유형]
            meta_match = re.search(r"\[(.*?)\]", plain)
            if meta_match:
                current_problem["question_type"] = meta_match.group(1)
            state = "await_passage_trigger"
            
        elif state == "await_passage_trigger":
            if "# Passage Start" in plain:
                state = "passage"
                
        elif state == "passage":
            if "# Options Start" in plain or "# Options  Start" in plain:
                state = "options"
            elif "# Vocab Start" in plain:
                state = "vocab"
            else:
                current_problem["passage"] += (styled + "\n").strip()
                
        elif state == "options":
            if "# answer" in plain:
                problems.append(current_problem)
                current_problem = None
                state = "idle"
            else:
                current_problem["options"] += styled + "\n"
                
        elif state == "vocab":
            if "# Options Start" in plain or "# Options  Start" in plain:
                state = "options"
            else:
                current_problem["vocab"] += (styled + "\n").strip()
    
    # 마지막 문제 추가
    if current_problem:
        problems.append(current_problem)
    
    # 4. 정답 추출
    answer_mode = False
    answer_string = ""
    
    for para in paragraphs:
        plain = para["plain"]
        
        if "# answer" in plain.lower():
            answer_mode = True
            continue
        
        if answer_mode:
            answer_string += plain + " "
    
    # 정답 정리 및 분할
    answer_string = answer_string.strip().replace("\n", "").replace(" ", "")
    answers = [s for s in answer_string.split(",") if s]
    
    # 정답 검증
    assert len(answers) == len(problems), f"❌ 정답 수({len(answers)})와 문제 수({len(problems)})가 다릅니다!"
    
    # 정답 할당
    for i, problem in enumerate(problems):
        problem["answer"] = answers[i]
    
    # 5. 텍스트 정리
    for problem in problems:
        problem["passage"] = problem.get("passage", "").strip()
        problem["options"] = problem.get("options", "").strip()
        problem["vocab"] = problem.get("vocab", "").strip()
    
    # 6. 데이터 정리
    data = problems
    
    # 7. JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 파싱 완료: {len(problems)}개 문제")
    print(f"✅ JSON 저장 완료: {output_path}")
    
    return data

# 사용 예시
if __name__ == "__main__":
    doc_path = "input_file.docx"
    json_output_path = "output_file.json"
    
    parsed_data = parse_docx_to_json(doc_path, json_output_path)