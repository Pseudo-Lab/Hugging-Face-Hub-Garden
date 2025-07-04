from prompt import AUGMENT_PROMPT
from litellm import batch_completion
import json
import os
os.environ["OPENAI_API_KEY"] =""
original_data_path = "meme_sample_with_questions.jsonl"


if __name__ == "__main__":
    question_list = []
    with open(original_data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                question_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    messages = []
    for question in question_list:
        title = question['title']
        content = question['content']
        example_questions = question['question']
        example_answers = question['answer']
        
        chat = AUGMENT_PROMPT.format(
            word=title,
            meaning=content,
            example_question=example_questions,
            example_answer=example_answers
        )
        messages.append([{"role": "user", "content": chat}])
    resps = batch_completion(
        model="gpt-4o",
        messages=messages,
        max_tokens=512,
    )
    answers = [i.choices[0].message.content for i in resps]
    
    # title과 answer를 포함하는 딕셔너리 리스트 생성
    result_data = []
    for question, answer in zip(question_list, answers):
        result_data.append({
            "title": question['title'],
            "question": answer
        })
    
    with open("understand_questions.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
