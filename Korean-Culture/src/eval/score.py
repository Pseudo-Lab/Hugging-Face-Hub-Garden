import json
from litellm import batch_completion
from prompt import UNDERSTAND_PROMPT, USAGE_PROMPT
from pydantic import BaseModel
from typing import Literal
import os
from datasets import load_dataset

os.environ["OPENAI_API_KEY"] = "sk-"  # Set your OpenAI API key


class ScoreModel(BaseModel):
    score: Literal[1,2,3,4,5]

class UsageModel(BaseModel):
    used: Literal['true', 'false']

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def dump_json(data, file_path):
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def batch_call_llm(chat_messages, scorer_name, response_model, n):
    try:
        responses = batch_completion(
            model=scorer_name,
            messages=chat_messages,
            max_tokens=512,
            temperature=0.7, 
            response_format=response_model,
            n=n
        )
        
        # 모든 메시지와 선택지들을 올바르게 처리
        result = []
        for response in responses:
            # 각 메시지에 대한 n개의 선택지 추출
            choices_content = [choice['message']['content'] for choice in response['choices']]
            result.append(choices_content)
        
        return result
    except Exception as e:
        print(f"Error during LLM call: {e}")
        # 에러 시 기본값 반환
        return [['{"score": 1}'] * n] * len(chat_messages)

def score_answers(scorer_name, model_name, score_mode, response_model=ScoreModel):
    # Load inference results
    infer_list = load_json(f"results/{score_mode}/infer_results/{model_name}_{score_mode}_inference_results.json")
    
    result_list = []
    
    if score_mode == "understand":
        meme_data_list = []
        with open("meme_sample_with_questions.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                meme_data_list.append(json.loads(line))
    
        messages = []
        for meme_dict, qa_dict in zip(meme_data_list, infer_list):
            prompt = UNDERSTAND_PROMPT.format(
                word=meme_dict['title'],
                meaning=meme_dict['content'],
                question=qa_dict['question'],
                response=qa_dict['answer']
            )            
            messages.append(
                [
                    {"role": "user", "content": prompt}
                ]
            )
            # break  # For testing, only process the first item
            
        resps = batch_call_llm(messages, scorer_name, ScoreModel, n=1)
        
        # 각 질문별로 개별 처리
        for i, qa_dict in enumerate(infer_list[:len(messages)]):
            if i < len(resps):
                try:
                    resp_json = json.loads(resps[i][0])  # n=1이므로 첫 번째 응답만 사용
                    score = resp_json['score']
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"JSON 파싱 오류 (질문 {i}): {e}, 응답: {resps[i][0] if i < len(resps) and resps[i] else 'No response'}")
                    score = 1  # 기본 점수
            else:
                score = 1  # 기본 점수
            
            result_list.append({
                'question': qa_dict['question'],
                'answer': qa_dict['answer'],
                'score': score
            })
        
        # 전체 평균 점수 계산
        if result_list:
            total_avg_score = round(sum(item['score'] for item in result_list) / len(result_list), 3)
        else:
            total_avg_score = 0.0
            
        output_dict = {
            "model_name": model_name,
            "average_score": total_avg_score,
            "results": result_list
        }
            
        file_path = f"results/{score_mode}/score_results/{model_name}-{scorer_name}-score_results.json"
        dump_json(output_dict, file_path)
        print(f"Results saved to: {file_path}")
        print(f"Total average score: {total_avg_score}")

    elif score_mode == "usage":
        messages = []
        usage_word_dataset = load_dataset('huggingface-KREW/KoCultre-Descriptions', split='train').filter(lambda x: x['category'] == 'meme')
        word_list_str = ""
        for item in usage_word_dataset:
            word_title = item['title']
            word_content = '.'.join(item['content'].split('.')[:2]) + '.' if len(item['content'].split('.')) > 1 else item['content'] # 첫 두 문장만 사용
            word_list_str += f"{word_title} : ({word_content})\n"
        
        for qa_dict in infer_list:
            prompt = USAGE_PROMPT.format(
                word_list=word_list_str,
                question=qa_dict['question'],
                answer=qa_dict['answer']
            )
            messages.append(
                [
                    {"role": "user", "content": prompt}
                ]
            )
        
        resps = batch_call_llm(messages, scorer_name, UsageModel, n=1)
        
        # 각 질문별로 개별 처리
        for i, qa_dict in enumerate(infer_list[:len(messages)]):
            if i < len(resps):
                try:
                    resp_json = json.loads(resps[i][0])  # n=1이므로 첫 번째 응답만 사용
                    usage_value = resp_json['used'] == 'true'
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"JSON 파싱 오류 (질문 {i}): {e}, 응답: {resps[i][0] if i < len(resps) and resps[i] else 'No response'}")
                    usage_value = False  # 기본값
            else:
                usage_value = False  # 기본값
            
            result_list.append({
                'question': qa_dict['question'],
                'answer': qa_dict['answer'],
                'used': usage_value
            })
        
        # 전체 사용률 계산
        if result_list:
            total_usage_rate = round(sum(item['used'] for item in result_list) / len(result_list), 3)
        else:
            total_usage_rate = 0.0
            
        output_dict = {
            "model_name": model_name,
            "total_usage_rate": total_usage_rate,
            "results": result_list
        }
            
        file_path = f"results/{score_mode}/score_results/{model_name}-{scorer_name}-score_results.json"
        dump_json(output_dict, file_path)
        print(f"Results saved to: {file_path}")
        print(f"Total usage rate: {total_usage_rate}")

if __name__ == "__main__":
    score_answers(scorer_name="gpt-4.1",
                  model_name="EXAONE-3.5-7.8B-Instruct-KoCulture-fulltrain-transformers",
                  score_mode="usage")
    score_answers(scorer_name="gpt-4.1",
                  model_name="kanana-1.5-8b-instruct-2505-KoCulture-fulltrain-transformers",
                  score_mode="usage")
    score_answers(scorer_name="gpt-4.1",
                model_name="Qwen3-8B-KoCulture-fulltrain-transformers",
                score_mode="usage")
    score_answers(scorer_name="gpt-4.1",
            model_name="gpt-4o",
            score_mode="usage")
        
