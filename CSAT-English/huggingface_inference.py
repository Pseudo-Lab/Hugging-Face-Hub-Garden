from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# 모델 로드
model = AutoPeftModelForCausalLM.from_pretrained(
    "huggingface-KREW/Qwen3-8B-Korean-Highschool-English-Exam",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 영어 지문 예시
passage = """
If the brain has already stored someone's face and name, why do we still end up 
remembering one and not the other? This is because the brain has a two-tier memory 
system at work when it comes to retrieving memories, giving rise to a common yet 
infuriating sensation: recognising someone but not being able to remember how or why, 
or what their name is.
"""

# 문제 생성 함수
def generate_question(passage, question_type):
    messages = [
        {
            "role": "user",
            "content": f"다음 영어 지문을 {question_type} 문제로 만들어주세요.\n\n지문:\n{passage}\n\n"
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"].to("cuda"), 
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# 제목 추론 문제 생성
result = generate_question(passage, "제목 추론")
print(result)
