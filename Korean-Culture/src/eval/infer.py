import argparse
from infer_module import InferenceModule
import json
import os

UNDERSTAND_TEST_SET_PATH = "understand_questions.json"
USAGE_TEST_SET_PATH = "usage_questions.json"
PREFIX = "친구와 채팅을 하고 있다고 가정하고 다음 질문에 밈과 유행어를 활용하여 대답하세요."

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="Name of the model to use for inference")
parser.add_argument("model_type", type=str, choices=["api", "local"], help="Type of model: 'api' or 'local'")
parser.add_argument("task_type", type=str, choices=["understand", "usage"], help="Type of task: 'understand' or 'usage'")



def main():
    args = parser.parse_args()
    
    question_list = []
    if args.task_type == "understand":
        with open(UNDERSTAND_TEST_SET_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                question_list = [datum['question'] for datum in data]
            except Exception as e:
                print(f"Error reading understand test set: {e}")
                return
        
        
    elif args.task_type == "usage":
        with open(USAGE_TEST_SET_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                question_list = data["questions"]
            except Exception as e:
                print(f"Error reading usage test set: {e}")
                return

    
    # Initialize the inference module with the provided model name and type
    inference_module = InferenceModule(model_name=args.model_name, model_type=args.model_type, prefix=PREFIX)
    
    # Run the inference process
    a_list = inference_module.infer(question_list)
    
    res_list = []
    for q, a in zip(question_list, a_list):
        res_list.append({
            "question": q,
            "answer": a
        })
    model_name = args.model_name.split("/")[-1] if args.model_type == "local" else args.model_name
    file_name = f"results/{args.task_type}/infer_results/{model_name}_{args.task_type}_inference_results.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(res_list, f, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    main()


