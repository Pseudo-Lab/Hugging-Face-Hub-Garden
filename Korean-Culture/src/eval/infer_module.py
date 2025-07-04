from litellm import batch_completion
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
os.environ["GEMINI_API_KEY"] = "your-api-key"


class InferenceModule:
    def __init__(self, model_name, model_type, prefix):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.prefix = prefix
        self.init_model()
        
    def init_model(self):
        if self.model_type == "api":
            print(f"Using API model: {self.model_name}")
            
        elif self.model_type == "local":
            print(f"Using local model: {self.model_name}")
            try:
                # Load local model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                  trust_remote_code=True).to("cuda")
                print(f"Successfully loaded local model: {self.model_name}")
            except KeyError as e:
                print(f"loading model via unsloth: {self.model_name}")
                from unsloth import FastLanguageModel
                # Fallback to unsloth for loading local models
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(self.model_name)
    
    def tokenize_for_local_models(self, input_text):
        assert self.tokenizer is not None, "Tokenizer is not initialized."
        assert self.model_type == "local", "This method is only for local models."
        
        tokenizer = self.tokenizer
        
        messages = [{"role": "user", "content": f"{self.prefix}: {input_text}"}]
        
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True,tokenize=False,enable_thinking=False)
        
        
    def infer(self, input_texts):
        if self.model_type == "api":
            try:
                # Use API for inference
                responses = batch_completion(
                    model=self.model_name,
                    messages=[[{"role": "user", "content": f"{self.prefix}: {text}"}] for text in input_texts],
                    max_tokens=512,
                    temperature=0.7
                )
                return [response['choices'][0]['message']['content'] for response in responses]
            except Exception as e:
                print(f"API inference error: {e}")
                return ["Error in API inference"] * len(input_texts)
        
        elif self.model_type == "local":
            tokenizer = self.tokenizer
            res_list = []
            
            try:
                # Use local model for inference
                input_texts = [self.tokenize_for_local_models(text) for text in input_texts]

                for text in tqdm(input_texts):
                    inputs = tokenizer(text, return_tensors="pt").to("cuda")
                    
                    with torch.no_grad(): 
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            temperature=0.7, 
                            top_p=0.8, 
                            top_k=20,
                            min_p=0,
                            repetition_penalty=1.15,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response_ids = outputs[0][len(inputs.input_ids[0]):]
                    response = tokenizer.decode(response_ids, skip_special_tokens=True)
                    res_list.append(response.strip())
                    
                    del inputs, outputs
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Local inference error: {e}")
                return ["Error in local inference"] * len(input_texts)
                
            return res_list
    