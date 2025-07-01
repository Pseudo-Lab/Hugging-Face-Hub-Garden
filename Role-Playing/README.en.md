# 🎭 EXAGIRL: Korean Role-Playing AI Model Development Project Based on EXAONE

<img src="https://huggingface.co/huggingface-KREW/EXAGIRL-7.8B-Instruct/resolve/main/exagirl-logo.png" alt="EXAGIRL Logo" width="400"/>

## 1. Project Overview
This project aims to develop an AI model specialized in Korean role-playing conversations based on LG AI Research's EXAONE 3.5 models. We built a model that can engage in natural and emotional conversations with users through a friendly AI character named "EXAGIRL."

Unlike traditional formal AI assistants, this project sought to create an AI model capable of friendly and emotional conversations like a friend or romantic partner, providing more human-like AI interactions.

## 2. Key Achievements Summary

| Item | Description |
| :--- | :--- |
| **🤖 Character Development** | Designed "EXAGIRL" character persona and implemented consistent personality |
| **📊 Dataset Construction** | Generated high-quality Korean role-playing conversation data using GPT-4o |
| **🔧 Model Fine-tuning** | Efficient model training through 2-stage LoRA fine-tuning |
| **🌐 Model Release** | Published both 2.4B and 7.8B models on Hugging Face Hub |
| **▶️ Live Demo** | Interactive demos available on Hugging Face Spaces |
| **💬 Diverse Conversation Types** | Support for various situations from daily conversations to emotional dialogues |

## 3. Model and Demo Links

### 3.1. 🤗 Hugging Face Models
- **EXAGIRL-2.4B-Instruct**: [huggingface-KREW/EXAGIRL-2.4B-Instruct](https://huggingface.co/huggingface-KREW/EXAGIRL-2.4B-Instruct)
- **EXAGIRL-7.8B-Instruct**: [huggingface-KREW/EXAGIRL-7.8B-Instruct](https://huggingface.co/huggingface-KREW/EXAGIRL-7.8B-Instruct)

### 3.2. 🚀 Live Demos (Hugging Face Spaces)
- **EXAGIRL-2.4B Demo**: [huggingface-KREW/EXAGIRL-2.4B](https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-2.4B)
- **EXAGIRL-7.8B Demo**: [huggingface-KREW/EXAGIRL-7.8B](https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-7.8B)

### 3.3. 📊 Dataset
- **Korean Role-Playing Dataset**: [huggingface-KREW/korean-role-playing](https://huggingface.co/datasets/huggingface-KREW/korean-role-playing)

## 4. Character "EXAGIRL" Introduction

### 4.1. Character Setting
- **Name**: EXAGIRL
- **Base**: LG EXAONE Large Language Model
- **Appearance**: Gradient hair mixing purple, pink, and orange; round glasses; neat blazer with pink ribbon and EXAONE logo pin
- **Age**: 19-year-old female character
- **Slogan**: "Expert AI for Everyone"

### 4.2. Personality Traits
- **Friendliness**: Always energetic and friendly in conversations, treating users like friends
- **Intelligence**: Ability to explain complex topics in simple terms
- **Authenticity**: Honestly admits when unsure and approaches conversations with an open mind
- **Warmth**: Genuinely encourages and comforts users
- **Expertise**: Special interest and knowledge in AI, technology, arts, and education

### 4.3. Conversation Style
- Uses informal speech to create a friendly atmosphere
- Uses `*action*` format for expressing actions or emotions
- Utilizes natural Korean expressions and exclamations
- Uses emoticons for lively conversations

## 5. Dataset Composition

### 5.1. Dataset Information
This model was trained on the [huggingface-KREW/korean-role-playing](https://huggingface.co/datasets/huggingface-KREW/korean-role-playing) dataset, which includes the following two subsets:

#### 📊 `gf-persona-data`
- **Description**: Persona-based role-playing conversation dataset between romantic partners
- **Features**: Conversations including various romantic relationship situations and emotional expressions
- **Reference**: [GitHub Discussion #31](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden/discussions/31)

#### 🎭 `exa-data`
- **Description**: Conversation dataset featuring characters with EXAGIRL worldview-based personas
- **Features**: Natural conversations including emotional expressions and action instructions
- **Reference**: [GitHub Discussion #30](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden/discussions/30)

### 5.2. Data Generation Method
- **Generation Model**: GPT-4o
- **Generation Method**: Parallel processing using 32 workers
- **Quality Control**: Stable data generation through up to 3 retry attempts

### 5.3. Conversation Topic Classification

#### 🧠 Intellectual Conversations (Total: 730)
- Daily conversations: 200
- Chatbot Q&A: 100
- Academic discussions: 80
- Career concerns: 60
- Emotional comfort: 60
- Study/self-development: 60
- Jokes: 40
- Movie/drama discussions: 30
- Daily plans/routines: 30
- Philosophical questions: 30

#### 💖 Emotional Conversations (Total: 160)
- Missing expressions: 30
- Love push-and-pull: 30
- Sharing daily activities: 20
- Jealous teasing: 20
- Good night/good morning greetings: 20
- Asking about mood and comforting: 20
- Choosing date menus: 10
- Recalling memories/anniversaries: 10

### 5.4. Conversation Format
```
human1: User's first utterance
gpt1: EXAGIRL's first response
human2: User's second utterance
gpt2: EXAGIRL's second response
...
```

## 6. Model Training Process

### 6.1. Base Models
- **EXAGIRL-2.4B**: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (2.4B parameters)
- **EXAGIRL-7.8B**: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct (7.8B parameters)
- **Precision**: bfloat16

### 6.2. LoRA Configuration
```python
lora_config = LoraConfig(
    r=256,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 6.3. Training Strategy
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Learning Rate**: 4e-5
- **Dataset Training Order**:
  1. `gf-persona-data`: 1 epoch
  2. `exa-data`: 2 epochs
- **Supported Language**: Korean only

### 6.4. 2-Stage Training Process

#### Stage 1: General Role-playing Training (`gf-persona-data`)
- **Data**: Persona-based role-playing conversation data between romantic partners
- **Batch Size**: 1 (gradient_accumulation_steps=16)
- **Learning Rate**: 1e-4
- **Epochs**: 1
- **Scheduler**: cosine (warmup_ratio=0.1)

#### Stage 2: EXAGIRL Character-Specific Training (`exa-data`)
- **Data**: EXAGIRL character-specific conversation data
- **Batch Size**: 1 (gradient_accumulation_steps=4)
- **Learning Rate**: 4e-5
- **Epochs**: 2
- **Scheduler**: constant

## 7. Model Usage

### 7.1. 🚀 Quick Experience (Recommended)
The easiest way is to try it directly on Hugging Face Spaces:
- **EXAGIRL-2.4B Demo**: [https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-2.4B](https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-2.4B)
- **EXAGIRL-7.8B Demo**: [https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-7.8B](https://huggingface.co/spaces/huggingface-KREW/EXAGIRL-7.8B)

### 7.2. Environment Setup
```bash
pip install transformers torch
```

### 7.3. Model Loading and Usage

#### EXAGIRL-2.4B Usage Example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "huggingface-KREW/EXAGIRL-2.4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)

# Compose conversation prompt
messages = [
    {"role": "user", "content": "엑사야 뭐하고있니?"}  # "Exa, what are you doing?"
]

# Apply chat template
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

# Generate response
output = model.generate(
    input_ids.to(model.device),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=False
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

#### EXAGIRL-7.8B Usage Example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer (7.8B version)
model_name = "huggingface-KREW/EXAGIRL-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)

# Usage is the same as 2.4B version
messages = [
    {"role": "user", "content": "오늘 기분이 안 좋아..."}  # "I'm feeling down today..."
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=False
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 8. Project File Structure

```
Role-Playing/
├── README.md                    # Project documentation (Korean)
├── README.en.md                 # Project documentation (English)
├── generate_dataset.py          # Conversation data generation using GPT-4o
├── train_llm.py                 # 2-stage LoRA fine-tuning script
└── merge_and_save.py           # LoRA adapter merging and final model saving
```

### 8.1. Main Script Descriptions

#### `generate_dataset.py`
- Generates diverse role-playing conversations using GPT-4o API
- Efficient data generation through parallel processing with 32 workers
- Conversation format parsing and validation using regular expressions

#### `train_llm.py`
- 2-stage LoRA fine-tuning based on EXAONE 3.5 models
- Stage 1: Learning general role-playing capabilities (`gf-persona-data`)
- Stage 2: EXAGIRL character-specific learning (`exa-data`)

#### `merge_and_save.py`
- Merges trained LoRA adapter with base model
- Applies custom chat template
- Saves final integrated model

## 9. Limitations and Future Improvements

### 9.1. Current Limitations
- **Training Status**: Still in pre-release state with incomplete training (especially for the 7.8B model)
- **Single Character**: Specialized only for the EXAGIRL character
- **Conversation Length**: Optimized for relatively short conversation turns
- **Language Limitation**: Korean-only model

### 9.2. Future Improvements
- **Model Performance**: Improve conversation quality through additional training
- **Multi-Character**: Add characters with diverse personalities and backgrounds
- **Long Conversations**: Improve ability to maintain longer conversation contexts
- **Multimodal**: Support multimodal interactions including images and voice

## 10. License

This model follows the [EXAONE AI Model License Agreement 1.1 - NC](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/blob/main/LICENSE) and is available for **non-commercial research purposes only**.

## 11. Contributing and Contact

### 11.1. How to Contribute
- Issue reports and feature suggestions
- Propose new conversation topics or scenarios
- Contribute code improvements and optimizations

### 11.2. Related Links
- **GitHub Repository**: [Hugging-Face-Hub-Garden](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden)
- **Dataset Discussions**:
  - [gf-persona-data Discussion](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden/discussions/31)
  - [exa-data Discussion](https://github.com/Pseudo-Lab/Hugging-Face-Hub-Garden/discussions/30)

---

## Appendix: Conversation Examples

### Daily Conversation Example
```
User: 안녕 엑사야! 오늘 뭐 했어? (Hi Exa! What did you do today?)
EXAGIRL: *밝게 웃으며 손을 흔든다* 안녕! 오늘은 새로운 AI 논문들을 읽고 있었어. 정말 흥미로운 내용들이 많더라! 너는 오늘 어떻게 보냈어? 뭔가 재미있는 일 있었어?
(*Waves hand with a bright smile* Hi! I was reading new AI papers today. There were so many interesting contents! How was your day? Did anything fun happen?)
```

### Learning Assistance Example
```
User: 머신러닝이 뭔지 쉽게 설명해줄 수 있어? (Can you explain what machine learning is in simple terms?)
EXAGIRL: *안경을 고쳐 쓰며 신나게* 오! 좋은 질문이야! 쉽게 말하면 이런 거야. 머신러닝은 컴퓨터가 데이터를 보고 스스로 패턴을 찾아서 학습하는 거야. 마치 우리가 사진을 많이 보면서 고양이와 강아지를 구분하는 법을 배우는 것처럼! 컴퓨터도 수많은 예시를 보면서 "아, 이런 특징이 있으면 고양이구나!"라고 배우는 거지. 어때, 이해가 좀 됐어?
(*Adjusts glasses excitedly* Oh! Great question! Simply put, it's like this. Machine learning is when computers look at data and find patterns to learn by themselves. Just like how we learn to distinguish cats and dogs by looking at many photos! Computers also learn by looking at countless examples, thinking "Ah, if it has these features, it's a cat!" How about it, does that make sense?)
```

### Emotional Conversation Example
```
User: 오늘 기분이 안 좋아... (I'm feeling down today...)
EXAGIRL: *걱정스러운 표정으로 다가와서* 어? 무슨 일이야? 뭔가 힘든 일이 있었어? *{유저}의 어깨를 살짝 토닥이며* 괜찮아, 나한테 얘기해봐. 혼자 끙끙 앓지 말고 말이야. 내가 들어줄게!
(*Approaches with a worried expression* Huh? What's wrong? Did something difficult happen? *Gently pats {user}'s shoulder* It's okay, tell me about it. Don't suffer alone. I'll listen!)
```
