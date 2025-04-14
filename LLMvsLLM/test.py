from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

model_id = "google/gemma-3-27b-it"

# Load model and processor
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# Read prompt from file
with open("prompt.txt", "r") as file:
    prompt_text = file.read().strip()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text}
        ]
    }
]

# Prepare inputs
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

# Generate response
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
