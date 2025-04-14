from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from tqdm import tqdm
import json
import csv

model_id = "google/gemma-3-27b-it"
num_iterations = 1  # Change this if you need

# Load model and processor once
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# Read single prompt from file
with open("prompt.txt", "r") as file:
    prompt_text = file.read().strip()

results = []

# Prepare messages once
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text}
        ]
    }
]

# Prepare inputs once (since prompt doesn't change)
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

# Run inference loop
for _ in tqdm(range(num_iterations), desc="Running Inference"):
    with torch.inference_mode():
        generation = model.generate(
            **inputs, max_new_tokens=500, do_sample=False
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    decoded = decoded.lstrip("```json")
    decoded = decoded.rstrip("```")
    print("THE DECODED ",decoded)

    try:
        output_json = json.loads(decoded)
        program = output_json.get("Program", "").strip()
        fault = output_json.get("Fault", "").strip()
    except json.JSONDecodeError as e:
        print(e)
        program = ""
        fault = ""

    results.append({
        "program": program,
        "fault": fault
    })

# Write results to CSV
with open("output.csv", "w", newline='') as csvfile:
    fieldnames = ["program", "fault"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
