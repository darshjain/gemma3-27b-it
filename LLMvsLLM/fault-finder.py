
import os
import csv
import torch
import json
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AdamW
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = "google/gemma-3-27b-it"
checkpoint_dir = "checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pt")
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map={"": "cuda:0"})
model.train()
processor = AutoProcessor.from_pretrained(model_id)
optimizer = AdamW(model.parameters(), lr=1e-5)
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
csv_filename = "output.csv"
rows = []
with open(csv_filename, "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        rows.append(row)
with open("prompts/prompt-check.txt", "r") as file:
    base_prompt_text = file.read().strip()
iteration = 0
for row in tqdm(rows, desc="Processing"):
    program_text = row["program"]
    expected_fault = row["fault"].strip()
    prompt_text = base_prompt_text + "\n" + program_text
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        generation = model.generate(**inputs, max_new_tokens=400, do_sample=True)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True).strip()
    try:
        output_json = json.loads(decoded)
        predicted_fault = output_json.get("FaultyLine", "").strip()
    except json.JSONDecodeError:
        predicted_fault = decoded
    if predicted_fault != expected_fault:
        ft_prompt = prompt_text + "\n" + expected_fault
        inputs_ft = processor(ft_prompt, return_tensors="pt").to(device, dtype=torch.bfloat16)
        inputs_ft["labels"] = inputs_ft["input_ids"].clone()
        outputs = model(**inputs_ft)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    iteration += 1
    if iteration % 5 == 0:
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, checkpoint_path)
