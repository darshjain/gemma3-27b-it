from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

model_id = "google/gemma-3-27b-it"

# Load model and processor once
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

while True:
    user_input = input("Please enter the prompt (or type 'exit' to quit) - ")
    if user_input.lower() == "exit":
        break

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_input}
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
        generation = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print("Here is your output -", decoded)
