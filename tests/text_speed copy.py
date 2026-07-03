import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your actual local BF16 path
model_id = "/home/nauyan/voice-agent-pipeline/models/gemma-4-E4B-it"

print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model weights...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    # Gemma 4 E4B uses a hybrid sliding window. Use sdpa for safety.
    attn_implementation="sdpa",
)

print("Warming up GPU cache...")
dummy_input = tokenizer("Warm up", return_tensors="pt").to("cuda")
with torch.inference_mode():
    _ = model.generate(**dummy_input, max_new_tokens=1, use_cache=True)
torch.cuda.synchronize()

questions = [
    "What is speed?",
    "How do I say hello in Italian?",
    "What is the weather today?",
    "Summarize the word computer in one sentence.",
    "What is the fastest land animal?",
]

print("\n--- Voice Agent Speed Test ---")
all_speeds = []
all_ttft = []

for index, question in enumerate(questions, start=1):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        start_event.record()
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            use_cache=True,
            do_sample=False,
        )
        end_event.record()
        torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    tokens_per_sec = input_len / (elapsed_ms / 1000.0)
    all_speeds.append(tokens_per_sec)
    all_ttft.append(elapsed_ms)

    print(f"Question {index}: {question}")
    print(f"  TTFT: {elapsed_ms:.2f} ms")
    print(f"  Speed: {tokens_per_sec:.2f} tokens/sec")
    print(f"  First token id: {output[0][-1].item()}")
    print("----------------------------------------")

average_speed = sum(all_speeds) / len(all_speeds)
average_ttft = sum(all_ttft) / len(all_ttft)
print(
    f"Average prompt speed over {len(questions)} questions: {average_speed:.2f} tokens/sec"
)
print(f"Average TTFT over {len(questions)} questions: {average_ttft:.2f} ms")
