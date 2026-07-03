import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your actual local BF16 path
MODEL_ID = "/home/nauyan/voice-agent-pipeline/models/gemma-4-E4B-it"

tokenizer = None
model = None

def init_llm():
    """Loads the model with SDPA attention and warms up the GPU cache."""
    global tokenizer, model
    
    print("⏳ Initializing Gemma-4 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("⏳ Loading Gemma-4 model weights (SDPA + BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    print("🔥 Warming up GPU cache...")
    dummy_input = tokenizer("Warm up", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        _ = model.generate(**dummy_input, max_new_tokens=1, use_cache=True)
    torch.cuda.synchronize()
    print("✅ Brain Engine Ready!")

def get_agent_response(user_text: str) -> str:
    """Takes transcribed text, runs inference, and returns voice-ready dialogue."""
    global tokenizer, model
    
    if model is None:
        raise RuntimeError("LLM not initialized. Call init_llm() first.")
    
    # STRICT VOICE PROMPT: Prevents the model from "thinking out loud" or using markdown
    messages = [
        {
            "role": "system", 
            "content": "You are a conversational AI. Respond directly to the user. Do NOT output any internal thoughts, reasoning, asterisks, or markdown. Output ONLY the exact words you will speak out loud. Keep answers brief and natural."
        },
        {"role": "user", "content": user_text}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # Timing the generation for real-time tracking
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        start_event.record()
        outputs = model.generate(
            **inputs,
            max_new_tokens=100, # Kept short so it responds instantly
            use_cache=True,
            do_sample=True,
            temperature=0.7,
        )
        end_event.record()
        torch.cuda.synchronize()

    # Calculate metrics
    elapsed_ms = start_event.elapsed_time(end_event)
    output_tokens = outputs[0].shape[0] - input_len
    tokens_per_sec = output_tokens / (elapsed_ms / 1000.0)
    
    print(f"⚡ Generation Speed: {tokens_per_sec:.2f} tokens/sec | Total Time: {elapsed_ms:.2f} ms")

    # Decode and clean the final text
    response_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    clean_text = response_text.replace("*", "").replace("[", "").replace("]", "").strip()
    
    return clean_text

# --- Independent Testing Block ---
# This runs only if you execute this file directly to test it
if __name__ == "__main__":
    init_llm()
    print("\n--- Testing Brain Logic ---")
    
    test_question = "If I am jogging on a treadmill at 6.5 speed, how long until I burn 300 calories?"
    print(f"🗣️ User: {test_question}")
    
    answer = get_agent_response(test_question)
    print(f"🤖 Agent: {answer}")