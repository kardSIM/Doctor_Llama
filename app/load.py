from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from trl import setup_chat_format



def load_model():

    model_name="meta-llama/Llama-3.2-3B-Instruct"
    peft_model = "youzarsif/Doctor_Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(peft_model)

    base_model_reload= AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
    model = PeftModel.from_pretrained(base_model_reload, peft_model)

    model = model.merge_and_unload()
    return model, tokenizer






def generate_rep(query, instruction, model,tokenizer):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": query}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens=150, 
        num_return_sequences=1, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.9,
        repetition_penalty=1.2 
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = text.split("assistant")[1].strip() if "assistant" in text else text.strip()   
    return response 