
import torch
import os
import sys
from transformers import AutoTokenizer

MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(MODEL_PATH)

print(f"Loading Reparameterized RcMoE explicitly from {MODEL_PATH}...")
try:
    from modeling_RcMoE import RcMoEForCausalLM, RcMoEConfig
    config = RcMoEConfig.from_pretrained(MODEL_PATH)
    
    # 1. 正常加载瘦身后的主模型到 GPU
    model = RcMoEForCausalLM.from_pretrained(
        MODEL_PATH, 
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    # 2. [核心] 挂载 LUT (Mmap 模式，瞬间完成，0 显存占用)
    lut_path = os.path.join(MODEL_PATH, "lut_nf4.pt")
    model.load_lut(lut_path)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Generating...")
    inputs = tokenizer(
        "Artificial Intelligence is a branch of computer", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
        
    print("-" * 30)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-" * 30)

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILED: {e}")
