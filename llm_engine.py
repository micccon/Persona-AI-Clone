from llama_cpp import Llama
import os

class LLMEngine:
    def __init__(self, model_path, n_ctx=4096):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
        
        print(f"🧠 Loading LLM from {model_path}...")
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1, # Offload all layers to GPU
            verbose=False
        )

    def generate(self, system_prompt, history, user_input):
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-20:]) # Keep last 20 messages context
        messages.append({"role": "user", "content": user_input})

        output = self.model.create_chat_completion(
            messages=messages,
            temperature=0.85,      
            top_p=0.92,            
            frequency_penalty=0.1, 
            presence_penalty=0.3,  
            max_tokens=600,        
            stream=False           
        )
        return output["choices"][0]["message"]["content"]