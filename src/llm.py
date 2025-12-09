from typing import List, Dict, Optional
import torch
import json

class LLMWrapper:
    def __init__(self, model_name: str, device: str = "cuda", mock: bool = False):
        self.mock = mock
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        if not self.mock:
            # Import here to avoid dependencies if just checking code structure
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # minimal 4bit config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Ensure pad token is set (often needed for Llama/Mistral)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device
            )

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
        if self.mock:
            return self._mock_generate(system_prompt, user_prompt)
        
        # Simple chat template construction
        # Note: Proper chat template usage depends on the specific model (Llama-3 vs Mistral vs etc)
        # Here we do a generic raw string concatenation for simplicity in this demo.
        # In production, use tokenizer.apply_chat_template if available.
        
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def _mock_generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Deterministic mock responses for testing logic flow.
        """
        sys = system_prompt[:20].lower()
        if "storyteller" in sys or "creative" in sys:
            return f"Once upon a time, [MOCK STORY] based on {user_prompt}. The end."
        
        if "semantic" in sys or "extract" in sys:
            # Return valid JSON for parsing
            return json.dumps({
                "agent": "MockAgent",
                "predicate": "MockAction",
                "patient": "MockPatient",
                "location": "MockPlace",
                "time": "MockTime"
            })
            
        if "speaker" in sys:
            return f"Hello, I am a mock speaker. I am responding to {user_prompt[:10]}..."
            
        if "detective" in sys or "recovery" in sys:
             return json.dumps({
                "agent": "MockAgent",
                "predicate": "MockAction",
                "patient": "MockPatient",
                "location": "MockPlace",
                "time": "MockTime"
            })
            
        return "Generic Mock Response"
