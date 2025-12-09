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
            # vLLM imports
            from vllm import LLM, SamplingParams
            
            # vLLM handles quantization and device mapping internally.
            # We assume CUDA_VISIBLE_DEVICES is set correctly by the worker.
            self.model = LLM(
                model=model_name,
                quantization="awq" if "awq" in model_name.lower() else None, 
                trust_remote_code=True,
                dtype="auto"
            )
            # No tokenizer needed explicitly for vLLM generation usually, 
            # but we can keep it if needed for checking tokens, though vLLM handles it.
            # We'll rely on vLLM's internal tokenization.

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
        if self.mock:
            return self._mock_generate(system_prompt, user_prompt)
        
        from vllm import SamplingParams

        # Simple chat template construction
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
        )

        # vLLM generate returns a list of RequestOutput objects
        outputs = self.model.generate([full_prompt], sampling_params, use_tqdm=False)
        
        # We only passed one prompt
        generated_text = outputs[0].outputs[0].text
        return generated_text.strip()

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
