import json
from typing import List, Optional
from llm import LLMWrapper
from data_models import DatasetEntry, Recovery
from prompt_templates import SYSTEM_PROMPTS
from judge import Judge

class RecoveryPipeline:
    def __init__(self, llm: LLMWrapper, k: int = 3):
        self.llm = llm
        self.judge = Judge(llm)
        self.k = k

    def run_recovery(self, entry: DatasetEntry) -> Recovery:
        dialogue_text = "\n".join(entry.dialogue.turns)
        
        # Generate guesses
        guesses = self._generate_guesses(dialogue_text)
        
        # Evaluate
        success = self.judge.check_recovery(entry.gold_semantics.hidden_event, guesses)
        
        return Recovery(guesses=guesses, success=success)

    def _generate_guesses(self, dialogue_text: str) -> List[str]:
        prompt = SYSTEM_PROMPTS["recovery_agent"].format(dialogue=dialogue_text, k=self.k)
        # Using generic system prompt
        response = self.llm.generate("You are a helpful assistant.", prompt)
        
        data = self._parse_json(response)
        guesses = data.get("guesses", [])
        
        # Ensure we have a list of strings
        if isinstance(guesses, list):
            return [str(g) for g in guesses]
        return []

    def _parse_json(self, raw: str) -> dict:
        import re
        match = re.search(r"```json(.*?)```", raw, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            try:
                start = raw.index('{')
                end = raw.rindex('}') + 1
                cleaned = raw[start:end]
            except ValueError:
                cleaned = raw.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}
