import json
from typing import Dict, Any, List, Optional
from llm import LLMWrapper
from prompt_templates import SYSTEM_PROMPTS

class Judge:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def check_story(self, hidden_event: str, story_text: str) -> bool:
        prompt = f"Hidden Event: {hidden_event}\nStory: {story_text}"
        response = self.llm.generate(SYSTEM_PROMPTS["judge_story"], prompt)
        data = self._parse_json(response)
        return data.get("valid", False)

    def check_dialogue(self, hidden_event: str, story_text: str, dialogue_text: str) -> bool:
        prompt = f"Hidden Event: {hidden_event}\nStory: {story_text}\nDialogue: {dialogue_text}"
        response = self.llm.generate(SYSTEM_PROMPTS["judge_dialogue"], prompt)
        data = self._parse_json(response)
        return data.get("valid", False)

    def check_recovery(self, hidden_event: str, guesses: List[str]) -> bool:
        guesses_str = json.dumps(guesses)
        prompt = f"Hidden Event: {hidden_event}\nGuesses: {guesses_str}"
        response = self.llm.generate(SYSTEM_PROMPTS["judge_recovery"], prompt, temperature=0.0) # Low temp for deterministic judgment
        data = self._parse_json(response)
        return data.get("match", False)

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        import re
        # Try to find a JSON block in markdown
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
