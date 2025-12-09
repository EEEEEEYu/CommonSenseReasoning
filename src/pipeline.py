import json
import random
import time
from typing import Optional, Dict

from llm import LLMWrapper
from data_models import Story, GoldSemantics, Dialogue, Recovery, DatasetEntry
from prompt_templates import SYSTEM_PROMPTS
from utils import generate_banlist, check_banlist, calculate_set_atom_metrics

class GenerationPipeline:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def run_single_iteration(self, event_hint: str) -> Optional[DatasetEntry]:
        try:
            # Step 1: Story Generation
            story_text = self._generate_story(event_hint)
            story = Story(text=story_text, hidden_event=event_hint)

            # Step 2: Gold Extraction
            gold_json = self._extract_gold(story_text)
            print("story_text", story_text)
            print("gold_json", gold_json)
            gold_semantics = GoldSemantics(**gold_json)

            # Step 3: Banlist Creation
            banlist = generate_banlist(event_hint)

            # Step 4: Dialogue Generation (with retry)
            dialogue = self._generate_dialogue(story_text, banlist)
            if not dialogue:
                return None  # Failed to generate valid dialogue after retries

            # Step 5: Recovery
            recovery_json = self._recover_semantics(dialogue)
            print(recovery_json)
            recovery = Recovery(predicted_semantics=GoldSemantics(**recovery_json))

            # Metrics
            metrics = calculate_set_atom_metrics(gold_json, recovery_json)

            return DatasetEntry(
                story=story,
                gold_semantics=gold_semantics,
                banlist=banlist,
                dialogue=dialogue,
                recovery=recovery,
                metrics=metrics
            )

        except Exception as e:
            print(f"Error in pipeline: {e}")
            return None

    def _generate_story(self, hint: str) -> str:
        prompt = f"Write a story about: {hint}"
        return self.llm.generate(SYSTEM_PROMPTS["storyteller"], prompt)

    def _extract_gold(self, story: str, max_retries: int = 3) -> dict:
        prompt = f"Story: {story}\n\nExtract the JSON semantics:"
        for _ in range(max_retries):
            raw = self.llm.generate(SYSTEM_PROMPTS["gold_extractor"], prompt)
            data = self._parse_json(raw)
            # Basic validation: check for required keys
            if data and "agent" in data and "predicate" in data:
                return data
        # Return empty if all retries fail, will likely cause downstream error but handled by try/except
        return {}

    def _generate_dialogue(self, story: str, banlist: list, max_retries: int = 3) -> Optional[Dialogue]:
        banlist_str = ", ".join(banlist)
        
        for attempt in range(max_retries):
            turns = []
            # Initialize with story context (invisible to later recovery, but needed for speakers)
            # We simulate a conversation of 4 turns
            
            # Speaker A
            p1 = f"Context: {story}. Banned words: {banlist_str}. You start the conversation."
            t1 = self.llm.generate(SYSTEM_PROMPTS["dialogue_speaker_1"], p1)
            if not check_banlist(t1, banlist): continue
            turns.append(t1)
            
            # Speaker B
            p2 = f"Context: {story}. Previous turn: {t1}. Banned words: {banlist_str}. Reply."
            t2 = self.llm.generate(SYSTEM_PROMPTS["dialogue_speaker_2"], p2)
            if not check_banlist(t2, banlist): continue
            turns.append(t2)
            
            # Speaker A
            p3 = f"Context: {story}. Previous turn: {t2}. Banned words: {banlist_str}. Reply."
            t3 = self.llm.generate(SYSTEM_PROMPTS["dialogue_speaker_1"], p3)
            if not check_banlist(t3, banlist): continue
            turns.append(t3)
            
             # Speaker B
            p4 = f"Context: {story}. Previous turn: {t3}. Banned words: {banlist_str}. Reply."
            t4 = self.llm.generate(SYSTEM_PROMPTS["dialogue_speaker_2"], p4)
            if not check_banlist(t4, banlist): continue
            turns.append(t4)
            
            return Dialogue(turns=turns)
            
        return None

    def _recover_semantics(self, dialogue: Dialogue, max_retries: int = 3) -> dict:
        # Join turns for the input
        conversation_text = "\n".join(dialogue.turns)
        prompt = f"Dialogue:\n{conversation_text}\n\nPredict the hidden event semantics as JSON:"
        
        for _ in range(max_retries):
            raw = self.llm.generate(SYSTEM_PROMPTS["recovery_agent"], prompt)
            data = self._parse_json(raw)
            # Basic validation
            if data and "agent" in data and "predicate" in data:
                return data
                
        return {}

    def _parse_json(self, raw: str) -> dict:
        import re
        # Try to find a JSON block in markdown
        match = re.search(r"```json(.*?)```", raw, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            # If no markdown block, try to find the first { and last }
            # This handles cases where the model returns just valid JSON or JSON with text around it
            try:
                start = raw.index('{')
                end = raw.rindex('}') + 1
                cleaned = raw[start:end]
            except ValueError:
                cleaned = raw.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: try to repair common issues if needed or just return empty
            return {}

