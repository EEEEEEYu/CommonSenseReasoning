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
            gold_semantics = GoldSemantics(**gold_json)

            # Step 3: Banlist Creation
            banlist = generate_banlist(event_hint)

            # Step 4: Dialogue Generation (with retry)
            dialogue = self._generate_dialogue(story_text, banlist)
            if not dialogue:
                return None  # Failed to generate valid dialogue after retries

            # Step 5: Recovery
            recovery_json = self._recover_semantics(dialogue)
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

    def _extract_gold(self, story: str) -> dict:
        prompt = f"Story: {story}\n\nExtract the JSON semantics:"
        raw = self.llm.generate(SYSTEM_PROMPTS["gold_extractor"], prompt)
        return self._parse_json(raw)

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

    def _recover_semantics(self, dialogue: Dialogue) -> dict:
        # Join turns for the input
        conversation_text = "\n".join(dialogue.turns)
        prompt = f"Dialogue:\n{conversation_text}\n\nPredict the hidden event semantics as JSON:"
        raw = self.llm.generate(SYSTEM_PROMPTS["recovery_agent"], prompt)
        return self._parse_json(raw)

    def _parse_json(self, raw: str) -> dict:
        # Basic cleanup to handle markdown json blocks usually returned by LLMs
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback for simple errors or partial generation
            # In a real system, we might use a parser library or regex extract
            # returning empty dict for resilience here
            return {}
