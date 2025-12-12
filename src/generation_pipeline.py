import json
import random
from typing import Optional, List

from llm import LLMWrapper
from data_models import Story, GoldSemantics, Dialogue, DatasetEntry
from prompt_templates import SYSTEM_PROMPTS
from utils import generate_banlist, check_banlist
from judge import Judge

class DataGenerationPipeline:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm
        self.judge = Judge(llm)

    def run_single_iteration(self, event_hint: str) -> Optional[DatasetEntry]:
        try:
            # Step 1: Story Generation
            story_text = self._generate_story(event_hint)
            
            # Step 1.5: Judge Story
            if not self.judge.check_story(event_hint, story_text):
                print(f"Story rejected by judge for event: {event_hint}")
                return None

            # Step 2: Extract Protagonist
            protagonist_name = self._extract_protagonist(story_text)
            
            story = Story(text=story_text, hidden_event=event_hint, protagonist_name=protagonist_name)
            gold_semantics = GoldSemantics(hidden_event=event_hint, protagonist_name=protagonist_name)

            # Step 3: Banlist Creation
            banlist = generate_banlist(event_hint)

            # Step 4: Dialogue Generation (Dynamic Turns)
            dialogue = self._generate_dialogue(story_text, event_hint, protagonist_name, banlist)
            if not dialogue:
                return None

            # Step 4.5: Judge Dialogue
            dialogue_text = "\n".join(dialogue.turns)
            if not self.judge.check_dialogue(event_hint, story_text, dialogue_text):
                print(f"Dialogue rejected by judge for event: {event_hint}")
                return None

            return DatasetEntry(
                story=story,
                gold_semantics=gold_semantics,
                banlist=banlist,
                dialogue=dialogue
            )

        except Exception as e:
            print(f"Error in generation pipeline: {e}")
            return None

    def _generate_story(self, hint: str) -> str:
        prompt = f"Hidden Event: {hint}"
        return self.llm.generate(SYSTEM_PROMPTS["storyteller"], prompt)

    def _extract_protagonist(self, story: str) -> str:
        prompt = f"Story: {story}"
        return self.llm.generate(SYSTEM_PROMPTS["protagonist_extractor"], prompt).strip()

    def _generate_dialogue(self, story: str, hidden_event: str, protagonist: str, banlist: list, max_retries: int = 3) -> Optional[Dialogue]:
        banlist_str = ", ".join(banlist)
        
        # Dynamic turns: 2 to 4
        num_turns = random.randint(2, 4)
        
        for attempt in range(max_retries):
            turns = []
            history = ""
            
            for i in range(num_turns):
                is_speaker_a = (i % 2 == 0)
                speaker_prompt_key = "dialogue_speaker_1" if is_speaker_a else "dialogue_speaker_2"
                speaker_label = "[Speaker A]" if is_speaker_a else "[Speaker B]"
                
                prompt = SYSTEM_PROMPTS[speaker_prompt_key].format(
                    story=story,
                    protagonist=protagonist,
                    hidden_event=hidden_event,
                    banlist_str=banlist_str,
                    history=history
                )
                
                turn_text = self.llm.generate("", prompt) # System prompt is embedded in the formatted string now? 
                # Wait, the previous code used system prompt key. 
                # My new prompts in SYSTEM_PROMPTS are full instructions. 
                # I should probably pass them as system prompt or user prompt. 
                # Let's pass empty system prompt and full user prompt, or use the key if LLMWrapper supports it.
                # Looking at LLMWrapper usage in original code: self.llm.generate(SYSTEM_PROMPTS["storyteller"], prompt)
                # It takes (system_prompt, user_prompt).
                # My new prompts are designed as system prompts mostly.
                # Let's adjust:
                
                # Actually, the prompts I wrote have placeholders like {story}. 
                # I should format them first.
                
                # Let's treat the formatted string as the system prompt (or user prompt if system is fixed).
                # The original code used: self.llm.generate(SYSTEM_PROMPTS["dialogue_speaker_1"], p1)
                # So I should probably keep the system prompt static and put context in user prompt?
                # But my new prompts have the context embedded.
                # Let's use the formatted string as the USER prompt and a generic system prompt, 
                # OR use the formatted string as the SYSTEM prompt and empty user prompt.
                # Let's go with: System Prompt = "You are a helpful assistant." (or similar generic), User Prompt = Formatted Instruction.
                # OR better: The LLMWrapper might expect specific args.
                
                # Let's check LLMWrapper signature if possible, but I don't have it open. 
                # Assuming generate(system, user).
                
                # I will use a generic system prompt for dialogue and put everything in user prompt for simplicity,
                # OR I can just pass the formatted string as the system prompt and "Go." as user prompt.
                
                # Let's try: 
                # System: "You are a roleplay actor."
                # User: <The formatted prompt>
                
                pass 
                
            # RETHINKING: The prompts in `prompt_templates.py` are strings. 
            # I should use them as templates.
            
            # Let's fix the loop.
            
            current_turns = []
            valid_attempt = True
            
            for i in range(num_turns):
                is_speaker_a = (i % 2 == 0)
                speaker_key = "dialogue_speaker_1" if is_speaker_a else "dialogue_speaker_2"
                speaker_label = "Speaker A" if is_speaker_a else "Speaker B"
                
                # Construct history for the prompt
                history_str = "\n".join(current_turns) if current_turns else "None"
                
                # Format the prompt
                # Note: The keys in SYSTEM_PROMPTS are the templates now.
                template = SYSTEM_PROMPTS[speaker_key]
                full_prompt = template.format(
                    story=story,
                    protagonist=protagonist,
                    hidden_event=hidden_event,
                    banlist_str=banlist_str,
                    history=history_str
                )
                
                # Generate
                # We pass the full prompt as the "system" instruction effectively, or just as user prompt.
                # To be safe with `llm.generate(sys, user)`, I'll pass:
                # sys = "You are a roleplay actor."
                # user = full_prompt
                turn_response = self.llm.generate("You are a roleplay actor.", full_prompt)
                
                # Check banlist
                if not check_banlist(turn_response, banlist):
                    valid_attempt = False
                    break
                
                current_turns.append(f"[{speaker_label}]: {turn_response}")
            
            if valid_attempt:
                return Dialogue(turns=current_turns)
                
        return None
