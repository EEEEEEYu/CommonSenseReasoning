from typing import List, Optional
from pydantic import BaseModel, Field

class Story(BaseModel):
    text: str = Field(..., description="The generated story text.")
    hidden_event: str = Field(..., description="The hidden event semantics used to generate the story.")
    protagonist_name: str = Field(..., description="The name of the protagonist in the story.")

class GoldSemantics(BaseModel):
    hidden_event: str = Field(..., description="The true hidden event.")
    protagonist_name: str = Field(..., description="The name of the protagonist.")

class Dialogue(BaseModel):
    turns: List[str] = Field(..., description="A list of strings representing the dialogue turns between two speakers.")

class Recovery(BaseModel):
    guesses: List[str] = Field(..., description="List of k guesses for the hidden event.")
    success: bool = Field(..., description="Whether the true hidden event was found in the guesses.")

class DatasetEntry(BaseModel):
    story: Story
    gold_semantics: GoldSemantics
    banlist: List[str]
    dialogue: Dialogue
    recovery: Optional[Recovery] = None # Optional because generation pipeline doesn't produce this
    metrics: Optional[dict] = None

