from typing import List, Optional
from pydantic import BaseModel, Field

class Story(BaseModel):
    text: str = Field(..., description="The generated story text.")
    hidden_event: str = Field(..., description="The hidden event semantics used to generate the story.")

class GoldSemantics(BaseModel):
    agent: List[str] = Field(default_factory=list, description="List of agents involved.")
    predicate: List[str] = Field(default_factory=list, description="List of actions or predicates.")
    patient: List[str] = Field(default_factory=list, description="List of entities affected by the action.")
    recipient: List[str] = Field(default_factory=list, description="List of entities receiving something.")
    location: List[str] = Field(default_factory=list, description="List of locations.")
    time: List[str] = Field(default_factory=list, description="List of time references.")
    instrument: List[str] = Field(default_factory=list, description="List of instruments used.")

class Dialogue(BaseModel):
    turns: List[str] = Field(..., description="A list of strings representing the dialogue turns between two speakers.")

class Recovery(BaseModel):
    predicted_semantics: GoldSemantics = Field(..., description="The Neo-Davidsonian semantics recovered from the dialogue.")

class DatasetEntry(BaseModel):
    story: Story
    gold_semantics: GoldSemantics
    banlist: List[str]
    dialogue: Dialogue
    recovery: Recovery
    metrics: dict
