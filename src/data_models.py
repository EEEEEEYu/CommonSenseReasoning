from typing import List, Optional
from pydantic import BaseModel, Field

class Story(BaseModel):
    text: str = Field(..., description="The generated story text.")
    hidden_event: str = Field(..., description="The hidden event semantics used to generate the story.")

class GoldSemantics(BaseModel):
    agent: str = Field(..., description="The agent of the event.")
    predicate: str = Field(..., description="The action or predicate.")
    patient: Optional[str] = Field(None, description="The entity affected by the action.")
    recipient: Optional[str] = Field(None, description="The entity receiving something.")
    location: Optional[str] = Field(None, description="Where the event took place.")
    time: Optional[str] = Field(None, description="When the event took place.")
    instrument: Optional[str] = Field(None, description="The instrument used.")

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
