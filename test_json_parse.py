
import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.pipeline import GenerationPipeline
from src.llm import LLMWrapper

class TestParsing(unittest.TestCase):
    def test_parse_clean_json(self):
        llm = LLMWrapper("mock", mock=True)
        pipeline = GenerationPipeline(llm)
        
        raw = '{"agent": "Bond", "predicate": "spies"}'
        parsed = pipeline._parse_json(raw)
        self.assertEqual(parsed, {"agent": "Bond", "predicate": "spies"})

    def test_parse_markdown_json(self):
        llm = LLMWrapper("mock", mock=True)
        pipeline = GenerationPipeline(llm)
        
        raw = 'Here is the result:\n```json\n{"agent": "Bond", "predicate": "spies"}\n```'
        parsed = pipeline._parse_json(raw)
        self.assertEqual(parsed, {"agent": "Bond", "predicate": "spies"})

    def test_parse_dirty_json(self):
        llm = LLMWrapper("mock", mock=True)
        pipeline = GenerationPipeline(llm)
        
        raw = 'Sure, here logic: {"agent": "Bond", "predicate": "spies"} Hope this helps!'
        parsed = pipeline._parse_json(raw)
        self.assertEqual(parsed, {"agent": "Bond", "predicate": "spies"})

if __name__ == '__main__':
    unittest.main()
