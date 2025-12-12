import os
import sys
import json
import shutil

# Add src and root to pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["vllm"] = MagicMock()
sys.modules["accelerate"] = MagicMock()

# Mock Judge to always pass
import judge
judge.Judge.check_story = MagicMock(return_value=True)
judge.Judge.check_dialogue = MagicMock(return_value=True)
judge.Judge.check_recovery = MagicMock(return_value=True)

# Mock LLM generation to return usable data
from src.llm import LLMWrapper
def mock_generate(self, system_prompt, user_prompt, **kwargs):
    if "protagonist" in user_prompt.lower() and "extract" in user_prompt.lower():
        return "Alice"
    if "guesses" in user_prompt.lower():
        return '{"guesses": ["event 1", "event 2", "event 3"]}'
    if "json" in user_prompt.lower():
        return '{}'
    return "Mocked content."

LLMWrapper.generate = mock_generate

from main import main
from unittest.mock import patch

def test_pipeline():
    print("Testing Generation Pipeline...")
    # Run generation in mock mode
    # We simulate command line args
    test_args_gen = [
        "main.py",
        "--mock",
        "--mode", "generate",
        "--iterations", "2",
        "--num_gpus", "1"
    ]
    
    with patch.object(sys, 'argv', test_args_gen):
        main()
        
    # Check if output file exists and has content
    output_file = "output_gpu_0.jsonl"
    if not os.path.exists(output_file):
        print("FAILED: Output file not created.")
        return
        
    with open(output_file, "r") as f:
        lines = f.readlines()
        if len(lines) != 2:
            print(f"FAILED: Expected 2 lines, got {len(lines)}")
            return
        
        # Validate content structure
        try:
            data = json.loads(lines[0])
            if "story" not in data or "dialogue" not in data:
                print("FAILED: Missing keys in generated data.")
                return
            print("Generation Output Validated.")
        except json.JSONDecodeError:
            print("FAILED: Invalid JSON in output.")
            return

    # Rename output file to input file for recovery
    input_file = "test_input.jsonl"
    shutil.copy(output_file, input_file)
    
    print("Testing Recovery Pipeline...")
    # Run recovery in mock mode
    test_args_rec = [
        "main.py",
        "--mock",
        "--mode", "recover",
        "--input_file", input_file,
        "--num_gpus", "1"
    ]
    
    with patch.object(sys, 'argv', test_args_rec):
        main()
        
    # Check output (it overwrites output_gpu_0.jsonl)
    with open(output_file, "r") as f:
        lines = f.readlines()
        if len(lines) != 2:
            print(f"FAILED: Expected 2 recovered lines, got {len(lines)}")
            return
            
        try:
            data = json.loads(lines[0])
            if "recovery" not in data or not data["recovery"]:
                print("FAILED: Recovery data missing.")
                return
            if "guesses" not in data["recovery"]:
                print("FAILED: Guesses missing in recovery.")
                return
            print("Recovery Output Validated.")
        except json.JSONDecodeError:
            print("FAILED: Invalid JSON in recovery output.")
            return

    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    test_pipeline()
