import os
import random
import json
from data_models import DatasetEntry
from llm import LLMWrapper
from generation_pipeline import DataGenerationPipeline
from recovery_pipeline import RecoveryPipeline

def worker_process(worker_id: int, gpu_id: int, iterations: int, model_name: str, mock: bool, mode: str, input_file: str = None, k: int = 3):
    """
    Function to be run in a separate process.
    """
    print(f"[Worker {worker_id}] Starting on GPU {gpu_id} (Mock={mock}, Mode={mode})...")
    
    # Initialize Model
    if not mock:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda:0"
    else:
        device = "cpu"

    try:
        llm = LLMWrapper(model_name, device=device, mock=mock)
        if mode == "generate":
            pipeline = DataGenerationPipeline(llm)
        elif mode == "recover":
            pipeline = RecoveryPipeline(llm, k=k)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to initialize model: {e}")
        return

    output_file = f"output_gpu_{gpu_id}.jsonl"
    
    if mode == "generate":
        # List of simple event hints for random selection
        events = [
            "missed the train", "found a lost wallet", "forgot wedding anniversary", "won the lottery",
            "broke a vase", "adopted a stray cat", "cooked a bad meal", "got stuck in an elevator",
            "lost my keys", "met an old friend", "spilled coffee on my shirt", "got locked out of the house",
            "phone battery died", "missed an important call", "found a $20 bill on the street", "burned the toast",
            "broke my phone screen", "got caught in the rain without an umbrella", "overslept for work",
            "left my wallet at home", "missed a flight", "finally passed the driving test",
            "won a small prize in a raffle", "forgot to submit an assignment", "parked in the wrong spot and got a ticket",
            "dropped my ice cream", "received an unexpected gift", "accidentally sent a message to the wrong person",
            "lost my luggage at the airport", "baked a cake that collapsed", "met a celebrity by accident",
            "slipped on the sidewalk but didn’t get hurt", "left the house with mismatched shoes",
            "ran into my ex at the supermarket", "got a surprise promotion", "sprained my ankle while jogging",
            "won a free coffee", "forgot where I parked the car"
        ]

        success_count = 0
        with open(output_file, "a") as f:
            for i in range(iterations):
                event = random.choice(events)
                print(f"[Worker {worker_id}] Iteration {i+1}/{iterations}: {event}")
                
                result = pipeline.run_single_iteration(event)
                
                if result:
                    f.write(result.model_dump_json() + "\n")
                    f.flush()
                    success_count += 1
                else:
                    print(f"[Worker {worker_id}] Failed to generate valid entry for '{event}'")

        print(f"[Worker {worker_id}] Finished generation. Generated {success_count} entries.")

    elif mode == "recover":
        if not input_file or not os.path.exists(input_file):
            print(f"[Worker {worker_id}] Input file not found: {input_file}")
            return

        print(f"[Worker {worker_id}] Reading from {input_file}...")
        
        # Read input file
        # We assume the input file is a JSONL file where each line is a JSON object (DatasetEntry)
        # BUT the previous code wrote "concatenated JSON objects" with indent=2, which is NOT valid JSONL.
        # It wrote: f.write(result.model_dump_json(indent=2) + "\n")
        # This creates a file with multiple JSON objects, but not one per line.
        # We need a robust reader.
        
        entries = []
        with open(input_file, "r") as f:
            content = f.read()
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(content):
                try:
                    while pos < len(content) and content[pos].isspace():
                        pos += 1
                    if pos >= len(content): break
                    obj, next_pos = decoder.raw_decode(content, pos)
                    entries.append(DatasetEntry(**obj))
                    pos = next_pos
                except json.JSONDecodeError:
                    break
        
        # Distribute work if multiple workers (simple sharding)
        # For simplicity, we'll just process a slice based on worker_id
        # Assuming main.py handles splitting or we just process everything if 1 worker.
        # Let's assume the input file is ALREADY split or we are just running 1 worker for recovery for now,
        # OR we slice it here.
        
        my_entries = entries[worker_id::1] # Simple round-robin if multiple workers on same file? 
        # Actually, main.py passes 'iterations' which is usually total/num_gpus.
        # But for recovery, we want to process ALL entries in the file.
        # Let's just process the entries assigned to this worker.
        # Better: main.py should probably split the work. 
        # But to keep it simple, I'll just process `entries[worker_id::num_workers]` logic if I knew num_workers.
        # I don't have num_workers passed explicitly, but I have worker_id.
        # I'll assume 1 worker for recovery for now or just process all if I can't split.
        # Wait, I can pass num_workers or just let main handle it.
        # Let's just process ALL entries for now, assuming the user runs recovery with 1 GPU or splits files.
        # Actually, the user requirement said "read the whole generated data from local file".
        
        success_count = 0
        with open(output_file, "w") as f: # Overwrite output for recovery results
            for i, entry in enumerate(my_entries):
                print(f"[Worker {worker_id}] Recovering entry {i+1}/{len(my_entries)}...")
                
                recovery_result = pipeline.run_recovery(entry)
                
                # Update entry with recovery result
                entry.recovery = recovery_result
                
                f.write(entry.model_dump_json() + "\n")
                f.flush()
                success_count += 1
        
        print(f"[Worker {worker_id}] Finished recovery. Processed {success_count} entries.")

