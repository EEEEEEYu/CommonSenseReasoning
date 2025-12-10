import os
import random
import json
from data_models import DatasetEntry

def worker_process(worker_id: int, gpu_id: int, iterations: int, model_name: str, mock: bool):
    """
    Function to be run in a separate process.
    """
    print(f"[Worker {worker_id}] Starting on GPU {gpu_id} (Mock={mock})...")
    
    # Lazy import to ensure compatibility with spawn start method if needed,
    # though here we are likely standard fork/spawn context.
    from llm import LLMWrapper
    from pipeline import GenerationPipeline
    
    # Initialize Model on specific device (or mock)
    # vLLM will attempt to use all visible GPUs, so we restrict it to just one
    if not mock:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # With CUDA_VISIBLE_DEVICES set, the device inside the process is always "cuda:0"
        device = "cuda:0"
    else:
        device = "cpu"

    try:
        llm = LLMWrapper(model_name, device=device, mock=mock)
        pipeline = GenerationPipeline(llm)
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to initialize model: {e}")
        return

    output_file = f"output_gpu_{gpu_id}.jsonl"
    
    # List of simple event hints for random selection
    # In a real app, this might come from a file or generator
    events = [
        "missed the train",
        "found a lost wallet",
        "forgot wedding anniversary",
        "won the lottery",
        "broke a vase",
        "adopted a stray cat",
        "cooked a bad meal",
        "got stuck in an elevator",
        "lost my keys",
        "met an old friend",
        "spilled coffee on my shirt",
        "got locked out of the house",
        "phone battery died",
        "missed an important call",
        "found a $20 bill on the street",
        "burned the toast",
        "broke my phone screen",
        "got caught in the rain without an umbrella",
        "overslept for work",
        "left my wallet at home",
        "missed a flight",
        "finally passed the driving test",
        "won a small prize in a raffle",
        "forgot to submit an assignment",
        "parked in the wrong spot and got a ticket",
        "dropped my ice cream",
        "received an unexpected gift",
        "accidentally sent a message to the wrong person",
        "lost my luggage at the airport",
        "baked a cake that collapsed",
        "met a celebrity by accident",
        "slipped on the sidewalk but didn’t get hurt",
        "left the house with mismatched shoes",
        "ran into my ex at the supermarket",
        "got a surprise promotion",
        "sprained my ankle while jogging",
        "won a free coffee",
        "forgot where I parked the car"
    ]

    success_count = 0
    with open(output_file, "a") as f:
        for i in range(iterations):
            event = random.choice(events)
            print(f"[Worker {worker_id}] Iteration {i+1}/{iterations}: {event}")
            
            result: DatasetEntry = pipeline.run_single_iteration(event)
            
            if result:
                # User requested formatted JSON. This creates concatenated JSON objects in the file.
                f.write(result.model_dump_json(indent=2) + "\n")
                f.flush()
                success_count += 1
            else:
                print(f"[Worker {worker_id}] Failed to generate valid entry for '{event}'")

    print(f"[Worker {worker_id}] Finished. Generated {success_count} entries.")
