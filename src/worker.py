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
        "missed the train", "found a lost wallet", "forgot wedding anniversary",
        "won the lottery", "broke a vase", "adopted a stray cat",
        "cooked a bad meal", "got stuck in an elevator"
    ]

    success_count = 0
    with open(output_file, "a") as f:
        for i in range(iterations):
            event = random.choice(events)
            print(f"[Worker {worker_id}] Iteration {i+1}/{iterations}: {event}")
            
            result: DatasetEntry = pipeline.run_single_iteration(event)
            
            if result:
                f.write(result.model_dump_json() + "\n")
                f.flush()
                success_count += 1
            else:
                print(f"[Worker {worker_id}] Failed to generate valid entry for '{event}'")

    print(f"[Worker {worker_id}] Finished. Generated {success_count} entries.")
