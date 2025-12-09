import argparse
import multiprocessing
import sys
import os
import json

# Add src to pythonpath so imports work easily from main
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.worker import worker_process
from src.data_models import DatasetEntry

def main():
    parser = argparse.ArgumentParser(description="NLP Data Generation Pipeline")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model name or path")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--iterations", type=int, default=10, help="Total iterations across all GPUs")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no GPU required)")
    
    args = parser.parse_args()

    # If mock, we ignore gpu count constraint but still start processes to test logic
    if args.mock:
        print("Running in MOCK mode.")
    
    iters_per_worker = args.iterations // args.num_gpus
    processes = []
    
    print(f"Starting {args.num_gpus} workers. Total iterations: {args.iterations}")

    for i in range(args.num_gpus):
        p = multiprocessing.Process(
            target=worker_process,
            args=(i, i, iters_per_worker, args.model, args.mock)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All workers finished. Aggregating manual review buffer...")
    
    # Aggregate first 10 examples for manual review
    manual_review_file = "manual_review_samples.txt"
    count = 0
    with open(manual_review_file, "w") as out:
        for i in range(args.num_gpus):
            out_file = f"output_gpu_{i}.jsonl"
            if os.path.exists(out_file):
                with open(out_file, "r") as f:
                    for line in f:
                        if count >= 10: break
                        try:
                            data = json.loads(line)
                            out.write(f"--- Sample {count+1} ---\n")
                            out.write(f"Story: {data['story']['text']}\n")
                            out.write(f"Gold: {json.dumps(data['gold_semantics'], indent=2)}\n")
                            out.write(f"Dialogue: {json.dumps(data['dialogue']['turns'], indent=2)}\n")
                            out.write(f"Recovered: {json.dumps(data['recovery']['predicted_semantics'], indent=2)}\n")
                            out.write("\n")
                            count += 1
                        except: pass
            if count >= 10: break
            
    print(f"Manual review samples saved to {manual_review_file}")

if __name__ == "__main__":
    # Ensure spawn method for CUDA compatibility
    multiprocessing.set_start_method("spawn", force=True)
    main()
