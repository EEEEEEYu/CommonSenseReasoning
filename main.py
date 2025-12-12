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
    parser.add_argument("--iterations", type=int, default=10, help="Total iterations across all GPUs (Generation mode)")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no GPU required)")
    parser.add_argument("--mode", type=str, default="generate", choices=["generate", "recover"], help="Pipeline mode")
    parser.add_argument("--input_file", type=str, default=None, help="Input file for recovery mode")
    parser.add_argument("--k", type=int, default=3, help="Number of guesses for recovery")
    
    args = parser.parse_args()

    # If mock, we ignore gpu count constraint but still start processes to test logic
    if args.mock:
        print("Running in MOCK mode.")
    
    iters_per_worker = args.iterations // args.num_gpus
    processes = []
    
    print(f"Starting {args.num_gpus} workers. Mode: {args.mode}")

    # Cleanup old output files to ensure we don't read stale data
    # Only cleanup if generating, or maybe always? 
    # If recovering, we write to output_gpu_X.jsonl too, so we should clean them.
    for i in range(args.num_gpus):
        out_file = f"output_gpu_{i}.jsonl"
        if os.path.exists(out_file):
            os.remove(out_file)

    for i in range(args.num_gpus):
        p = multiprocessing.Process(
            target=worker_process,
            args=(i, i, iters_per_worker, args.model, args.mock, args.mode, args.input_file, args.k)
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
                    file_content = f.read().strip()
                    if not file_content: continue
                    
                    decoder = json.JSONDecoder()
                    pos = 0
                    while pos < len(file_content) and count < 10:
                        try:
                            # Skip whitespace
                            while pos < len(file_content) and file_content[pos].isspace():
                                pos += 1
                            if pos >= len(file_content): break
                            
                            data, next_pos = decoder.raw_decode(file_content, pos)
                            pos = next_pos
                            
                            out.write(f"--- Sample {count+1} ---\n")
                            out.write(f"Story: {data['story']['text']}\n")
                            out.write(f"Hidden Event: {data['gold_semantics']['hidden_event']}\n")
                            out.write(f"Protagonist: {data['gold_semantics']['protagonist_name']}\n")
                            out.write(f"Dialogue: {json.dumps(data['dialogue']['turns'], indent=2)}\n")
                            
                            if 'recovery' in data and data['recovery']:
                                out.write(f"Guesses: {json.dumps(data['recovery']['guesses'], indent=2)}\n")
                                out.write(f"Success: {data['recovery']['success']}\n")
                            
                            out.write("\n")
                            count += 1
                        except json.JSONDecodeError:
                            break
            if count >= 10: break
            
    print(f"Manual review samples saved to {manual_review_file}")

if __name__ == "__main__":
    # Ensure spawn method for CUDA compatibility
    multiprocessing.set_start_method("spawn", force=True)
    main()
