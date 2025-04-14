import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from jinja2 import Template
import argparse
import json
import os
import gc
import numpy as np
import sys
from src.utils import print_rank0
# Add current directory to path to find local modules
sys.path.append(os.path.dirname(__file__))

try:
    from preprocess import prepare_test_data
    from evaluation import evaluate_model
except ImportError as e:
    print_rank0(f"Error importing local modules: {e}")
    print_rank0("Ensure preprocess.py and evaluation.py are in the same directory or accessible.")
    sys.exit(1)

def load_model_for_eval(base_model_path, adapter_path=None, quantization_bits=0, compute_dtype_str="bf16", merge_adapter=False):
    """Loads model for evaluation, potentially applying an adapter."""
    print_rank0(f"--- Loading Model Configuration ---")
    print_rank0(f"Base Model Path: {base_model_path}")
    print_rank0(f"Adapter Path: {adapter_path if adapter_path else 'N/A'}")
    print_rank0(f"Load Base Quantization: {quantization_bits if quantization_bits > 0 else 'Full Precision'}")
    if quantization_bits == 0:
        print_rank0(f"Compute Dtype: {compute_dtype_str}")
    print_rank0(f"Merge Adapter Before Eval: {merge_adapter}")
    print_rank0(f"---------------------------------")

    bnb_config = None
    load_params = {}
    compute_dtype = torch.float32

    if quantization_bits == 4:
        compute_dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )
        load_params["quantization_config"] = bnb_config
        print_rank0("Loading base model in 4-bit quantization...")
    elif quantization_bits == 8:
        compute_dtype = torch.float16
        load_params["load_in_8bit"] = True
        print_rank0("Loading base model in 8-bit quantization...")
    else:
        if compute_dtype_str == "bf16" and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            print_rank0("Using bfloat16 compute dtype.")
        elif compute_dtype_str == "fp16":
            compute_dtype = torch.float16
            print_rank0("Using float16 compute dtype.")
        else:
            compute_dtype = torch.float32
            print_rank0("Using float32 compute dtype.")
        load_params["torch_dtype"] = compute_dtype
        print_rank0(f"Loading base model in full precision ({compute_dtype})...")

    try:
        print_rank0("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto", # Automatically distribute across available GPUs
            **load_params
        )
        print_rank0("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Add pad token if missing
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                print_rank0("Tokenizer missing pad token, setting to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                print_rank0("Warning: Tokenizer missing pad_token and eos_token. Adding a default pad token '<PAD>'.")
                tokenizer.add_special_tokens({'pad_token': '<PAD>'})
                if quantization_bits == 0:
                    print_rank0("Resizing token embeddings for new pad token.")
                    model.resize_token_embeddings(len(tokenizer))

        if adapter_path:
            if not os.path.exists(adapter_path):
                 raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
            print_rank0(f"Loading LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)
            print_rank0("Adapter loaded.")
            if merge_adapter:
                print_rank0("Merging adapter into base model...")
                model = model.merge_and_unload()
                print_rank0("Adapter merged and unloaded.")

        model.eval() # Set model to evaluation mode
        print_rank0("Model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        print_rank0(f"Error loading model/tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned language model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base pre-trained model.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the LoRA adapter checkpoint (directory containing adapter_config.json).")
    parser.add_argument("--quantization_bits", type=int, default=0, choices=[0, 4, 8], help="Quantization bits for loading the BASE model (0 for full precision). Default=0.")
    parser.add_argument("--compute_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Compute dtype if quantization_bits=0. Default=bf16.")
    parser.add_argument("--merge_adapter", action='store_true', help="Merge the adapter into the base model before evaluation.")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Path to save evaluation metrics as JSON.")
    args = parser.parse_args()

    # --- Load Model and Tokenizer ---
    model, tokenizer = load_model_for_eval(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        quantization_bits=args.quantization_bits,
        compute_dtype_str=args.compute_dtype,
        merge_adapter=args.merge_adapter
    )

    if model is None or tokenizer is None:
        print_rank0("Exiting due to model loading failure.")
        sys.exit(1)

    # --- Prepare Dataset and Template ---
    print_rank0("Loading test dataset...")
    try:
        test_dataset = prepare_test_data() # Assuming this function loads your test data
        print_rank0(f"Loaded {len(test_dataset)} test samples.")
    except Exception as e:
        print_rank0(f"Error loading test data using prepare_test_data(): {e}")
        sys.exit(1)

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            template = Template(tokenizer.chat_template)
            print_rank0("Using chat template from tokenizer.")
        except Exception as e:
            print_rank0(f"Warning: Error processing tokenizer.chat_template: {e}. Proceeding without template.")
            template = None
    else:
        print_rank0("Warning: Tokenizer does not have a chat_template defined. Generation might be suboptimal.")
        template = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_rank0("Starting evaluation...")
    try:
        evaluation_metrics = evaluate_model(model, tokenizer, test_dataset, template)
    except Exception as e:
        print_rank0(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print_rank0(f"Saving evaluation results to {args.output_file}...")
    serializable_metrics = {}
    for k, v in evaluation_metrics.items():
        if isinstance(v, (np.float32, np.float64)):
            serializable_metrics[k] = float(v)
        elif isinstance(v, (np.int32, np.int64, np.bool_)):
             serializable_metrics[k] = int(v) if not isinstance(v, np.bool_) else bool(v)
        elif v is None or isinstance(v, (str, int, float, bool, list, dict)):
             serializable_metrics[k] = v
        else:
             print_rank0(f"Warning: Skipping non-serializable metric '{k}' of type {type(v)}")
             serializable_metrics[k] = str(v)

    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)
        print_rank0("Results saved successfully.")
    except Exception as e:
        print_rank0(f"Error saving results to JSON: {e}")

    print_rank0("Evaluation finished.")

if __name__ == "__main__":
    main()