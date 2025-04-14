# QLoRA Fine-tuning and Evaluation

This project aims to provide a comprehensive workflow for fine-tuning pre-trained language models and evaluating their performance. It supports efficient fine-tuning using LoRA or QLoRA technology and offers tools for assessing model performance.

(🥹🥹🥹🥹Mainly because I'm currently working on a course assignment for my graduate studies, which involves QLoRA. I couldn't find any code for single-machine multi-GPU distributed training online, so I wrote the code myself and then organized it.🥹🥹🥹🥹🥹🥹)

## Project Structure
```
.
├── src
│   ├── eval.py           # Evaluation script
│   ├── evaluation.py     # Functions for evaluation
│   ├── preprocess.py     # Data preprocessing script
│   ├── train.py          # Training script
│   └── utils.py          # Utility functions
├── scripts
│   ├── eval.sh           # Shell script for evaluation
│   └── train.sh          # Shell script for training
└── README.md         # Project documentation
```

## Environment Setup
Ensure you have installed the necessary Python libraries. You can use the following command for installation:
```bash
pip install torch transformers peft datasets nltk rouge_score jinja2
```

## Usage Instructions

### Data Preparation
This project uses the `Alignment-Lab-AI/Lawyer-Instruct` dataset by default. You can replace it with your own dataset. Simply modify the `prepare_data_splits` function in the `src/preprocess.py` file and replace it with code to load your own dataset.

### Model Training
Use the `scripts/train.sh` script to train the model:
```bash
bash scripts/train.sh
```
You can modify the parameters in `scripts/train.sh` as needed, such as:
- `LORA_R`: The rank parameter for LoRA.
- `QUANTIZATION_BITS`: The number of quantization bits.
- `USE_QUANTIZATION`: Whether to use quantization.
- `OUTPUT_PATH`: The path to save the fine-tuned model.

### Model Evaluation
Use the `scripts/eval.sh` script to evaluate the model:
```bash
bash scripts/eval.sh
```
Similarly, you can modify the parameters in `scripts/eval.sh` as needed, such as:
- `BASE_MODEL_PATH`: The path to the pre-trained model.
- `ADAPTER_PATH`: The path to the LoRA adapter.
- `QUANTIZATION_BITS`: The number of quantization bits.
- `COMPUTE_DTYPE`: The data type for computation.
- `MERGE_ADAPTER`: Whether to merge the adapter before evaluation.

### Customizing Evaluation Scripts
If you want to use your own evaluation script, you can replace the function implementations in the `src/evaluation.py` file. Ensure that the evaluation functions called in the `src/eval.py` and `src/train.py` files are consistent with your modified functions.

## Code Explanation
### `src/preprocess.py`
- `prepare_data_splits`: Splits the dataset into training, validation, and test sets.
- `prepare_test_data`: Prepares the test dataset.
- `prepare_data`: Prepares the training and validation datasets.
- `preprocess`: Preprocesses the data, including encoding and label processing.

### `src/train.py`
- `load_args`: Loads command-line arguments.
- `load_model_and_tokenizer`: Loads the pre-trained model and tokenizer.
- `preprocess_model`: Applies LoRA technology to the model.
- `main`: The main function responsible for model training and evaluation.

### `src/eval.py`
- `load_model_for_eval`: Loads the model and tokenizer for evaluation.
- `main`: The main function responsible for model evaluation and result saving.

### `src/evaluation.py`
- `generate`: Generates model outputs.
- `calculate_bleu`: Calculates the BLEU score.
- `calculate_ppl`: Calculates the perplexity.
- `evaluate_model`: Evaluates the model's performance.
- `print_evaluation_comparison`: Prints the comparison of evaluation results before and after fine-tuning.

### `src/utils.py`
- `print_rank0`: Prints text only on rank 0 in a distributed setting.
- `str2bool`: Converts a string to a boolean value.
- `print_trainable_parameters`: Prints the number of trainable parameters in the model.
- `clear_memory`: Clears GPU memory and garbage collects.

## Notes
- Ensure that your GPU environment is correctly configured and has sufficient memory to run the model.
- When using quantization, ensure that your hardware supports the corresponding number of quantization bits.
- If you encounter any issues, check the log files for detailed information.

By following the above steps, you can easily use this project for language model fine-tuning and evaluation, and customize it according to your needs. 