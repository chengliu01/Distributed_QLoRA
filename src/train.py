import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from jinja2 import Template
import math
import os
from preprocess import prepare_data, prepare_test_data, InstructDataset, DataCollatorForSupervisedDataset
from src.evaluation import evaluate_model, print_evaluation_comparison
import argparse
from src.utils import print_rank0, str2bool, print_trainable_parameters, clear_memory


def load_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")
    parser.add_argument("--evaluate", type=str2bool, nargs='?', const=True, default=False, 
                        help="Whether to evaluate the model before and after fine-tuning (true/false).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--lora_r", type=str, required=True, help="LoRA rank parameter.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the fine-tuned model.")

    parser.add_argument("--use_quantization", type=str2bool, nargs='?', const=True, default=False, 
                        help="Whether use quantized (true/false)")
    parser.add_argument("--quantization_bits", type=int, default=4, choices=[4, 8], 
                        help=" (4 or 8 bit), only used when use_quantization is true.")
    parser.add_argument("--lora_target_modules", type=str, default=None, 
                        help="LoRA target modules, separated by commas. If None, all modules will be used.")

    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer to use.")
    parser.add_argument("--gradient_checkpointing", type=str2bool, nargs='?', const=True, default=False, 
                        help="Whether to use gradient checkpointing (true/false).")

    args = parser.parse_args()
    return args


def load_model_and_tokenizer(model_path, use_quantization=False, quantization_bits=4):
    params = {}
    device_map_option = None
    is_distributed = 'LOCAL_RANK' in os.environ
    local_rank = int(os.environ['LOCAL_RANK']) if is_distributed else 0

    if use_quantization:
        if quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            params["quantization_config"] = bnb_config
            if is_distributed:
                device_map_option = {'': f'cuda:{local_rank}'}
            print(f"Rank {local_rank}: Loading model in 4-bit quantization. Device map: {device_map_option}")
        elif quantization_bits == 8:
            params["load_in_8bit"] = True
            if is_distributed:
                device_map_option = {'': f'cuda:{local_rank}'}
            print(f"Rank {local_rank}: Loading model in 8-bit quantization. Device map: {device_map_option}")
    else:
        params["torch_dtype"] = torch.float16
        print(f"Loading model without quantization, using dtype: {params['torch_dtype']}.")
    
    try:
        if device_map_option:
            # 适合多卡，每个卡片加载一个完整的模型，各自进行梯度计算
            # 此时的全局批次大小就是进程指定的批次大小 * 进程数目 = per_device_train_batch_size * num_gpus 
            params["device_map"] = device_map_option
        else:
            # 适合单卡，或者多卡共享模型， 模型会被“拆解”到多张卡上
            # 此时的全局批次大小就是这个单进程指定的批次大小 = per_device_train_batch_size
            params["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **params,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 
        return model, tokenizer
    except Exception as e:
        print_rank0(f"Error loading model: {e}")
        return None, None


def preprocess_model(model, lora_r=16, target_modules=None):
    config_dict = {
        "r": lora_r,
        "lora_alpha": 2*lora_r,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    
    if target_modules:
        if isinstance(target_modules, str):
            target_modules = [m.strip() for m in target_modules.split(',')]
        config_dict["target_modules"] = target_modules
    
    config = LoraConfig(**config_dict)
    model = get_peft_model(model, config)
    return model
    

def main():
    args = load_args()

    model, tokenizer = load_model_and_tokenizer(
        model_path = args.model_path, 
        use_quantization=args.use_quantization,
        quantization_bits = args.quantization_bits
    )

    train_dataset, val_dataset = prepare_data()
    if args.evaluate:
        test_dataset = prepare_test_data()
        template = Template(tokenizer.chat_template)
        print_rank0("Evaluating model before fine-tuing...")
        pre_finetune_metrics = evaluate_model(model, tokenizer, test_dataset, template)
        clear_memory()
    
    # whether to use quantization
    if args.use_quantization:
        model = prepare_model_for_kbit_training(model)

    model = preprocess_model(model, lora_r=int(args.lora_r), target_modules=args.lora_target_modules)

    # whether to use gradient_checkpointing
    if args.gradient_checkpointing:
        print_rank0("Enabling input require grads for gradient checkpointing with PEFT...")
        model.enable_input_require_grads()

    print_trainable_parameters(model)
    
    # InstructDataset could be customized
    train_dataset = InstructDataset(train_dataset, tokenizer, args.max_seq_length)
    val_dataset = InstructDataset(val_dataset, tokenizer, args.max_seq_length)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    training_arguments = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim='paged_adamw_32bit',
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=-1,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        fp16=False,
        bf16=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    print_rank0("Evaluating model before training...")
    eval_results_before = trainer.evaluate()
    print_rank0(f"Perplexity: {math.exp(eval_results_before['eval_loss']):.2f}")

    print_rank0("Training model...")
    trainer.train()

    print_rank0("Evaluating model after training...")
    eval_results_final = trainer.evaluate()
    print_rank0(f"Perplexity: {math.exp(eval_results_final['eval_loss']):.2f}, Improvement: {math.exp(eval_results_final['eval_loss']) - math.exp(eval_results_before['eval_loss']):.2f}")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print_rank0(f"Model saved to {output_path}")
    
    # need to define evaluate_model function(Customize it)
    if args.evaluate:
        clear_memory()
        print_rank0("Loading fine-tuning model for evaluation...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, output_path)
        model = model.merge_and_unload()
        model.eval()
        
        print_rank0("Evaluating fine-tuning model...")
        post_finetune_metrics = evaluate_model(model, tokenizer, test_dataset, template)
        print_evaluation_comparison(pre_finetune_metrics, post_finetune_metrics)


if __name__ == "__main__":
    main()