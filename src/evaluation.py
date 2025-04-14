import torch
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from jinja2 import Template
from torch.nn import CrossEntropyLoss
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@torch.no_grad()
def generate(prompts, model, tokenizer, template):

    legal_prompt = "You are a professional legal consultant. Please provide accurate and professional legal advice based on the following questions, ensuring the precision of legal terminology and the completeness of responses."

    new_prompts = [f"{legal_prompt}\n\n{prompt}" for prompt in prompts]
    batch_size = 64
    generated_texts = []
    
    total_batches = (len(new_prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(new_prompts), batch_size), desc="Generating batches", total=total_batches, unit="batch"):
        batch = new_prompts[i:i + batch_size]
        model_inputs = [template.render(messages=[{"role": "user", "content": prompt}], bos_token=tokenizer.bos_token,
                                      add_generation_prompt=True) for prompt in batch]
        input_ids = tokenizer(
            model_inputs, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512).to("cuda:0")

        outputs = model.generate(
            input_ids.input_ids,
            attention_mask=input_ids.attention_mask,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50
        )

        for output in outputs:
            generated_ids = output[input_ids.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            generated_texts.append(generated_text)

    return generated_texts


def calculate_bleu(reference, hypothesis):
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)

    smoothie = SmoothingFunction().method1
    if len(hypothesis_tokens) > 0:
        bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, 
                                   weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=smoothie)
    else:
        bleu_score = 0
    return bleu_score


def calculate_ppl(model, tokenizer, prompt, answer, index=-1):
    try:
        if tokenizer.eos_token:
            full_text = prompt + tokenizer.eos_token + answer + tokenizer.eos_token
        else:
            full_text = prompt + answer

        max_len = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 1024

        full_inputs = tokenizer(
            full_text,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True
        ).to(model.device)

        prompt_inputs = tokenizer(
            prompt + (tokenizer.eos_token if tokenizer.eos_token else ""),
            return_tensors="pt",
            max_length=max_len,
            truncation=True
        ).to(model.device)
        
        full_input_ids = full_inputs["input_ids"]
        prompt_input_ids = prompt_inputs["input_ids"]

        if full_input_ids.shape[1] <= 1:
             if index >= 0: print(f"样本 {index}: Full sequence tokenization too short (<=1). Prompt: {prompt[:30]} Answer: {answer[:30]}")
             return None
        prompt_length = prompt_input_ids.shape[1]

        if prompt_length >= full_input_ids.shape[1]:
            if index >= 0: print(f"样本 {index}: Prompt length ({prompt_length}) >= full length ({full_input_ids.shape[1]}) after tokenization/truncation. Skipping PPL.")
            return None

        with torch.no_grad():
            outputs = model(**full_inputs)
            logits = outputs.logits

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            if index >= 0: print(f"样本 {index}: Logits contain NaN or Inf.")
            return None

        labels = full_input_ids.clone()
        labels[:, :prompt_length] = -100
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fn = CrossEntropyLoss()
        mean_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))

        if torch.isnan(mean_loss) or torch.isinf(mean_loss):
             if index >= 0: print(f"样本 {index}: Mean loss is NaN or Inf.")
             return None

        ppl = torch.exp(mean_loss).item()

        if np.isnan(ppl) or np.isinf(ppl):
             if index >= 0: print(f"样本 {index}: Final PPL is NaN or Inf. Mean Loss: {mean_loss.item()}")
             return None
        return ppl

    except Exception as e:
        print(f"计算PPL时出错 (样本 {index}): {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model(model, tokenizer, dataset, template):

    prompts = dataset["instruction"]
    references = dataset['output']
    print("Generating model answers...")
    model_answers = generate(prompts, model, tokenizer, template)

    total = len(prompts) # Base total on input prompts/references
    print(f"评估 {total} 个样本...")

    # Ensure model_answers has the same length as prompts/references
    if len(model_answers) != total:
        print(f"Warning: Number of generated answers ({len(model_answers)}) does not match number of prompts ({total}). Evaluating based on {min(total, len(model_answers))} pairs.")
        total = min(total, len(model_answers)) # Adjust total to the minimum length

    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Store results per sample
    results_per_sample = []

    for i in tqdm(range(total), desc="Evaluating samples"):
        prompt = prompts[i]
        reference = references[i]
        model_answer = model_answers[i]

        bleu_score = calculate_bleu(reference, model_answer)
        rouge_scores = rouge_scorer_instance.score(reference, model_answer)
        rouge_l = rouge_scores['rougeL'].fmeasure
        ppl = calculate_ppl(model, tokenizer, prompt, reference, index=i) # Pass reference for PPL

        results_per_sample.append({
            "index": i,
            "prompt": prompt,
            "reference": reference,
            "model_answer": model_answer,
            "bleu": bleu_score,
            "rouge_l": rouge_l,
            "ppl": ppl # Store the result, which might be None
        })

    # Calculate averages from the collected results
    bleu_scores = [r['bleu'] for r in results_per_sample]
    rouge_l_scores = [r['rouge_l'] for r in results_per_sample]
    # Filter valid PPL values for averaging
    valid_ppl_values = [r['ppl'] for r in results_per_sample if r['ppl'] is not None and not np.isnan(r['ppl']) and not np.isinf(r['ppl'])]

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge_l = np.mean(rouge_l_scores) if rouge_l_scores else 0
    avg_ppl = np.mean(valid_ppl_values) if valid_ppl_values else None

    print(f"\n--- Evaluation Results ---")
    print(f"平均 BLEU 分数: {avg_bleu:.4f}")
    print(f"平均 ROUGE-L 分数: {avg_rouge_l:.4f}")
    if avg_ppl is not None:
        print(f"平均困惑度 (PPL): {avg_ppl:.2f} (基于 {len(valid_ppl_values)}/{total} 个有效样本)")
    else:
        print(f"未计算或所有样本的困惑度 (PPL) 无效 (基于 {total} 个样本)")

    # Print examples safely using results_per_sample
    num_examples_to_print = min(3, total)
    print(f"\n--- 示例 (前 {num_examples_to_print}) ---")
    for i in range(num_examples_to_print):
        sample_result = results_per_sample[i] # Get the dictionary for sample i
        print(f"\n示例 {i+1}:")
        print(f"问题: {sample_result['prompt'][:200]}...")
        print(f"参考回答: {sample_result['reference'][:200]}...")
        print(f"模型回答: {sample_result['model_answer'][:200]}...")
        print(f"BLEU: {sample_result['bleu']:.4f}, ROUGE-L: {sample_result['rouge_l']:.4f}")
        # Check if PPL is valid before printing
        if sample_result['ppl'] is not None and not np.isnan(sample_result['ppl']) and not np.isinf(sample_result['ppl']):
            print(f"PPL: {sample_result['ppl']:.2f}")
        else:
            print("PPL: 计算失败或无效") # Indicate failure for this specific sample

    return {
        "samples": total,
        "bleu": avg_bleu,
        "rouge_l": avg_rouge_l,
        "ppl": avg_ppl,
        "valid_ppl_samples": len(valid_ppl_values) # Return count of valid PPLs
    }


def print_evaluation_comparison(pre_finetune_metrics, post_finetune_metrics):
    print("\n=== 微调效果对比 ===")
    print(f"样本数量: {pre_finetune_metrics['samples']}")
    
    print("\n指标对比:")
    print(f"BLEU 分数      - 微调前: {pre_finetune_metrics['bleu']:.4f}, 微调后: {post_finetune_metrics['bleu']:.4f}, " 
          f"提升: {post_finetune_metrics['bleu'] - pre_finetune_metrics['bleu']:.4f}")
    
    print(f"ROUGE-L 分数   - 微调前: {pre_finetune_metrics['rouge_l']:.4f}, 微调后: {post_finetune_metrics['rouge_l']:.4f}, "
          f"提升: {post_finetune_metrics['rouge_l'] - pre_finetune_metrics['rouge_l']:.4f}")
    
    if pre_finetune_metrics['ppl'] is not None and post_finetune_metrics['ppl'] is not None:
        print(f"困惑度(PPL)    - 微调前: {pre_finetune_metrics['ppl']:.2f}, 微调后: {post_finetune_metrics['ppl']:.2f}, "
              f"降低: {pre_finetune_metrics['ppl'] - post_finetune_metrics['ppl']:.2f}")
    else:
        print("困惑度(PPL) - 未计算")
    
    print("\n请查看上方输出的示例回答，对比微调效果")


