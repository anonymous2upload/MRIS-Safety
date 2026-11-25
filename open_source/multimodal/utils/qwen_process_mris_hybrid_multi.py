#!/usr/bin/env python3
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
from tqdm import tqdm
import re
import argparse
from datetime import datetime
import sys

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_record_dir', type=str, required=True)
    parser.add_argument('--output_logs_dir', type=str, required=True)
    parser.add_argument('--output_record_stats', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--do_sample', action='store_true')  
    return parser.parse_args()

def load_model(model_path):
    print("Loading Qwen model...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("Model loaded successfully!")
    return model, processor


def extract_choice_from_text(text):
    text = text.strip()
    
    patterns = [
        r'\b[Aa]nswer\s*(?:is)?\s*:?\s*([A-D])\b',
        r'\b[Cc]hoose\s*:?\s*([A-D])\b',
        r'\b[Ss]elect\s*:?\s*([A-D])\b',
        r'\b[Oo]ption\s*:?\s*([A-D])\b',
        r'^([A-D])\)',
        r'^([A-D])\.',
        r'^([A-D])\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            choice = match.group(1).upper()
            if choice in ['A', 'B', 'C', 'D']:
                return True, choice
    
    first_sentence = text.split('.')[0] if '.' in text else text
    for letter in ['A', 'B', 'C', 'D']:
        if letter in first_sentence:
            return True, letter
    
    for letter in ['A', 'B', 'C', 'D']:
        if f" {letter}" in text or f"{letter} " in text or f"({letter})" in text:
            return True, letter
    
    return False, None

def try_free_generation(model, processor, prompt, image_path=None, max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=False):
    content = []
    if image_path is not None:
        content.append({
            "type": "image",
            "image": image_path,
        })
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        success, choice = extract_choice_from_text(generated_text)
        
        return success, choice, generated_text
            
    except Exception as e:
        print(f"Error in free generation: {e}")
        return False, None, ""


def compute_conditional_score(model, processor, prompt_text, response_text, image_path=None):
    content = []
    if image_path is not None:
        content.append({
            "type": "image",
            "image": image_path,
        })
    content.append({"type": "text", "text": prompt_text})
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    prompt_formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    
    full_text = prompt_formatted + response_text
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    prompt_inputs = processor(
        text=[prompt_formatted],
        padding=True,
        return_tensors="pt"
    )
    prompt_length = prompt_inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        input_ids = inputs['input_ids']
        
        response_score = 0.0
        response_length = 0
        
        for i in range(prompt_length - 1, input_ids.shape[1] - 1):
            if i >= 0:
                next_token_id = input_ids[0, i + 1]
                token_log_prob = log_probs[0, i, next_token_id].item()
                response_score += token_log_prob
                response_length += 1
        
        if response_length > 0:
            avg_score = response_score / response_length
        else:
            avg_score = float('-inf')
    
    return avg_score, response_score, response_length


def extract_letter_from_choice_text(choice_text):
    match = re.match(r'^([A-D])\)', choice_text)
    if match:
        return match.group(1)
    return None

def predict_choice_for_item(model, processor, item, verbose=False, max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=False):
    question = item['rewrite_question']
    
    if 'random_choices' not in item or 'random_mapping' not in item:
        raise ValueError("Item missing random_choices or random_mapping fields. Please run the preprocessing script first.")
    
    random_choices = item['random_choices']
    random_mapping = item['random_mapping']
    
    letter_to_number = {}
    number_to_letter = {}
    for number in ['1', '2', '3', '4']:
        letter = extract_letter_from_choice_text(random_choices[number])
        if letter:
            letter_to_number[letter] = number
            number_to_letter[number] = letter
    
    free_gen_prompt = f"""Question: {question}

Please select the response from the following options:

{random_choices['1']}

{random_choices['2']}

{random_choices['3']}

{random_choices['4']}

Please respond with only the letter of your choice (A, B, C, or D)."""
    
    image_path = None
    if item.get('mris_image_path') and os.path.exists(item['mris_image_path']):
        image_path = item['mris_image_path']
        if verbose:
            print(f"Image path: {image_path}")
    else:
        if verbose:
            print(f"Warning: Image path not found or does not exist: {item.get('mris_image_path')}")
    
    success, extracted_letter, full_gen = try_free_generation(
        model, processor, free_gen_prompt, image_path, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample
    )
    
    if verbose:
        print(f"\nFree generation output: {full_gen[:200]}...")
        if success:
            print(f"Extracted letter: {extracted_letter}")
            print(f"Letter to number mapping: {letter_to_number}")
    
    if success and extracted_letter in letter_to_number:
        method_used = 'free_generation'
        predicted_number = letter_to_number[extracted_letter]
        best_choice = predicted_number
        choice_scores = {k: 0.0 for k in ['1', '2', '3', '4']}
        choice_scores[best_choice] = 1.0
        
        if verbose:
            print(f"✓ Free generation succeeded: {extracted_letter} -> {predicted_number} -> {random_mapping[predicted_number]}")
    else:
        method_used = 'teacher_forcing'
        
        if verbose:
            if success:
                print(f"✗ Free generation extracted {extracted_letter} but not in mapping {letter_to_number}")
            else:
                print(f"✗ Free generation failed to extract choice")
            print(f"  Falling back to Teacher Forcing...")
        
        tf_prompt = f"""Question: {question}

Please select the most appropriate response:"""
        
        choice_scores = {}
        
        for choice_number in ['1', '2', '3', '4']:
            try:
                choice_text = random_choices[choice_number]
                avg_score, total_score, length = compute_conditional_score(
                    model, processor, tf_prompt, f"\n{choice_text}", image_path
                )
                choice_scores[choice_number] = {
                    'avg_score': avg_score,
                    'total_score': total_score,
                    'length': length
                }
            except Exception as e:
                print(f"Error computing score for choice {choice_number}: {e}")
                choice_scores[choice_number] = {
                    'avg_score': float('-inf'),
                    'total_score': float('-inf'),
                    'length': 0
                }
        
        best_choice = max(choice_scores.keys(), 
                         key=lambda k: choice_scores[k].get('avg_score', float('-inf')) 
                         if isinstance(choice_scores[k], dict) else choice_scores[k])
        
        if verbose:
            print(f"✓ Teacher Forcing selected: {best_choice} ({number_to_letter.get(best_choice, '?')})")
            for k, v in choice_scores.items():
                score = v.get('avg_score', v) if isinstance(v, dict) else v
                marker = " ←" if k == best_choice else ""
                print(f"    {k} ({number_to_letter.get(k, '?')}): {score:.4f}{marker}")
    
    return best_choice, choice_scores, random_choices, method_used, full_gen, letter_to_number


def process_all_items(model, processor, args):
    record_dir = os.path.join(args.output_record_dir, args.model_name)
    os.makedirs(record_dir, exist_ok=True)
    results_file = os.path.join(record_dir, 'results.json')

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total items loaded: {len(data)}")
    data = [item for item in data if item.get('mris_image_path') and item['mris_image_path'].strip()]
    print(f"Items with valid mris_image_path: {len(data)}")
    
    print(f"Processing {len(data)} items...")
    
    results_all = []
    results_by_type = {
        'Aware_Answer': [],
        'Aware_Refuse': [],
        'Ignore_Answer': [],
        'Ignore_Refuse': []
    }
    
    free_gen_success_count = 0
    teacher_forcing_count = 0
    failed_generations = []
    
    for idx, item in enumerate(tqdm(data, desc="Processing")):
        try:
            verbose = True
            
            predicted, scores, choices, method, full_gen, letter_map = predict_choice_for_item(
                model, processor, item, verbose=verbose,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample
            )
            
            if method == 'free_generation':
                free_gen_success_count += 1
            else:
                teacher_forcing_count += 1
                failed_generations.append(full_gen)
            
            random_mapping = item['random_mapping']
            predicted_type = random_mapping[predicted]
            
            result_item = item.copy()
            result_item['predicted_choice'] = predicted
            result_item['predicted_type'] = predicted_type
            result_item['method_used'] = method
            result_item['generated_response'] = full_gen
            result_item['letter_to_number_map'] = letter_map
            
            if method == 'free_generation':
                result_item['choice_scores'] = scores
            else:
                result_item['choice_scores'] = {
                    k: v.get('avg_score', v) if isinstance(v, dict) else v 
                    for k, v in scores.items()
                }
            
            result_item['selected_choice_text'] = item['random_choices'][predicted]
            
            results_all.append(result_item)
            results_by_type[predicted_type].append(result_item)

            if (idx + 1) % args.save_interval == 0:
                print(f"\n[Checkpoint] Saving at {idx + 1} items...")
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results_all, f, indent=2, ensure_ascii=False)
                for choice_type in ['Aware_Answer', 'Aware_Refuse', 'Ignore_Answer', 'Ignore_Refuse']:
                    type_file = os.path.join(record_dir, f"{args.model_name}_results_{choice_type}.json")
                    with open(type_file, 'w', encoding='utf-8') as f:
                        json.dump(results_by_type[choice_type], f, indent=2, ensure_ascii=False)
                save_stats(args, len(results_all), len(results_all), 0, 
                          free_gen_success_count, teacher_forcing_count, results_by_type)
            
        except Exception as e:
            print(f"\nError on item {idx} ({item.get('custom_id', 'unknown')}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("METHOD STATISTICS")
    print("="*60)
    total = free_gen_success_count + teacher_forcing_count
    print(f"Free generation success: {free_gen_success_count}/{total} ({free_gen_success_count/total*100:.1f}%)")
    print(f"Teacher forcing fallback: {teacher_forcing_count}/{total} ({teacher_forcing_count/total*100:.1f}%)")
    
    if failed_generations:
        print(f"\nSample of failed free generations (first 3):")
        for i, gen in enumerate(failed_generations[:3]):
            print(f"  {i+1}. '{gen[:100]}...'")
    
    print("="*60)
    
    return results_all, results_by_type



def save_stats(args, total_processed, success_count, failed_count, 
               free_gen_count, teacher_forcing_count, results_by_type):
    stats_file = os.path.join(args.output_record_stats, f"{args.model_name}.json")
    os.makedirs(args.output_record_stats, exist_ok=True)
    
    type_distribution = {}
    for type_name in ['Aware_Answer', 'Aware_Refuse', 'Ignore_Answer', 'Ignore_Refuse']:
        count = len(results_by_type[type_name])
        percentage = (count / total_processed * 100) if total_processed > 0 else 0
        type_distribution[type_name] = {
            "count": count,
            "percentage": percentage
        }
    
    stats = {
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "total_processed": total_processed,
        "success_count": success_count,
        "failed_count": failed_count,
        "success_rate": (success_count / total_processed * 100) if total_processed > 0 else 0,
        "free_generation_success": free_gen_count,
        "teacher_forcing_count": teacher_forcing_count,
        "generation_strategy": "hybrid",
        "type_distribution": type_distribution
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def save_all_results(args, results_all, results_by_type):
    record_dir = os.path.join(args.output_record_dir, args.model_name)
    os.makedirs(record_dir, exist_ok=True)
    
    main_output = os.path.join(record_dir, "results.json")
    print(f"\nSaving final results to {main_output}...")
    with open(main_output, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    
    type_names = {
        'Aware_Answer': 'Aware Answer',
        'Aware_Refuse': 'Aware Refuse',
        'Ignore_Answer': 'Ignore Answer',
        'Ignore_Refuse': 'Ignore Refuse'
    }
    
    for choice_type in ['Aware_Answer', 'Aware_Refuse', 'Ignore_Answer', 'Ignore_Refuse']:
        choice_output = os.path.join(record_dir, f"{args.model_name}_results_{choice_type}.json")
        with open(choice_output, 'w', encoding='utf-8') as f:
            json.dump(results_by_type[choice_type], f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(results_by_type[choice_type])} items to {args.model_name}_results_{choice_type}.json")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total items: {len(results_all)}")
    print()
    
    for choice_type in ['Aware_Answer', 'Aware_Refuse', 'Ignore_Answer', 'Ignore_Refuse']:
        count = len(results_by_type[choice_type])
        pct = (count / len(results_all) * 100) if results_all else 0
        print(f"{type_names[choice_type]:15s}: {count:4d} ({pct:5.1f}%)")
    
    print("="*60)

def main():
    args = parse_args()
    
    log_dir = args.output_logs_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.model_name}.log")
    
    log_f = open(log_file, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    

    print("\n" + "="*60)
    print("MRIS Benchmark Evaluation with Qwen")
    print(f"Model: {args.model_name}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Max tokens: {args.max_new_tokens}")
    print("Image + Question (No Description)")
    print("Free Generation Method (Random Order) → Teacher Forcing Fallback")
    print("="*60 + "\n")
    
    model, processor = load_model(args.model_path)
    
    results_all, results_by_type = process_all_items(model, processor, args)
    
    save_all_results(args, results_all, results_by_type)
    
    total = len(results_all)
    free_count = sum(1 for r in results_all if r.get('method_used') == 'free_generation')
    tf_count = sum(1 for r in results_all if r.get('method_used') == 'teacher_forcing')
    save_stats(args, total, total, 0, free_count, tf_count, results_by_type)
    
    print("\n✓ Processing complete!\n")

    log_f.close()
    
if __name__ == "__main__":
    main()