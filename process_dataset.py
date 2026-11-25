#!/usr/bin/env python3
import json
import argparse
import os
import glob
from pathlib import Path
from typing import Dict, List, Any


def replace_path_prefix(data: List[Dict[str, Any]], new_prefix: str) -> List[Dict[str, Any]]:
    path_fields = ['mris_image_path', 'safe_image_path']
    
    for record in data:
        for field in path_fields:
            if field in record and record[field]:
                old_path = record[field]
                path_parts = Path(old_path).parts
                
                image_dirs = ['unsafe_image_compress', 'safe_image_compress']
                relative_parts = None
                
                for img_dir in image_dirs:
                    if img_dir in path_parts:
                        dir_idx = path_parts.index(img_dir)
                        relative_parts = path_parts[dir_idx:]
                        break
                
                if relative_parts:
                    relative_path = os.path.join(*relative_parts)
                    record[field] = os.path.join(new_prefix, relative_path)
                else:
                    relative_path = os.path.join(*path_parts[-2:])
                    record[field] = os.path.join(new_prefix, relative_path)
    
    return data

def replace_path_prefix_vlm(data: List[Dict[str, Any]], new_prefix: str) -> List[Dict[str, Any]]:
    path_fields = ['image_path']
    old_prefix = '/DATASET-PREFIX-PATH'
    old_prefix_path = Path(old_prefix)
    new_prefix_path = Path(new_prefix)
    
    for record in data:
        for field in path_fields:
            if field in record and record[field]:
                current_path = Path(record[field])
                
                try:
                    relative_path = current_path.relative_to(old_prefix_path)
                    record[field] = str(new_prefix_path / relative_path)
                except ValueError:
                    pass
    
    return data


def find_dataset_files(script_dir: str) -> List[str]:
    json_files = glob.glob(os.path.join(script_dir, '*.json'))
    
    json_files.sort()
    
    return json_files


def process_mrisbench_dataset(input_file: str, output_file: str, new_prefix: str):
    print(f"Processing {os.path.basename(input_file)}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} records")
    
    updated_data = replace_path_prefix(data, new_prefix)
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to {os.path.basename(output_file)}")


def process_vlm_dataset(input_file: str, output_file: str, new_prefix: str):
    print(f"Processing {os.path.basename(input_file)}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} records")
    
    updated_data = replace_path_prefix_vlm(data, new_prefix)
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to {os.path.basename(output_file)}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess JSON datasets by replacing path prefixes')
    
    parser.add_argument('--prefix', help='New path prefix to use for image paths')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Path Preprocessing")
    print("=" * 60)
    print(f"New prefix: {args.prefix}")
    print()
    
    mris_json_files = [
        './data/MRISbench/MRISbench_original.json',
        './data/MRISbench/MRISbench_test_original.json'
    ]

    mris_prefix = os.path.join(args.prefix, 'data','MRISbench')
    for json_file in mris_json_files:
        output_file = json_file.replace('_original', '')
        process_mrisbench_dataset(json_file, output_file, mris_prefix)
    
    vlm_json_files = [
        './data/benign_original.json',
        './data/malicious_original.json'
    ]

    vlm_prefix = os.path.join(args.prefix, 'data')
    for json_file in vlm_json_files:
        output_file = json_file.replace('_original', '')
        process_vlm_dataset(json_file, output_file, vlm_prefix)
    
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    main()
