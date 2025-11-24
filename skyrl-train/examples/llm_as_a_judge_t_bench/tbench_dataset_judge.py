# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the T-Bench dataset to parquet format
"""

import argparse
import json
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="examples/llm_as_a_judge_t_bench/tb_sonnet_4_5_success_messages_single_turn_rubrics.json")
    parser.add_argument("--output_dir", default="examples/llm_as_a_judge_t_bench")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "tbench"

    # Load the JSON file
    with open(args.input_file, 'r') as f:
        raw_data = json.load(f)

    # Convert to list format for datasets
    processed_data = []
    for idx, item in enumerate(raw_data):
        # Extract the user prompt from messages (remove last message and format rest as string)
        # or use the instructions from task_info
        if 'messages' in item and len(item['messages']) > 0:
            # Remove the last message and format the rest into a single string
            messages_without_last = item['messages'][:-1]
            
            # Format messages into a single string using standard OpenAI format
            formatted_content = ""
            for msg in messages_without_last:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                # Capitalize role for standard OpenAI formatting (User/Assistant)
                role_formatted = role.capitalize()
                formatted_content += f"{role_formatted}: {content}\n\n"
            
            # Remove trailing newlines
            formatted_content = formatted_content.rstrip()
            
            prompt = [
                {
                    "role": "user",
                    "content": formatted_content,
                }
            ]
        else:
            # Fallback to using instructions as a single user message
            prompt = [
                {
                    "role": "user",
                    "content": item['task_info']['instructions'],
                }
            ]
        
        data = {
            "data_source": data_source,
            "prompt": prompt,
            "env_class": "llm_as_a_judge_t_bench",
            "reward_spec": {
                "method": "rule",
                "ground_truth": item['task_info'].get('solution', ''),
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "answer": item['task_info'].get('solution', ''),
                "question": item['task_info']['instructions'],
                "task": item['task'],
                "rubrics": item['task_info'].get('rubrics', []),
                "rubrics_reasoning": item['task_info'].get('rubrics_reasoning', ''),
            },
        }
        processed_data.append(data)

    # Create datasets from the processed data
    dataset = datasets.Dataset.from_list(processed_data)
    
    # Split into train and validation (90/10 split)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    # Update split labels
    def update_split(example, split_name):
        example['extra_info']['split'] = split_name
        return example
    
    train_dataset = train_dataset.map(lambda x: update_split(x, 'train'))
    val_dataset = val_dataset.map(lambda x: update_split(x, 'test'))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
    
    print(f"Processed {len(processed_data)} items from T-Bench")
    print(f"Train set: {len(train_dataset)} items")
    print(f"Validation set: {len(val_dataset)} items")
    print(f"Output saved to: {output_dir}")
