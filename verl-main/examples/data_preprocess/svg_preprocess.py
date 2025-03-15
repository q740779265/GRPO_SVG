import datasets
import argparse

def get_answer(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
        
    else:
        retval = string[idx:right_brace_idx + 1]

    left = "\\boxed{"
    try:
        assert retval[:len(left)] == left, f"unmatched format: {retval}"
        assert retval[-1] == "}", f"unmatched format: {retval}"
        return retval
    except:
        return None

def read_dataset_bespoke(dataset_path: str, save_dir: str='../datasets/bespoke/', remove_none_answer: bool=False):
    print(f"Loading the dataset from {dataset_path}...", flush=True)
    dataset = datasets.load_dataset(dataset_path)
    print(dataset)
    train_dataset = dataset['train']
    # print(train_dataset['conversations'][0]['value'])
    # train_dataset['prompt'] = train_dataset['conversations'][0]['value']

    # Split into train and validation sets


    def make_map_fn(split):
        def process_fn(example, idx):
            question = 'SVG illustration of '+ example['description']
            ground = 'SVG illustration of '+ example['description']
            data = {
                "data_source": 'svg_prompt',
                "prompt": question,
                "ability": "svg",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn
    train_dataset = train_dataset.map(make_map_fn('train'), with_indices=True)
    if remove_none_answer:
        train_dataset = train_dataset.filter(lambda x: x['reward_model']['ground_truth'] is not None)
    # print(train_dataset[0]['prompt'], '\n', train_dataset[0]['final answer'])
    for i in range(20):
        print(train_dataset[i]['reward_model']['ground_truth'])
        print('-'*40)
    # 将数据集储存为parquet格式
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    train_dataset.to_parquet(f"{save_dir}/train.parquet")
    print(f'save train_dataset in {save_dir}/train.parquet')
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='/home/wxf/lzq/twq/verl-main/examples/datasets/svg')
    parser.add_argument('--dataset_path', default='/home/wxf/lzq/data/datasets/SVG@')

    args = parser.parse_args()
    read_dataset_bespoke(dataset_path=args.dataset_path, save_dir=args.save_dir, remove_none_answer=True)