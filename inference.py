import argparse
import torch
from modeling_mm import MMForCausalLM
import json


def move_tensors_to_cuda(data_dict, gpu_id):
    if torch.cuda.is_available():
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(f'cuda:{gpu_id}')
    else:
        print("CUDA is not available.")
    return data_dict

def process_prompt(video_path: str, text: str):
    video_dict = {
        "path": video_path,
        "source": "localpath",
        "file_type": 'V'
    }
    return f"<img_start>{json.dumps(video_dict)}<img_end>\nQuestion: {text}\nAnswer:"


def eval_single_node(model, tokenizer, video_path, text, max_frame_number=None, gpu_id=0):
    print(model.config)
    processor = model.bind_processor(tokenizer,
                                     config=model.config, max_frame_number=max_frame_number)
    prompt = process_prompt(video_path, text)
    proc_ret = processor(prompt)
    proc_ret = move_tensors_to_cuda(proc_ret, gpu_id)
    with torch.inference_mode():
        output_ids = model.generate(**proc_ret,
                                do_sample=False,
                                num_beams=1,
                                max_new_tokens=256,
                                use_cache=True)
    outputs = tokenizer.batch_decode(
        output_ids[:, proc_ret["input_ids"].shape[1]:],
        skip_special_tokens=True)[0].strip()
    return outputs
    

def load_model(model_path, gpu_id):
    model = MMForCausalLM.from_pretrained(model_path,
                                            device_map=f'cuda:{gpu_id}',
                                            torch_dtype=torch.bfloat16).eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        truncation_side="left",
        use_fast=False,
        trust_remote_code=True,
    )
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="path/to/aitqe")
    parser.add_argument("--video_path", type=str, default="./test.png")
    parser.add_argument("--question", type=str, default="Please describe the video in detail.")
    parser.add_argument("--max_frame_number", type=int, default=64)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.gpu_id)
    output = eval_single_node(model, tokenizer, args.video_path, args.question, args.max_frame_number, args.gpu_id)
    print("="*100)
    print("[Input]")
    print(f"Video: {args.video_path}")
    print(f"Question: {args.question}")
    print("="*100)
    print("[Output]")
    print(output)


if __name__ == "__main__":
    main()

