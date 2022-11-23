import torch
import clip
import json 
from tqdm import tqdm
import pickle
import os
from argparse import ArgumentParser

mode = "train"
gpu_idx = 0
split = False
before = False

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

# def save_dict(di_, filename_):
#     with open(filename_, 'wb') as f:
#         pickle.dump(di_, f)

# def load_dict(filename_):
#     with open(filename_, 'rb') as f:
#         ret_di = pickle.load(f)
#     return ret_di


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--caption_root',
        type=str,
        default=''
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='anet_clip_b16/'
    )
    # parser.add_argument(
    #     '--num-proc',
    #     type=int,
    #     default=1
    # )
    # parser.add_argument(
    #     '--rank',
    #     type=int,
    #     default=0
    # )
    args = parser.parse_args()


    if not os.path.exists(os.path.join(args.output_root, "sent_feature")):
        os.mkdir(os.path.join(args.output_root, "sent_feature"))

    device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    captions = load_json(args.caption_root)

    for name, val in tqdm(captions.items()):
        output_path = os.path.join(
            args.output_root, "sent_feature",
            name + '.json'
        )

        if os.path.exists(output_path):
            continue
        
        sens = val['sentences']
        text = clip.tokenize(sens, truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)

        save_json(text_features.cpu().numpy().tolist(), output_path)