import torch
import clip
import json 
import PIL
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import os
import glob


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def sort_key(name):
    return int(name.split("/")[-1].split(".")[0])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--frame-root',
        type=str,
        default='anet_raw_frames/'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='anet_clip_b16/'
    )
    parser.add_argument(
        '--dset_name',
        type=str,
    )
    parser.add_argument(
        '--num-proc',
        type=int,
        default=1
    )
    parser.add_argument(
        '--rank',
        type=int,
        default=0
    )
    args = parser.parse_args()

    frame_root = args.frame_root
    output_root = args.output_root
    
    if not os.path.exists(output_root):
        os.mkdir(output_root)

    assert args.rank < args.num_proc, 'Rank index must be in [0, {})'.format(args.num_proc)

    n_processed = 0
    dirnames = sorted(os.listdir(frame_root))
    start_step, end_step = np.linspace(0, len(dirnames), args.num_proc + 1)[args.rank:args.rank+2]
    start_step, end_step = int(start_step), int(end_step)

    device = f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    voc = load_json(f"./preprocess/{args.dset_name}_clip_word2idx.json").keys()
    word2idx = {v:i for i, v in enumerate(voc)}
    idx2word = {int(v): k for k, v in word2idx.items()}
    tokens = word2idx.keys()
    text = clip.tokenize(tokens).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    if not os.path.exists(os.path.join(output_root, "lang_feature")):
        os.mkdir(os.path.join(output_root, "lang_feature"))
    if not os.path.exists(os.path.join(output_root, "frame_feature")):
        os.mkdir(os.path.join(output_root, "frame_feature"))

    for i, vidname in tqdm(enumerate(dirnames[start_step:end_step])):
        video_path = os.path.join(frame_root, vidname)
        output_path = os.path.join(
            output_root, "lang_feature",
            vidname + '.json'
        )

        if os.path.exists(output_path):
            continue

        lang = []

        frame_feat_path = os.path.join(
            output_root, "frame_feature",
            vidname + '.npy'
        )

        if os.path.exists(frame_feat_path):
            image_features = torch.tensor(np.load(frame_feat_path), device=device) 
        else:
            try:
                video = [preprocess(PIL.Image.open(f)) for f in sorted(glob.glob(video_path + "/*.jpg"), key=sort_key)]
                image_input = torch.tensor(np.stack(video), device=device) 
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
            except Exception as e:
                print(f"[*] error at {vidname}")
                print(e)
                continue
        
        with torch.no_grad():
            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            # shape = [global_batch_size, global_batch_size]
            probs = logits_per_image.softmax(dim=-1)
            sorted_, indices = torch.sort(probs, dim=1, descending=True)
            sorted_, indices = sorted_[:,:100], indices[:,:100]

            for index in indices:
                lang.append([idx2word[int(i.cpu().numpy())] for i in index])

        if not os.path.exists(frame_feat_path):
            np.save(frame_feat_path, image_features.cpu().numpy())

        save_json(lang, output_path)