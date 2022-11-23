import clip
import json 
import torch
import argparse

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default="anet", choices=["anet", "yc2"])
    opt = parser.parse_args()

    voc = load_json(f"../cache/{opt.dset_name}_word2idx.json").keys()
    word2idx = {v:i for i, v in enumerate(voc)}
    tokens = word2idx.keys()
    text = clip.tokenize(tokens).to("cuda")
    clip_model, _ = clip.load("ViT-B/16", device="cuda")
    with torch.no_grad():
        text_features = clip_model.encode_text(text).cpu().numpy()

    torch.save(text_features, f"../cache/{opt.dset_name}_vocab_clip.pt")