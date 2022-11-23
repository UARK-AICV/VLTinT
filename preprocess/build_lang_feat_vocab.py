import argparse
import os
import nltk
import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def build_vocab_idx(word_insts, min_word_count):
    full_vocab = set(w for sent in word_insts for w in sent)
    print("[Info] Original Vocabulary size =", len(full_vocab))

    word2idx = {}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count >= min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print("[Info] Trimmed vocabulary size = {},".format(len(word2idx)),
          "each with minimum occurrence = {}".format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default="anet", choices=["anet", "yc2"])
    parser.add_argument("--min_word_count", type=int, default=6)

    opt = parser.parse_args()
    
    # load, merge, clean, split data
    if opt.dset_name == "anet":
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../densevid_eval/anet_data/train.json")
    elif opt.dset_name == "yc2":
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../densevid_eval/yc2_data/yc2_train_anet_format.json")

    # load, merge, clean, split data
    train_data = load_json(data_path)
    all_sentences = flat_list_of_lists([v["sentences"] for k, v in train_data.items()])
    all_sentences = [nltk.tokenize.word_tokenize(sen.lower())
                     for sen in all_sentences]
    word2idx = build_vocab_idx(all_sentences, opt.min_word_count)
    word2idx_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 
        "{}_clip_word2idx.json".format(opt.dset_name)
    )
    save_json(word2idx, word2idx_path, save_pretty=True)
    print("[Info] Finish.")

if __name__ == "__main__":
    main()
