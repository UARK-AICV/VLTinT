#!/usr/bin/env bash
# Usage:
#   $ bash {SCRIPT.sh} {DATASET} [Any flags available in train.py, could also be empty]
#   DATASET: `anet` or `yc2`
#   Note the additional flags added will overwrite the specified flags below,
#   i.e., if `--exp_id run1` is specified, it will overwrite `--exp_id init` below.
# Examples:
#   anet debug mode: $ bash scripts/train.sh anet --debug
#   yc2 training mode: $ bash scripts/train.sh yc2

dset_name=$1  # [anet, yc2]
use_env=false
use_agent=false
use_lang=false
use_tint=false
while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v=true
   fi
  shift
done

data_dir="$PWD/densevid_eval/${dset_name}_data"
v_feat_dir="$PWD/video_feature/${dset_name}_trainval"
dur_file="$PWD/video_feature/${dset_name}_duration_frame.csv"
word2idx_path="$PWD/cache/${dset_name}_word2idx.json"
glove_path="$PWD/cache/${dset_name}_vocab_clip.pt"

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "[*] Running training on ${dset_name} dataset"
if [[ ${dset_name} == "anet" ]]; then
    max_n_sen=6
    max_t_len=22  # including "BOS" and "EOS"
    max_v_len=100
elif [[ ${dset_name} == "yc2" ]]; then
    max_n_sen=12
    max_t_len=22  # including "BOS" and "EOS"
    max_v_len=100
else
    echo "Wrong option for your first argument, select between anet and yc2"
    exit 1
fi

echo "[*] Using options: use_env ${use_env}, use_agent ${use_agent}, use_lang ${use_lang}, use_tint ${use_tint}"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
extra_args=()
extra_args+=(--recurrent)

python src/train.py \
--use_env ${use_env} \
--use_agent ${use_agent} \
--use_lang ${use_lang} \
--use_tint ${use_tint} \
--dset_name ${dset_name} \
--data_dir ${data_dir} \
--video_feature_dir ${v_feat_dir} \
--v_duration_file ${dur_file} \
--word2idx_path ${word2idx_path} \
--glove_path ${glove_path} \
--max_n_sen ${max_n_sen} \
--max_t_len ${max_t_len} \
--max_v_len ${max_v_len} \
--exp_id init \
${extra_args[@]} 