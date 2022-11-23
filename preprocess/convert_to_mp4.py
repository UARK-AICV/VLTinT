import cv2
import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--video-root',
        type=str,
        default='error_videos/'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='fixed_rescaled_videos/'
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

    video_root = args.video_root
    output_root = args.output_root
    assert args.rank < args.num_proc, 'Rank index must be in [0, {})'.format(args.num_proc)

    n_processed = 0
    filenames = sorted(os.listdir(video_root))
    start_step, end_step = np.linspace(0, len(filenames), args.num_proc + 1)[args.rank:args.rank+2]
    start_step, end_step = int(start_step), int(end_step)

    for i, filename in tqdm(enumerate(filenames[start_step:end_step])):
        if filename.endswith(".mp4"):
            os.symlink(os.path.join(video_root, filename), os.path.join(output_root, filename))
        elif filename.endswith(".mkv"):
            os.system(f"ffmpeg -i {os.path.join(video_root, filename)} -c:a aac {os.path.join(output_root, filename.replace('.mkv', '.mp4'))}")
        elif filename.endswith(".webm"):
            os.system(f"ffmpeg -i {os.path.join(video_root, filename)} {os.path.join(output_root, filename.replace('.webm', '.mp4'))}")
        else:
            raise ValueError(f"[*] unsupported file type: {filename.split('.')[-1]}")

