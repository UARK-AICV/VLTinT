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
        '--frame-dir',
        type=str,
        default='../raw_frames/'
    )
    parser.add_argument(
        '--target-n-frames',
        type=int,
        default=1600
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
    target_n_frames = args.target_n_frames

    assert args.rank < args.num_proc, 'Rank index must be in [0, {})'.format(args.num_proc)
    frame_dir = args.frame_dir
    os.makedirs(frame_dir, exist_ok=True)

    n_processed = 0
    filenames = sorted(os.listdir(video_root))
    start_step, end_step = np.linspace(0, len(filenames), args.num_proc + 1)[args.rank:args.rank+2]
    start_step, end_step = int(start_step), int(end_step)

    for i, filename in tqdm(enumerate(filenames[start_step:end_step])):
        # tic = time.time()
        video_path = os.path.join(video_root, filename)
        target_video_path = os.path.join(
            output_root,
            os.path.splitext(filename)[0] + '.mp4'
        )

        if os.path.exists(target_video_path) and os.path.exists(os.path.join(frame_dir, filename.split(".")[0])):
            continue

        in_video = cv2.VideoCapture(video_path)
        num_frames = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if os.path.isfile(target_video_path):
            out_frames = cv2.VideoCapture(target_video_path).get(cv2.CAP_PROP_FRAME_COUNT)
            if out_frames == target_n_frames:
                print('Detected %s. %d/%d.' % (
                    os.path.basename(video_path),
                    i + 1,
                    len(filenames),
                ))
                continue
        n_processed += 1

        out_video = cv2.VideoWriter(
            filename=target_video_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=30.,
            frameSize=(width, height),
            isColor=True
        )

        flag = num_frames < target_n_frames
        if num_frames == 0:
            print(f'[*] {video_path} has no frames.')
            with open('error_videos.txt', 'a') as f:
                f.write(f'{filename} has no frames.\n')
            continue

        os.makedirs(os.path.join(frame_dir, filename.split(".")[0]))
        frame_idx, error_counts = 0, 0
        middle_idx = 0
        target_frames = np.bincount(np.round(np.linspace(0, num_frames - 1, target_n_frames)).astype(np.int))

        for i in range(num_frames):
            success, frame = in_video.read()
            if target_frames[i] != 0:
                if success:
                    for c in range(target_frames[i]):
                        out_video.write(frame)
                        if (frame_idx + 8) % 16 == 0:
                            middle_idx += 1
                            cv2.imwrite(os.path.join(frame_dir, filename.split(".")[0], f'{middle_idx}.jpg'), frame)
                        frame_idx += 1
                else:
                    print(f'[*] Error reading frame {i}/{num_frames}')
                    with open('error_videos.txt', 'a') as f:
                        f.write(f'{filename} error reading frame {i+1}/{num_frames}\n')
                    if i+1 >= num_frames:
                        break
                    target_frames[i+1] += target_frames[i]
                    error_counts += 1

        in_video.release()
        out_video.release()

        try:
            assert frame_idx == target_n_frames, f'Frame count mismatch. Got {frame_idx} frames for {filename}'
            assert middle_idx == target_n_frames // 16, f'Middle frame count mismatch Got {middle_idx} frames for {filename}'
        except AssertionError as e:
            print(e)
            os.system(f"rm -rf {os.path.join(frame_dir, filename.split('.')[0])}")
            os.system(f"rm {target_video_path}")
            with open('error_videos.txt', 'a') as f:
                f.write(f'{filename} {e}\n')