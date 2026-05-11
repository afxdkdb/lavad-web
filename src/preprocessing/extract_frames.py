import argparse
import os
import subprocess
from pathlib import Path

import cv2


def extract_frames(video_path, frames_dir):
    video_name = Path(video_path).stem

    video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    has_frames = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    cap.release()

    if has_frames:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(video_frames_dir, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        cap.release()
    else:
        cmd = ['ffmpeg', '-i', video_path, '-start_number', '0', '-q:v', '2',
               f'{video_frames_dir}/%06d.jpg', '-y']
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            frame_count = len([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
        else:
            frame_count = 0

    print(f"Extracted {frame_count} frames from {video_path} to {video_frames_dir}")
    return video_name, frame_count


def main(videos_dir, frames_dir, annotations_file):
    os.makedirs(frames_dir, exist_ok=True)

    with open(annotations_file, "w") as f:
        for root, dirs, files in os.walk(videos_dir):
            for video_file in files:
                if video_file.endswith(".avi") or video_file.endswith(".mp4"):
                    video_path = os.path.join(root, video_file)
                    video_relative = os.path.relpath(video_path, videos_dir)
                    video_relative_stem = Path(video_relative).stem
                    _, num_frames = extract_frames(video_path, frames_dir)
                    f.write(f"{video_relative_stem} 0 {num_frames - 1} 0\n")


def main_from_annotation(videos_dir, frames_dir, input_annotation_file, output_annotation_file):
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_annotation_file), exist_ok=True)

    with open(input_annotation_file, 'r') as infile, open(output_annotation_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            video_relative = parts[0]
            start_frame = parts[1]

            video_path_mp4 = os.path.join(videos_dir, video_relative + ".mp4")
            video_path_avi = os.path.join(videos_dir, video_relative + ".avi")

            if os.path.exists(video_path_mp4):
                video_path = video_path_mp4
            elif os.path.exists(video_path_avi):
                video_path = video_path_avi
            else:
                print(f"Video not found: {video_relative}")
                continue

            video_name = os.path.basename(video_relative)
            _, num_frames = extract_frames(video_path, frames_dir)
            outfile.write(f"{video_relative} 0 {num_frames - 1} 0\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Directory path to the videos.",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory path to the frames.",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        required=True,
        help="Path to the annotations file.",
    )
    parser.add_argument(
        "--input_annotation_file",
        type=str,
        help="Path to input annotation file (for extracting specific videos only).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="scan",
        choices=["scan", "annotation"],
        help="Mode: 'scan' scans all videos, 'annotation' extracts only videos in input_annotation_file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "annotation" and args.input_annotation_file:
        main_from_annotation(
            args.videos_dir,
            args.frames_dir,
            args.input_annotation_file,
            args.annotations_file
        )
    else:
        main(args.videos_dir, args.frames_dir, args.annotations_file)
