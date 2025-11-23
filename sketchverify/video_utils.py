"""
Video utility functions for saving and rendering
"""
import os
import numpy as np
import cv2
import imageio
from typing import List, Dict
from tqdm import tqdm
from PIL import Image

from data_structures import TrajectoryChunk


def save_frames_as_video(frames: List[np.ndarray], output_path: str, fps: int = 30, height: int = 1024, width: int = 1024):
    """Save a list of frames as an MP4 video file"""
    if not frames:
        print("No frames to save")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frame = cv2.resize(frame, (height, width))
            writer.append_data(frame)

    print(f"Video saved to: {output_path}")


def save_candidate_videos(candidates: List[TrajectoryChunk],
                         chunk_generator: 'ChunkBasedGenerator',
                         detection_map: Dict,
                         object_imgs: Dict,
                         background_image: Image.Image,
                         output_skill_dir: str,
                         phase_idx: int,
                         phase_name: str,
                         best_chunk: TrajectoryChunk):
    """
    Render and save videos for all candidate trajectories

    Args:
        candidates: List of trajectory chunks
        chunk_generator: Generator instance with rendering capabilities
        detection_map: Object detection mapping
        object_imgs: Object images for compositing
        background_image: Background image
        output_skill_dir: Output directory
        phase_idx: Phase index
        phase_name: Phase name
        best_chunk: The selected best chunk
    """
    for cand_idx, candidate in enumerate(tqdm(candidates, desc="  Rendering candidate videos")):
        cand_frames = []

        # Render each frame in this candidate
        for frame_data in candidate.frames:
            rendered = chunk_generator.generate_frame_image(
                frame_data, detection_map, object_imgs, background_image
            )
            cand_frames.append(np.array(rendered))

        # Save candidate video
        is_best = (candidate == best_chunk)
        suffix = "_BEST" if is_best else f"_candidate_{cand_idx}"
        cand_video_path = os.path.join(output_skill_dir, "chunks",
                                       f"chunk_{phase_idx + 1}_{phase_name}{suffix}.mp4")
        save_frames_as_video(cand_frames, cand_video_path, fps=8)

        print(f"    Saved candidate {cand_idx} video: {cand_video_path}")
