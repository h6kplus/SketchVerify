"""
Main pipeline for chunk-based grounded planning with test-time search
"""
import os
import json
import argparse
import time
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

from vlm_verifier import VLMVerifier
from object_proposal import ObjectProposalGenerator
from object_detector import GroundedObjectDetector
from flux_generator import FluxControlRemovalGenerator
from chunk_generator import ChunkBasedGenerator
from video_utils import save_frames_as_video, save_candidate_videos


def process_single_sample(text_prompt, frame1_path, output_dir, args,
                          proposal_generator, object_detector, flux_generator,
                          chunk_generator):
    """
    Process a single text prompt and frame1 image

    Args:
        text_prompt: Text description of the video to generate
        frame1_path: Path to the first frame image
        output_dir: Directory to save outputs
        args: Command line arguments
        proposal_generator: ObjectProposalGenerator instance
        object_detector: GroundedObjectDetector instance
        flux_generator: FluxControlRemovalGenerator instance
        chunk_generator: ChunkBasedGenerator instance

    Returns:
        dict: Results including trajectory, global plan, etc.
    """
    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"Processing: {text_prompt}")
    print(f"Frame1: {frame1_path}")
    print(f"{'='*80}\n")

    # Step 1: Generate object proposals
    print("Step 1: Generating object proposals...")
    start_processing_time = time.time()
    proposals = proposal_generator.generate_proposals(frame1_path, text_prompt)

    if "error" in proposals:
        raise ValueError(f"Proposal generation failed: {proposals['error']}")

    detection_targets = proposals.get("static_objects", []) + proposals.get("moving_objects", [])
    moving_objects = proposals.get("moving_objects", [])

    print(f"Proposed detection targets: {detection_targets}")
    print(f"Proposed moving objects: {moving_objects}")
    print(f"Generation time: {time.time() - start_processing_time:.2f} seconds")

    # Step 2: Initial detection and segmentation on frame 1
    print("\nStep 2: Detecting and segmenting objects in frame 1...")
    start_processing_time = time.time()
    frame1_image = Image.open(frame1_path).convert("RGB").resize((1024, 1024))

    detections = object_detector.detect_and_segment(
        frame1_image,
        detection_targets,
        threshold=args.threshold
    )

    print(f"Successfully detected {len(detections)} objects: {[d.label for d in detections]}")

    if not detections:
        raise ValueError("No objects detected, cannot continue")

    print(f"Detection time: {time.time() - start_processing_time:.2f} seconds")

    # Step 3: Generate background
    print("\nStep 3: Generating clean background...")
    start_processing_time = time.time()
    moving_detections = object_detector.detect_and_segment_moving_objects(
        frame1_image, proposals, threshold=args.threshold
    )

    if moving_detections:
        print(f"Detected {len(moving_detections)} moving objects to remove")

        background_image, _, _, _ = flux_generator.generate_background_with_flux_control(
            frame1_image,
            moving_detections,
            proposals,
            guidance_scale=args.flux_guidance_scale,
            num_inference_steps=args.flux_inference_steps,
            target_size=(1024, 1024),
            use_feathering=args.use_feathering,
            output_path=output_dir if args.save_visualizations else None
        )

        print(f"Generated clean background by removing {len(moving_detections)} objects")
    else:
        print("No moving objects detected, using original image as background")
        background_image = frame1_image

    # Create detection mapping and store object images
    detection_map = {d.label: d for d in detections}

    if args.save_visualizations:
        os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
        background_save_path = os.path.join(output_dir, "generated_background.png")
        background_image.save(background_save_path)

    print(f"Background generation time: {time.time() - start_processing_time:.2f} seconds")

    # Store object images
    print("\nStep 4: Storing object images...")
    start_processing_time = time.time()
    if args.save_visualizations:
        obj_save_dir = os.path.join(output_dir, "object_images")
        os.makedirs(obj_save_dir, exist_ok=True)

    object_imgs = {}
    saved = []
    for detection in detections:
        obj_name = detection.label
        if obj_name in saved:
            continue
        saved.append(obj_name)

        if detection.mask is None or not np.any(detection.mask):
            height, width = frame1_image.height, frame1_image.width
            mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = int(detection.box.xmin), int(detection.box.ymin), int(detection.box.xmax), int(detection.box.ymax)
            mask[y1:y2, x1:x2] = 255
            detection.mask = mask[np.newaxis, :, :]

        img_np = np.array(frame1_image)
        obj_rgba = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
        obj_rgba[:,:,:3] = img_np
        obj_rgba[:,:,3] = detection.mask[0,:,:] * 255

        x1, y1, x2, y2 = int(detection.box.xmin), int(detection.box.ymin), int(detection.box.xmax), int(detection.box.ymax)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)

        cropped_obj = obj_rgba[y1:y2, x1:x2]
        object_imgs[obj_name] = cropped_obj

        if args.save_visualizations:
            obj_filename = os.path.join(obj_save_dir, f"{obj_name.replace('.', '')}.png")
            cv2.imwrite(obj_filename, cv2.cvtColor(cropped_obj, cv2.COLOR_RGBA2BGRA))

    print(f"Time to store object images: {time.time() - start_processing_time:.2f} seconds")

    # Step 5: Create global movement plan
    print("\nStep 5: Creating global movement plan...")
    start_processing_time = time.time()
    global_plan = chunk_generator.create_global_movement_plan(
        frame1_image, text_prompt, detections, total_frames=args.total_frames
    )

    print("Global plan created successfully")
    moving_objects_list = global_plan.get('moving_objects', [])
    static_objects_list = global_plan.get('static_objects', [])
    print(f"- Moving objects: {moving_objects_list}")
    print(f"- Static objects: {static_objects_list}")

    # Initialize scorer with rendering components for VLM verification
    print("Initializing trajectory scorer with VLM verification...")
    chunk_generator.initialize_scorer(detection_map, object_imgs, background_image)
    print(f"Global plan creation time: {time.time() - start_processing_time:.2f} seconds")

    # Step 6: CHUNK-BASED generation with test-time search
    print("\nStep 6: Chunk-based trajectory generation with test-time search...")
    start_processing_time = time.time()

    # Parse phases from global plan
    phases = []
    if 'frame_allocation' in global_plan:
        for phase_name, phase_desc in global_plan['frame_allocation'].items():
            start, end = chunk_generator.parse_phase_frames(phase_desc)
            if start and end:
                phases.append((phase_name, start, end))
                print(f"  Phase: {phase_name}, frames {start}-{end}")

    if not phases:
        print("  Warning: Could not parse phases, using default chunking")
        # Default: split into 3 equal chunks
        chunk_size = args.total_frames // 3
        phases = [
            ('approach_phase', 1, chunk_size),
            ('action_phase', chunk_size + 1, chunk_size * 2),
            ('completion_phase', chunk_size * 2 + 1, args.total_frames)
        ]

    # Initialize trajectory with frame 1 - ONLY include moving objects
    moving_detections = []
    for d in detections:
        label_clean = d.label.rstrip('.')
        if label_clean in moving_objects_list:
            moving_detections.append(d)

    print(f"Frame 1: Tracking {len(moving_detections)} moving objects (ignoring {len(detections) - len(moving_detections)} static objects)")

    full_trajectory = [{
        'frame_number': 1,
        'objects': [{'label': d.label,
                    'box': [d.box.xmin / 1024, d.box.ymin / 1024,
                           d.box.xmax / 1024, d.box.ymax / 1024],
                    'state': 'initial'} for d in moving_detections],
        'caption': 'Initial frame - moving objects at starting positions'
    }]

    current_frame_image = frame1_image

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    if args.save_visualizations:
        os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)

    # Generate each chunk with test-time search
    for phase_idx, (phase_name, chunk_start, chunk_end) in enumerate(phases):
        print(f"\n=== Phase {phase_idx + 1}/{len(phases)}: {phase_name} (frames {chunk_start}-{chunk_end}) ===")

        # Generate multiple candidates for this chunk
        candidates = chunk_generator.generate_chunk_candidates(
            start_frame_image=current_frame_image,
            text_prompt=text_prompt,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            phase_name=phase_name,
            num_candidates=args.num_candidates,
            previous_trajectory=full_trajectory,
            detection_map=detection_map,
            object_imgs=object_imgs,
            background_image=background_image
        )

        if not candidates:
            print(f"  Failed to generate candidates for phase {phase_name}, skipping...")
            break

        print(f"  Generated {len(candidates)} candidates")

        # Determine boundary conditions
        start_boundary = full_trajectory[-1] if full_trajectory else None

        # End boundary from global plan (if available)
        end_boundary = None
        if global_plan and 'movement_plans' in global_plan:
            end_boundary = {'objects': []}
            for obj_name, plan in global_plan['movement_plans'].items():
                if 'key_positions' in plan:
                    key_pos = plan['key_positions'].get(f'frame_{chunk_end}')
                    if key_pos:
                        end_boundary['objects'].append({
                            'label': obj_name,
                            'box': key_pos
                        })

        # Select best candidate
        best_chunk, score_breakdown = chunk_generator.select_best_chunk(
            candidates,
            start_boundary=start_boundary,
            end_boundary=end_boundary
        )

        print(f"  Selected chunk with score: {best_chunk.score:.4f}")

        # Add chunk frames to full trajectory
        full_trajectory.extend(best_chunk.frames[1:] if chunk_start > 1 else best_chunk.frames)

        # Update current frame image for next chunk
        if best_chunk.frames:
            last_frame_data = best_chunk.frames[-1]
            current_frame_image = chunk_generator.generate_frame_image(
                last_frame_data, detection_map, object_imgs, background_image
            )

        # Save chunk visualizations
        if args.save_visualizations:
            chunk_viz_path = os.path.join(output_dir, "chunks",
                                         f"chunk_{phase_idx + 1}_{phase_name}_BEST.json")
            with open(chunk_viz_path, 'w') as cf:
                json.dump({
                    'phase': phase_name,
                    'frames': best_chunk.frames,
                    'score': best_chunk.score,
                    'score_breakdown': score_breakdown,
                    'selected': True,
                    'num_candidates': len(candidates)
                }, cf, indent=2)

            if args.save_all_candidates:
                print(f"  Saving all {len(candidates)} candidates for analysis...")
                ranked_candidates = sorted(enumerate(candidates),
                                          key=lambda x: x[1].score,
                                          reverse=True)

                for orig_idx, candidate in enumerate(candidates):
                    is_best = (candidate == best_chunk)
                    rank = next(i for i, (_, c) in enumerate(ranked_candidates, 1)
                              if c == candidate)

                    if hasattr(candidate, 'score') and candidate.score > 0:
                        cand_score = candidate.score
                        _, cand_breakdown = chunk_generator.scorer.score_trajectory_chunk(
                            candidate, start_boundary, end_boundary
                        )
                    else:
                        cand_score, cand_breakdown = chunk_generator.scorer.score_trajectory_chunk(
                            candidate, start_boundary, end_boundary
                        )

                    cand_viz_path = os.path.join(output_dir, "chunks",
                                                f"chunk_{phase_idx + 1}_{phase_name}_candidate_{orig_idx}.json")
                    with open(cand_viz_path, 'w') as cf:
                        json.dump({
                            'phase': phase_name,
                            'candidate_index': orig_idx,
                            'rank': rank,
                            'total_candidates': len(candidates),
                            'frames': candidate.frames,
                            'score': float(cand_score),
                            'score_breakdown': {k: float(v) for k, v in cand_breakdown.items()},
                            'selected': is_best
                        }, cf, indent=2)

                if args.save_all_candidates_videos:
                    print(f"  Rendering videos for all {len(candidates)} candidates...")
                    save_candidate_videos(candidates, chunk_generator, detection_map,
                                         object_imgs, background_image, output_dir,
                                         phase_idx, phase_name, best_chunk)

    print(f"Chunk-based generation time: {time.time() - start_processing_time:.2f} seconds")

    # Step 7: Interpolate and render
    print("\nStep 7: Interpolating trajectories and rendering final frames...")

    print(f"Interpolating from {len(full_trajectory)} trajectory frames to 81 frames...")
    interpolated_trajectory = chunk_generator.interpolate_trajectory_to_target_count(
        full_trajectory, target_count=81
    )
    print(f"Successfully interpolated to {len(interpolated_trajectory)} frames")

    # Render all frames
    print("Rendering all interpolated frames...")
    all_rendered_frames = []
    for interp_frame_data in tqdm(interpolated_trajectory, desc="Rendering frames"):
        rendered_frame = chunk_generator.generate_frame_image(
            interp_frame_data,
            detection_map,
            object_imgs,
            background_image
        )
        all_rendered_frames.append(np.array(rendered_frame))

    # Save frames
    print("Saving interpolated frames...")
    for idx, frame in enumerate(all_rendered_frames):
        cv2.imwrite(
            os.path.join(output_dir, "frames", f"frame_{idx:04d}.png"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )

    # Save video
    save_frames_as_video(all_rendered_frames, os.path.join(output_dir, "video.mp4"), fps=16)

    # Prepare results
    result = {
        "prompt": text_prompt,
        "frame1_path": frame1_path,
        "proposals": proposals,
        "global_plan": global_plan,
        "phases": [{"name": p[0], "start": p[1], "end": p[2]} for p in phases],
        "num_candidates_per_chunk": args.num_candidates,
        "predicted_trajectory": full_trajectory,
        "interpolated_trajectory": interpolated_trajectory,
        "num_predicted_frames": len(full_trajectory),
        "num_interpolated_frames": len(interpolated_trajectory),
        "output_directory": output_dir
    }

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ Successfully processed in {elapsed/60:.2f} minutes")
    print(f"  - Generated {len(full_trajectory)} trajectory frames across {len(phases)} chunks")
    print(f"  - Interpolated to {len(interpolated_trajectory)} frames")
    print(f"  - Outputs saved to: {output_dir}")
    print(f"{'='*80}\n")

    return result


def main(args):
    # Initialize components
    print("Initializing Chunk-Based Pipeline with Test-Time Search...")

    # Create OpenAI client
    client = OpenAI(api_key=args.api_key)

    # Create VLM verifier with optional Gemini support
    vlm_verifier = VLMVerifier(
        openai_client=client,
        gemini_api_key=args.gemini_api_key if hasattr(args, 'gemini_api_key') and args.gemini_api_key else None
    )

    proposal_generator = ObjectProposalGenerator(client)
    object_detector = GroundedObjectDetector(device="cuda" if torch.cuda.is_available() else "cpu")

    # Initialize FLUX Control Removal generator
    flux_generator = FluxControlRemovalGenerator(device="cuda" if torch.cuda.is_available() else "cpu")

    chunk_generator = ChunkBasedGenerator(client, object_detector, vlm_verifier=vlm_verifier,
                                         temperature=args.candidate_temperature,
                                         top_p=args.candidate_top_p,
                                         diversity_threshold=args.diversity_threshold)

    # SINGLE SAMPLE MODE: Process one text prompt and image
    if args.text_prompt and args.frame1_image:
        print("\n" + "="*80)
        print("SINGLE SAMPLE MODE")
        print("="*80)

        # Verify frame1 image exists
        if not os.path.exists(args.frame1_image):
            raise FileNotFoundError(f"Frame1 image not found: {args.frame1_image}")

        # Create output directory
        output_dir = args.output_path if args.output_path else "output"
        os.makedirs(output_dir, exist_ok=True)

        # Process the single sample
        result = process_single_sample(
            text_prompt=args.text_prompt,
            frame1_path=args.frame1_image,
            output_dir=output_dir,
            args=args,
            proposal_generator=proposal_generator,
            object_detector=object_detector,
            flux_generator=flux_generator,
            chunk_generator=chunk_generator
        )

        # Save result JSON
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output_file}")

        return result

    # BATCH MODE: Process prompts from files
    elif args.prompt_path:
        print("\n" + "="*80)
        print("BATCH MODE")
        print("="*80)

        # Create output directory
        output_dir = os.path.dirname(args.output_file) if args.output_file else ""
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result = {}

        # Process prompts
        if os.path.isfile(args.prompt_path):
            prompt_files = [os.path.basename(args.prompt_path)]
            prompt_dir = os.path.dirname(args.prompt_path)
        else:
            prompt_files = [f for f in os.listdir(args.prompt_path) if f.endswith('.txt')]
            prompt_dir = args.prompt_path


        for file in tqdm(prompt_files, desc="Processing prompt files"):
            if file.startswith('.'):
                continue
            if file != args.skill + ".txt":
                print(f"Skipping file {file} as it does not match the skill {args.skill}")
                continue

            result[file] = []
            skill_name = file.split('.')[0]

            print(f"\nProcessing skill: {skill_name}")

            with open(os.path.join(prompt_dir, file), "r", encoding="utf-8") as f:
                for i, line in tqdm(enumerate(f), desc=f"Processing prompts in {file}"):
                    if i >= args.max_prompts:
                        break
                    if i < args.min_prompts:
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    text_prompt = line

                    # Find corresponding frame1 image
                    if not args.frame1_images_path:
                        print("Error: frame1_images_path is required in batch mode")
                        continue

                    possible_paths = [
                        os.path.join(args.frame1_images_path, skill_name, f"{i+1:04d}.jpg"),
                        os.path.join(args.frame1_images_path, skill_name, f"{i+1:04d}.png"),
                        os.path.join(args.frame1_images_path, f"{skill_name}_{i+1:04d}.jpg"),
                        os.path.join(args.frame1_images_path, f"{skill_name}_{i+1:04d}.png"),
                    ]

                    frame1_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            frame1_path = path
                            break

                    if not frame1_path:
                        print(f"Warning: Frame1 image not found for {skill_name} prompt {i+1}")
                        continue

                    # Determine output directory
                    output_skill_dir = os.path.join(args.output_path, skill_name, f"{i+1:04d}")

                    # Process this sample
                    try:
                        prompt_result = process_single_sample(
                            text_prompt=text_prompt,
                            frame1_path=frame1_path,
                            output_dir=output_skill_dir,
                            args=args,
                            proposal_generator=proposal_generator,
                            object_detector=object_detector,
                            flux_generator=flux_generator,
                            chunk_generator=chunk_generator
                        )

                        # Add batch-specific metadata
                        prompt_result["index"] = i + 1
                        result[file].append(prompt_result)

                        # Save intermediate results
                        if args.output_file:
                            with open(args.output_file, "w") as f_out:
                                json.dump(result, f_out, indent=2)

                    except Exception as e:
                        print(f"Error processing prompt {i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        # Save final results
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)

        print(f"\nChunk-Based Pipeline with Test-Time Search completed!")
        print(f"Results saved to: {args.output_file}")
        print(f"Videos and frames saved to: {args.output_path}")

    else:
        raise ValueError("Either provide --text_prompt and --frame1_image for single sample mode, "
                        "or --prompt_path and --frame1_images_path for batch mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk-Based Planning Pipeline with Test-Time Search + VLM Verification\n\n"
                    "Two modes available:\n"
                    "  1. SINGLE SAMPLE: --text_prompt + --frame1_image\n"
                    "  2. BATCH: --prompt_path + --frame1_images_path",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--api_key", type=str, required=True,
                       help="OpenAI API key")

    # Single sample mode arguments
    parser.add_argument("--text_prompt", type=str, default=None,
                       help="[SINGLE MODE] Text prompt describing the video to generate")
    parser.add_argument("--frame1_image", type=str, default=None,
                       help="[SINGLE MODE] Path to the first frame image")

    # Batch mode arguments
    parser.add_argument("--prompt_path", type=str, default=None,
                       help="[BATCH MODE] Path to prompt file or directory")
    parser.add_argument("--frame1_images_path", type=str, default=None,
                       help="[BATCH MODE] Path to frame1 images directory")
    parser.add_argument("--skill", type=str, default="robotics",
                       help="[BATCH MODE] Skill name to process")
    parser.add_argument("--max_prompts", type=int, default=20,
                       help="[BATCH MODE] Maximum number of prompts to process")
    parser.add_argument("--min_prompts", type=int, default=0,
                       help="[BATCH MODE] Minimum prompt index to start from")

    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file path (optional, defaults to 'result.json')")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output directory for videos and frames (optional, defaults to 'output')")

    # VLM arguments
    parser.add_argument("--gemini_api_key", type=str, default=None,
                       help="Google Gemini API key for physical plausibility verification (optional)")

    # Pipeline arguments
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="Detection confidence threshold")
    parser.add_argument("--total_frames", type=int, default=21,
                       help="Total number of trajectory frames to generate")
    parser.add_argument("--num_candidates", type=int, default=3,
                       help="Number of candidate trajectories per chunk (test-time search)")
    parser.add_argument("--candidate_temperature", type=float, default=1.0,
                       help="Temperature for trajectory generation (higher = more diverse, 0.0-2.0)")
    parser.add_argument("--candidate_top_p", type=float, default=None,
                       help="Top-p sampling for diversity (0.0-1.0, optional)")
    parser.add_argument("--diversity_threshold", type=float, default=0.05,
                       help="Minimum trajectory distance for diversity filtering (0.0 = no filtering)")
    parser.add_argument("--save_visualizations", action="store_true",
                       help="Save visualization images and best chunk data")
    parser.add_argument("--save_all_candidates", action="store_true",
                       help="Save all trajectory candidates (not just best) as JSON files")
    parser.add_argument("--save_all_candidates_videos", action="store_true",
                       help="Render and save videos for all candidates (requires --save_all_candidates)")

    # FLUX arguments
    parser.add_argument("--flux_guidance_scale", type=float, default=3.5,
                       help="FLUX guidance scale")
    parser.add_argument("--flux_inference_steps", type=int, default=28,
                       help="FLUX inference steps")
    parser.add_argument("--use_feathering", action="store_true", default=True,
                       help="Apply feathering to masks")

    args = parser.parse_args()

    # Set defaults for output paths if not provided
    if not args.output_file:
        args.output_file = "result.json"
    if not args.output_path:
        args.output_path = "output"

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Using CPU (will be slower)")

    main(args)
