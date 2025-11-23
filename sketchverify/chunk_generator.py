"""
Chunk-based trajectory generator with test-time search
"""
import re
import json
import copy
import ast
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from data_structures import TrajectoryChunk
from trajectory_scorer import TrajectoryScorer
from vlm_verifier import VLMVerifier
from utils import encode_pil_image, COORDS_GUIDE


class ChunkBasedGenerator:
    """Generate trajectories chunk-by-chunk with test-time search"""

    def __init__(self, client, detector, target_size=(1024, 1024), vlm_verifier: Optional[VLMVerifier] = None,
                 temperature: float = 1.0, top_p: Optional[float] = None, diversity_threshold: float = 0.1):
        self.client = client
        self.detector = detector
        self.target_size = target_size
        self.global_plan = None
        self.scorer = None
        self.vlm_verifier = vlm_verifier
        self.temperature = temperature
        self.top_p = top_p
        self.diversity_threshold = diversity_threshold  # Min L2 distance between candidates
        # These will be set during trajectory generation
        self.detection_map = None
        self.object_imgs = None
        self.background_image = None

    def compute_trajectory_distance(self, traj1: TrajectoryChunk, traj2: TrajectoryChunk) -> float:
        """
        Compute L2 distance between two trajectory chunks

        Args:
            traj1, traj2: Trajectory chunks to compare

        Returns:
            Average L2 distance across all frames and objects
        """
        if len(traj1.frames) != len(traj2.frames):
            return float('inf')

        distances = []
        for f1, f2 in zip(traj1.frames, traj2.frames):
            if 'objects' not in f1 or 'objects' not in f2:
                continue

            # Create object dictionaries for matching
            objs1 = {obj['label']: obj['box'] for obj in f1['objects']}
            objs2 = {obj['label']: obj['box'] for obj in f2['objects']}

            # Compare matching objects
            for label in objs1:
                if label in objs2:
                    box1 = np.array(objs1[label])
                    box2 = np.array(objs2[label])
                    # Compute center distance
                    center1 = (box1[:2] + box1[2:]) / 2
                    center2 = (box2[:2] + box2[2:]) / 2
                    dist = np.linalg.norm(center1 - center2)
                    distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def filter_diverse_candidates(self, candidates: List[TrajectoryChunk]) -> List[TrajectoryChunk]:
        """
        Filter candidates to ensure diversity

        Args:
            candidates: List of trajectory chunks

        Returns:
            Filtered list with diverse trajectories
        """
        if len(candidates) <= 1:
            return candidates

        # Start with first candidate
        diverse_candidates = [candidates[0]]

        for candidate in candidates[1:]:
            # Check if this candidate is sufficiently different from all selected ones
            is_diverse = True
            for selected in diverse_candidates:
                dist = self.compute_trajectory_distance(candidate, selected)
                if dist < self.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diverse_candidates.append(candidate)

        print(f"    Diversity filtering: {len(candidates)} → {len(diverse_candidates)} candidates")
        return diverse_candidates

    def create_global_movement_plan(self, frame1_image, text_prompt, detections, total_frames=20):
        """Create global movement plan using structured tool calling"""

        base64_image = encode_pil_image(frame1_image)

        system_prompt = (
            f"You are a video motion planning expert. "
            f"You have EXACTLY {total_frames} frames to complete the ENTIRE task."
        )

        objects_info = []
        for d in detections:
            box_norm = [d.box.xmin/1024, d.box.ymin/1024, d.box.xmax/1024, d.box.ymax/1024]
            objects_info.append(f"- {d.label}: currently at {box_norm}")
        objects_text = "\n".join(objects_info)

        user_text = f"""
Text prompt: "{text_prompt}"
Available frames: EXACTLY {total_frames}.
Current objects in frame 1:
{objects_text}

Return ONLY a call to the function with a complete plan that finishes by frame {total_frames}.
"""

        # Define tool schema
        movement_plan_tool = {
            "type": "function",
            "function": {
                "name": "submit_movement_plan",
                "description": "Return the full movement plan that completes the task by the final frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_breakdown": {
                            "type": "object",
                            "properties": {
                                "complete_objective": {"type": "string"},
                                "phase_1": {"type": "string"},
                                "phase_2": {"type": "string"},
                                "phase_3": {"type": "string"},
                                "success_criteria": {"type": "string"}
                            },
                            "required": ["complete_objective","phase_1","phase_2","phase_3","success_criteria"]
                        },
                        "frame_allocation": {
                            "type": "object",
                            "properties": {
                                "approach_phase": {"type": "string"},
                                "action_phase": {"type": "string"},
                                "completion_phase": {"type": "string"}
                            },
                            "required": ["approach_phase","action_phase","completion_phase"]
                        },
                        "moving_objects": {"type": "array", "items": {"type": "string"}},
                        "static_objects": {"type": "array", "items": {"type": "string"}},
                        "detailed_timeline": {
                            "type": "object",
                            "properties": {
                                "frame_1": {"type": "string"},
                                "frame_3": {"type": "string"},
                                "frame_6": {"type": "string"},
                                "frame_8": {"type": "string"},
                                "frame_10": {"type": "string"},
                                "frame_12": {"type": "string"},
                                "frame_15": {"type": "string"},
                                "frame_18": {"type": "string"},
                                **{f"frame_{total_frames}": {"type": "string"}}
                            },
                            "required": ["frame_1","frame_3","frame_6","frame_8","frame_10","frame_12","frame_15","frame_18", f"frame_{total_frames}"]
                        },
                        "movement_plans": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "movement_type": {"type": "string", "enum": ["linear","curved","complex"]},
                                    "total_distance": {"type": "string"},
                                    "movement_phases": {
                                        "type": "object",
                                        "properties": {
                                            "phase_1_frames": {"type": "string"},
                                            "phase_2_frames": {"type": "string"},
                                            "phase_3_frames": {"type": "string"}
                                        },
                                        "required": ["phase_1_frames","phase_2_frames","phase_3_frames"]
                                    },
                                    "key_positions": {
                                        "type": "object",
                                        "properties": {
                                            "frame_1":  {"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},
                                            "frame_6":  {"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},
                                            "frame_10": {"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},
                                            "frame_15": {"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},
                                            **{f"frame_{total_frames}": {"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}}
                                        },
                                        "required": ["frame_1","frame_6","frame_10","frame_15", f"frame_{total_frames}"]
                                    }
                                },
                                "required": ["movement_type","total_distance","movement_phases","key_positions"]
                            }
                        },
                        "completion_verification": {"type": "string"}
                    },
                    "required": ["task_breakdown","frame_allocation","moving_objects","static_objects","detailed_timeline","movement_plans","completion_verification"]
                }
            }
        }

        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            tools=[movement_plan_tool],
            tool_choice={"type": "function", "function": {"name": "submit_movement_plan"}}
        )

        msg = response.choices[0].message
        tool_args_json = msg.tool_calls[0].function.arguments
        plan = json.loads(tool_args_json)

        # Normalize labels
        plan["moving_objects"] = [re.sub(r'\([^)]*\)', '', s).strip().lower() for s in plan.get("moving_objects", [])]
        plan["static_objects"] = [re.sub(r'\([^)]*\)', '', s).strip().lower() for s in plan.get("static_objects", [])]

        self.global_plan = plan
        # Scorer will be created with rendering components once they're available
        self.scorer = None

        return plan

    def initialize_scorer(self, detection_map: Dict, object_imgs: Dict, background_image: Image.Image):
        """Initialize the scorer with rendering components for VLM verification"""
        self.detection_map = detection_map
        self.object_imgs = object_imgs
        self.background_image = background_image

        # Create scorer with VLM verification support
        self.scorer = TrajectoryScorer(
            global_plan=self.global_plan,
            vlm_verifier=self.vlm_verifier,
            chunk_generator=self,
            detection_map=detection_map,
            object_imgs=object_imgs,
            background_image=background_image
        )
        print("TrajectoryScorer initialized with VLM verification support")

    def parse_phase_frames(self, phase_description: str) -> Tuple[int, int]:
        """Parse frame range from phase description like 'frames 1-6' or 'Frames 1-6'"""
        frame_pattern = r'(?:frames\s+)?(\d+)-(\d+)'
        match = re.search(frame_pattern, phase_description, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def generate_chunk_candidates(self, start_frame_image: Image.Image,
                                  text_prompt: str,
                                  chunk_start: int,
                                  chunk_end: int,
                                  phase_name: str,
                                  num_candidates: int,
                                  previous_trajectory: List[Dict],
                                  detection_map: Dict,
                                  object_imgs: Dict,
                                  background_image: Image.Image) -> List[TrajectoryChunk]:
        """
        Generate multiple candidate trajectories for a chunk

        Args:
            start_frame_image: Starting frame image
            text_prompt: Original text prompt
            chunk_start: Start frame number
            chunk_end: End frame number
            phase_name: Name of the phase (approach/action/completion)
            num_candidates: Number of candidates to generate
            previous_trajectory: Full trajectory history before this chunk
            detection_map: Detection map
            object_imgs: Object images for rendering
            background_image: Background image

        Returns:
            List of TrajectoryChunk candidates
        """
        print(f"\nGenerating {num_candidates} candidates IN PARALLEL for chunk frames {chunk_start}-{chunk_end} ({phase_name})")

        try:
            # PARALLEL GENERATION: Generate all N candidates in a single API call!
            all_chunk_frames = self._generate_chunk_trajectories_parallel(
                start_frame_image=start_frame_image,
                text_prompt=text_prompt,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                phase_name=phase_name,
                previous_trajectory=previous_trajectory,
                num_candidates=num_candidates
            )

            candidates = []
            for idx, chunk_frames in enumerate(all_chunk_frames):
                if chunk_frames:  # Only add if parsing succeeded
                    print(f"  ✓ Candidate {idx + 1}/{num_candidates}: {len(chunk_frames)} frames generated")
                    chunk = TrajectoryChunk(
                        start_frame=chunk_start,
                        end_frame=chunk_end,
                        phase_name=phase_name,
                        frames=chunk_frames
                    )
                    candidates.append(chunk)
                else:
                    print(f"  ✗ Candidate {idx + 1}/{num_candidates}: Failed to parse")

            # Apply diversity filtering if threshold > 0
            if self.diversity_threshold > 0 and len(candidates) > 1:
                candidates = self.filter_diverse_candidates(candidates)

            return candidates

        except Exception as e:
            print(f"  ⚠ Parallel generation failed: {e}")
            print(f"  Falling back to sequential generation...")

            # FALLBACK: Sequential generation if parallel fails
            return self._generate_chunk_candidates_sequential(
                start_frame_image, text_prompt, chunk_start, chunk_end,
                phase_name, num_candidates, previous_trajectory
            )

    def _generate_chunk_candidates_sequential(self, start_frame_image: Image.Image,
                                              text_prompt: str,
                                              chunk_start: int,
                                              chunk_end: int,
                                              phase_name: str,
                                              num_candidates: int,
                                              previous_trajectory: List[Dict]) -> List[TrajectoryChunk]:
        """
        Fallback: Generate candidates sequentially (original approach)

        This is used when parallel generation fails. Uses temperature variation
        across candidates for diversity.
        """
        candidates = []

        for candidate_idx in range(num_candidates):
            print(f"  Generating candidate {candidate_idx + 1}/{num_candidates} (sequential)...")

            try:
                chunk_frames = self._generate_single_chunk_trajectory(
                    start_frame_image=start_frame_image,
                    text_prompt=text_prompt,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    phase_name=phase_name,
                    previous_trajectory=previous_trajectory,
                    temperature=0.7 + candidate_idx * 0.1  # Vary temperature for diversity
                )

                chunk = TrajectoryChunk(
                    start_frame=chunk_start,
                    end_frame=chunk_end,
                    phase_name=phase_name,
                    frames=chunk_frames
                )

                candidates.append(chunk)

            except Exception as e:
                print(f"    Failed to generate candidate {candidate_idx + 1}: {e}")
                continue

        # Apply diversity filtering if threshold > 0
        if self.diversity_threshold > 0 and len(candidates) > 1:
            candidates = self.filter_diverse_candidates(candidates)

        return candidates

    def _generate_chunk_trajectories_parallel(self, start_frame_image: Image.Image,
                                             text_prompt: str,
                                             chunk_start: int,
                                             chunk_end: int,
                                             phase_name: str,
                                             previous_trajectory: List[Dict],
                                             num_candidates: int) -> List[List[Dict]]:
        """
        Generate multiple chunk trajectories in parallel using OpenAI's n parameter

        This is much more efficient than sequential generation!
        """
        # Get phase context from global plan
        phase_description = ""
        if self.global_plan and 'frame_allocation' in self.global_plan:
            phase_allocation = self.global_plan['frame_allocation']
            for p_name, p_desc in phase_allocation.items():
                if p_name == phase_name:
                    phase_description = p_desc
                    break

        # Get moving/static objects
        moving_objects = self.global_plan.get('moving_objects', []) if self.global_plan else []
        static_objects = self.global_plan.get('static_objects', []) if self.global_plan else []

        # Build context from previous trajectory
        history_text = ""
        if previous_trajectory:
            recent_frames = previous_trajectory[-3:]
            history_text = "\n\nRecent trajectory:\n"
            for frame_data in recent_frames:
                if 'objects' in frame_data:
                    objects_str = ", ".join([f"{obj['label']}: {obj['box']}" for obj in frame_data['objects']])
                    caption = frame_data.get('caption', 'No caption')
                    history_text += f"Frame_{frame_data['frame_number']}: {objects_str}  Caption: {caption}\n"

        # Encode current frame image
        base64_image = encode_pil_image(start_frame_image)

        total_frames = self.global_plan.get('detailed_timeline', {})
        total_frames_num = max([int(k.split('_')[1]) for k in total_frames.keys()]) if total_frames else 20

        # Create prompt for chunk generation
        system_prompt = f"""You are a video motion planning expert generating trajectories for frames {chunk_start} to {chunk_end}.

Current phase: {phase_name}
Phase description: {phase_description}
Total frames in video: {total_frames_num}

**COORDINATE SYSTEM:**
{COORDS_GUIDE}

**DIRECTIONAL MAPPINGS:**
- RIGHT: x1 += delta; x2 += delta
- LEFT:  x1 -= delta; x2 -= delta
- UP:    y1 -= delta; y2 -= delta
- DOWN:  y1 += delta; y2 += delta

Focus ONLY on moving objects: {moving_objects}
Ignore static objects in outputs: {static_objects}
"""

        user_prompt = f"""Text prompt: "{text_prompt}"

{history_text}

Generate a smooth trajectory from frame {chunk_start} to frame {chunk_end} for this {phase_name}.

IMPORTANT: Since multiple trajectories will be generated, explore DIFFERENT valid motion paths.
Consider variations in:
- Path shape (straight, curved, arc)
- Speed profile (constant, accelerating, decelerating)
- Intermediate waypoints (different approaches to the goal)

For EACH frame from {chunk_start} to {chunk_end}, output:
Frame_N: [["object_name", [x1, y1, x2, y2]], ...], caption: <description>

Requirements:
- Smooth motion (delta 0.03-0.08 per frame)
- Consistent with phase objectives
- Maintain object sizes
- Only include moving objects: {moving_objects}
"""

        # Generate N candidates in parallel using the n parameter
        api_params = {
            "model": "gpt-4.1",
            "temperature": self.temperature,
            "n": num_candidates,  # KEY: Generate multiple responses in one call!
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        }

        # Add top_p if specified (for nucleus sampling)
        if self.top_p is not None:
            api_params["top_p"] = self.top_p

        response = self.client.chat.completions.create(**api_params)

        # Parse each candidate response
        all_chunk_frames = []
        for choice in response.choices:
            content = choice.message.content
            chunk_frames = self._parse_chunk_response(content, chunk_start, chunk_end)
            all_chunk_frames.append(chunk_frames)

        return all_chunk_frames

    def _generate_single_chunk_trajectory(self, start_frame_image: Image.Image,
                                         text_prompt: str,
                                         chunk_start: int,
                                         chunk_end: int,
                                         phase_name: str,
                                         previous_trajectory: List[Dict],
                                         temperature: float = 0.7) -> List[Dict]:
        """
        Generate a single trajectory for a chunk (used for fallback)

        This method only generates trajectory data (bounding boxes), not rendered images.
        Frame rendering happens later during the final interpolation step.
        """

        # Get phase context from global plan
        phase_description = ""
        if self.global_plan and 'frame_allocation' in self.global_plan:
            phase_allocation = self.global_plan['frame_allocation']
            for p_name, p_desc in phase_allocation.items():
                if p_name == phase_name:
                    phase_description = p_desc
                    break

        # Get moving/static objects
        moving_objects = self.global_plan.get('moving_objects', []) if self.global_plan else []
        static_objects = self.global_plan.get('static_objects', []) if self.global_plan else []

        # Build context from previous trajectory
        history_text = ""
        if previous_trajectory:
            # Use last few frames for context
            recent_frames = previous_trajectory[-3:]
            history_text = "\n\nRecent trajectory:\n"
            for frame_data in recent_frames:
                if 'objects' in frame_data:
                    objects_str = ", ".join([f"{obj['label']}: {obj['box']}" for obj in frame_data['objects']])
                    caption = frame_data.get('caption', 'No caption')
                    history_text += f"Frame_{frame_data['frame_number']}: {objects_str}  Caption: {caption}\n"

        # Encode current frame image
        base64_image = encode_pil_image(start_frame_image)

        total_frames = self.global_plan.get('detailed_timeline', {})
        total_frames_num = max([int(k.split('_')[1]) for k in total_frames.keys()]) if total_frames else 20

        # Create prompt for chunk generation
        system_prompt = f"""You are a video motion planning expert generating trajectories for frames {chunk_start} to {chunk_end}.

Current phase: {phase_name}
Phase description: {phase_description}
Total frames in video: {total_frames_num}

**COORDINATE SYSTEM:**
{COORDS_GUIDE}

**DIRECTIONAL MAPPINGS:**
- RIGHT: x1 += delta; x2 += delta
- LEFT:  x1 -= delta; x2 -= delta
- UP:    y1 -= delta; y2 -= delta
- DOWN:  y1 += delta; y2 += delta

Focus ONLY on moving objects: {moving_objects}
Ignore static objects in outputs: {static_objects}
"""

        user_prompt = f"""Text prompt: "{text_prompt}"

{history_text}

Generate a smooth trajectory from frame {chunk_start} to frame {chunk_end} for this {phase_name}.

IMPORTANT: Since multiple trajectories will be generated, explore DIFFERENT valid motion paths.
Consider variations in:
- Path shape (straight, curved, arc)
- Speed profile (constant, accelerating, decelerating)
- Intermediate waypoints (different approaches to the goal)

For each object, you should maintain its size (box dimensions) and only change its position unless you are specifically instructed to resize.

For EACH frame from {chunk_start} to {chunk_end}, output:
Frame_N: [["object_name", [x1, y1, x2, y2]], ...], caption: <description>

Requirements:
- Smooth motion (delta 0.03-0.08 per frame)
- Consistent with phase objectives
- Maintain object sizes
- Only include moving objects: {moving_objects}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                temperature=temperature,  # Use temperature for diversity in sequential fallback
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            )

            content = response.choices[0].message.content

            # Parse the response to extract frame-by-frame data
            chunk_frames = self._parse_chunk_response(content, chunk_start, chunk_end)

            return chunk_frames

        except Exception as e:
            print(f"Error generating chunk trajectory: {e}")
            raise

    def _parse_chunk_response(self, response_text: str, chunk_start: int, chunk_end: int) -> List[Dict]:
        """Parse LLM response to extract frame data for the entire chunk"""
        response_text = response_text.lower()
        frames = []

        # Pattern to match Frame_N: [...] caption: ... (comma optional)
        # Use negative lookahead to prevent matching past "caption:" when using DOTALL
        # This ensures we stop at the first ]] before "caption:", not the last ] in the response
        for frame_num in range(chunk_start, chunk_end + 1):
            pattern = rf"frame_{frame_num}:\s*.*?(\[\[.*?\]\]\])\s*.*?caption:\s*(.*?)(?:\n|$)"
            match = re.search(pattern, response_text, re.DOTALL)

            if match:
                try:
                    objects_str = match.group(1)
                    caption = match.group(2).strip()
                    raw_objects = ast.literal_eval(objects_str)

                    objects = []
                    for label, box in raw_objects:
                        objects.append({
                            'label': label,
                            'box': box,
                            'state': 'moving'  # Default state
                        })

                    frame_data = {
                        'frame_number': frame_num,
                        'objects': objects,
                        'caption': caption
                    }

                    frames.append(frame_data)

                except Exception as e:
                    print(f"    Failed to parse Frame_{frame_num}: {e}")
                    print("    Raw data:", match.group(0))
                    print("objects_str:", objects_str)
                    continue
            else:
                print(f"    Warning: Frame_{frame_num} not found in response")

        return frames

    def select_best_chunk(self, candidates: List[TrajectoryChunk],
                         start_boundary: Optional[Dict] = None,
                         end_boundary: Optional[Dict] = None) -> Tuple[TrajectoryChunk, Dict]:
        """
        Select best trajectory chunk from candidates using scoring

        Returns:
            (best_chunk, score_breakdown)
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            score, breakdown = self.scorer.score_trajectory_chunk(candidates[0], start_boundary, end_boundary)
            candidates[0].score = score
            return candidates[0], breakdown

        # Score all candidates
        scored_candidates = []
        for candidate in candidates:
            score, breakdown = self.scorer.score_trajectory_chunk(candidate, start_boundary, end_boundary)
            candidate.score = score
            scored_candidates.append((candidate, score, breakdown))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return best candidate
        best_chunk, best_score, best_breakdown = scored_candidates[0]

        print(f"\n  Selected best candidate with score: {best_score:.4f}")
        print(f"    Breakdown: {best_breakdown}")

        return best_chunk, best_breakdown

    def generate_frame_image(self, frame_data, detection_map, object_imgs, background_image):
        """Generate the actual frame image based on frame_data"""
        # Start with background
        frame_background = background_image.copy()

        # Get moving objects from global plan
        moving_objects = self.global_plan.get('moving_objects', []) if self.global_plan else []

        # Place moving objects at their predicted positions
        for obj_data in frame_data['objects']:
            obj_name = obj_data['label']
            obj_name_with_dot = obj_name+'.' if not obj_name.endswith('.') else obj_name
            obj_name = obj_name.rstrip('.')

            if obj_name_with_dot not in detection_map:
                continue

            if obj_name not in moving_objects:
                continue

            new_bbox = obj_data['box']  # [x1, y1, x2, y2] normalized

            # Skip if box is too small
            bbox_width = new_bbox[2] - new_bbox[0]
            bbox_height = new_bbox[3] - new_bbox[1]
            if bbox_width < 0.01 or bbox_height < 0.01:
                continue

            detection = detection_map[obj_name_with_dot]

            # Plot the moving object at its new position
            if obj_name_with_dot in object_imgs:
                frame_background = self.plot_image(frame_background, object_imgs[obj_name_with_dot], detection, new_bbox)

        return frame_background

    def plot_image(self, background_image, object_image, detection, box2):
        """Plot object on background image at new position"""
        box1 = np.array([detection.box.xmin, detection.box.ymin, detection.box.xmax, detection.box.ymax], dtype=float)
        box1 /= 1024
        box1_width = box1[2] - box1[0]
        box1_height = box1[3] - box1[1]
        box2_width = box2[2] - box2[0]
        box2_height = box2[3] - box2[1]

        if box2_height < 0.01 or box2_width < 0.01:
            return background_image

        # Scale handling
        longer_side = max(box2_width, box2_height)
        if longer_side < 0.15:
            scale_factor = 0.15 / longer_side
        else:
            scale_factor = 1

        box2_width = box2_width * scale_factor
        box2_height = box2_height * scale_factor

        scale_factor = min(box2_width / box1_width, box2_height / box1_height)

        new_width = int(object_image.shape[1] * scale_factor)
        new_height = int(object_image.shape[0] * scale_factor)

        scaled_img = Image.fromarray(object_image).resize((new_width, new_height), Image.Resampling.LANCZOS)

        background = copy.deepcopy(background_image)

        center_x = (box2[0] + box2[2]) / 2
        center_y = (box2[1] + box2[3]) / 2

        bg_width = int(box2_width * 1024)
        bg_height = int(box2_height * 1024)

        x_start = int(center_x * 1024 - bg_width / 2)
        y_start = int(center_y * 1024 - bg_height / 2)

        scaled_img = scaled_img.resize((bg_width, bg_height), Image.Resampling.LANCZOS)

        background.paste(scaled_img, (x_start, y_start), scaled_img)
        return background

    def interpolate_trajectory_to_target_count(self, trajectory_history: List[Dict], target_count: int = 81) -> List[Dict]:
        """Interpolate trajectory data to reach target frame count"""
        if not trajectory_history:
            return []

        if len(trajectory_history) == 1:
            return [copy.deepcopy(trajectory_history[0]) for _ in range(target_count)]

        original_count = len(trajectory_history)

        if original_count >= target_count:
            indices = np.linspace(0, original_count - 1, target_count, dtype=int)
            return [copy.deepcopy(trajectory_history[i]) for i in indices]

        # Interpolate
        interpolated_trajectory = []
        target_indices = np.linspace(0, original_count - 1, target_count)

        for interp_frame_num, target_idx in enumerate(target_indices):
            left_idx = int(np.floor(target_idx))
            right_idx = int(np.ceil(target_idx))

            if left_idx >= original_count - 1:
                frame_data = copy.deepcopy(trajectory_history[-1])
                frame_data['frame_number'] = interp_frame_num + 1
                interpolated_trajectory.append(frame_data)
                continue

            if left_idx == right_idx:
                frame_data = copy.deepcopy(trajectory_history[left_idx])
                frame_data['frame_number'] = interp_frame_num + 1
                interpolated_trajectory.append(frame_data)
                continue

            alpha = target_idx - left_idx
            beta = 1.0 - alpha

            left_frame = trajectory_history[left_idx]
            right_frame = trajectory_history[right_idx]

            interpolated_frame = {
                'frame_number': interp_frame_num + 1,
                'objects': [],
                'caption': left_frame.get('caption', ''),
                'reasoning': f"Interpolated between frame {left_idx + 1} and {right_idx + 1} (alpha={alpha:.3f})"
            }

            left_objects = {obj['label']: obj for obj in left_frame.get('objects', [])}
            right_objects = {obj['label']: obj for obj in right_frame.get('objects', [])}

            for label in left_objects.keys():
                left_obj = left_objects[label]
                if label in right_objects:
                    right_obj = right_objects[label]

                    left_box = np.array(left_obj['box'])
                    right_box = np.array(right_obj['box'])
                    interpolated_box = (beta * left_box + alpha * right_box).tolist()

                    interpolated_obj = {
                        'label': label,
                        'box': interpolated_box,
                        'state': left_obj.get('state', 'unchanged')
                    }
                    interpolated_frame['objects'].append(interpolated_obj)
                else:
                    interpolated_frame['objects'].append(copy.deepcopy(left_obj))

            for label in right_objects.keys():
                if label not in left_objects:
                    interpolated_frame['objects'].append(copy.deepcopy(right_objects[label]))

            interpolated_trajectory.append(interpolated_frame)

        return interpolated_trajectory
