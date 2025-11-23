"""
Trajectory scoring for test-time search
"""
import numpy as np
from typing import Dict, List, Optional
from PIL import Image

from data_structures import TrajectoryChunk


class TrajectoryScorer:
    """Score trajectory candidates for test-time search"""

    def __init__(self, global_plan: Dict, vlm_verifier: Optional['VLMVerifier'] = None,
                 chunk_generator: Optional['ChunkBasedGenerator'] = None,
                 detection_map: Optional[Dict] = None,
                 object_imgs: Optional[Dict] = None,
                 background_image: Optional[Image.Image] = None):
        self.global_plan = global_plan
        self.vlm_verifier = vlm_verifier
        self.chunk_generator = chunk_generator
        self.detection_map = detection_map
        self.object_imgs = object_imgs
        self.background_image = background_image

    def score_trajectory_chunk(self, chunk: TrajectoryChunk,
                               start_boundary: Optional[Dict] = None,
                               end_boundary: Optional[Dict] = None) -> float:
        """
        Score a trajectory chunk based on multiple criteria

        Args:
            chunk: TrajectoryChunk to score
            start_boundary: Expected start state (frame data with object positions)
            end_boundary: Expected end state (frame data with object positions)

        Returns:
            Combined score (higher is better)
        """
        scores = {}

        # 1. Boundary consistency (40% weight)
        scores['boundary'] = self._score_boundary_consistency(chunk, start_boundary, end_boundary)

        # 2. Motion smoothness (30% weight)
        scores['smoothness'] = self._score_motion_smoothness(chunk)

        # 3. Global plan alignment (20% weight)
        scores['plan_alignment'] = self._score_plan_alignment(chunk)

        # 4. Physical plausibility (10% weight)
        scores['plausibility'] = self._score_physical_plausibility(chunk)

        # Weighted combination
        weights = {
            'boundary': 0.0,
            'smoothness': 0.0,
            'plan_alignment': 0.5,
            'plausibility': 0.5
        }

        total_score = sum(scores[key] * weights[key] for key in scores)

        return total_score, scores

    def _score_boundary_consistency(self, chunk: TrajectoryChunk,
                                   start_boundary: Optional[Dict],
                                   end_boundary: Optional[Dict]) -> float:
        """Score how well chunk respects boundary conditions"""
        if not chunk.frames:
            return 0.0

        score = 0.0
        count = 0

        # Check start boundary
        if start_boundary and 'objects' in start_boundary:
            start_frame = chunk.frames[0]
            if 'objects' in start_frame:
                start_score = self._compare_object_positions(
                    start_boundary['objects'],
                    start_frame['objects']
                )
                score += start_score
                count += 1

        # Check end boundary
        if end_boundary and 'objects' in end_boundary:
            end_frame = chunk.frames[-1]
            if 'objects' in end_frame:
                end_score = self._compare_object_positions(
                    end_boundary['objects'],
                    end_frame['objects']
                )
                score += end_score
                count += 1

        return score / count if count > 0 else 1.0

    def _compare_object_positions(self, expected_objects: List[Dict],
                                 actual_objects: List[Dict]) -> float:
        """Compare object positions between expected and actual"""
        expected_dict = {obj['label']: obj['box'] for obj in expected_objects}
        actual_dict = {obj['label']: obj['box'] for obj in actual_objects}

        scores = []
        for label in expected_dict:
            if label in actual_dict:
                expected_box = np.array(expected_dict[label])
                actual_box = np.array(actual_dict[label])

                # L2 distance normalized by diagonal
                distance = np.linalg.norm(expected_box - actual_box)
                diagonal = np.sqrt(2)  # Max possible distance in normalized coords
                similarity = 1.0 - min(distance / diagonal, 1.0)
                scores.append(similarity)

        return np.mean(scores) if scores else 0.0

    def _score_motion_smoothness(self, chunk: TrajectoryChunk) -> float:
        """Score trajectory smoothness using velocity and acceleration"""
        if len(chunk.frames) < 3:
            return 1.0

        # Extract trajectories for each object
        object_trajectories = {}
        for frame in chunk.frames:
            if 'objects' not in frame:
                continue
            for obj in frame['objects']:
                label = obj['label']
                if label not in object_trajectories:
                    object_trajectories[label] = []
                # Use center of bounding box
                box = obj['box']
                center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                object_trajectories[label].append(center)

        # Compute smoothness for each object
        smoothness_scores = []
        for label, trajectory in object_trajectories.items():
            if len(trajectory) < 3:
                continue

            trajectory = np.array(trajectory)

            # Compute velocities
            velocities = np.diff(trajectory, axis=0)

            # Compute accelerations
            accelerations = np.diff(velocities, axis=0)

            # Smoothness = low acceleration variance
            accel_variance = np.var(accelerations)
            smoothness = np.exp(-accel_variance * 100)  # Scale and convert to [0,1]
            smoothness_scores.append(smoothness)

        return np.mean(smoothness_scores) if smoothness_scores else 1.0

    def _score_plan_alignment(self, chunk: TrajectoryChunk) -> float:
        """Score alignment with global plan expectations using VLM verification"""

        # If VLM verifier is available and we have rendering capabilities, use VLM verification
        if (self.vlm_verifier and self.chunk_generator and
            self.detection_map and self.object_imgs and self.background_image):

            try:
                # Render first and last frames of the chunk
                first_frame_data = chunk.frames[0]
                last_frame_data = chunk.frames[-1]

                first_frame_img = self.chunk_generator.generate_frame_image(
                    first_frame_data, self.detection_map, self.object_imgs, self.background_image
                )
                last_frame_img = self.chunk_generator.generate_frame_image(
                    last_frame_data, self.detection_map, self.object_imgs, self.background_image
                )

                # Get phase description and end goal from global plan
                phase_description = ""
                end_goal = ""

                if self.global_plan and 'frame_allocation' in self.global_plan:
                    phase_allocation = self.global_plan['frame_allocation']
                    phase_description = phase_allocation.get(chunk.phase_name, "")

                if self.global_plan and 'detailed_timeline' in self.global_plan:
                    timeline = self.global_plan['detailed_timeline']
                    end_goal = timeline.get(f'frame_{chunk.end_frame}', "")

                # Use VLM to verify alignment
                vlm_score, explanation = self.vlm_verifier.verify_plan_alignment(
                    first_frame_img, last_frame_img,
                    chunk.phase_name, phase_description, end_goal
                )

                return vlm_score

            except Exception as e:
                print(f"    Warning: VLM plan alignment failed, using heuristic: {e}")
                # Fall back to heuristic method

        # Heuristic method (original implementation)
        if not self.global_plan or 'movement_plans' not in self.global_plan:
            return 1.0

        movement_plans = self.global_plan['movement_plans']

        # Check if objects move in expected directions during this phase
        scores = []
        for obj_name, plan in movement_plans.items():
            if 'key_positions' not in plan:
                continue

            # Find expected positions for this chunk's frames
            key_positions = plan['key_positions']

            # Get trajectory for this object in the chunk
            obj_trajectory = self._extract_object_trajectory(chunk, obj_name)
            if not obj_trajectory:
                continue

            # Compare with key positions
            alignment_score = self._compute_trajectory_alignment(
                obj_trajectory, key_positions, chunk.start_frame, chunk.end_frame
            )
            scores.append(alignment_score)

        return np.mean(scores) if scores else 1.0

    def _extract_object_trajectory(self, chunk: TrajectoryChunk, obj_name: str) -> List[np.ndarray]:
        """Extract trajectory for a specific object"""
        trajectory = []
        for frame in chunk.frames:
            if 'objects' not in frame:
                continue
            for obj in frame['objects']:
                if obj['label'].rstrip('.') == obj_name.rstrip('.'):
                    box = obj['box']
                    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    trajectory.append(center)
                    break
        return trajectory

    def _compute_trajectory_alignment(self, trajectory: List[np.ndarray],
                                     key_positions: Dict, start_frame: int,
                                     end_frame: int) -> float:
        """Compute alignment between trajectory and key positions"""
        if not trajectory or not key_positions:
            return 1.0

        # Find relevant key positions
        relevant_keys = []
        for key, pos in key_positions.items():
            frame_num = int(key.split('_')[1])
            if start_frame <= frame_num <= end_frame:
                relevant_keys.append((frame_num, pos))

        if not relevant_keys:
            return 1.0

        # Compute distances to expected positions
        scores = []
        for frame_num, expected_box in relevant_keys:
            # Map frame_num to trajectory index
            traj_idx = frame_num - start_frame
            if 0 <= traj_idx < len(trajectory):
                actual_center = trajectory[traj_idx]
                expected_center = np.array([
                    (expected_box[0] + expected_box[2]) / 2,
                    (expected_box[1] + expected_box[3]) / 2
                ])
                distance = np.linalg.norm(actual_center - expected_center)
                score = 1.0 - min(distance / np.sqrt(2), 1.0)
                scores.append(score)

        return np.mean(scores) if scores else 1.0

    def _score_physical_plausibility(self, chunk: TrajectoryChunk) -> float:
        """Score physical plausibility using VLM verification with rendered video"""

        # If VLM verifier is available and we have rendering capabilities, use VLM verification
        if (self.vlm_verifier and self.vlm_verifier.gemini_model and
            self.chunk_generator and self.detection_map and
            self.object_imgs and self.background_image):

            try:
                # Render all frames in the chunk
                rendered_frames = []
                for frame_data in chunk.frames:
                    frame_img = self.chunk_generator.generate_frame_image(
                        frame_data, self.detection_map, self.object_imgs, self.background_image
                    )
                    rendered_frames.append(np.array(frame_img))

                # Use VLM (Gemini) to verify physical plausibility
                vlm_score, explanation = self.vlm_verifier.verify_physical_plausibility(rendered_frames)

                return vlm_score

            except Exception as e:
                print(f"    Warning: VLM physical plausibility check failed, using heuristic: {e}")
                # Fall back to heuristic method

        # Heuristic method (original implementation)
        if len(chunk.frames) < 2:
            return 1.0

        # Check for reasonable velocities
        max_velocity_per_frame = 0.15  # Max 15% of screen per frame in normalized coords

        scores = []
        for i in range(len(chunk.frames) - 1):
            current_frame = chunk.frames[i]
            next_frame = chunk.frames[i + 1]

            if 'objects' not in current_frame or 'objects' not in next_frame:
                continue

            current_objs = {obj['label']: obj['box'] for obj in current_frame['objects']}
            next_objs = {obj['label']: obj['box'] for obj in next_frame['objects']}

            for label in current_objs:
                if label not in next_objs:
                    continue

                current_box = np.array(current_objs[label])
                next_box = np.array(next_objs[label])

                # Compute center movement
                current_center = (current_box[:2] + current_box[2:]) / 2
                next_center = (next_box[:2] + next_box[2:]) / 2
                velocity = np.linalg.norm(next_center - current_center)

                # Score based on velocity
                if velocity > max_velocity_per_frame:
                    # Penalize high velocities
                    score = max(0, 1.0 - (velocity - max_velocity_per_frame) * 5)
                else:
                    score = 1.0

                scores.append(score)

        return np.mean(scores) if scores else 1.0
