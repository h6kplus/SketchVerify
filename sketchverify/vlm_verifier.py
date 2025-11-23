"""
Vision Language Model verifier for trajectory scoring
"""
import re
import json
import time
import os
import tempfile
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import imageio
import google.generativeai as genai

from utils import encode_pil_image


class VLMVerifier:
    """Vision Language Model verifier for trajectory scoring"""

    def __init__(self, openai_client, gemini_api_key: Optional[str] = None):
        self.openai_client = openai_client
        self.gemini_model = None

        # Initialize Gemini if API key provided
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            # Use gemini-2.5-flash (fast, cost-effective) or gemini-2.5-pro (higher quality)
            # For robotics tasks, consider gemini-robotics-er-1.5-preview
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            print("VLMVerifier: Gemini 2.5-flash initialized successfully")
        else:
            print("VLMVerifier: Gemini API key not provided, physical plausibility will use heuristics")

    def verify_plan_alignment(self, first_frame: Image.Image, last_frame: Image.Image,
                              phase_name: str, phase_description: str,
                              end_goal: str) -> Tuple[float, str]:
        """
        Verify if the last frame aligns with the end goal of the current phase

        Args:
            first_frame: Starting frame of the chunk
            last_frame: Ending frame of the chunk
            phase_name: Name of the phase (e.g., 'approach_phase')
            phase_description: Description of what should happen in this phase
            end_goal: Expected end state for this phase

        Returns:
            (score [0,1], explanation)
        """
        print(f"    VLM: Verifying plan alignment for {phase_name}...")

        # Encode images
        first_b64 = encode_pil_image(first_frame)
        last_b64 = encode_pil_image(last_frame)

        system_prompt = """You are an expert at evaluating video motion trajectories.
Your task is to verify if the motion from the first frame to the last frame aligns with the expected phase goal.

Rate the alignment on a scale of 0.0 to 1.0 where:
- 1.0 = Perfect alignment, the last frame clearly achieves the phase goal
- 0.7-0.9 = Good alignment with minor deviations
- 0.4-0.6 = Partial alignment, some aspects correct
- 0.0-0.3 = Poor alignment, does not achieve the goal

Return ONLY a JSON object with:
{
  "score": <float between 0 and 1>,
  "explanation": "<brief explanation of why this score was given>"
}"""

        user_prompt = f"""Phase: {phase_name}
Phase Description: {phase_description}
Expected End Goal: {end_goal}

Please compare the FIRST frame (starting state) with the LAST frame (ending state).
Does the last frame show that the phase goal has been achieved?

Consider:
- Object positions relative to the goal
- Whether objects moved in the expected direction
- Whether the motion is consistent with the phase description
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Using gpt-4o for vision tasks
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "FIRST FRAME (starting state):"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_b64}"}},
                            {"type": "text", "text": "LAST FRAME (ending state):"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_b64}"}},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            )

            content = response.choices[0].message.content

            # Parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get('score', 0.5))
                explanation = result.get('explanation', 'No explanation provided')

                # Clamp score to [0, 1]
                score = max(0.0, min(1.0, score))

                print(f"    VLM: Plan alignment score = {score:.3f} - {explanation}")
                return score, explanation
            else:
                print(f"    VLM: Failed to parse JSON, defaulting to 0.5")
                return 0.5, "Failed to parse VLM response"

        except Exception as e:
            print(f"    VLM: Plan alignment verification failed: {e}")
            return 0.5, f"Error: {str(e)}"

    def verify_physical_plausibility(self, frames: List[np.ndarray]) -> Tuple[float, str]:
        """
        Verify if the video sequence obeys physical laws using Gemini

        Args:
            frames: List of frame images (numpy arrays)

        Returns:
            (score [0,1], explanation)
        """
        if not self.gemini_model:
            print("    VLM: Gemini not available, skipping physical plausibility check")
            return 1.0, "Gemini not configured, using heuristics only"

        print(f"    VLM: Verifying physical plausibility with Gemini ({len(frames)} frames)...")

        try:
            # Create temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
                tmp_path = tmp_video.name

            # Save frames as video
            with imageio.get_writer(tmp_path, fps=8, codec='libx264') as writer:
                for frame in frames:
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    writer.append_data(frame)

            # Upload video to Gemini
            print(f"    VLM: Uploading video to Gemini...")
            video_file = genai.upload_file(path=tmp_path)

            # Wait for the file to be processed and become ACTIVE
            max_wait_time = 60  # Maximum 60 seconds
            start_time = time.time()
            while video_file.state.name != "ACTIVE":
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(f"Video file processing timeout after {max_wait_time}s (state: {video_file.state.name})")
                if video_file.state.name == "FAILED":
                    raise RuntimeError(f"Video file processing failed: {video_file.state}")
                print(f"    VLM: Waiting for video to be processed (current state: {video_file.state.name})...")
                time.sleep(1)
                video_file = genai.get_file(video_file.name)
            print(f"    VLM: Video ready (state: {video_file.state.name})")

            prompt = """Analyze this video sequence and evaluate whether the motion obeys physical laws.

**IMPORTANT NOTE**: This video is generated by copy & paste composition - each frame is created by
pasting objects onto a background. Therefore, please focus on evaluating the **movement trajectories
and positions of individual objects** across frames, not visual quality, shadows, or composition artifacts.

Consider for each moving object:
1. **Smooth motion**: Do objects move smoothly without sudden jumps or teleportation?
2. **Realistic velocities**: Are the speeds reasonable for the objects shown?
3. **Natural acceleration**: Do objects accelerate/decelerate naturally?
4. **Trajectory consistency**: Do object paths make physical sense?
5. **Collision/interaction physics**: Do objects interact realistically when they get close?
6. **Gravity/orientation**: Are objects properly oriented and affected by gravity if applicable?

Rate the physical plausibility on a scale of 0.0 to 1.0 where:
- 1.0 = Perfectly realistic, obeys all physical laws
- 0.7-0.9 = Mostly realistic with minor issues
- 0.4-0.6 = Some unrealistic aspects but acceptable
- 0.0-0.3 = Highly unrealistic, violates physics (teleportation, impossible speeds, etc.)

Return ONLY a JSON object:
{
  "score": <float between 0 and 1>,
  "explanation": "<brief explanation focusing on object movement quality, highlight any physics violations>"
}"""

            response = self.gemini_model.generate_content([video_file, prompt])

            # Clean up temporary local file and Gemini uploaded file
            try:
                os.unlink(tmp_path)
            except:
                pass

            try:
                genai.delete_file(video_file.name)
                print(f"    VLM: Cleaned up uploaded video file")
            except:
                pass

            # Parse response
            content = response.text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get('score', 0.5))
                explanation = result.get('explanation', 'No explanation provided')

                # Clamp score to [0, 1]
                score = max(0.0, min(1.0, score))

                print(f"    VLM: Physical plausibility score = {score:.3f} - {explanation}")
                return score, explanation
            else:
                print(f"    VLM: Failed to parse Gemini response, defaulting to 0.7")
                return 0.7, "Failed to parse Gemini response"

        except Exception as e:
            print(f"    VLM: Physical plausibility verification failed: {e}")
            return 0.7, f"Error: {str(e)}"
