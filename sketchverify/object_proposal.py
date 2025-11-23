"""
Object proposal generation using GPT-4 Vision
"""
import re
import json

from utils import encode_image


class ObjectProposalGenerator:
    """Step 1: Generate object proposals using GPT-4 Vision"""

    def __init__(self, client):
        self.client = client

    def generate_proposals(self, image_path, text_prompt):
        """Analyze frame1 and text prompt to propose what objects should be added/moved"""
        base64_image = encode_image(image_path)

        system_prompt = '''You are an expert in video generation and object placement. Given the first frame of a video and a text description, propose what objects should be added, moved, or animated to fulfill the text prompt.

Your task:
1. Analyze the current frame and identify existing objects
2. Based on the text prompt, determine what new objects need to be added
3. Identify which objects (existing or new) should move/animate
4. Consider realistic object placement and movement
5. Focus on larger objects that significantly impact the scene, and treat objects as whole entities if they are attached together.
6. For each proposal, use the minimal necessary detail to convey the idea, the length should be 1-3 words and as short as possible

Output JSON format:
{
  "scene_analysis": "Brief description of the current frame",
  "existing_objects": ["list", "of", "objects", "already", "in", "frame"],
  "objects_to_add": [
    {
      "name": "object_name",
      "reasoning": "why this object is needed",
      "movement_type": "static/linear/curved/complex",
      "priority": "high/medium/low"
    }
  ],
  "moving_objects": ["objects", "that", "should", "move"],
  "static_objects": ["objects", "that", "should", "stay", "static"],
}'''

        user_prompt = f'''
Text prompt: "{text_prompt}"

Analyze this frame and the text prompt. What objects should be added or animated to fulfill this prompt?

Focus on:
- What objects are missing that the prompt requires
- Which objects should move to create the described action
- Which objects should remain static as background
- Realistic object placement and timing
'''

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )

            content = response.choices[0].message.content

            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    proposals = json.loads(json_match.group())
                    for moving_object in proposals["moving_objects"]:
                        moving_object = re.sub(r'\([^)]*\)', '', moving_object).strip().lower()
                    return proposals
                else:
                    return {"error": "No valid JSON found in response", "raw": content}
            except json.JSONDecodeError as e:
                return {"error": f"JSON decode error: {e}", "raw_response": content}

        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}
