"""
Grounded object detection and segmentation using Grounding DINO + SAM
"""
import torch
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline
from transformers import SamModel, SamProcessor

from data_structures import DetectionResult, BoundingBox


class GroundedObjectDetector:
    """Step 2: Detect and segment proposed objects using Grounding DINO + SAM"""

    def __init__(self, detector_id="IDEA-Research/grounding-dino-base",
                 segmenter_id="facebook/sam-vit-base", device="cuda"):
        self.device = device
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self._load_models()

    def _load_models(self):
        """Load detection and segmentation models"""
        print("Loading Grounding DINO and SAM models...")

        self.object_detector = hf_pipeline(
            model=self.detector_id,
            task="zero-shot-object-detection",
            device=0 if self.device == "cuda" else -1
        )

        self.segmentator = SamModel.from_pretrained(self.segmenter_id).to(self.device)
        self.processor = SamProcessor.from_pretrained(self.segmenter_id)

        print("Models loaded successfully!")

    def detect_and_segment(self, image, labels, threshold=0.3):
        """Complete pipeline: detect + segment"""
        print(labels)
        if not labels:
            return []

        # Ensure labels end with periods for DINO
        formatted_labels = [label if label.endswith(".") else label+"." for label in labels]

        try:
            # Detection
            detections = self.object_detector(image, candidate_labels=formatted_labels, threshold=threshold)
            detections = [DetectionResult.from_dict(det) for det in detections]

            if not detections:
                return detections

            # Segmentation
            boxes = [[det.box.xmin, det.box.ymin, det.box.xmax, det.box.ymax] for det in detections]
            inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.segmentator(**inputs)

            masks = self.processor.post_process_masks(
                masks=outputs.pred_masks,
                original_sizes=inputs.original_sizes,
                reshaped_input_sizes=inputs.reshaped_input_sizes
            )[0]

            # Add masks to detections
            for i, detection in enumerate(detections):
                if i < len(masks):
                    mask_np = masks[i].cpu().numpy().squeeze()
                    mask_np = (mask_np > 0.5).astype(np.uint8)
                    detection.mask = mask_np

            return detections

        except Exception as e:
            print(f"Detection/Segmentation failed: {e}")
            return []

    def extract_moving_objects_from_proposals(self, proposals):
        """Extract moving objects from proposals"""
        moving_objects = set()

        # From moving_objects list
        if "moving_objects" in proposals:
            moving_objects.update(proposals["moving_objects"])

        # Clean object names
        cleaned_objects = [obj.lower().strip() for obj in moving_objects
                          if obj.lower().strip() not in ['the', 'a', 'an', 'and', 'or', 'in', 'on', 'at']]
        return cleaned_objects

    def detect_and_segment_moving_objects(self, image, proposals, threshold=0.3):
        """Detect and segment moving objects with fallback strategies"""
        moving_objects = self.extract_moving_objects_from_proposals(proposals)

        if not moving_objects:
            return []

        # Try detection with multiple strategies
        detection_results = []

        # Strategy 1: Original names
        detection_results.extend(self._try_detection(image, moving_objects, threshold))

        # Strategy 2: Lower threshold if needed
        if not detection_results and threshold > 0.1:
            detection_results.extend(self._try_detection(image, moving_objects, threshold=0.15))

        if not detection_results:
            return []

        # Filter to keep only the highest confidence detection for each object type
        detection_results = self._filter_best_detections_per_object(detection_results, moving_objects)

        # Segmentation
        return self._segment_detections(image, detection_results)

    def _try_detection(self, image, object_labels, threshold):
        """Try detection with given labels"""
        if not object_labels:
            return []

        formatted_labels = [label if label.endswith(".") else label+"." for label in object_labels]

        try:
            detections = self.object_detector(image, candidate_labels=formatted_labels, threshold=threshold)
            return [DetectionResult.from_dict(det) for det in detections]
        except:
            return []

    def _filter_best_detections_per_object(self, detection_results, target_objects):
        """Filter detections to keep only the highest confidence detection for each target object."""
        # Group detections by object type
        object_detections = {}

        for detection in detection_results:
            # Find which target object this detection corresponds to
            matched_object = self._match_detection_to_target_object(detection.label, target_objects)

            if matched_object:
                if matched_object not in object_detections:
                    object_detections[matched_object] = []
                object_detections[matched_object].append(detection)

        # Keep only the highest confidence detection for each object type
        filtered_detections = []
        for obj_name, detections in object_detections.items():
            # Sort by confidence score (descending) and take the best one
            best_detection = max(detections, key=lambda d: d.score)
            filtered_detections.append(best_detection)
            print(f"Selected best detection for '{obj_name}': {best_detection.label} (score: {best_detection.score:.3f})")

        return filtered_detections

    def _match_detection_to_target_object(self, detection_label, target_objects):
        """Match a detection label to the closest target object name."""
        detection_label_lower = detection_label.lower().strip('.')

        # Direct match
        for target_obj in target_objects:
            if target_obj.lower() == detection_label_lower:
                return target_obj

        # Check if detection label contains target object or vice versa
        for target_obj in target_objects:
            target_lower = target_obj.lower()
            if (target_lower in detection_label_lower or
                detection_label_lower in target_lower):
                return target_obj

        # If no match found, return the detection label itself
        return detection_label_lower

    def _segment_detections(self, image, detection_results):
        """Generate segmentation masks using SAM"""
        boxes = [[det.box.xmin, det.box.ymin, det.box.xmax, det.box.ymax] for det in detection_results]

        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.segmentator(**inputs)

        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        img_height, img_width = np.array(image).shape[:2]

        for i, detection in enumerate(detection_results):
            if i < len(masks):
                mask_tensor = masks[i]
                mask_np = mask_tensor.cpu().numpy()

                # Handle multi-channel masks
                if mask_np.ndim == 3:
                    if mask_np.shape[0] in [1, 3, 4]:  # Channels first
                        mask_np = mask_np[0]
                    elif mask_np.shape[-1] in [1, 3, 4]:  # Channels last
                        mask_np = mask_np[:, :, 0]
                    else:
                        mask_np = mask_np.squeeze()
                elif mask_np.ndim == 1:
                    if mask_np.size == img_height * img_width:
                        mask_np = mask_np.reshape(img_height, img_width)
                    else:
                        continue
                elif mask_np.ndim != 2:
                    mask_np = mask_np.squeeze()
                    if mask_np.ndim != 2:
                        continue

                # Resize if needed
                if mask_np.shape != (img_height, img_width):
                    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((img_width, img_height), Image.Resampling.NEAREST)
                    mask_np = np.array(mask_pil).astype(np.float32) / 255.0

                mask_binary = (mask_np > 0.5).astype(np.uint8)
                detection.mask = mask_binary

        return detection_results
