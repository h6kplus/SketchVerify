"""
Data structures for the grounded planning pipeline
"""
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


@dataclass
class TrajectoryChunk:
    """Represents a chunk of trajectory frames"""
    start_frame: int
    end_frame: int
    phase_name: str
    frames: List[Dict]  # List of frame data dicts
    score: float = 0.0

    def __len__(self):
        return len(self.frames)
