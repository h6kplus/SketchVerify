# Sketch Verify - Modular Architecture

This directory contains the refactored, modular version of the chunk-based grounded planning pipeline with test-time search.

## Module Structure

The code has been organized into focused modules for better maintainability:

### Core Data Structures
- **`data_structures.py`** - Data classes (BoundingBox, DetectionResult, TrajectoryChunk)

### Utilities
- **`utils.py`** - Image encoding, normalization, compositing, and coordinate system guide
- **`video_utils.py`** - Video saving and rendering utilities

### Pipeline Components
- **`vlm_verifier.py`** - VLMVerifier class for trajectory verification using OpenAI and Gemini
- **`object_proposal.py`** - ObjectProposalGenerator for generating object proposals with GPT-4 Vision
- **`object_detector.py`** - GroundedObjectDetector for detection and segmentation using Grounding DINO + SAM
- **`flux_generator.py`** - FluxControlRemovalGenerator for background generation
- **`trajectory_scorer.py`** - TrajectoryScorer for scoring trajectory candidates
- **`chunk_generator.py`** - ChunkBasedGenerator for chunk-based trajectory generation with test-time search

### Main Pipeline
- **`pipeline.py`** - Main entry point with CLI interface
