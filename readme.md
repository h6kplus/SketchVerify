# SketchVerify: Planning with Sketch-Guided Verification for Physics-Aware Video Generation

**[Yidong Huang](https://h6kplus.github.io/owenhuang.github.io/)<sup>1</sup>**, **[Zun Wang](https://zunwang1.github.io/)<sup>1</sup>**, **[Han Lin](https://hl-hanlin.github.io/)<sup>1</sup>**, **[Dong-Ki Kim](https://dkkim93.github.io/)<sup>2</sup>**, **[Shayegan Omidshafiei](https://www.linkedin.com/in/shayegan/)<sup>2</sup>**, **[Jaehong Yoon](https://jaehong31.github.io/)<sup>3</sup>**, **[Yue Zhang](https://zhangyuejoslin.github.io/)<sup>1</sup>**, **[Mohit Bansal](https://www.cs.unc.edu/~mbansal/)<sup>1</sup>**

<sup>1</sup> UNC Chapel Hill  <sup>2</sup> Field AI  <sup>3</sup> Nanyang Technological University


[Paper](https://arxiv.org/abs/2511.17450)  |   [Website](https://sketchverify.github.io/)

Recent video generation approaches increasingly rely on planning intermediate control signals, such as object trajectories, to improve temporal coherence and motion fidelity. However, these methods mostly employ single-shot plans that are typically limited to simple motions, or iterative refinement, which requires multiple calls to the video generator and incurs high computational cost. To overcome these limitations, we propose SketchVerify, a training-free, sketch-verification-based planning framework that improves motion planning quality with dynamically coherent trajectories (i.e., physically plausible and instruction-consistent motions) prior to full video generation, by introducing a test-time sampling and verification loop. Given a prompt and a reference image, our method predicts multiple candidate motion plans and ranks them using a vision–language verifier that jointly evaluates semantic alignment with the instruction and physical plausibility. To efficiently score candidate motion plans, we render each trajectory as a lightweight video sketch by compositing objects over a static background, which bypasses the need for expensive, repeated diffusion-based synthesis while achieving comparable performance. We iteratively refine the motion plan until a satisfactory one is identified, which is then passed to a trajectory-conditioned generator for final synthesis. Experiments on WorldModelBench and PhyWorldBench demonstrate that SketchVerify significantly improves motion quality, physical realism, and long-term consistency compared to strong baselines, while being substantially more efficient. Ablations further show that scaling up the number of trajectory candidates and using multimodal sketch-based verification consistently enhances overall performance.

<p align="center">
<img src="./assets/teaser.gif" alt="teaser image"/>
</p>

##  Todos

- [x] Open-source **single-sample inference pipeline**
- [ ] Open-source **batch processing scripts** for WorldModelBench and PhyWorldBench
- [ ] Open-source **PhyWorldBench utilities**:
  - script for pre-processing
  - script for generating first frames
  - generated first frames for testing


## Environment Setup 

1. Setup image generation and video generation environments. 

```
pip install -r requirements.txt
```

2. Setup Grounded-Segment-Anything environments.

```
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/

python -m pip install -e segment_anything

pip install --no-build-isolation -e GroundingDINO

git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

3. Setup ATI

```
git clone https://github.com/bytedance/ATI.git
cd ATI
pip install .
huggingface-cli download bytedance-research/ATI --local-dir ./Wan2.1-ATI-14B-480P
cp ./Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth ./Wan2.1-ATI-14B-480P/
cp ./Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth ./Wan2.1-ATI-14B-480P/
cp ./Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth ./Wan2.1-ATI-14B-480P/
cp -r ./Wan2.1-I2V-14B-480P/xlm-roberta-large ./Wan2.1-ATI-14B-480P/
cp -r ./Wan2.1-I2V-14B-480P/google ./Wan2.1-ATI-14B-480P/
```

## Stage1: SketchVerify Trajectory Planning

1. The whole SketchVerify pipeline is in `./skechverify`, and definition of each component is defined [here](sketchverify/README.md). For a singel sample, we can use this script to generate the trajectory plan for a input text prompt and image. 

```
CUDA_VISIBLE_DEVICES=0  python sketchverify/pipeline.py \
    --text_prompt "A robot arm picks up a cube" \
    --frame1_image assets/sample.jpg
    --api_key OPENAI_API_KEY
    --gemini_api_key GEMINI_API_KEY
    --output_file result.json
    --output_path output
```

* `api_key`: OpenAI API key. 

* `gemini_api_key`: Gemini API key.

* `text_prompt`: Input text prompts.

* `frame1_image`: The input condition image.

* `output_file`: Path to save the output plan.

* `output_path`: Path to save the intermediate images and video sketches.

2. Turn output trajectory plan to the format of ATI input


```
 python sketchverify/convert.py \
    --input_plan result.json \
    --frame1_image assets/sample.jpg
    --output_path ati
```

* `input_plan`: Same as `output_file` in last step.


* `frame1_image`: The input condition image.

* `output_path`: Path to save the trajectory plan in ATI input format. Should include `input.jpg`, `plan.pth`, `test.yaml`.

## Stage2: Video Generation with TrajectoryPlan

1. Generate videos with the video sketch as condition in noise initialization 

```
cd ATI
./run_example.sh -c Wan2.1-ATI-14B-480P -p ../ati/test.yaml -o output
```

* `-c`: Path to the ATI checkpoint

* `-p`: Path to hte saved trajectory plan

* `-o`: Path to output video saving directory


## Citation

If you find this work useful, please consider citing:

```bibtex

@misc{huang2025sketchverify,
  title         = {SketchVerify: Planning with Sketch-Guided Verification for Physics-Aware Video Generation},
  author        = {Huang, Yidong and Wang, Zun and Lin, Han and Kim, Dong-Ki and Omidshafiei, Shayegan and Yoon, Jaehong and Zhang, Yue and Bansal, Mohit},
  year          = {2025},
  eprint        = {2511.17450},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

```
