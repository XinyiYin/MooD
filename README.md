# MooD

Official implementation of **MooD: Perception-Enhanced Efficient Affective Image Editing via Continuous Valence-Arousal Modeling**.

MooD is a Valence-Arousal (VA)-driven framework for fine-grained and efficient affective image editing. Given a source image and a continuous target VA pair, MooD retrieves an affectively aligned reference image and edits the source through two complementary channels:

- visual affective cue transfer with an image adapter;
- perception-enhanced semantic guidance with the Affective Semantic Projection Network (ASP-Net).

The framework is designed to preserve the source content while providing continuous emotional control without requiring long text prompts or iterative reasoning at inference time.

## Highlights

- **Continuous affect control:** directly uses Valence-Arousal values as editing conditions.
- **VA-aware retrieval:** retrieves references that are both emotionally aligned and visually compatible with the source image.
- **Dual-channel guidance:** combines IP-Adapter visual transfer with ASP-Net semantic guidance in the SDXL text embedding space.
- **Efficient inference:** uses a lightweight editing pipeline suitable for large-scale affective image editing.
- **AffectSet:** a VA-annotated visual affective dataset constructed from EmoSet for model optimization and evaluation.

## Repository Structure

```text
MooD/
├── affect_encoder/
│   ├── encoder.py              # Emotion encoder architecture
│   └── train_encoder.py        # Emotion encoder training
├── inference/
│   ├── inference.py            # Dual-channel image editing
│   └── metrics.py              # Quantitative evaluation
├── model/
│   ├── att_loss.py             # VA-aware attribute loss
│   ├── dataset.py              # AffectSet data loader
│   └── model.py                # Mapper, projector, and VA regressor
├── precompute/
│   ├── attr_list.json          # Affect-related object attributes
│   └── attr_va_stats.pt        # Precomputed affective attribute weights
├── train.py                    # Main MooD training entry point
├── train.sh                    # Multi-GPU training example
└── requirements.txt
```

## Installation

The code was validated with Python 3.12.3, PyTorch 2.9.0, and CUDA 12.8.

```bash
conda create -n mood python=3.12.3 -y
conda activate mood

# Install the CUDA-enabled PyTorch build first.
pip install torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Install the remaining dependencies.
pip install -r requirements.txt
```



## Required Models and Assets

Prepare the following models for the workflow you intend to run:

| Asset | Purpose | Required for |
| --- | --- | --- |
| Stable Diffusion XL base model | Diffusion backbone | Training and inference |
| IP-Adapter for SDXL | Visual affective cue transfer | Inference |
| OpenAI CLIP weights | Attribute embedding and evaluation | Training and evaluation |

**Additional asset:** Download `attr_va_stats.pt` from the [GitHub Releases](https://github.com/XinyiYin/MooD/releases) page. This file is required for both training and inference.




## Data Preparation

The main training script expects the following directory structure:

```text
DATA_ROOT/
├── image/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── annotation/
    ├── 000000.json
    ├── 000001.json
    └── ...
```

Each annotation file should contain continuous VA labels and an object list:

```json
{
  "valence": 0.85,
  "arousal": 0.65,
  "object": ["Flower", "Plant"]
}
```

## Train the Emotion Encoder

Set `image_dir`, `annotation_dir`, and `save_dir` in `affect_encoder/train_encoder.py`, then run from the repository root:

```bash
python -m affect_encoder.train_encoder
```

The resulting `best_model.pth` is used by the main MooD training and evaluation pipelines.

## Train MooD

Before training:

1. set the EmotionEncoder checkpoint path in `train.py`;
2. set the CLIP model path used for attribute embeddings in `train.py`;
3. fill in the dataset, model, output, and precomputed-statistics paths in `train.sh`;
4. run `accelerate config` once for the current machine.

Start training with:

```bash
bash train.sh
```

The main outputs include periodic Accelerate checkpoints, `modemo_final.pth`, and the best validation checkpoint under `best_model/modemo_best.pth`.

## Inference

Input images are expected to be named `1.png`, `2.png`, and so on. Run single-image inference from the repository root:

```bash
python -m inference.inference \
  --input_dir /path/to/input_images \
  --output_dir /path/to/output_images \
  --sdxl_path /path/to/sdxl \
  --ip_adapter_path /path/to/IP-Adapter \
  --ip_adapter_weight_name ip-adapter_sdxl.bin \
  --attr_va_stats_path /path/to/attr_va_stats.pt \
  --checkpoint_path /path/to/modemo_best.pth \
  --image_idx 1 \
  --target_valence 0.5 \
  --target_arousal 0.4 \
  --save_comparison
```

Important arguments:

- `--target_valence` and `--target_arousal`: target emotion coordinates;
- `--ip_scale`: strength of the visual transfer channel;
- `--norm_scale`: strength of the semantic token channel;
- `--va_threshold`: VA-space retrieval radius;
- `--strength`: SDXL image-to-image editing strength.

## Evaluation

Evaluate generated images with PSNR, SSIM, perceptual similarity, CLIP image similarity, and VA prediction error:

```bash
python -m inference.metrics \
  --input_dir /path/to/input_images \
  --output_dir /path/to/output_images \
  --results_save_dir /path/to/evaluation_results \
  --alexnet_weights /path/to/alexnet_weights.pth \
  --lpips_weights /path/to/lpips_weights.pth \
  --clip_weights /path/to/CLIP-ViT-L-14.pt \
  --emotion_encoder_path /path/to/emotion_encoder/best_model.pth \
  --target_valence 0.5 \
  --target_arousal 0.4
```

Use `--measure_efficiency` together with the inference model paths to report runtime and peak GPU memory.


## Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{11740160,
  author={Yin, Xinyi and Wang, Yiduo and Hu, Tingqi and Si, Meicong and Shi, Yunyun and Chen, Shi and Wang, Hao and Xue, Junxiao and Wu, Xuecheng},
  journal={IEEE Transactions on Computational Social Systems}, 
  title={MooD: Perception-Enhanced Efficient Affective Image Editing via Continuous Valence–Arousal Modeling}, 
  year={2026},
  pages={1-11},
  doi={10.1109/TCSS.2026.3737407}
}

```

## Acknowledgements

This project builds upon Stable Diffusion XL, Diffusers, Transformers, IP-Adapter, OpenAI CLIP, and EmoSet. We thank the authors and maintainers of these projects and datasets.

## License

The code is released under the [Apache License 2.0](LICENSE). Third-party models, datasets, and pretrained weights remain subject to their respective licenses and terms of use.
