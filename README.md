---
license: cc-by-nc-4.0
gated: true
extra_gated_heading: SeFi-Image Non-Commercial License Agreement
extra_gated_prompt: >-
  By clicking "Agree and access repository", you acknowledge that you have
  read and agree to the Creative Commons Attribution-NonCommercial 4.0
  International license (CC BY-NC 4.0). You agree to use SeFi-Image
  checkpoints for non-commercial purposes only and to comply with all
  applicable laws and responsible AI use requirements.
extra_gated_fields:
  I agree to use SeFi-Image checkpoints for non-commercial use only: checkbox
extra_gated_button_content: Agree and access repository
language:
- en
- zh
pipeline_tag: text-to-image
library_name: sefi
tags:
- text-to-image
- image-generation
- safetensors
- bilingual-text-rendering
- semantic-first-diffusion
- gated
---

# SeFi-Image

<p>
  <a href="https://jmliu206.github.io/sefi-web/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-5c6bc0" height="22" /></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2606.22568"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2606.22568-b31b1b" height="22" /></a>
  &nbsp;
  <a href="https://github.com/jmliu206/SeFi-Image"><img alt="Inference Code" src="https://img.shields.io/badge/Inference-Code-181717?logo=github&logoColor=white" height="22" /></a>
  &nbsp;
  <a href="https://huggingface.co/SeFi-Image"><img alt="Hugging Face Models" src="https://img.shields.io/badge/Hugging%20Face-Models-yellow" height="22" /></a>
</p>

**SeFi-Image** is a text-to-image foundation model family built with
**Semantic-First Diffusion**. It separates generation into semantic and texture
latent streams, denoising semantic structure slightly ahead of texture details.
This design gives the texture stream a cleaner structural anchor and improves
the reconstruction-generation trade-off in latent diffusion.

<table>
  <tr>
    <td width="50%"><img src="assets/teaser_canvas.jpg" alt="SeFi-Image generated examples"></td>
    <td width="50%"><img src="assets/teaser_canvas_2.png" alt="More SeFi-Image generated examples"></td>
  </tr>
</table>

## Highlights

<table>
  <tr>
    <td width="33%" align="center">
      <img src="assets/icon_semantic.png" width="58" alt="Semantic-first generation icon"><br>
      <b>Semantic-first generation</b><br>
      <sub>Semantic latents denoise ahead of texture latents, providing a cleaner structural anchor for image synthesis.</sub>
    </td>
    <td width="33%" align="center">
      <img src="assets/icon_speed.png" width="58" alt="Faster training icon"><br>
      <b>Faster training</b><br>
      <sub>The 5B model reaches strong benchmark performance with about <b>125K A800 GPU hours</b>.</sub>
    </td>
    <td width="33%" align="center">
      <img src="assets/icon_tradeoff.png" width="58" alt="Generation-reconstruction trade-off icon"><br>
      <b>Better generation-reconstruction trade-off</b><br>
      <sub>A high-fidelity texture latent preserves reconstruction detail, while a compact semantic latent simplifies generation.</sub>
    </td>
  </tr>
</table>

## Performance

The following numbers follow the main evaluation tables in the SeFi-Image
technical report and summarize SeFi-Image-5B across representative benchmarks.

![SeFi-Image-5B performance overview](assets/performance_overview.png)

## Model Zoo

| Family | Model | Checkpoint | Steps | Guidance |
| :--- | :--- | :--- | :---: | :---: |
| Base | SeFi-Image-1B-Base | [SeFi-Image/SeFi-Image-1B-Base](https://huggingface.co/SeFi-Image/SeFi-Image-1B-Base) | 50 | 4.0 |
| Base | SeFi-Image-2B-Base | [SeFi-Image/SeFi-Image-2B-Base](https://huggingface.co/SeFi-Image/SeFi-Image-2B-Base) | 50 | 4.0 |
| Base | SeFi-Image-5B-Base | [SeFi-Image/SeFi-Image-5B-Base](https://huggingface.co/SeFi-Image/SeFi-Image-5B-Base) | 50 | 4.0 |
| RL | SeFi-Image-5B-RL | [SeFi-Image/SeFi-Image-5B-RL](https://huggingface.co/SeFi-Image/SeFi-Image-5B-RL) | 50 | 4.0 |
| Turbo | SeFi-Image-1B-turbo | [SeFi-Image/SeFi-Image-1B-turbo](https://huggingface.co/SeFi-Image/SeFi-Image-1B-turbo) | 4 | 1.0 |
| Turbo | SeFi-Image-2B-turbo | [SeFi-Image/SeFi-Image-2B-turbo](https://huggingface.co/SeFi-Image/SeFi-Image-2B-turbo) | 4 | 1.0 |
| Turbo | SeFi-Image-5B-turbo | [SeFi-Image/SeFi-Image-5B-turbo](https://huggingface.co/SeFi-Image/SeFi-Image-5B-turbo) | 4 | 1.0 |

## Quick Start

Install the SeFi inference code and dependencies from the SeFi-Image inference
repository, then pass a Hugging Face checkpoint repo id:

```bash
python inference.py \
  --checkpoint SeFi-Image/SeFi-Image-5B-Base \
  --prompt "A red apple on a wooden table." \
  --output-dir outputs/inference/sefi_5b_base
```

Turbo checkpoints use the same command pattern:

```bash
python inference.py \
  --checkpoint SeFi-Image/SeFi-Image-5B-turbo \
  --prompt "A blue ceramic mug on a white desk." \
  --steps 4 \
  --output-dir outputs/inference/sefi_5b_turbo
```

Python API:

```python
from sefi import SEFIInferencePipeline

pipe = SEFIInferencePipeline.from_pretrained(
    "SeFi-Image/SeFi-Image-5B-Base",
)
images = pipe(
    "A red apple on a wooden table.",
    seed=42,
)
images[0].save("example.png")
```

Turbo checkpoints use the same API:

```python
from sefi import SEFIInferencePipeline

pipe = SEFIInferencePipeline.from_pretrained(
    "SeFi-Image/SeFi-Image-5B-turbo",
)
images = pipe(
    "A blue ceramic mug on a white desk.",
    num_inference_steps=4,
    guidance_scale=1.0,
    seed=123,
)
images[0].save("turbo_example.png")
```

## Intended Use

SeFi-Image is intended for research and creative text-to-image generation,
including prompt following, bilingual text rendering, style exploration, and
model development. The Base checkpoints are suitable starting points for
fine-tuning and analysis. Turbo checkpoints are intended for fast generation.
The RL checkpoint is intended for stronger alignment-oriented generation.

## Citation

If you find SeFi-Image useful, please cite the paper:

```bibtex
@misc{sefiteam2026sefiimagetexttoimagefoundationmodel,
      title={SeFi-Image: A Text-to-Image Foundation Model with Semantic-First Diffusion}, 
      author={SeFi-Team},
      year={2026},
      eprint={2606.22568},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.22568}, 
}
```
