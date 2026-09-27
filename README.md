# Stable Diffusion Sampling & Scheduler Analysis

This project explores how Stable Diffusion XL (SDXL) generates images during the reverse diffusion process and how different inference settings affect image quality and runtime.

The experiments focus on reverse denoising visualization, classifier-free guidance, sampling steps, and scheduler comparisons.

## Project Overview

The project investigates several practical questions in diffusion-based image generation:

- How does an image emerge from Gaussian noise during reverse diffusion?
- How does classifier-free guidance affect prompt alignment and image naturalness?
- How many inference steps are needed before additional computation produces diminishing returns?
- How do Euler and DDIM schedulers compare in runtime and visual quality?

## Reverse Diffusion Visualization

Intermediate latent representations were extracted during the denoising process and decoded using the SDXL VAE.

The experiments showed that:

- Early steps mainly contain noise
- Global image structure becomes recognizable around step 20
- Later steps primarily refine textures, lighting, and fine details

This suggests that SDXL first establishes the overall composition before refining local details.

## Guidance Scale Analysis

Two classifier-free guidance settings were compared:

| Guidance Scale | Prompt Alignment | Naturalness | Visual Artifacts |
| --- | --- | --- | --- |
| CFG 3.0 | Moderate | High | Minimal |
| CFG 15.0 | Stronger | Lower | More high-contrast artifacts |

Higher guidance improved prompt alignment, but excessive guidance also produced oversaturation and less natural-looking images.

## Scheduler & Sampling Analysis

The project compares Euler Discrete and DDIM schedulers across different numbers of inference steps.

| Scheduler | Steps | Approx. Runtime | Observation |
| --- | ---: | ---: | --- |
| Euler | 10 | ~2.4 s | Fast convergence and stable structure |
| Euler | 25 | ~5.6 s | Better high-frequency detail |
| Euler | 50 | ~8.8 s | Fine details with diminishing improvement |
| DDIM | 10 | ~2.4 s | Good overall structure |
| DDIM | 25 | ~5.7 s | High fidelity and smooth output |
| DDIM | 50 | ~8.9 s | High quality but more computationally expensive |

The experiments showed diminishing visual improvements as the number of sampling steps increased.

## Experimental Setup

- Model: Stable Diffusion XL
- Library: Hugging Face Diffusers
- Dataset / Prompts: DrawBench
- Schedulers: Euler Discrete and DDIM
- Guidance scales tested: 3.0 and 15.0
- Fixed random seeds were used for controlled comparisons

## Technologies

- Python
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion XL
- NumPy
- Matplotlib

## Key Takeaways

- SDXL generates global structure before refining local details.
- Higher classifier-free guidance can improve prompt alignment but may reduce naturalness.
- Increasing sampling steps improves quality only up to a point.
- Euler and DDIM show similar runtime scaling under the tested settings.
- Lower-step configurations can provide a better speed-quality trade-off for many prompts.

## Repository Structure

```text
stable-diffusion-sampling-analysis/
├── README.md
├── notebooks/
│   └── stable_diffusion_analysis.ipynb
├── results/
│   ├── reverse_diffusion/
│   ├── guidance_comparison/
│   └── scheduler_comparison/
└── .gitignore
