# Stable Diffusion XL: Denoising Trajectory, Guidance Scale and Sampler Cost

Individual project exploring how Stable Diffusion XL (SDXL) builds an image during reverse diffusion
and how inference settings affect image quality and runtime.

**Questions**

1. In what order does the image take shape during denoising?
2. How does the classifier-free guidance (CFG) scale trade prompt alignment against naturalness?
3. How many sampling steps are worth their cost, and does the scheduler matter?

## Setup

| Setting | Value |
| --- | --- |
| Model | Stable Diffusion XL via Hugging Face Diffusers |
| Resolution | 1024 × 1024 |
| Prompts | 5 sampled from 10 DrawBench prompts |
| Trajectory | 50 steps; latents decoded with the SDXL VAE at step indices 10, 20, 30, 40, 49 |
| Guidance | CFG 3.0 vs. 15.0, 50 steps |
| Schedulers | Euler discrete and DDIM; complete runs at 10, 25 and 50 steps, plus 20 and 80 |
| Seed | Fixed across all runs |

## Results

### Denoising trajectory

![Car trajectory](results/traj_car.png)

![Cats trajectory](results/traj_cats.png)

In these trajectories, the main objects became recognizable around step index 20, while secondary
elements and fine texture appeared in later steps. Both examples also show a counting error that
remained in the final image: "a white sheep" produced a flock, and "three cats and one dog" produced
two cats and one dog.

### Guidance scale

![Statue, CFG 3.0 vs 15.0](results/cfg_statue.png)

![Tile sphere, CFG 3.0 vs 15.0](results/cfg_sphere.png)

| Aspect | CFG 3.0 | CFG 15.0 |
| --- | --- | --- |
| Prompt content | Animal resembles a lion as much as a cat | Clearly a cat, man mid-stride over it |
| Color and contrast | Muted tones, soft shadows | More saturated, harder shadows |
| Sphere setting and texture | Indoor scene, irregular tiles | Studio background, regular tile grid |

### Runtime

Average wall-clock time per image over the five prompts:

| Scheduler | 10 steps | 25 steps | 50 steps |
| --- | ---: | ---: | ---: |
| Euler | 2.4 s | 5.6 s | 8.8 s |
| DDIM | 2.4 s | 5.7 s | 8.9 s |

- Five times as many steps took about 3.7 times as long, consistent with a fixed per-image cost plus
  a per-step cost.
- Euler and DDIM took nearly the same time in this setup.
- 25 steps added most of the visible detail; 50 steps was only slightly sharper, and 80 steps was
  hard to distinguish from 50. A complete 20-step Euler run took about half the time of a 50-step run.

## Limitations

- Image quality was judged visually on five prompts with one seed.
- Hardware, precision and warm-up were not recorded with the timings, so runtimes are relative
  comparisons within one setup.

## Repository Structure

```text
stable-diffusion-sampling-analysis/
├── README.md
├── requirements.txt
├── Main.py                 # trajectory, guidance and scheduler experiments
└── results/
    ├── traj_car.png        # decoded latents, "A red car and a white sheep."
    ├── traj_cats.png       # decoded latents, "Three cats and one dog sitting on the grass."
    ├── cfg_statue.png      # CFG 3.0 vs. 15.0
    └── cfg_sphere.png      # CFG 3.0 vs. 15.0
```

## Running

```bash
pip install -r requirements.txt
python Main.py
```

A CUDA GPU is required.

## References

- Podell, D., et al. (2024). [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952). ICLR 2024.
- Ho, J., and Salimans, T. (2021). [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
- Song, J., Meng, C., and Ermon, S. (2021). [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502). ICLR 2021.
- Saharia, C., et al. (2022). [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) (DrawBench). NeurIPS 2022.
