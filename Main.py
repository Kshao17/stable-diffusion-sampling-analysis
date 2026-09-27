import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler
import os
import sys
import time
import random
import pandas as pd
from PIL import Image
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if p not in ['', script_dir, '/root', '/root/assignment4_run']]
os.environ['HF_DATASETS_OFFLINE'] = '0'

drawbench_prompts = [
    "Greek statue of a man tripping over a cat.",
    "A red car and a white sheep.",
    "A bird scaring a scarecrow.",
    "One cat and one dog sitting on the grass.",
    "Three cats and one dog sitting on the grass.",
    "A triangular orange picture frame. An orange picture frame in the shape of a triangle.",
    "A sphere made of kitchen tile. A sphere with the texture of kitchen tile.",
    "A stack of 3 books. A green book is on the top, sitting on a red book. The red book is in the middle, sitting on a blue book. The blue book is on the bottom.",
    "A device consisting of a circular canopy of cloth on a folding metal frame supported by a central rod, used as protection against rain or sometimes sun.",
    "New York Skyline with 'NeurIPS' written with fireworks on the sky."
]
random.seed(42)
sample_prompts = random.sample(drawbench_prompts, 5)
print(f"Sampled Prompts: {sample_prompts}")
device = "cuda"

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

def get_latents_callback(p_idx, cfg_val):
    def callback(pipe, step, timestep, callback_kwargs):
        save_steps = [10, 20, 30, 40, 49]   

        if step in save_steps:
            latents = callback_kwargs["latents"]
            with torch.no_grad():
                scaled_latents = latents / pipe.vae.config.scaling_factor
                decoded = pipe.vae.decode(scaled_latents, return_dict=False)[0]

                decoded = (decoded / 2 + 0.5).clamp(0, 1)

                image_array = decoded[0].detach().cpu().permute(1, 2, 0).float().numpy()
                image_array = (image_array * 255).round().astype("uint8")
                image = Image.fromarray(image_array)

            save_dir = f"results/problem1/prompt_{p_idx}/cfg_{cfg_val}"
            os.makedirs(save_dir, exist_ok=True)
            image.save(f"{save_dir}/step_{step}.png")
        return callback_kwargs

    return callback

guidance_scales = [3.0, 7.5]
seed = 2026

print("Starting Problem 1...")
for p_idx, prompt in enumerate(sample_prompts):
    for cfg in guidance_scales:
        generator = torch.Generator(device).manual_seed(seed)
        print(f"Generating Problem 1: Prompt {p_idx}, CFG {cfg}")
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=cfg,
            generator=generator,
            callback_on_step_end=get_latents_callback(p_idx, cfg)
        ).images[0]
        
        final_dir = f"results/problem1/prompt_{p_idx}/cfg_{cfg}"
        os.makedirs(final_dir, exist_ok=True)
        image.save(f"{final_dir}/final.png")

schedulers = {
    "Euler": EulerDiscreteScheduler.from_config(pipe.scheduler.config),
    "DDIM": DDIMScheduler.from_config(pipe.scheduler.config)
}
step_counts = [10, 25, 50]
performance_data = []

print("\nStarting Problem 2...")
for s_name, s_obj in schedulers.items():
    pipe.scheduler = s_obj
    for steps in step_counts:
        for p_idx, prompt in enumerate(sample_prompts):
            generator = torch.Generator(device).manual_seed(seed) 
            
            start_time = time.time()
            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=7.5, 
                generator=generator
            ).images[0]
            end_time = time.time()
            
            runtime = end_time - start_time 
            performance_data.append({"Scheduler": s_name, "Steps": steps, "Prompt_Idx": p_idx, "Runtime": runtime})
            
            save_dir = f"results/problem2/{s_name}/steps_{steps}"
            os.makedirs(save_dir, exist_ok=True)
            image.save(f"{save_dir}/prompt_{p_idx}.png")
            print(f"Finished: {s_name}, Steps {steps}, Prompt {p_idx}, Time: {runtime:.2f}s")

df_perf = pd.DataFrame(performance_data)
summary_table = df_perf.groupby(['Scheduler', 'Steps'])['Runtime'].mean().reset_index()
summary_table.to_csv("results/problem2_performance_summary.csv", index=False)
print("\nAll tasks completed. Summary table saved.")