# Independent Study ML Diffusion

Key finding:
For generating new poses of existing characters, the reference image does most of the work. FLUX Kontext takes a reference sprite and edits the pose while keeping the character consistent. The LoRA adds style consistency on top, most useful for generating new characters or backgrounds from text alone using the frmcrtoon trigger word.

Phase 1: 
API-based generation (FLUX Kontext Pro via Replicate)
Started by calling the FLUX Kontext Pro model through the Replicate API at ~$0.005 per image. Built Python scripts (sprite_gen.py, sprite_animator.py, background_gen.py) that sent a reference image + prompt and received a generated image back. This worked but became costly at scale. Generating 134 training images plus animation frames adds up quickly.

Phase 2: 
Moving to local/cloud inference
Switched to running FLUX.1-Kontext-dev open source model locally first. Hit a major problem: laptop has an Intel Iris Xe integrated GPU with no CUDA support — PyTorch couldn't run the model at all. Moved to Google Colab instead.

Problems on Colab free tier (T4 GPU, 15GB VRAM):

FLUX.1-Kontext-dev is 33GB (wouldn't fit on the T4)
Tried enable_model_cpu_offload() to split between GPU and RAM. Model loaded but ran out of VRAM during inference
Tried reducing image resolution to 512x512 ... still failed
Upgraded to Colab Pro with A100 GPU (40GB VRAM) which solved the memory issue

Phase 3: 

LoRA fine-tuning

Collected 134 training images (all 5 characters, multiple poses each) with auto-generated captions using the trigger word frmcrtoon
Ran train_dreambooth_lora_flux.py from HuggingFace diffusers on the A100
Hit another VRAM error even on A100 — fixed by adding --gradient_checkpointing, reducing --rank from 16 to 4, and adding --use_8bit_adam
Training ran 1000 steps over ~1 hour, loss dropped from (0.42) to (0.146)
Downloaded trained weights as pytorch_lora_weights.safetensors (~200MB)











