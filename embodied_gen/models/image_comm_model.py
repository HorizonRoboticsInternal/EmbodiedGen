import os
from abc import ABC, abstractmethod

import torch
from diffusers import (
    ChromaPipeline,
    Cosmos2TextToImagePipeline,
    DPMSolverMultistepScheduler,
    FluxPipeline,
    KolorsPipeline,
    StableDiffusion3Pipeline,
)
from diffusers.quantizers import PipelineQuantizationConfig
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, SiglipProcessor

__all__ = [
    "build_hf_image_pipeline",
]


class BasePipelineLoader(ABC):
    def __init__(self, device="cuda"):
        self.device = device

    @abstractmethod
    def load(self):
        pass


class BasePipelineRunner(ABC):
    def __init__(self, pipe):
        self.pipe = pipe

    @abstractmethod
    def run(self, prompt: str, **kwargs) -> Image.Image:
        pass


# ===== SD3.5-medium =====
class SD35Loader(BasePipelineLoader):
    def load(self):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to(self.device)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        return pipe


class SD35Runner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> Image.Image:
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Cosmos2 =====
class CosmosLoader(BasePipelineLoader):
    def __init__(
        self,
        model_id="nvidia/Cosmos-Predict2-2B-Text2Image",
        local_dir="weights/cosmos2",
        device="cuda",
    ):
        super().__init__(device)
        self.model_id = model_id
        self.local_dir = local_dir

    def _patch(self):
        def patch_model(cls):
            orig = cls.from_pretrained

            def new(*args, **kwargs):
                kwargs.setdefault("attn_implementation", "flash_attention_2")
                kwargs.setdefault("torch_dtype", torch.bfloat16)
                return orig(*args, **kwargs)

            cls.from_pretrained = new

        def patch_processor(cls):
            orig = cls.from_pretrained

            def new(*args, **kwargs):
                kwargs.setdefault("use_fast", True)
                return orig(*args, **kwargs)

            cls.from_pretrained = new

        patch_model(AutoModelForCausalLM)
        patch_processor(SiglipProcessor)

    def load(self):
        self._patch()
        snapshot_download(
            repo_id=self.model_id,
            local_dir=self.local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            },
            components_to_quantize=["text_encoder", "transformer", "unet"],
        )

        pipe = Cosmos2TextToImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=config,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        return pipe


class CosmosRunner(BasePipelineRunner):
    def run(self, prompt: str, negative_prompt=None, **kwargs) -> Image.Image:
        return self.pipe(
            prompt=prompt, negative_prompt=negative_prompt, **kwargs
        ).images


# ===== Kolors =====
class KolorsLoader(BasePipelineLoader):
    def load(self):
        pipe = KolorsPipeline.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        return pipe


class KolorsRunner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> Image.Image:
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Flux =====
class FluxLoader(BasePipelineLoader):
    def load(self):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        return pipe.to(self.device)


class FluxRunner(BasePipelineRunner):
    def run(self, prompt: str, **kwargs) -> Image.Image:
        return self.pipe(prompt=prompt, **kwargs).images


# ===== Chroma =====
class ChromaLoader(BasePipelineLoader):
    def load(self):
        return ChromaPipeline.from_pretrained(
            "lodestones/Chroma", torch_dtype=torch.bfloat16
        ).to(self.device)


class ChromaRunner(BasePipelineRunner):
    def run(self, prompt: str, negative_prompt=None, **kwargs) -> Image.Image:
        return self.pipe(
            prompt=prompt, negative_prompt=negative_prompt, **kwargs
        ).images


PIPELINE_REGISTRY = {
    "sd35": (SD35Loader, SD35Runner),
    "cosmos": (CosmosLoader, CosmosRunner),
    "kolors": (KolorsLoader, KolorsRunner),
    "flux": (FluxLoader, FluxRunner),
    "chroma": (ChromaLoader, ChromaRunner),
}


def build_hf_image_pipeline(name: str, device="cuda") -> BasePipelineRunner:
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unsupported model: {name}")
    loader_cls, runner_cls = PIPELINE_REGISTRY[name]
    pipe = loader_cls(device=device).load()

    return runner_cls(pipe)


if __name__ == "__main__":
    model_name = "sd35"
    runner = build_hf_image_pipeline(model_name)
    # NOTE: Just for pipeline testing, generation quality at low resolution is poor.
    images = runner.run(
        prompt="A robot holding a sign that says 'Hello'",
        height=512,
        width=512,
        num_inference_steps=10,
        guidance_scale=6,
        num_images_per_prompt=1,
    )

    for i, img in enumerate(images):
        img.save(f"image_{model_name}_{i}.jpg")
