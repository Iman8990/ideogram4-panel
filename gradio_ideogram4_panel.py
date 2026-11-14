"""Ideogram 4 — Gradio panel on Modal (teaching build).

Standalone app, separate from ideogram4-model / ideogram4-telegram-bot;
reuses their weights volume and the 'ideogram4-model-auth' secret
(HF_TOKEN + IDEOGRAM_API_KEY).

Deploy:  modal deploy gradio_ideogram4_panel.py
Panel login: attach secret 'ideogram4-panel-auth' (PANEL_USER, PANEL_PASSWORD).
"""

import base64
import io
import json
import math
import os
import random
import time

import modal

MODEL_REPO = "ideogram-ai/ideogram-4-nf4-diffusers"
MODEL_DIR = "/models"
PROMPT_ENHANCER_REPO = "diffusers/qwen3-vl-8b-instruct-lm-head"
MAGIC_PROMPT_URL = "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt"
MAX_SEED = 2**31 - 1

app = modal.App("ideogram4-panel")

vol = modal.Volume.from_name("ideogram4-model-weights", create_if_missing=True)
secrets = [modal.Secret.from_name("ideogram4-model-auth")]
panel_secrets = secrets + [modal.Secret.from_name("ideogram4-panel-auth")]

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libgomp1", "git")
    .pip_install(
        "huggingface_hub",
        "fastapi",
        "httpx",
        "torch",
        "torchvision",
        "pillow",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "outlines",  # for schema-constrained local prompt upsampling
        "bitsandbytes",  # required for nf4 quantization
        "gradio",
    )
    .pip_install("git+https://github.com/huggingface/diffusers.git")
)


@app.cls(
    gpu="A100-40GB",
    image=image,
    volumes={MODEL_DIR: vol},
    secrets=secrets,
    max_containers=1,
    timeout=3600,
)
class ModelServer:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import Ideogram4Pipeline, Ideogram4PromptEnhancerHead

        hf_token = os.environ.get("HF_TOKEN", "")

        try:
            prompt_enhancer_head = Ideogram4PromptEnhancerHead.from_pretrained(
                PROMPT_ENHANCER_REPO,
                torch_dtype=torch.bfloat16,
                cache_dir=MODEL_DIR,
                token=hf_token,
            )
        except Exception as e:
            print(f"Warning: could not load prompt enhancer head: {e}")
            prompt_enhancer_head = None

        self.pipe = Ideogram4Pipeline.from_pretrained(
            MODEL_REPO,
            prompt_enhancer_head=prompt_enhancer_head,
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_DIR,
            token=hf_token,
        ).to("cuda")

        _ = self.pipe(
            "a red apple",
            height=512,
            width=512,
            prompt_upsampling=False,
        ).images[0]
        print("Ideogram 4 warmup done")

    def _remote_upsample(self, prompt, width, height):
        """Hosted Magic Prompt API -> structured JSON caption string."""
        import httpx

        api_key = os.environ.get("IDEOGRAM_API_KEY", "")
        if not api_key:
            raise RuntimeError("IDEOGRAM_API_KEY not set")
        d = math.gcd(int(width), int(height)) or 1
        aspect_ratio = f"{int(width) // d}x{int(height) // d}"
        resp = httpx.post(
            MAGIC_PROMPT_URL,
            headers={"Api-Key": api_key, "Content-Type": "application/json"},
            json={"text_prompt": prompt, "aspect_ratio": aspect_ratio},
            timeout=120.0,
        )
        resp.raise_for_status()
        jp = resp.json().get("json_prompt")
        if not jp:
            raise RuntimeError("Ideogram API returned no json_prompt")
        jp.pop("aspect_ratio", None)
        for el in jp.get("compositional_deconstruction", {}).get("elements", []):
            if isinstance(el, dict):
                el.pop("bbox", None)
        return json.dumps(jp, ensure_ascii=False, separators=(",", ":"))

    def _inject_palette(self, caption_text, color_palette):
        """Add hex colors to style_description.color_palette of a JSON caption."""
        if not color_palette:
            return caption_text
        try:
            caption = json.loads(caption_text)
        except Exception:
            return caption_text
        style = caption.get("style_description")
        if isinstance(style, dict):
            style["color_palette"] = list(color_palette)[:16]
            return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
        return caption_text

    @modal.method()
    def generate(
        self,
        prompt,
        height=1024,
        width=1024,
        seed=None,
        upsampler="remote",
        num_inference_steps=48,
        guidance_schedule=None,
        guidance_scale=7.0,
        mu=0.0,
        std=1.5,
        color_palette=None,
    ):
        import torch

        height = int(height)
        width = int(width)
        num_inference_steps = int(num_inference_steps)

        gen = torch.Generator(device="cuda")
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            seed = gen.seed()

        caption = prompt
        use_local_upsampling = False

        if upsampler == "remote":
            try:
                caption = self._remote_upsample(prompt, width, height)
            except Exception as e:
                print(f"Remote upsample failed, falling back to local: {e}")
                use_local_upsampling = True
        elif upsampler == "local":
            use_local_upsampling = True

        pipe_kwargs = {
            "height": height,
            "width": width,
            "generator": gen,
            "num_inference_steps": num_inference_steps,
            "mu": float(mu),
            "std": float(std),
        }
        # A per-step schedule overrides the constant guidance scale.
        if guidance_schedule:
            pipe_kwargs["guidance_schedule"] = [float(g) for g in guidance_schedule]
        else:
            pipe_kwargs["guidance_scale"] = float(guidance_scale)

        if use_local_upsampling:
            # Explicit call so the panel can show the caption fed to the model.
            try:
                upsampled = self.pipe.upsample_prompt(prompt, height=height, width=width)
                caption = upsampled[0] if isinstance(upsampled, (list, tuple)) else str(upsampled)
                caption = self._inject_palette(caption, color_palette)
                image = self.pipe(caption, prompt_upsampling=False, **pipe_kwargs).images[0]
            except Exception as e:
                print(f"Explicit local upsample failed, using pipeline upsampling: {e}")
                caption = json.dumps({"prompt": prompt, "note": "upsampled locally inside the pipeline"})
                image = self.pipe(prompt, prompt_upsampling=True, **pipe_kwargs).images[0]
        else:
            caption = self._inject_palette(caption, color_palette)
            image = self.pipe(caption, prompt_upsampling=False, **pipe_kwargs).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {
            "image_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "seed": seed,
            "caption": caption,
        }


# ---------------- Gradio panel ----------------

# Sampler presets from the model docs.
MODES = {
    "Turbo · 12 steps": dict(num_inference_steps=12, guidance_schedule=[7.0] * 11 + [3.0] * 1, mu=0.5, std=1.75),
    "Default · 20 steps": dict(num_inference_steps=20, guidance_schedule=[7.0] * 18 + [3.0] * 2, mu=0.0, std=1.75),
    "Quality · 48 steps": dict(num_inference_steps=48, guidance_schedule=[7.0] * 45 + [3.0] * 3, mu=0.0, std=1.5),
    "Custom": None,
}

UPSAMPLERS = ["Ideogram (remote)", "Qwen (local)", "Off — direct JSON / plain"]

SIZE_PRESETS = {
    "Square — 1024×1024 (1:1)": (1024, 1024),
    "Square 2K — 2048×2048 (1:1, max quality)": (2048, 2048),
    "Landscape — 1536×1024 (3:2)": (1536, 1024),
    "Portrait — 1024×1536 (2:3)": (1024, 1536),
    "Widescreen — 1920×1088 (16:9)": (1920, 1088),
    "Ultrawide — 2048×768 (21:9)": (2048, 768),
    "Phone wallpaper — 1024×1792 (9:16)": (1024, 1792),
    "Social banner — 1600×400 (4:1)": (1600, 400),
}

JSON_EXAMPLE = json.dumps(
    {
        "high_level_description": "A golden retriever riding a skateboard down a sunny sidewalk.",
        "style_description": {
            "aesthetics": "warm, playful, vibrant",
            "lighting": "bright afternoon sunlight, long soft shadows",
            "photo": "shallow depth of field, eye-level, 85mm lens",
            "medium": "photograph",
            "color_palette": ["#F5C542", "#87CEEB", "#4A4A4A", "#FFFFFF", "#2E8B57"],
        },
        "compositional_deconstruction": {
            "background": "A sun-drenched suburban sidewalk lined with green hedges and a white picket fence. Dappled light filters through overhead trees.",
            "elements": [
                {
                    "type": "obj",
                    "bbox": [200, 300, 800, 900],
                    "desc": "A golden retriever with a fluffy coat, standing on a red skateboard with all four paws. Its tongue is out and ears are flapping in the wind.",
                },
                {
                    "type": "obj",
                    "bbox": [250, 750, 750, 950],
                    "desc": "A worn red skateboard with black wheels rolling along the concrete sidewalk.",
                },
            ],
        },
    },
    ensure_ascii=False,
    indent=2,
)


def build_panel():
    import gradio as gr

    def apply_preset(name):
        w, h = SIZE_PRESETS[name]
        return w, h

    def parse_palette(text):
        colors = []
        for token in (text or "").replace(",", " ").split():
            c = token.strip().upper()
            if c.startswith("#") and len(c) == 7:
                colors.append(c)
        return colors[:16]

    def generate(prompt, mode="Default · 20 steps", upsampler_label=UPSAMPLERS[0], width=1024,
                 height=1024, seed=0, randomize_seed=True, palette_text="", c_steps=48,
                 c_guidance=7.0, c_mu=0.0, c_std=1.5):
        prompt = (prompt or "").strip()
        if not prompt:
            yield None, None, None, "Please write a prompt first."
            return
        if randomize_seed or seed is None or int(seed) < 0:
            seed = random.randint(0, MAX_SEED)
        seed = int(seed)

        if mode == "Custom":
            params = dict(
                num_inference_steps=int(c_steps),
                guidance_schedule=None,
                guidance_scale=float(c_guidance),
                mu=float(c_mu),
                std=float(c_std),
            )
        else:
            preset = MODES[mode]
            params = dict(
                num_inference_steps=preset["num_inference_steps"],
                guidance_schedule=preset["guidance_schedule"],
                guidance_scale=7.0,
                mu=preset["mu"],
                std=preset["std"],
            )

        upsampler = {"Ideogram (remote)": "remote", "Qwen (local)": "local"}.get(upsampler_label, "off")

        call = ModelServer().generate.spawn(
            prompt,
            height=int(height),
            width=int(width),
            seed=seed,
            upsampler=upsampler,
            color_palette=parse_palette(palette_text),
            **params,
        )
        started = time.time()
        yield None, None, None, "Model is loading / generating… (first run after idle takes longer)"
        while True:
            try:
                result = call.get(timeout=10)
                break
            except TimeoutError:
                yield None, None, None, f"Generating… {int(time.time() - started)}s"

        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(result["image_b64"])))
        try:
            caption_out = json.loads(result["caption"])
        except Exception:
            caption_out = {"caption": result["caption"]}
        yield img, result["seed"], caption_out, f"Done in {int(time.time() - started)}s"

    with gr.Blocks(title="Ideogram 4") as demo:
        gr.Markdown("# Ideogram 4")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=4)
                mode = gr.Radio(
                    choices=list(MODES.keys()),
                    value="Default · 20 steps",
                    label="Mode (speed ↔ quality)",
                )
                run = gr.Button("Generate", variant="primary")
                with gr.Accordion("Advanced", open=False):
                    upsampler = gr.Radio(
                        choices=UPSAMPLERS,
                        value=UPSAMPLERS[0],
                        label="Prompt upsampler",
                        info="Remote = Ideogram Magic Prompt API · Local = on-device Qwen · Off = prompt as-is (JSON caption).",
                    )
                    size = gr.Dropdown(
                        choices=list(SIZE_PRESETS.keys()),
                        value="Square — 1024×1024 (1:1)",
                        label="Size preset",
                    )
                    with gr.Row():
                        width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                        height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=0, precision=0)
                        randomize = gr.Checkbox(label="Randomize seed", value=True)
                    palette = gr.Textbox(
                        label="Color palette (optional, hex — comma separated)",
                        placeholder="#1B1B2F, #E43F5A, #F5F5F5",
                        value="",
                    )
                    gr.Markdown("**Custom mode sampling**")
                    with gr.Row():
                        c_steps = gr.Slider(4, 64, value=48, step=1, label="Steps")
                        c_guidance = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="Guidance scale")
                    with gr.Row():
                        c_mu = gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="mu (schedule mean)")
                        c_std = gr.Slider(0.1, 3.0, value=1.5, step=0.05, label="std (schedule spread)")
                with gr.Accordion("JSON caption example (for Upsampler = Off)", open=False):
                    gr.Code(value=JSON_EXAMPLE, language="json", label="Structured caption — copy, edit, paste into Prompt")
            with gr.Column():
                out_image = gr.Image(label="Output", type="pil")
                out_seed = gr.Number(label="Seed used", interactive=False)
                out_caption = gr.JSON(label="Caption fed to the model (upsampled when enabled)")
                status = gr.Textbox(label="Status", interactive=False)

        gr.Examples(
            examples=[
                ["a ginger cat wearing a tiny wizard hat reading a spellbook"],
                ["an isometric illustration of a tiny city floating in the clouds"],
                ["a golden retriever on a skateboard"],
                [JSON_EXAMPLE],
            ],
            inputs=[prompt],
            outputs=[out_image, out_seed, out_caption, status],
            fn=generate,
            cache_examples=False,
        )

        size.change(apply_preset, inputs=size, outputs=[width, height])
        run.click(
            generate,
            inputs=[prompt, mode, upsampler, width, height, seed, randomize,
                    palette, c_steps, c_guidance, c_mu, c_std],
            outputs=[out_image, out_seed, out_caption, status],
        )
    return demo


# max_containers=1: Gradio's queue keeps state in memory — join and stream
# must hit the same container, or the stream 404s mid-generation.
# fastapi/gradio are imported inside this function on purpose: they only
# exist in the Modal image, so `modal deploy` on a fresh machine needs
# nothing but the modal package itself.
@app.function(image=image, secrets=panel_secrets, timeout=1800, max_containers=1, scaledown_window=300)
@modal.asgi_app()
def gradio_app():
    import gradio as gr
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    web_app = FastAPI()

    @web_app.middleware("http")
    async def basic_auth(request: Request, call_next):
        user = os.environ.get("PANEL_USER", "")
        password = os.environ.get("PANEL_PASSWORD", "")
        if user and password:
            auth = request.headers.get("authorization", "")
            ok = False
            if auth.startswith("Basic "):
                try:
                    decoded = base64.b64decode(auth[6:]).decode("utf-8")
                    u, _, p = decoded.partition(":")
                    ok = u == user and p == password
                except Exception:
                    ok = False
            if not ok:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Login required"},
                    headers={"WWW-Authenticate": "Basic"},
                )
        return await call_next(request)

    return gr.mount_gradio_app(web_app, build_panel(), path="/")
