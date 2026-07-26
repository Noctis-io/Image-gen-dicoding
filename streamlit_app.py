import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import gc
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stable Diffusion Pipeline", layout="wide")

AUTHOR_NAME = "Feri Putra"
HF_TOKEN = os.environ.get("HF_TOKEN", st.secrets.get("HF_TOKEN", ""))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"

st.sidebar.title(f"Stable Diffusion Pipeline")
st.sidebar.caption(f"By {AUTHOR_NAME} | Device: {DEVICE.upper()}")

mode = st.sidebar.selectbox(
    "Mode",
    ["Text-to-Image (Basic)", "Text-to-Image (Advanced)", "Batch Generation",
     "Inpainting", "Outpainting", "Two-Stage Generation"]
)

@st.cache_resource
def load_base_pipe():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        use_auth_token=HF_TOKEN or None
    )
    pipe = pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
    return pipe

@st.cache_resource
def load_inpaint_pipe():
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        use_auth_token=HF_TOKEN or None
    )
    pipe = pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
    return pipe

@st.cache_resource
def load_img2img_pipe():
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        use_auth_token=HF_TOKEN or None
    )
    pipe = pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
    return pipe

def generate_image(pipe, prompt, negative_prompt="", seed=42, guidance_scale=7.5, num_inference_steps=50):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    image = pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        generator=generator, guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    return image

def inpaint(image, mask, prompt, negative_prompt="", seed=9, guidance_scale=7.5, num_inference_steps=50):
    pipe = load_inpaint_pipe()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    result = pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        image=image, mask_image=mask,
        generator=generator, guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    return result

def prepare_outpainting(image, direction="right", expand_pixels=128, fill_color=(0, 0, 0)):
    w, h = image.size
    if direction == "right":
        new_img = Image.new("RGB", (w + expand_pixels, h), fill_color)
        new_img.paste(image, (0, 0))
        mask = Image.new("L", (w + expand_pixels, h), 0)
        ImageDraw.Draw(mask).rectangle([w, 0, w + expand_pixels, h], fill=255)
    elif direction == "left":
        new_img = Image.new("RGB", (w + expand_pixels, h), fill_color)
        new_img.paste(image, (expand_pixels, 0))
        mask = Image.new("L", (w + expand_pixels, h), 0)
        ImageDraw.Draw(mask).rectangle([0, 0, expand_pixels, h], fill=255)
    elif direction == "up":
        new_img = Image.new("RGB", (w, h + expand_pixels), fill_color)
        new_img.paste(image, (0, expand_pixels))
        mask = Image.new("L", (w, h + expand_pixels), 0)
        ImageDraw.Draw(mask).rectangle([0, 0, w, expand_pixels], fill=255)
    elif direction == "down":
        new_img = Image.new("RGB", (w, h + expand_pixels), fill_color)
        new_img.paste(image, (0, 0))
        mask = Image.new("L", (w, h + expand_pixels), 0)
        ImageDraw.Draw(mask).rectangle([0, h, w, h + expand_pixels], fill=255)
    return new_img, mask

def do_outpainting(image, prompt, direction, expand_pixels=128, seed=9, guidance_scale=7.5, num_inference_steps=50):
    expanded_img, mask = prepare_outpainting(image, direction, expand_pixels)
    return inpaint(expanded_img, mask, prompt, seed=seed, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

st.title(f"🎨 Stable Diffusion Pipeline")
st.markdown(f"---")

if mode == "Text-to-Image (Basic)":
    st.header("Basic Text-to-Image Generation")

    col1, col2 = st.columns([1, 2])

    with col1:
        prompt = st.text_area("Prompt", "an astronaut standing on the Earth, outer space view")
        negative_prompt = st.text_area("Negative Prompt", "photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, grainy, unfinished, 8K, highly detailed")
        seed = st.number_input("Seed", value=42, min_value=0, max_value=999999)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.0, 0.5)
        steps = st.slider("Inference Steps", 10, 200, 50, 5)

        if st.button("Generate", type="primary"):
            with st.spinner("Generating..."):
                pipe = load_base_pipe()
                img = generate_image(pipe, prompt, negative_prompt, seed, guidance_scale, steps)
                st.session_state.basic_result = img

    with col2:
        if "basic_result" in st.session_state:
            st.image(st.session_state.basic_result, caption=prompt, use_container_width=True)
            buf = BytesIO()
            st.session_state.basic_result.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "output.png", "image/png")

elif mode == "Text-to-Image (Advanced)":
    st.header("Advanced Text-to-Image Generation")

    col1, col2 = st.columns([1, 2])

    with col1:
        prompt = st.text_area("Prompt", "an astronaut standing on the Earth, outer space view")
        negative_prompt = st.text_area("Negative Prompt", "painting, oil painting, 2d, sketch, drawing, cartoon, illustration, 3d render, blurry, low quality, ugly, grainy, abstract")
        seed = st.number_input("Seed", value=123, min_value=0, max_value=999999)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 8.0, 0.5)
        steps = st.slider("Inference Steps", 10, 200, 150, 5)

        if st.button("Generate", type="primary"):
            with st.spinner("Generating..."):
                pipe = load_base_pipe()
                img = generate_image(pipe, prompt, negative_prompt, seed, guidance_scale, steps)
                st.session_state.advanced_result = img

    with col2:
        if "advanced_result" in st.session_state:
            st.image(st.session_state.advanced_result, caption=prompt, use_container_width=True)
            buf = BytesIO()
            st.session_state.advanced_result.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "output.png", "image/png")

elif mode == "Batch Generation":
    st.header("Batch Generation (2x2 Grid)")

    col1, col2 = st.columns([1, 2])

    with col1:
        prompt = st.text_area("Prompt", "an astronaut standing on the Earth, outer space view")
        negative_prompt = st.text_area("Negative Prompt", "photorealistic, realistic, photograph, 3d render, blurry, low quality, ugly, grainy")
        seed = st.number_input("Base Seed", value=42, min_value=0, max_value=999999)
        n_images = st.selectbox("Number of Images", [1, 2, 4, 6, 9], index=2)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        steps = st.slider("Inference Steps", 10, 200, 50, 5)

        if st.button("Generate", type="primary"):
            with st.spinner(f"Generating {n_images} images..."):
                pipe = load_base_pipe()
                images = []
                for i in range(n_images):
                    g = torch.Generator(device=DEVICE).manual_seed(seed + i)
                    img = pipe(
                        prompt=prompt, negative_prompt=negative_prompt,
                        generator=g, guidance_scale=guidance_scale,
                        num_inference_steps=steps
                    ).images[0]
                    images.append(img)
                st.session_state.batch_result = images

    with col2:
        if "batch_result" in st.session_state:
            import math
            images = st.session_state.batch_result
            cols = min(3, len(images))
            rows = math.ceil(len(images) / cols)
            for r in range(rows):
                row_cols = st.columns(cols)
                for c in range(cols):
                    idx = r * cols + c
                    if idx < len(images):
                        row_cols[c].image(images[idx], use_container_width=True)

elif mode == "Inpainting":
    st.header("Image Inpainting")

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        mask_uploaded = st.file_uploader("Upload Mask (optional)", type=["png", "jpg", "jpeg"], key="mask")
        prompt = st.text_area("Prompt", "a beautiful garden with flowers")
        negative_prompt = st.text_area("Negative Prompt", "ugly, blurry, low quality")
        seed = st.number_input("Seed", value=9, min_value=0, max_value=999999)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        steps = st.slider("Inference Steps", 10, 200, 50, 5)

        mask_mode = st.radio("Mask Mode", ["Upload mask", "Create rectangle mask (center)"])

        if uploaded and st.button("Inpaint", type="primary"):
            image = Image.open(uploaded).convert("RGB")
            if mask_uploaded:
                mask = Image.open(mask_uploaded).convert("L")
            elif mask_mode == "Create rectangle mask (center)":
                w, h = image.size
                mask = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)
            else:
                st.error("Please upload a mask or select rectangle mode")
                st.stop()

            with st.spinner("Inpainting..."):
                result = inpaint(image, mask, prompt, negative_prompt, seed, guidance_scale, steps)
                st.session_state.inpaint_result = result
                st.session_state.inpaint_original = image

    with col2:
        if "inpaint_result" in st.session_state:
            orig, res = st.columns(2)
            with orig:
                st.image(st.session_state.inpaint_original, caption="Original", use_container_width=True)
            with res:
                st.image(st.session_state.inpaint_result, caption="Result", use_container_width=True)
            buf = BytesIO()
            st.session_state.inpaint_result.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "inpaint_result.png", "image/png")

elif mode == "Outpainting":
    st.header("Image Outpainting")

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        prompt = st.text_area("Prompt", "continue the landscape with mountains and sky")
        direction = st.selectbox("Direction", ["right", "left", "up", "down"])
        expand_pixels = st.slider("Expand Pixels", 64, 512, 128, 32)
        seed = st.number_input("Seed", value=9, min_value=0, max_value=999999)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        steps = st.slider("Inference Steps", 10, 200, 50, 5)

        if uploaded and st.button("Outpaint", type="primary"):
            image = Image.open(uploaded).convert("RGB")
            with st.spinner("Outpainting..."):
                result = do_outpainting(image, prompt, direction, expand_pixels, seed, guidance_scale, steps)
                st.session_state.outpaint_result = result
                st.session_state.outpaint_original = image

    with col2:
        if "outpaint_result" in st.session_state:
            orig, res = st.columns(2)
            with orig:
                st.image(st.session_state.outpaint_original, caption="Original", use_container_width=True)
            with res:
                st.image(st.session_state.outpaint_result, caption="Result", use_container_width=True)
            buf = BytesIO()
            st.session_state.outpaint_result.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "outpaint_result.png", "image/png")

elif mode == "Two-Stage Generation":
    st.header("Two-Stage Generation (Base + Refinement)")

    col1, col2 = st.columns([1, 2])

    with col1:
        prompt = st.text_area("Prompt", "a beautiful fantasy landscape")
        negative_prompt = st.text_area("Negative Prompt", "ugly, blurry, low quality")
        seed = st.number_input("Seed", value=42, min_value=0, max_value=999999)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        steps = st.slider("Inference Steps", 10, 200, 50, 5)

        if st.button("Generate", type="primary"):
            with st.spinner("Stage 1/2 - Base generation..."):
                pipe_base = load_base_pipe()
                generator = torch.Generator(device=DEVICE).manual_seed(seed)
                init_image = pipe_base(
                    prompt=prompt, negative_prompt=negative_prompt,
                    generator=generator, guidance_scale=guidance_scale,
                    num_inference_steps=steps, denoising_end=0.8
                ).images[0]

            with st.spinner("Stage 2/2 - Refinement..."):
                pipe_refiner = load_img2img_pipe()
                refined = pipe_refiner(
                    prompt=prompt, negative_prompt=negative_prompt,
                    image=init_image, generator=generator,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps, strength=0.2
                ).images[0]

                st.session_state.two_stage_base = init_image
                st.session_state.two_stage_refined = refined

    with col2:
        if "two_stage_refined" in st.session_state:
            base, ref = st.columns(2)
            with base:
                st.image(st.session_state.two_stage_base, caption="Base", use_container_width=True)
            with ref:
                st.image(st.session_state.two_stage_refined, caption="Refined", use_container_width=True)
            buf = BytesIO()
            st.session_state.two_stage_refined.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "refined_output.png", "image/png")

st.sidebar.markdown("---")
if st.sidebar.button("Clear Memory"):
    clear_memory()
    st.sidebar.success("Memory cleared!")

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Device:** {DEVICE.upper()}\n\n"
    f"**Author:** {AUTHOR_NAME}\n\n"
    "⚠️ GPU dengan CUDA sangat direkomendasikan."
)
