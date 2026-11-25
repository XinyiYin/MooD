import os
import gc
import argparse
import numpy as np
import torch
from PIL import Image

from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
)
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

from model.model import EmotionMapper, CLIPLtoGProjector

class FlatRetriever:
    def __init__(self, stats_path: str, device: str = "cpu"):
        raw = torch.load(stats_path, map_location="cpu", weights_only=False)

        all_va = []
        all_emo_features = []
        all_clip_features = []
        all_image_paths = []
        all_attr_names = []

        for attr_name, stat in raw.items():
            va_arr = stat["va_array"]
            emo_feats = stat["emo_features"]
            clip_feats = stat["clip_features"]
            paths = stat.get("image_paths", [None] * len(va_arr))

            if isinstance(emo_feats, torch.Tensor):
                emo_feats = list(emo_feats)
            if isinstance(clip_feats, torch.Tensor):
                clip_feats = list(clip_feats)

            for i in range(len(va_arr)):
                if paths[i] is None:
                    continue
                all_va.append(va_arr[i])
                all_emo_features.append(emo_feats[i])
                all_clip_features.append(clip_feats[i])
                all_image_paths.append(paths[i])
                all_attr_names.append(attr_name)

        self.all_va = np.stack(all_va, axis=0)
        self.all_emo_features = torch.stack(all_emo_features).float()
        self.all_clip_features = torch.stack(all_clip_features).float()
        self.all_image_paths = all_image_paths
        self.all_attr_names = all_attr_names
        self.device = device

        norms = self.all_clip_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.all_clip_features_normed = (self.all_clip_features / norms).to(device)

        M = self.all_va.shape[0]

    def retrieve(self, target_va, source_clip_feature,
                 va_threshold=0.3, top_k_images=1, verbose=True):
        target_va_np = target_va.cpu().numpy()

        va_dists = np.sqrt(((self.all_va - target_va_np) ** 2).sum(axis=1))
        mask = va_dists < va_threshold
        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            candidate_indices = np.argsort(va_dists)[:200]

        src = source_clip_feature.to(self.device).float()
        src = src / src.norm().clamp(min=1e-8)

        cand_normed = self.all_clip_features_normed[candidate_indices]
        cos_sims = (cand_normed @ src).cpu().numpy()

        ranked = np.argsort(-cos_sims)[:top_k_images]

        selected = []
        for rank, r in enumerate(ranked):
            global_idx = candidate_indices[r]
            if verbose:
                print(f"    #{rank+1} [{self.all_attr_names[global_idx]}] "
                      f"VA=({self.all_va[global_idx][0]:.3f},{self.all_va[global_idx][1]:.3f})  "
                      f"va_dist={va_dists[global_idx]:.4f}  cos_sim={cos_sims[r]:.4f}")
            selected.append({
                "image_path":  self.all_image_paths[global_idx],
                "attr_name":   self.all_attr_names[global_idx],
                "va":          self.all_va[global_idx],
                "emo_feature": self.all_emo_features[global_idx],
                "clip_feature": self.all_clip_features[global_idx],
                "cos_sim":     float(cos_sims[r]),
                "va_dist":     float(va_dists[global_idx]),
            })
        return selected


def setup_placeholder_tokens(tokenizer, text_encoder, placeholder_token,
                             initializer_token, num_vectors):
    placeholder_tokens = [placeholder_token] + [
        f"{placeholder_token}_{i}" for i in range(1, num_vectors)
    ]
    tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))

    init_id = tokenizer.encode(initializer_token, add_special_tokens=False)[0]
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for tid in placeholder_token_ids:
            token_embeds[tid] = token_embeds[init_id].clone()
    return placeholder_token_ids


def inject_tokens(text_encoder, placeholder_token_ids, pred_tokens):
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for i, tid in enumerate(placeholder_token_ids):
        token_embeds[tid] = pred_tokens[0, i]


class ModEmoInferenceEngine:
    def __init__(self, args, device=None):
        self.args = args
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.retriever = FlatRetriever(
            stats_path=args.attr_va_stats_path,
            device=self.device,
        )

        checkpoint_path = getattr(args, "checkpoint_path", None)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.mapper = EmotionMapper(
            input_dim=768,
            output_dim=768,
            num_vectors=args.num_vectors,
            num_layers=args.num_mapper_layers,
            dropout=0.1,
        ).to(self.device, dtype=self.dtype).eval()
        self.mapper.load_state_dict(ckpt["mapper"])

        self.clip_g_projector = CLIPLtoGProjector(
            input_dim=768,
            output_dim=1280,
            num_vectors=args.num_vectors,
        ).to(self.device, dtype=self.dtype).eval()
        self.clip_g_projector.load_state_dict(ckpt["clip_g_projector"])
        del ckpt

        tokenizer_l = CLIPTokenizer.from_pretrained(
            args.sdxl_path, subfolder="tokenizer"
        )
        tokenizer_g = CLIPTokenizer.from_pretrained(
            args.sdxl_path, subfolder="tokenizer_2"
        )
        self.text_encoder_l = CLIPTextModel.from_pretrained(
            args.sdxl_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.text_encoder_g = CLIPTextModelWithProjection.from_pretrained(
            args.sdxl_path,
            subfolder="text_encoder_2",
            torch_dtype=self.dtype,
        ).to(self.device)

        self.placeholder_token_ids_l = setup_placeholder_tokens(
            tokenizer_l,
            self.text_encoder_l,
            args.placeholder_token,
            args.initializer_token,
            args.num_vectors,
        )
        self.placeholder_token_ids_g = setup_placeholder_tokens(
            tokenizer_g,
            self.text_encoder_g,
            args.placeholder_token,
            args.initializer_token,
            args.num_vectors,
        )

        image_encoder_path = os.path.join(
            args.ip_adapter_path, "sdxl_models", "image_encoder"
        )
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path,
            torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.clip_image_processor = CLIPImageProcessor(
            size={"shortest_edge": 224},
            crop_size={"height": 224, "width": 224},
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            args.sdxl_path,
            text_encoder=self.text_encoder_l,
            text_encoder_2=self.text_encoder_g,
            tokenizer=tokenizer_l,
            tokenizer_2=tokenizer_g,
            image_encoder=self.clip_image_encoder,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.load_ip_adapter(
            args.ip_adapter_path,
            subfolder="sdxl_models",
            weight_name=args.ip_adapter_weight_name,
        )
        self.pipe.set_ip_adapter_scale(args.ip_scale)
        self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)

        tokens_str = " ".join(
            [args.placeholder_token]
            + [f"{args.placeholder_token}_{i}" for i in range(1, args.num_vectors)]
        )
        self.prompt = f"{args.prompt_prefix}, {tokens_str}"

        normal_embeds_l = self.text_encoder_l.get_input_embeddings().weight.data
        normal_embeds_g = self.text_encoder_g.get_input_embeddings().weight.data
        self.target_norm_l = normal_embeds_l.norm(dim=-1).mean()
        self.target_norm_g = normal_embeds_g.norm(dim=-1).mean()

    @torch.no_grad()
    def generate(self, input_image, target_va=None, seed=None, verbose=True):

        args = self.args
        if target_va is None:
            target_va = torch.tensor(
                [args.target_valence, args.target_arousal], dtype=torch.float32
            )
        elif not isinstance(target_va, torch.Tensor):
            target_va = torch.tensor(target_va, dtype=torch.float32)
        target_va = target_va.detach().cpu().float()

        w, h = input_image.size
        ratio = args.resolution / max(w, h)
        new_w = int(w * ratio) // 8 * 8
        new_h = int(h * ratio) // 8 * 8
        input_image_resized = input_image.resize((new_w, new_h), Image.LANCZOS)

        clip_pixel = self.clip_image_processor(
            images=input_image, return_tensors="pt"
        ).pixel_values.to(device=self.device, dtype=self.dtype)
        source_clip_feature = self.clip_image_encoder(
            pixel_values=clip_pixel
        ).image_embeds.squeeze(0).float()

        results = self.retriever.retrieve(
            target_va=target_va,
            source_clip_feature=source_clip_feature,
            va_threshold=args.va_threshold,
            top_k_images=args.top_k_images,
            verbose=verbose,
        )

        if len(results) == 1:
            emo_embed = results[0]["emo_feature"].unsqueeze(0)
        else:
            feats = torch.stack([item["emo_feature"] for item in results])
            weights = torch.tensor([item["cos_sim"] for item in results])
            weights = torch.softmax(weights, dim=0)
            emo_embed = (weights.unsqueeze(-1) * feats).sum(0, keepdim=True)
        emo_embed = emo_embed.to(device=self.device, dtype=self.dtype)

        pred_tokens_l = self.mapper(emo_embed)
        pred_tokens_g = self.clip_g_projector(pred_tokens_l)
        for token_idx in range(pred_tokens_l.shape[1]):
            norm_l = pred_tokens_l[0, token_idx].norm()
            if norm_l > 0:
                pred_tokens_l[0, token_idx] = (
                    pred_tokens_l[0, token_idx] / norm_l
                    * self.target_norm_l * args.norm_scale
                )
            norm_g = pred_tokens_g[0, token_idx].norm()
            if norm_g > 0:
                pred_tokens_g[0, token_idx] = (
                    pred_tokens_g[0, token_idx] / norm_g
                    * self.target_norm_g * args.norm_scale
                )

        inject_tokens(
            self.text_encoder_l, self.placeholder_token_ids_l, pred_tokens_l
        )
        inject_tokens(
            self.text_encoder_g, self.placeholder_token_ids_g, pred_tokens_g
        )

        reference_images = [
            Image.open(item["image_path"]).convert("RGB") for item in results
        ]

        run_seed = args.seed if seed is None else seed
        output_image = self.pipe(
            prompt=self.prompt,
            negative_prompt=args.negative_prompt,
            image=input_image_resized,
            ip_adapter_image=reference_images[0],
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device=self.device).manual_seed(run_seed),
        ).images[0]

        return {
            "image": output_image,
            "retrieval": results,
            "reference_images": reference_images,
            "input_image_resized": input_image_resized,
        }

    def close(self):
        del self.pipe
        del self.clip_image_encoder
        del self.text_encoder_l, self.text_encoder_g
        del self.mapper, self.clip_g_projector, self.retriever
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def run_inference(args):
    engine = ModEmoInferenceEngine(args)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.image_idx is not None:
        image_indices = [args.image_idx]
    else:
        image_indices = list(range(1, args.num_images + 1))

    for img_idx in image_indices:
        img_name = f"{img_idx}.png"
        img_path = os.path.join(args.input_dir, img_name)

        input_image = Image.open(img_path).convert("RGB")

        generated = engine.generate(input_image, seed=args.seed, verbose=True)

        output_image = generated["image"]
        results = generated["retrieval"]
        reference_images = generated["reference_images"]
        input_image_resized = generated["input_image_resized"]

        for ref_idx, (retrieved, ref_image) in enumerate(
            zip(results, reference_images)
        ):
            ref_name = (
                f"{img_idx}_ref_{ref_idx}_{retrieved['attr_name']}"
                f"_cs{retrieved['cos_sim']:.3f}.png"
            )
            ref_image.save(os.path.join(output_dir, ref_name))

        va_str = f"v{args.target_valence:.2f}_a{args.target_arousal:.2f}"
        out_name = (
            f"{img_idx}_dual_{va_str}"
            f"_ips{args.ip_scale:.1f}_ns{args.norm_scale:.1f}"
            f"_str{args.strength:.1f}.png"
        )
        out_path = os.path.join(output_dir, out_name)
        output_image.save(out_path)
        print(f"  → {out_path}")

        if args.save_comparison:
            tile = 512
            gap = 10
            comparison_images = [input_image_resized] + reference_images + [output_image]
            comp = Image.new(
                "RGB",
                (
                    tile * len(comparison_images)
                    + gap * (len(comparison_images) - 1),
                    tile,
                ),
                (255, 255, 255),
            )
            for index, image in enumerate(comparison_images):
                comp.paste(
                    image.resize((tile, tile), Image.LANCZOS),
                    (index * (tile + gap), 0),
                )
            comp.save(
                os.path.join(output_dir, f"{img_idx}_comparison_{va_str}.png")
            )

    engine.close()
    print(f"\n{'=' * 60}")
    print("Reasoning Completed!")
    print(f"Results: {output_dir}")
    print(f"{'=' * 60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ModEmo Dual-Channel: Mapper Text Tokens + IP-Adapter Image Cross-Attention")

    parser.add_argument("--input_dir", type=str, default="   ")
    parser.add_argument("--output_dir", type=str, default="   ")
    parser.add_argument("--sdxl_path", type=str, default="   ")
    parser.add_argument("--ip_adapter_path", type=str, default="   ")
    parser.add_argument("--ip_adapter_weight_name", type=str, default="   ")
    parser.add_argument("--attr_va_stats_path", type=str, default="   ")
    parser.add_argument("--checkpoint_path", type=str, default="   ")

    parser.add_argument("--target_valence", type=float, default=0.5)
    parser.add_argument("--target_arousal", type=float, default=0.4)

    parser.add_argument("--ip_scale", type=float, default=0.6)

    parser.add_argument("--norm_scale", type=float, default=5.0)
    parser.add_argument("--prompt_prefix", type=str, default="best quality, high quality")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, deformed")

    parser.add_argument("--num_vectors", type=int, default=4)
    parser.add_argument("--num_mapper_layers", type=int, default=3)
    parser.add_argument("--placeholder_token", type=str, default="<emo>")
    parser.add_argument("--initializer_token", type=str, default="emotion")

    parser.add_argument("--va_threshold", type=float, default=0.3)
    parser.add_argument("--top_k_images", type=int, default=1)

    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_images", type=int, default=415)
    parser.add_argument("--save_comparison", action="store_true")
    parser.add_argument("--image_idx", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
