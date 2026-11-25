import os
import math
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from tqdm import tqdm
from PIL import Image

from model.model import EmotionMapper, CLIPLtoGProjector, VARegressor
from affect_encoder.encoder import EmotionEncoder
from model.att_loss import VAAttrLoss, AttrIndexer, extract_attr_clip_embeddings


# TODO: supplement the file path of emotion encoder
class ImageEncoder(nn.Module):
    def __init__(self, clip_dim=768, emo_dim=768):
        super(ImageEncoder, self).__init__()
        self.resnet = EmotionEncoder()
        ckpt = torch.load(
            "   ",
            map_location="cpu", weights_only=False
        )
        self.resnet.load_state_dict(ckpt["model_state_dict"])
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[0:4])
        for param in self.resnet.parameters():
            param.requires_grad = False

        self._proj_initialized = False
        self.proj = None

    def forward(self, x):
        out = self.resnet(x)
        if out.dim() == 4:
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = out.view(out.size(0), -1)
        if out.shape[-1] != 768:
            if not self._proj_initialized:
                self.proj = nn.Linear(out.shape[-1], 768).to(out.device)
                self.proj.requires_grad_(False)
                self.proj.eval()
                self._proj_initialized = True
            out = self.proj(out)
        return out


class ModEmoDataset(Dataset):
    def __init__(self, data_root, tokenizer_l, tokenizer_g,
                 placeholder_token="<emo>", num_vectors=4,
                 size=1024, center_crop=True, latent_cache_dirs=None,
                 attr_indexer=None, emo_cache_dir=None):
        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop
        self.tokenizer_l = tokenizer_l
        self.tokenizer_g = tokenizer_g
        self.placeholder_token = placeholder_token
        self.num_vectors = num_vectors
        self.attr_indexer = attr_indexer

        tokens = [placeholder_token] + [f"{placeholder_token}_{i}" for i in range(1, num_vectors)]
        self.placeholder_string = " ".join(tokens)


        self.transform_sdxl = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.transform_emo = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        annotation_dir = os.path.join(data_root, "annotation")

        all_image_ids = []
        for filename in sorted(os.listdir(annotation_dir)):
            if filename.endswith('.json'):
                img_id = filename[:-5]
                all_image_ids.append(img_id)

        self.image_ids = []
        self.image_objects = {}
        self.va_cache = {}
        self.image_paths = {}

        image_dir = os.path.join(data_root, "image")
        _IMG_EXTS = ['.jpg', '.png', '.jpeg', '.bmp']
        attr_set = set(attr_indexer.attr_names) if attr_indexer is not None else None
        skipped_attr = 0
        skipped_va = 0
        skipped_img = 0

        for img_id in all_image_ids:
            ann_path = os.path.join(annotation_dir, f"{img_id}.json")
            with open(ann_path, "r") as f:
                ann = json.load(f)

            if "valence" not in ann or "arousal" not in ann:
                skipped_va += 1
                continue

            if attr_set is not None:
                objects = ann.get("object", [])
                matched = [f"obj_{obj}" for obj in objects if f"obj_{obj}" in attr_set]
                if not matched:
                    skipped_attr += 1
                    continue
                self.image_objects[img_id] = matched

            found_path = None
            for ext in _IMG_EXTS:
                p = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(p):
                    found_path = p
                    break
            if found_path is None:
                skipped_img += 1
                continue

            self.image_ids.append(img_id)
            self.va_cache[img_id] = {
                "valence": ann["valence"],
                "arousal": ann["arousal"],
            }
            self.image_paths[img_id] = found_path


        self._input_ids_l = tokenizer_l(
            self.placeholder_string,
            padding="max_length", truncation=True,
            max_length=77, return_tensors="pt"
        ).input_ids.squeeze(0)

        self._input_ids_g = tokenizer_g(
            self.placeholder_string,
            padding="max_length", truncation=True,
            max_length=77, return_tensors="pt"
        ).input_ids.squeeze(0)

        self.latent_cache_dirs = []
        self.use_cached_latents = False
        if latent_cache_dirs:
            if isinstance(latent_cache_dirs, str):
                latent_cache_dirs = [d.strip() for d in latent_cache_dirs.split(",") if d.strip()]
            for d in latent_cache_dirs:
                if os.path.isdir(d):
                    n_cached = len([f for f in os.listdir(d) if f.endswith("_latent.pt")])
                    self.latent_cache_dirs.append(d)
            if self.latent_cache_dirs:
                self.use_cached_latents = True

        self.emo_cache_dir = None
        if emo_cache_dir and os.path.isdir(emo_cache_dir):
            n_emo = len([f for f in os.listdir(emo_cache_dir) if f.endswith('_emo.pt')])
            self.emo_cache_dir = emo_cache_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        image_path = self.image_paths[img_id]

        image = Image.open(image_path).convert("RGB")

        va = self.va_cache[img_id]

        if self.emo_cache_dir is not None:
            emo_path = os.path.join(self.emo_cache_dir, f"{img_id}_emo.pt")
            if os.path.exists(emo_path):
                emo_pixel_values = torch.load(emo_path, weights_only=True)
            else:
                emo_pixel_values = self.transform_emo(image)
        else:
            emo_pixel_values = self.transform_emo(image)

        result = {
            "emo_pixel_values": emo_pixel_values,
            "input_ids_l": self._input_ids_l.clone(),
            "input_ids_g": self._input_ids_g.clone(),
            "valence": torch.tensor(va["valence"], dtype=torch.float32),
            "arousal": torch.tensor(va["arousal"], dtype=torch.float32),
        }

        if self.attr_indexer is not None:
            if img_id in self.image_objects:
                result["target_attr"] = self.attr_indexer.encode(self.image_objects[img_id])
            else:
                result["target_attr"] = self.attr_indexer.encode([])

        if self.use_cached_latents:
            latent_loaded = False
            for cache_dir in self.latent_cache_dirs:
                latent_path = os.path.join(cache_dir, f"{img_id}_latent.pt")
                if os.path.exists(latent_path):
                    result["cached_latents"] = torch.load(latent_path, weights_only=True)
                    latent_loaded = True
                    break
            if not latent_loaded:
                result["pixel_values"] = self.transform_sdxl(image)
        else:
            result["pixel_values"] = self.transform_sdxl(image)

        return result


def setup_placeholder_tokens(tokenizer, text_encoder, placeholder_token,
                             initializer_token, num_vectors, token_dim):
    placeholder_tokens = [placeholder_token]
    for i in range(1, num_vectors):
        placeholder_tokens.append(f"{placeholder_token}_{i}")

    num_added = tokenizer.add_tokens(placeholder_tokens)
    assert num_added == num_vectors, (
        f"Tokenizer already contains {placeholder_token}. Use a different token."
    )

    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    assert len(token_ids) == 1, "Initializer token must be a single token."
    initializer_token_id = token_ids[0]

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for tid in placeholder_token_ids:
            token_embeds[tid] = token_embeds[initializer_token_id].clone()

    return placeholder_token_ids


def save_placeholder_embeds(text_encoder, placeholder_token_ids):
    embeds = text_encoder.get_input_embeddings().weight.data
    return {tid: embeds[tid].clone() for tid in placeholder_token_ids}


def restore_placeholder_embeds(text_encoder, saved_embeds):
    embeds = text_encoder.get_input_embeddings().weight.data
    for tid, val in saved_embeds.items():
        embeds[tid] = val


def get_latest_checkpoint(output_dir):
    ckpt_dirs = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
                ckpt_dirs.append((step, os.path.join(output_dir, name)))
            except ValueError:
                continue
    if not ckpt_dirs:
        return None, 0
    ckpt_dirs.sort(key=lambda x: x[0])
    return ckpt_dirs[-1][1], ckpt_dirs[-1][0]



@torch.no_grad()
def validate(
    accelerator, val_dataloader, emotion_encoder, mapper, clip_g_projector,
    va_regressor, text_encoder_l, text_encoder_g, vae, unet, noise_scheduler,
    placeholder_token_ids_l, placeholder_token_ids_g,
    va_attr_loss_fn, weight_dtype, args,
):
    mapper.eval()
    clip_g_projector.eval()
    va_regressor.eval()

    te_l = accelerator.unwrap_model(text_encoder_l)
    te_g = accelerator.unwrap_model(text_encoder_g)

    saved_l = save_placeholder_embeds(te_l, placeholder_token_ids_l)
    saved_g = save_placeholder_embeds(te_g, placeholder_token_ids_g)

    total_loss = 0.0
    total_recon = 0.0
    total_attr = 0.0
    total_va = 0.0
    num_batches = 0

    for batch in val_dataloader:
        emo_embed = emotion_encoder(
            batch["emo_pixel_values"].to(dtype=weight_dtype)
        ).to(dtype=weight_dtype)

        pred_tokens_l = mapper(emo_embed)
        pred_tokens_g = clip_g_projector(pred_tokens_l)

        token_embeds_l = te_l.get_input_embeddings().weight.data
        for i, tid in enumerate(placeholder_token_ids_l):
            token_embeds_l[tid] = pred_tokens_l[0, i].detach()

        token_embeds_g = te_g.get_input_embeddings().weight.data
        for i, tid in enumerate(placeholder_token_ids_g):
            token_embeds_g[tid] = pred_tokens_g[0, i].detach()

        if "cached_latents" in batch:
            latents = batch["cached_latents"].to(device=accelerator.device, dtype=weight_dtype)
        else:
            with torch.no_grad():
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        max_timestep = min(500, noise_scheduler.config.num_train_timesteps)
        timesteps = torch.randint(
            0, max_timestep,
            (bsz,), device=latents.device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        output_l = text_encoder_l(batch["input_ids_l"])
        encoder_hidden_states_l = output_l.last_hidden_state.to(dtype=weight_dtype)

        output_g = text_encoder_g(batch["input_ids_g"])
        encoder_hidden_states_g = output_g.last_hidden_state.to(dtype=weight_dtype)
        pooled_output_g = output_g.text_embeds.to(dtype=weight_dtype)

        encoder_hidden_states = torch.cat(
            [encoder_hidden_states_l, encoder_hidden_states_g], dim=-1
        )

        add_time_ids = torch.tensor(
            [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
            dtype=weight_dtype, device=latents.device
        ).repeat(bsz, 1)

        model_pred = unet(
            noisy_latents, timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={
                "text_embeds": pooled_output_g,
                "time_ids": add_time_ids,
            },
        ).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

        loss_reconstruction = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        va_target = torch.stack([batch["valence"], batch["arousal"]], dim=-1).to(accelerator.device)

        if va_attr_loss_fn is not None:
            mapper_pooled = pred_tokens_l.mean(dim=1)
            target_attr = batch.get("target_attr", None)
            if target_attr is not None:
                target_attr = target_attr.to(accelerator.device)
            loss_attr, _ = va_attr_loss_fn(
                z=mapper_pooled.float(), va=va_target,
                target_attr=target_attr.float() if target_attr is not None else None
            )
        else:
            loss_attr = torch.tensor(0.0, device=accelerator.device)

        pred_va = va_regressor(pred_tokens_l.float(), pred_tokens_g.float())
        loss_va = F.mse_loss(pred_va, va_target)

        loss_total = (
            args.recon_weight * loss_reconstruction +
            args.attr_weight * loss_attr +
            args.va_weight * loss_va
        )

        total_loss += loss_total.item()
        total_recon += loss_reconstruction.item()
        total_attr += loss_attr.item()
        total_va += loss_va.item()
        num_batches += 1

    restore_placeholder_embeds(te_l, saved_l)
    restore_placeholder_embeds(te_g, saved_g)

    mapper.train()
    clip_g_projector.train()
    va_regressor.train()

    n = max(num_batches, 1)
    return {
        "val_loss": total_loss / n,
        "val_recon": total_recon / n,
        "val_attr": total_attr / n,
        "val_va": total_va / n,
    }


def main(args):
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.output_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_l = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer"
    )
    text_encoder_l = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder"
    )

    tokenizer_g = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer_2"
    )
    text_encoder_g = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model, subfolder="text_encoder_2"
    )

    if args.latent_cache_dirs:
        vae = None
    else:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model, subfolder="vae"
        )
        vae.enable_slicing()
        vae.enable_tiling()
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )

    emotion_encoder = ImageEncoder(clip_dim=768, emo_dim=768)
    emotion_encoder.eval()
    emotion_encoder.requires_grad_(False)

    mapper = EmotionMapper(
        input_dim=768,
        output_dim=768,
        num_vectors=args.num_vectors,
        num_layers=args.num_mapper_layers,
        dropout=args.mapper_dropout,
    )

    clip_g_projector = CLIPLtoGProjector(
        input_dim=768,
        output_dim=1280,
        num_vectors=args.num_vectors,
    )

    va_regressor = VARegressor(
        clip_l_dim=768,
        clip_g_dim=1280,
        hidden_dim=256,
    )

    if vae is not None:
        vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_l.requires_grad_(False)
    text_encoder_g.requires_grad_(False)

    text_encoder_l.get_input_embeddings().weight.requires_grad_(True)
    text_encoder_g.get_input_embeddings().weight.requires_grad_(True)

    placeholder_token_ids_l = setup_placeholder_tokens(
        tokenizer_l, text_encoder_l,
        args.placeholder_token, args.initializer_token,
        args.num_vectors, token_dim=768
    )

    placeholder_token_ids_g = setup_placeholder_tokens(
        tokenizer_g, text_encoder_g,
        args.placeholder_token, args.initializer_token,
        args.num_vectors, token_dim=1280
    )

    trainable_params = (
        list(mapper.parameters()) +
        list(clip_g_projector.parameters()) +
        list(va_regressor.parameters())
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    attr_indexer = AttrIndexer.from_json(args.attr_list_path)

    full_dataset = ModEmoDataset(
        data_root=args.train_data_dir,
        tokenizer_l=tokenizer_l,
        tokenizer_g=tokenizer_g,
        placeholder_token=args.placeholder_token,
        num_vectors=args.num_vectors,
        size=args.resolution,
        latent_cache_dirs=args.latent_cache_dirs,
        attr_indexer=attr_indexer,
        emo_cache_dir=args.emo_cache_dir,
    )

    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(args.seed if args.seed else 42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    if accelerator.is_main_process:
        print(f"Dataset split: train set {train_size}, validation set {val_size}")

    _use_persistent = args.dataloader_num_workers > 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=_use_persistent,
        prefetch_factor=2 if _use_persistent else None,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=_use_persistent,
        prefetch_factor=2 if _use_persistent else None,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    (
        text_encoder_l, text_encoder_g,
        optimizer, train_dataloader, val_dataloader, lr_scheduler,
        mapper, clip_g_projector, va_regressor,
    ) = accelerator.prepare(
        text_encoder_l, text_encoder_g,
        optimizer, train_dataloader, val_dataloader, lr_scheduler,
        mapper, clip_g_projector, va_regressor,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    emotion_encoder.to(accelerator.device, dtype=weight_dtype)

    # TODO: supplement the file path of CLIP
    attr_embeddings = extract_attr_clip_embeddings(
        attr_names=attr_indexer.attr_names,
        clip_model_path="   ",
        templates=["a photo of {}", "an image with {}"]
    )
    va_attr_loss_fn, _ = VAAttrLoss.from_precomputed(
        stats_path=args.attr_va_stats_path,
        attr_embeddings=attr_embeddings,
        attr_list_path=args.attr_list_path,
        temperature=0.07,
        weight_sharpness=args.attr_weight_sharpness,
        weight_floor=0.01
    )
    va_attr_loss_fn = va_attr_loss_fn.to(accelerator.device)
    use_attr_loss = True

    best_val_loss = float("inf")
    patience_counter = 0

    assert args.train_batch_size == 1, (
        "train_batch_size must be 1. The embedding table in Textual Inversion is globally shared, "
        "so conditions from different samples may get mixed up when batch_size > 1. "
        "Use gradient_accumulation_steps to increase the effective batch size."
    )

    te_l_unwrapped = accelerator.unwrap_model(text_encoder_l)
    te_g_unwrapped = accelerator.unwrap_model(text_encoder_g)
    init_embeds_l = save_placeholder_embeds(te_l_unwrapped, placeholder_token_ids_l)
    init_embeds_g = save_placeholder_embeds(te_g_unwrapped, placeholder_token_ids_g)

    with torch.no_grad():
        all_embeds_l = te_l_unwrapped.get_input_embeddings().weight.data
        mask_l = torch.ones(all_embeds_l.shape[0], dtype=torch.bool)
        mask_l[0] = False
        for tid in placeholder_token_ids_l:
            mask_l[tid] = False
        ref_norm_l = all_embeds_l[mask_l].norm(dim=-1).mean().item()

        all_embeds_g = te_g_unwrapped.get_input_embeddings().weight.data
        mask_g = torch.ones(all_embeds_g.shape[0], dtype=torch.bool)
        mask_g[0] = False
        for tid in placeholder_token_ids_g:
            mask_g[tid] = False
        ref_norm_g = all_embeds_g[mask_g].norm(dim=-1).mean().item()

    if accelerator.is_main_process:
        print(f"Reference embedding norms for SDXL: L={ref_norm_l:.4f}, G={ref_norm_g:.4f}")

    global_step = 0
    resume_step = 0
    first_epoch = 0

    resume_from = args.resume_from_checkpoint
    if resume_from == "latest":
        resume_from, _ = get_latest_checkpoint(args.output_dir)
        if resume_from is None:
            if accelerator.is_main_process:
                print("No existing checkpoint was located. Training commences from scratch.")
        else:
            if accelerator.is_main_process:
                print(f"Automatically locating the most recent checkpoint: {resume_from}")

    if resume_from is not None and os.path.isdir(resume_from):
        if accelerator.is_main_process:
            print(f"Training resumes from the checkpoint: {resume_from}")

        accelerator.load_state(resume_from)

        global_step = int(os.path.basename(resume_from).split("-")[-1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch
        resume_step_in_loader = resume_step * args.gradient_accumulation_steps

        if accelerator.is_main_process:
            print(
                f"  Resuming at global_step={global_step}, "
                f"epoch={first_epoch}, "
                f"skipping {resume_step_in_loader} dataloader batches within the epoch"
            )

        best_ckpt_path = os.path.join(args.output_dir, "best_model", "modemo_best.pth")
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_val_loss = best_ckpt.get("val_loss", float("inf"))
            if accelerator.is_main_process:
                print(f"  Loaded historical best val_loss={best_val_loss:.6f}")
    else:
        if resume_from is not None:
            if accelerator.is_main_process:
                print(f"Warning: The specified checkpoint path {resume_from} does not exist. Training commences from scratch.")


    if accelerator.is_main_process:
        accelerator.init_trackers("modemo_training")

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Training")

    for epoch in range(first_epoch, args.num_train_epochs):
        mapper.train()
        clip_g_projector.train()
        va_regressor.train()

        active_dataloader = train_dataloader
        if epoch == first_epoch and resume_step > 0:
            resume_step_in_loader = resume_step * args.gradient_accumulation_steps
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step_in_loader
            )
            if accelerator.is_main_process:
                print(f"Epoch {epoch}: Skipping the first {resume_step_in_loader} batches,"
                      f"resuming from batch {resume_step_in_loader}")

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(mapper, clip_g_projector, va_regressor):
                te_l = accelerator.unwrap_model(text_encoder_l)
                te_g = accelerator.unwrap_model(text_encoder_g)
                restore_placeholder_embeds(te_l, init_embeds_l)
                restore_placeholder_embeds(te_g, init_embeds_g)

                with torch.no_grad():
                    emo_embed = emotion_encoder(
                        batch["emo_pixel_values"].to(dtype=weight_dtype)
                    ).to(dtype=weight_dtype)

                pred_tokens_l = mapper(emo_embed)
                pred_tokens_g = clip_g_projector(pred_tokens_l)

                token_embeds_l = te_l.get_input_embeddings().weight
                token_embeds_g = te_g.get_input_embeddings().weight

                for i, tid in enumerate(placeholder_token_ids_l):
                    token_embeds_l.data[tid] = pred_tokens_l[0, i].detach()
                for i, tid in enumerate(placeholder_token_ids_g):
                    token_embeds_g.data[tid] = pred_tokens_g[0, i].detach()

                if "cached_latents" in batch:
                    latents = batch["cached_latents"].to(device=accelerator.device, dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        latents = vae.encode(
                            batch["pixel_values"].to(dtype=weight_dtype)
                        ).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                max_timestep = min(500, noise_scheduler.config.num_train_timesteps)
                timesteps = torch.randint(
                    0, max_timestep,
                    (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                output_l = text_encoder_l(batch["input_ids_l"])
                encoder_hidden_states_l = output_l.last_hidden_state.to(dtype=weight_dtype)

                output_g = text_encoder_g(batch["input_ids_g"])
                encoder_hidden_states_g = output_g.last_hidden_state.to(dtype=weight_dtype)
                pooled_output_g = output_g.text_embeds.to(dtype=weight_dtype)

                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states_l, encoder_hidden_states_g], dim=-1
                )

                add_time_ids = torch.tensor(
                    [[args.resolution, args.resolution, 0, 0,
                      args.resolution, args.resolution]],
                    dtype=weight_dtype, device=latents.device
                ).repeat(bsz, 1)

                model_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": pooled_output_g,
                        "time_ids": add_time_ids,
                    },
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type: {noise_scheduler.config.prediction_type}"
                    )

                loss_reconstruction = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )

                va_target = torch.stack(
                    [batch["valence"], batch["arousal"]], dim=-1
                ).to(accelerator.device)

                if use_attr_loss:
                    mapper_pooled = pred_tokens_l.mean(dim=1)
                    target_attr = batch.get("target_attr", None)
                    if target_attr is not None:
                        target_attr = target_attr.to(accelerator.device)
                    loss_attr, attr_info = va_attr_loss_fn(
                        z=mapper_pooled.float(), va=va_target.float(),
                        target_attr=target_attr.float() if target_attr is not None else None
                    )
                else:
                    loss_attr = torch.tensor(0.0, device=accelerator.device)
                    attr_info = {"weight_mean": 0.0}

                pred_va = va_regressor(
                    pred_tokens_l.float(), pred_tokens_g.float()
                )
                loss_va = F.mse_loss(pred_va, va_target)

                token_norms_l = pred_tokens_l.float().norm(dim=-1)
                token_norms_g = pred_tokens_g.float().norm(dim=-1)
                loss_norm = (
                    ((token_norms_l - ref_norm_l) ** 2).mean() +
                    ((token_norms_g - ref_norm_g) ** 2).mean()
                )

                loss_recon_scaled = args.recon_weight * loss_reconstruction

                grad_from_l, grad_from_g = torch.autograd.grad(
                    loss_recon_scaled,
                    [token_embeds_l, token_embeds_g],
                    retain_graph=False,
                    create_graph=False,
                )

                if grad_from_l is None and grad_from_g is None:
                    if accelerator.is_main_process and global_step % 100 == 0:
                        print(f"WARNING step {global_step}: embedding grad is None, bridge loss=0 this step")

                loss_bridge = torch.tensor(0.0, device=accelerator.device)

                if grad_from_l is not None:
                    for i, tid in enumerate(placeholder_token_ids_l):
                        g = grad_from_l[tid].detach()
                        t = pred_tokens_l[0, i]
                        loss_bridge = loss_bridge + torch.sum(g * t)

                if grad_from_g is not None:
                    for i, tid in enumerate(placeholder_token_ids_g):
                        g = grad_from_g[tid].detach()
                        t = pred_tokens_g[0, i]
                        loss_bridge = loss_bridge + torch.sum(g * t)

                loss_bridge = loss_bridge / args.num_vectors

                loss_phase2 = (
                    loss_bridge
                    + args.va_weight * loss_va
                    + args.norm_weight * loss_norm
                )
                if use_attr_loss:
                    loss_phase2 = loss_phase2 + args.attr_weight * loss_attr

                accelerator.backward(loss_phase2)

                if token_embeds_l.grad is not None:
                    token_embeds_l.grad.zero_()
                if token_embeds_g.grad is not None:
                    token_embeds_g.grad.zero_()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                loss_total = (
                    args.recon_weight * loss_reconstruction.detach() +
                    args.attr_weight * loss_attr.detach() +
                    args.va_weight * loss_va.detach() +
                    args.norm_weight * loss_norm.detach()
                )

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    log_dict = {
                        "loss/total": loss_total.item(),
                        "loss/reconstruction": loss_reconstruction.item(),
                        "loss/attribute": loss_attr.item() if use_attr_loss else 0.0,
                        "loss/va": loss_va.item(),
                        "loss/bridge": loss_bridge.item(),
                        "loss/norm": loss_norm.item(),
                        "norm/pred_l": token_norms_l.mean().item(),
                        "norm/pred_g": token_norms_g.mean().item(),
                        "norm/ref_l": ref_norm_l,
                        "norm/ref_g": ref_norm_g,
                        "attr/weight_mean": attr_info["weight_mean"],
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    for tracker in accelerator.trackers:
                        for k, v in log_dict.items():
                            tracker.writer.add_scalar(k, v, global_step)

                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        torch.save({
                            "mapper": accelerator.unwrap_model(mapper).state_dict(),
                            "clip_g_projector": accelerator.unwrap_model(clip_g_projector).state_dict(),
                            "va_regressor": accelerator.unwrap_model(va_regressor).state_dict(),
                        }, os.path.join(save_path, "modemo_modules.pth"))
                        if accelerator.is_main_process:
                            print(f"  -> save checkpoint: {save_path}")

            logs = {
                "loss": loss_total.item(),
                "recon": loss_reconstruction.item(),
                "attr": loss_attr.item() if use_attr_loss else 0.0,
                "va": loss_va.item(),
                "bridge": loss_bridge.item(),
                "norm_l": f"{token_norms_l.mean().item():.2f}/{ref_norm_l:.2f}",
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if (epoch + 1) % args.val_every_n_epochs == 0:
            if accelerator.is_main_process:
                print(f"\n Epoch {epoch + 1}: Running validation")

            val_metrics = validate(
                accelerator, val_dataloader, emotion_encoder,
                mapper, clip_g_projector, va_regressor,
                text_encoder_l, text_encoder_g, vae, unet, noise_scheduler,
                placeholder_token_ids_l, placeholder_token_ids_g,
                va_attr_loss_fn,
                weight_dtype, args,
            )

            val_loss_tensor = torch.tensor(
                val_metrics["val_loss"], device=accelerator.device
            )
            val_loss_tensor = accelerator.reduce(val_loss_tensor, reduction="mean")
            val_loss = val_loss_tensor.item()

            if accelerator.is_main_process:
                print(
                    f"Epoch {epoch + 1}: val_loss={val_loss:.6f} "
                    f"(best={best_val_loss:.6f}, "
                    f"patience={patience_counter}/{args.patience})"
                )

                for tracker in accelerator.trackers:
                    tracker.writer.add_scalar("val/loss", val_loss, epoch)
                    tracker.writer.add_scalar("val/recon", val_metrics["val_recon"], epoch)
                    tracker.writer.add_scalar("val/va", val_metrics["val_va"], epoch)
                    tracker.writer.add_scalar("val/attr", val_metrics["val_attr"], epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_save_path = os.path.join(args.output_dir, "best_model")
                    os.makedirs(best_save_path, exist_ok=True)
                    torch.save({
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "val_loss": val_loss,
                        "mapper": accelerator.unwrap_model(mapper).state_dict(),
                        "clip_g_projector": accelerator.unwrap_model(clip_g_projector).state_dict(),
                        "va_regressor": accelerator.unwrap_model(va_regressor).state_dict(),
                    }, os.path.join(best_save_path, "modemo_best.pth"))
                    print(f"  -> save the best model (val_loss={val_loss:.6f})")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(
                            f"  -> Early stopping triggered."
                        )

            should_stop = torch.tensor(
                [1 if patience_counter >= args.patience else 0],
                device=accelerator.device
            )
            should_stop = accelerator.reduce(should_stop, reduction="max")

            if should_stop.item() >= 1:
                if accelerator.is_main_process:
                    print("Training terminated across all processes.")
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save({
            "mapper": accelerator.unwrap_model(mapper).state_dict(),
            "clip_g_projector": accelerator.unwrap_model(clip_g_projector).state_dict(),
            "va_regressor": accelerator.unwrap_model(va_regressor).state_dict(),
        }, os.path.join(args.output_dir, "modemo_final.pth"))

        print(f"Training completed. Final model → {args.output_dir}/modemo_final.pth")
        print(f"Best model with val_loss={best_val_loss:.6f} → {args.output_dir}/best_model/")

    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="ModEmo Training")

    parser.add_argument("--pretrained_model", type=str, default="  ")
    parser.add_argument("--num_vectors", type=int, default=4)
    parser.add_argument("--num_mapper_layers", type=int, default=3)
    parser.add_argument("--mapper_dropout", type=float, default=0.1)
    parser.add_argument("--placeholder_token", type=str, default="<emo>")
    parser.add_argument("--initializer_token", type=str, default="emotion")

    parser.add_argument("--train_data_dir", type=str, default="   ")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--attr_weight", type=float, default=0.5)
    parser.add_argument("--va_weight", type=float, default=0.1)
    parser.add_argument("--norm_weight", type=float, default=0.01)

    parser.add_argument("--attr_list_path", type=str, default="   ")
    parser.add_argument("--attr_va_stats_path", type=str, default="   ")
    parser.add_argument("--attr_weight_sharpness", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_every_n_epochs", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="   ")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_cache_dirs", type=str, default=None)
    parser.add_argument("--emo_cache_dir", type=str, default=None)

    parser.add_argument("--resume_from_checkpoint", type=str, default="latest")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)