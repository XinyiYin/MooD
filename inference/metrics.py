import os
import json
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    from scipy.ndimage import uniform_filter

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssim_channels = []
    for c in range(img1.shape[2]):
        x = img1[:, :, c]
        y = img2[:, :, c]

        mu_x = uniform_filter(x, size=11)
        mu_y = uniform_filter(y, size=11)

        sigma_x_sq = uniform_filter(x ** 2, size=11) - mu_x ** 2
        sigma_y_sq = uniform_filter(y ** 2, size=11) - mu_y ** 2
        sigma_xy = uniform_filter(x * y, size=11) - mu_x * mu_y

        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = numerator / denominator
        ssim_channels.append(ssim_map.mean())

    return float(np.mean(ssim_channels))


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self, weights_path: str):
        super().__init__()
        from torchvision.models import alexnet

        model = alexnet(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        self.slice1 = nn.Sequential(*list(model.features.children())[:2])
        self.slice2 = nn.Sequential(*list(model.features.children())[2:5])
        self.slice3 = nn.Sequential(*list(model.features.children())[5:8])
        self.slice4 = nn.Sequential(*list(model.features.children())[8:10])
        self.slice5 = nn.Sequential(*list(model.features.children())[10:12])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]


class LPIPS(nn.Module):
    def __init__(self, alexnet_weights: str, lpips_weights: str):
        super().__init__()
        self.net = AlexNetFeatureExtractor(alexnet_weights)

        channels = [64, 192, 384, 256, 256]
        self.lins = nn.ModuleList()
        for ch in channels:
            lin = nn.Sequential(
                nn.Dropout(inplace=False),
                nn.Conv2d(ch, 1, 1, stride=1, padding=0, bias=False),
            )
            self.lins.append(lin)

        lpips_state = torch.load(lpips_weights, map_location="cpu", weights_only=True)

        new_state = {}
        for key, val in lpips_state.items():
            new_key = key
            if key.startswith("lin") and not key.startswith("lins"):
                parts = key.split(".")
                layer_idx = parts[0].replace("lin", "")
                remaining = ".".join(parts[2:])
                new_key = f"lins.{layer_idx}.{remaining}"
            elif key.startswith("lins."):
                new_key = key.replace(".model.", ".")
            new_state[new_key] = val

        try:
            self.lins.load_state_dict(new_state, strict=True)
        except Exception:
            for i, lin in enumerate(self.lins):
                conv = lin[1]
                for k, v in lpips_state.items():
                    if f"lin{i}" in k and "weight" in k and v.shape == conv.weight.shape:
                        conv.weight.data = v.clone()
                        break

        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def _normalize(x, eps=1e-10):
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
        return x / norm

    def forward(self, x, y):
        feats_x = self.net(x)
        feats_y = self.net(y)

        total = 0
        for i, (fx, fy) in enumerate(zip(feats_x, feats_y)):
            fx_norm = self._normalize(fx)
            fy_norm = self._normalize(fy)
            diff = (fx_norm - fy_norm) ** 2
            score = self.lins[i](diff)
            total = total + score.mean(dim=[2, 3])

        return total.squeeze(1)


class CLIPImageSimilarity:
    def __init__(self, clip_weights: str, device: str = "cuda"):
        self.device = device

        import clip as clip_module
        self.model, self.preprocess = clip_module.load(
            clip_weights, device=device, jit=False
        )
        self.model.eval()

    @torch.no_grad()
    def compute_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        t1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        t2 = self.preprocess(img2).unsqueeze(0).to(self.device)

        feat1 = self.model.encode_image(t1).float()
        feat2 = self.model.encode_image(t2).float()

        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)

        sim = (feat1 * feat2).sum(dim=-1)
        return sim.item()


class VAPredictorFromEncoder:
    def __init__(self, encoder_path: str, device: str = "cuda"):
        self.device = device

        try:
            from affect_encoder.encoder import EmotionEncoder
            self.encoder = EmotionEncoder()
            ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
            self.encoder.load_state_dict(ckpt["model_state_dict"])
            self.encoder = self.encoder.to(device).eval()
            print(f"[VA Predictor] EmotionEncoder loaded successfully: {encoder_path}")

            self._use_full_encoder = True

        except Exception as e:
            print(f"[VA Predictor] Failed to load EmotionEncoder: {e}")
            self._use_full_encoder = False

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def predict_va(self, img: Image.Image):
        x = self.transform(img).unsqueeze(0).to(self.device)

        if self._use_full_encoder:
            out = self.encoder(x)
            if isinstance(out, dict):
                v = out.get("valence", out.get("va", out.get("output", None)))
                if v is not None:
                    if v.shape[-1] >= 2:
                        return v[0, 0].item(), v[0, 1].item()
            elif isinstance(out, torch.Tensor):
                if out.shape[-1] == 2:
                    return out[0, 0].item(), out[0, 1].item()
                elif out.shape[-1] > 2:
                    return out[0, 0].item(), out[0, 1].item()

        return None, None


def measure_efficiency(args, device):
    try:
        from inference.inference import ModEmoInferenceEngine
    except ImportError:
        from inference import ModEmoInferenceEngine

    test_img_path = None
    for idx in range(1, args.num_images + 1):
        candidate = os.path.join(args.input_dir, f"{idx}.png")
        if os.path.exists(candidate):
            test_img_path = candidate
            break

    test_image = Image.open(test_img_path).convert("RGB")

    engine = ModEmoInferenceEngine(args, device=device)
    try:
        result_metrics = {
            "num_inference_steps": args.num_inference_steps,
        }

        target_va = (args.target_valence, args.target_arousal)

        def run_once(run_seed):
            return engine.generate(
                input_image=test_image,
                target_va=target_va,
                seed=run_seed,
                verbose=False,
            )["image"]

        _ = run_once(args.seed)

        is_cuda = device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        times = []
        for run_idx in range(args.num_efficiency_runs):
            if is_cuda:
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            _ = run_once(args.seed + run_idx)
            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            print(f"    Run {run_idx + 1}/{args.num_efficiency_runs}: {elapsed:.3f}s")

        avg_time = float(np.mean(times))
        result_metrics["avg_time_sec"] = avg_time
        result_metrics["std_time_sec"] = float(np.std(times))
        result_metrics["throughput_img_per_sec"] = 1.0 / avg_time
        if is_cuda:
            result_metrics["peak_gpu_memory_GB"] = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            )
        return result_metrics
    finally:
        engine.close()


def find_output_image(output_dir: str, img_idx: int,
                      target_v: float = None, target_a: float = None):
    if target_v is not None and target_a is not None:
        exact_name = f"{img_idx}_edited_v{target_v:.3f}_a{target_a:.3f}.png"
        exact_path = os.path.join(output_dir, exact_name)
        if os.path.exists(exact_path):
            return exact_path

        exact_name2 = f"{img_idx}_dual_v{target_v:.2f}_a{target_a:.2f}"
        for f in os.listdir(output_dir):
            if f.startswith(exact_name2) and f.endswith(".png"):
                return os.path.join(output_dir, f)

    for f in sorted(os.listdir(output_dir)):
        if (f.startswith(f"{img_idx}_") and f.endswith(".png")
                and "comparison" not in f and "_ref_" not in f):
            return os.path.join(output_dir, f)

    return None


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_model = LPIPS(
        alexnet_weights=args.alexnet_weights,
        lpips_weights=args.lpips_weights,
    ).to(device).eval()

    lpips_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ])

    clip_sim = CLIPImageSimilarity(
        clip_weights=args.clip_weights,
        device=str(device),
    )

    va_predictor = None
    if args.emotion_encoder_path and os.path.exists(args.emotion_encoder_path):
        try:
            va_predictor = VAPredictorFromEncoder(
                encoder_path=args.emotion_encoder_path,
                device=str(device),
            )
        except Exception as e:
            print(f"  Failed to load VA Predictor: {e}")

    results = []
    psnr_list = []
    ssim_list = []
    lpips_list = []
    clip_i_list = []
    v_err_list = []
    a_err_list = []

    skipped = 0

    for img_idx in tqdm(range(1, args.num_images + 1), desc="evaluating"):
        input_path = os.path.join(args.input_dir, f"{img_idx}.png")
        if not os.path.exists(input_path):
            for ext in [".jpg", ".jpeg"]:
                alt = os.path.join(args.input_dir, f"{img_idx}{ext}")
                if os.path.exists(alt):
                    input_path = alt
                    break
            else:
                skipped += 1
                continue

        output_path = find_output_image(
            args.output_dir, img_idx,
            args.target_valence, args.target_arousal
        )
        if output_path is None:
            skipped += 1
            continue

        input_img = Image.open(input_path).convert("RGB")
        output_img = Image.open(output_path).convert("RGB")

        out_w, out_h = output_img.size
        input_img_resized = input_img.resize((out_w, out_h), Image.LANCZOS)

        input_np = np.array(input_img_resized)
        output_np = np.array(output_img)
        psnr_val = compute_psnr(input_np, output_np)
        psnr_list.append(psnr_val)

        ssim_val = compute_ssim(input_np, output_np)
        ssim_list.append(ssim_val)

        t_in = lpips_transform(input_img_resized).unsqueeze(0).to(device)
        t_out = lpips_transform(output_img).unsqueeze(0).to(device)
        with torch.no_grad():
            lpips_val = lpips_model(t_in, t_out).item()
        lpips_list.append(lpips_val)

        clip_i_val = clip_sim.compute_similarity(input_img_resized, output_img)
        clip_i_list.append(clip_i_val)

        v_err = None
        a_err = None

        if va_predictor is not None:
            pred_v, pred_a = va_predictor.predict_va(output_img)
            if pred_v is not None and pred_a is not None:
                v_err = abs(pred_v - args.target_valence)
                a_err = abs(pred_a - args.target_arousal)
                v_err_list.append(v_err)
                a_err_list.append(a_err)

        results.append({
            "idx": img_idx,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "clip_i": clip_i_val,
            "v_err": v_err,
            "a_err": a_err,
        })

    print("\n" + "=" * 60)
    print("Summary of Evaluation Results")
    print("=" * 60)
    print()

    metrics = {}

    if psnr_list:
        finite_psnr = [p for p in psnr_list if p != float('inf')]
        mean_psnr = np.mean(finite_psnr) if finite_psnr else float('inf')
        std_psnr = np.std(finite_psnr) if finite_psnr else 0.0
        metrics["PSNR"] = mean_psnr
        print(f"  PSNR  ↑ : {mean_psnr:.4f} ± {std_psnr:.4f}")

    if ssim_list:
        mean_ssim = np.mean(ssim_list)
        std_ssim = np.std(ssim_list)
        metrics["SSIM"] = mean_ssim
        print(f"  SSIM  ↑ : {mean_ssim:.4f} ± {std_ssim:.4f}")

    if lpips_list:
        mean_lpips = np.mean(lpips_list)
        std_lpips = np.std(lpips_list)
        metrics["LPIPS"] = mean_lpips
        print(f"  LPIPS ↓ : {mean_lpips:.4f} ± {std_lpips:.4f}")

    if clip_i_list:
        mean_clip = np.mean(clip_i_list)
        std_clip = np.std(clip_i_list)
        metrics["CLIP-I"] = mean_clip
        print(f"  CLIP-I↑ : {mean_clip:.4f} ± {std_clip:.4f}")

    if v_err_list:
        mean_v = np.mean(v_err_list)
        std_v = np.std(v_err_list)
        metrics["Valence_Error"] = mean_v
        print(f"  V Err ↓ : {mean_v:.4f} ± {std_v:.4f}")

    if a_err_list:
        mean_a = np.mean(a_err_list)
        std_a = np.std(a_err_list)
        metrics["Arousal_Error"] = mean_a
        print(f"  A Err ↓ : {mean_a:.4f} ± {std_a:.4f}")

    print()

    if args.measure_efficiency:
        print("=" * 60)
        print("Measurement of generation efficiency")
        print("=" * 60)

        efficiency_metrics = measure_efficiency(args, device)
        metrics.update(efficiency_metrics)

        print(f"  Avg Time/Image ↓ : {efficiency_metrics.get('avg_time_sec', 0):.3f} s")
        print(f"  Throughput ↑     : {efficiency_metrics.get('throughput_img_per_sec', 0):.3f} img/s")
        if efficiency_metrics.get('peak_gpu_memory_GB'):
            print(f"  Peak GPU Memory  : {efficiency_metrics['peak_gpu_memory_GB']:.2f} GB")
        print()

    save_dir = args.results_save_dir or args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "eval_per_image.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "psnr", "ssim", "lpips", "clip_i", "v_err", "a_err"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Result of each sample→ {csv_path}")

    summary = {
        "num_evaluated": len(results),
        "num_skipped": skipped,
        "target_valence": args.target_valence,
        "target_arousal": args.target_arousal,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    json_path = os.path.join(save_dir, "eval_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary of results → {json_path}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="ModEmo Evaluation")

    parser.add_argument("--input_dir", type=str, default="   ")
    parser.add_argument("--output_dir", type=str, default="   ")
    parser.add_argument("--results_save_dir", type=str, default="   ",)

    parser.add_argument("--target_valence", type=float, default=0.5)
    parser.add_argument("--target_arousal", type=float, default=0.4)

    parser.add_argument("--num_images", type=int, default=415)

    parser.add_argument("--alexnet_weights", type=str, default="   ")
    parser.add_argument("--lpips_weights", type=str, default="   ")
    parser.add_argument("--clip_weights", type=str, default="   ")

    parser.add_argument("--emotion_encoder_path", type=str, default="   ")
    parser.add_argument("--checkpoint_path", type=str, default="   ")

    parser.add_argument("--measure_efficiency", action="store_true")
    parser.add_argument("--sdxl_path", type=str, default="   ")
    parser.add_argument("--ip_adapter_path", type=str, default="   ")
    parser.add_argument("--ip_adapter_weight_name", type=str, default="   ")
    parser.add_argument("--attr_va_stats_path", type=str, default="   ")
    parser.add_argument("--num_efficiency_runs", type=int, default=5)
    parser.add_argument("--num_vectors", type=int, default=4)
    parser.add_argument("--num_mapper_layers", type=int, default=3)

    parser.add_argument("--ip_scale", type=float, default=0.6)
    parser.add_argument("--norm_scale", type=float, default=5.0)
    parser.add_argument("--placeholder_token", type=str, default="<emo>")
    parser.add_argument("--initializer_token", type=str, default="emotion")
    parser.add_argument("--prompt_prefix", type=str, default="best quality, high quality")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, deformed")
    parser.add_argument("--va_threshold", type=float, default=0.3)
    parser.add_argument("--top_k_images", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
