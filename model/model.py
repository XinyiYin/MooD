import torch
import torch.nn as nn
import torch.nn.functional as F
from affect_encoder.encoder import EmotionEncoder

class image_encoder(nn.Module):
    def __init__(self, clip_dim=768, emo_dim=768):
        super(image_encoder, self).__init__()
        self.resnet = EmotionEncoder()
        state = torch.load("  ")
        self.resnet.load_state_dict(state)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[1:-2])

    def forward(self, x):
        out = self.resnet(x)
        return out


class EmotionMapper(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, num_vectors=4,
                 num_layers=3, use_layernorm=True, dropout=0.1):
        super().__init__()
        self.num_vectors = num_vectors
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim
        hidden_dim = (input_dim + output_dim * num_vectors) // 2

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim * num_vectors))
        self.mlp = nn.Sequential(*layers)

    def forward(self, emo_embed):
        out = self.mlp(emo_embed)
        return out.view(-1, self.num_vectors, self.output_dim)


class CLIPLtoGProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=1280, num_vectors=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, clip_l_tokens):
        return self.proj(clip_l_tokens)


class VARegressor(nn.Module):
    def __init__(self, clip_l_dim=768, clip_g_dim=1280, hidden_dim=256):
        super().__init__()

        self.clip_l_branch = nn.Sequential(
            nn.Linear(clip_l_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.clip_g_branch = nn.Sequential(
            nn.Linear(clip_g_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )

    def forward(self, clip_l_embeds, clip_g_embeds):
        if len(clip_l_embeds.shape) == 3:
            clip_l_embeds = torch.mean(clip_l_embeds, dim=1)
        if len(clip_g_embeds.shape) == 3:
            clip_g_embeds = torch.mean(clip_g_embeds, dim=1)

        l_features = self.clip_l_branch(clip_l_embeds)
        g_features = self.clip_g_branch(clip_g_embeds)

        combined = torch.cat([l_features, g_features], dim=-1)
        va_pred = self.shared_head(combined)

        return va_pred


class ModEmoModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = image_encoder(
            clip_dim=768,
            emo_dim=args.emo_dim
        )
        self.mapper = EmotionMapper(
            input_dim=args.emo_dim,
            output_dim=768,
            num_vectors=args.num_vectors,
            num_layers=args.num_mapper_layers,
            dropout=args.mapper_dropout,
        )
        self.clip_g_projector = CLIPLtoGProjector(
            input_dim=768,
            output_dim=1280,
            num_vectors=args.num_vectors,
        )
        self.va_regressor = VARegressor(
            clip_l_dim=768,
            clip_g_dim=1280,
            hidden_dim=256
        )

        self.encoder.requires_grad_(False)

    def get_trainable_parameters(self):
        params = []
        params += list(self.mapper.parameters())
        params += list(self.clip_g_projector.parameters())
        params += list(self.va_regressor.parameters())
        return params

    def forward(self, pixel_values, return_va_pred=False):
        with torch.no_grad():
            emo_embed = self.encoder(pixel_values)

        clip_l_tokens = self.mapper(emo_embed)  # (B, N, 768)
        clip_g_tokens = self.clip_g_projector(clip_l_tokens)  # (B, N, 1280)

        if return_va_pred:
            va_pred = self.va_regressor(clip_l_tokens, clip_g_tokens)
            return clip_l_tokens, clip_g_tokens, va_pred

        return clip_l_tokens, clip_g_tokens

