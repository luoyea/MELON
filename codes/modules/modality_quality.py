import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityQualityScorer(nn.Module):
    """
    Estimate modality quality score q_m in [0, 1] from:
    - user modality embedding
    - item modality embedding
    - raw modality feature
    """
    def __init__(self, emb_dim, raw_dim, hidden_dim=128):
        super(ModalityQualityScorer, self).__init__()
        in_dim = emb_dim * 2 + raw_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, user_emb, item_emb, raw_feat):
        x = torch.cat([user_emb, item_emb, raw_feat], dim=1)
        h = F.relu(self.fc1(x))
        q = torch.sigmoid(self.fc2(h)).squeeze(1)
        return q


class AspectDynamicGate(nn.Module):
    """
    Build aspect-level modality weights w_{a,m} with softmax normalization.
    """
    def __init__(self, quality_scale=1.0):
        super(AspectDynamicGate, self).__init__()
        self.quality_scale = quality_scale

    def forward(self, u_img_a, i_img_a, u_txt_a, i_txt_a, q_img, q_txt, use_aqg=True, fixed_weight=False):
        """
        Shapes:
          u_img_a, i_img_a, u_txt_a, i_txt_a: [B, A, D_a]
          q_img, q_txt: [B]
        Return:
          w_img, w_txt: [B, A]
        """
        batch_size, n_aspects, _ = u_img_a.shape

        if fixed_weight:
            fixed = torch.full((batch_size, n_aspects), 0.5, device=u_img_a.device, dtype=u_img_a.dtype)
            return fixed, fixed

        if use_aqg:
            img_logits = torch.sum(u_img_a * i_img_a, dim=2)
            txt_logits = torch.sum(u_txt_a * i_txt_a, dim=2)
        else:
            # No aspect-aware dynamics: use global modality logits and broadcast.
            img_logits = torch.zeros((batch_size, n_aspects), device=u_img_a.device, dtype=u_img_a.dtype)
            txt_logits = torch.zeros((batch_size, n_aspects), device=u_img_a.device, dtype=u_img_a.dtype)

        img_logits = img_logits + self.quality_scale * q_img.unsqueeze(1)
        txt_logits = txt_logits + self.quality_scale * q_txt.unsqueeze(1)

        logits = torch.stack([img_logits, txt_logits], dim=2)  # [B, A, 2]
        weights = torch.softmax(logits, dim=2)
        w_img = weights[:, :, 0]
        w_txt = weights[:, :, 1]
        return w_img, w_txt


class AspectFusion(nn.Module):
    """
    Fuse aspect-level modality matching scores with dynamic weights.
    """
    def __init__(self):
        super(AspectFusion, self).__init__()

    def forward(self, u_img_a, i_img_a, u_txt_a, i_txt_a, w_img, w_txt):
        """
        Shapes:
          u_img_a, i_img_a, u_txt_a, i_txt_a: [B, A, D_a]
          w_img, w_txt: [B, A]
        Return:
          s_mq: [B]
        """
        img_score = torch.sum(u_img_a * i_img_a, dim=2)
        txt_score = torch.sum(u_txt_a * i_txt_a, dim=2)
        aspect_score = w_img * img_score + w_txt * txt_score
        s_mq = torch.sum(aspect_score, dim=1)
        return s_mq
