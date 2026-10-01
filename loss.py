import torch
from torch import nn
from torch.nn import functional as F

class EraContrastiveLoss(nn.Module):
    """Supervised Era Contrastive (EC) Loss (Eq. 3 in He et al., 2024 / Khosla et al., 2020).

    Forces embeddings belonging to the same era class to be pulled together on the unit
    hypersphere while pushing apart embeddings from different era classes.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Embeddings are on the unit hypersphere: z_i . z_j is exact cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Subtract row-wise max for numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Mask out self-contrast (diagonal)
        logits_mask = torch.ones_like(sim) - torch.eye(batch_size, device=embeddings.device)

        # Mask for positive pairs (same label, excluding self)
        label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() * logits_mask

        # Log-probability: log [ exp(sim_ij / tau) / sum_{k != i} exp(sim_ik / tau) ]
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))

        # Mean over positive pairs for each anchor that has at least one positive
        pos_count = label_mask.sum(1)
        valid_anchors = pos_count > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        mean_log_prob_pos = (label_mask * log_prob).sum(1) / pos_count.clamp_min(1.0)
        loss = -mean_log_prob_pos[valid_anchors].mean()
        return loss


def supcon_loss(z, labels, temperature=.1):
    """Standard SupCon with all non-self views in the denominator."""
    z = F.normalize(z.float(), dim=-1)
    logits = z @ z.T / temperature
    n = logits.shape[0]
    self_mask = torch.eye(n, device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = positive.sum(1)
    valid = count > 0
    if not valid.any():
        raise ValueError("SupCon batch contains no positive pair")
    return -(log_prob.masked_fill(~positive, 0).sum(1)[valid] / count[valid]).mean()
