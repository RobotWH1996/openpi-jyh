import torch
import torch.nn.functional as F

def compute_rwbc_weights(rewards, beta=2.0, clip_max=10.0):
    """
    Converts offline scalar rewards [0,1] into normalized batch weights.
    beta: Controls how aggressively to prioritize good trajectories.
          Higher beta = more focus on high-reward demos.
    """
    # Ensure rewards is a tensor
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
    weights = torch.exp(beta * rewards)
    weights = torch.clamp(weights, max=clip_max)
    
    # Normalize by batch mean to keep the learning rate stable
    # Adding 1e-8 prevents division by zero
    return weights / (weights.mean() + 1e-8)


def compute_pi05_rwbc_loss(v_pred, v_target, rewards, beta=2.0):
    """
    Drop-in replacement for the Flow Matching MSE loss in OpenPI's pi0.5.
    
    Args:
        v_pred: Predicted velocity field from action expert (Batch, Time, Action_Dim)
        v_target: Target velocity field (Batch, Time, Action_Dim)
        rewards: The heuristic reward scores for this batch (Batch,)
        beta: RWBC exponent factor
        
    Returns:
        weighted_loss: The scalar loss to call .backward() on.
        avg_weight: For logging purposes to wandb/tensorboard.
    """
    # 1. Get sample weights for the batch
    weights = compute_rwbc_weights(rewards, beta=beta).to(v_pred.device)
    
    # 2. Unreduced MSE loss
    # reduction='none' is CRITICAL so we can weight each sample individually
    # Shape: (Batch, Time, Action_Dim)
    base_loss = F.mse_loss(v_pred, v_target, reduction='none')
    
    # 3. Average across time and action dimensions to get 1 scalar loss per sample
    # Shape: (Batch,)
    loss_per_sample = base_loss.mean(dim=[1, 2])
    
    # 4. Apply weights and compute final batch loss
    weighted_loss = (loss_per_sample * weights).mean()
    
    return weighted_loss, weights.mean().item()
