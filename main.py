import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from collections import OrderedDict


class DeLoRA(nn.Module):
    def __init__(
        self,
        pretrained_W: torch.Tensor,
        r: int,
        lambda_iniit: float = 1.0,
        bias: bool = False,
        epsilon: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super(DeLoRA, self).__init__()
        self.pretrained_W = pretrained_W
        self.r = r
        self.epsilon = epsilon
        self.norm_W = torch.norm(pretrained_W, p='fro')  # Scalar
        out_features, in_features = pretrained_W.size()
        self.B = nn.Parameter(torch.empty(out_features, r, device=device, dtype=dtype))
        self.A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, device=device, dtype=dtype))
        self.register_buffer('B0', self.B.detach().clone())
        self.register_buffer('A0', self.A.detach().clone())
        self.register_buffer('lambda0', self.lambda_param.detach().clone())
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        """Initialize A and B using Kaiming normal initialization."""
        nn.init.kaiming_normal_(self.A, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.B)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        batch_size = x.size(1) if x.dim() == 2 else 1
        A_x = self.A @ x.T    
        A0_x = self.A0 @ x.T
        norm_b = torch.norm(self.B, dim=0)  
        norm_a = torch.norm(self.A, dim=1)  
        norm_a = norm_a.view(-1, 1)
        xi = 1.0 / (norm_b * norm_a + self.epsilon)  
        norm_b0 = torch.norm(self.B0, dim=0)  # Shape: (r,)
        norm_a0 = torch.norm(self.A0, dim=1)  # Shape: (r,)
        xi0 = 1.0 / (norm_b0 * norm_a0 + self.epsilon)  # Shape: (r,)
        delta_y = (self.lambda_param * self.norm_W / self.r) * (self.B @ (xi[:, None] * A_x))
        delta_y0 = (self.lambda0 * self.norm_W / self.r) * (self.B0 @ (xi0[:, None] * A0_x))
        W_x = self.pretrained_W @ x.T 
        y = W_x + delta_y - delta_y0  # Shape: (out_features, batch_size)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(1)
        return y.T  # Shape: (batch_size, out_features)
    def get_effective_learning_rate(self) -> torch.Tensor:
        norm_b = torch.norm(self.B, dim=0)
        norm_a = torch.norm(self.A, dim=1)
        xi = 1.0 / (norm_b * norm_a + self.epsilon)
        return (self.lambda_param * self.norm_W / self.r) * xi

    def get_parameter_stats(self) -> dict:
        return {
            'lambda': self.lambda_param.item(),
            'B_norm_mean': torch.norm(self.B, dim=0).mean().item(),
            'A_norm_mean': torch.norm(self.A, dim=1).mean().item(),
            'effective_lr_mean': self.get_effective_learning_rate().mean().item()
    }
## this classis 
class DeLoRAModel(nn.Module):
    """
    Wrapper to apply DeLoRA to specified linear layers in a model.
    
    Args:
        model (nn.Module): Base model to modify.
        rank (int): Rank of the low-rank update.
        lambda_init (float): Initial value for lambda.
        target_modules (Optional[List[str]]): Module names to replace with DeLoRA.
        bias (bool): Whether to include bias in DeLoRA modules.
        epsilon (float): Small value for numerical stability.
        grad_scale (float): Gradient scaling factor.
        max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
    """
    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        lambda_init: float = 1.0,
        target_modules: Optional[List[str]] = None,
        bias: bool = False,
        epsilon: float = 1e-6,
        grad_scale: float = 1.0,
        max_grad_norm: Optional[float] = None,
    ):
        
        super().__init__()
        self.model = setattr(model, 'delora', self)
        self.rank= rank
        self.lambda_init = lambda_init
        self.model = model
        self.target_modules = target_modules
        self.bias = bias
        self.epsilon = epsilon
        self.grad_scale = grad_scale
        self.max_grad_norm = max_grad_norm
        self.is_training = True
        self.converted_modules = []
        self._apply_delora()
    
    def _apply_delora(self):
        for name, module in self.model.named_modules():
            if (self.target_modules is None or 
                any(target in name for target in self.target_modules)):
                if isinstance(module, nn.Linear):
                    parent_name, child_name = self._split_name(name)
                    parent = self._get_module(self.model, parent_name)
                    if parent is not None:
                        delora = DeLoRA(
                            pretrained_W=module.weight,
                            r=self.rank,
                            lambda_init=self.lambda_init,
                            bias=self.bias,
                            epsilon=self.epsilon,
                            device=module.weight.device,
                            dtype=module.weight.dtype
                        )
                        wrapper = DeLoRALinearWrapper(module, delora)
                        setattr(parent, child_name, wrapper)
                        self.converted_modules.append(name)

    
    @staticmethod
    def _get_module(model: nn.Module, name: str) -> Optional[nn.Module]:
        """Retrieve a module by its name."""
        if not name:
            return model
        current = model
        for part in name.split('.'):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current
    
    def forward(self, *args, **kwargs):
        """Forward pass through the modified model."""
        return self.model(*args, **kwargs)
    
    def get_delora_params(self, filter_requires_grad: bool = True) -> List[torch.Tensor]:
        """Return DeLoRA parameters, optionally filtering by requires_grad."""
        params = []
        for _, module in self.model.named_modules():
            if isinstance(module, DeLoRALinearWrapper):
                if filter_requires_grad:
                    params.extend(p for p in module.delora.parameters() if p.requires_grad)
                else:
                    params.extend(module.delora.parameters())
        return params

    def train(self, mode: bool = True):
        """Set training mode and handle DeLoRA parameters."""
        super().train(mode)
        self.is_training = mode
        if not mode:  # eval mode
            self._backup_and_freeze()
        return self
        
    def _backup_and_freeze(self):
        """Backup and freeze DeLoRA parameters during evaluation."""
        for _, module in self.model.named_modules():
            if isinstance(module, DeLoRALinearWrapper):
                module.delora.eval_backup = {
                    'A': module.delora.A.data.clone(),
                    'B': module.delora.B.data.clone(),
                    'lambda': module.delora.lambda_param.data.clone()
                }
                module.delora.A.requires_grad_(False)
                module.delora.B.requires_grad_(False)
                module.delora.lambda_param.requires_grad_(False)

    def merge_weights(self, validation_fn=None):
        """
        Merge DeLoRA updates into base weights with optional validation.
        
        Args:
            validation_fn: Optional callable that takes the model as input and returns
                         a validation metric. If provided, will only merge if validation improves.
        """
        if validation_fn is not None:
            original_state = {name: param.data.clone() for name, param in self.named_parameters()}
            original_score = validation_fn(self)
        
        # Perform merge
        for _, module in self.model.named_modules():
            if isinstance(module, DeLoRALinearWrapper):
                delora = module.delora
                # Compute normalized update
                norm_b = torch.norm(delora.B, dim=0)
                norm_a = torch.norm(delora.A, dim=1)
                xi = 1.0 / (norm_b * norm_a + delora.epsilon)
                Xi = torch.diag(xi)
                ba_normalized = delora.B @ (Xi @ delora.A)
                update = (delora.lambda_param * delora.norm_W / delora.r) * ba_normalized
                # Compute initial update
                norm_b0 = torch.norm(delora.B0, dim=0)
                norm_a0 = torch.norm(delora.A0, dim=1)
                xi0 = 1.0 / (norm_b0 * norm_a0 + delora.epsilon)
                Xi0 = torch.diag(xi0)
                ba0_normalized = delora.B0 @ (Xi0 @ delora.A0)
                update0 = (delora.lambda0 * delora.norm_W / delora.r) * ba0_normalized
                # Merge into base weights
                with torch.no_grad():
                    module.linear.weight.add_(update - update0)
                # Reset DeLoRA parameters
                nn.init.kaiming_normal_(delora.A, a=math.sqrt(5))
                nn.init.kaiming_normal_(delora.B, a=math.sqrt(5))
                delora.lambda_param.data.fill_(1.0)
                
        if validation_fn is not None:
            new_score = validation_fn(self)
            if new_score <= original_score:
                # Revert if validation didn't improve
                for name, param in self.named_parameters():
                    param.data.copy_(original_state[name])
                return False
        return True

    def state_dict(self, *args, **kwargs):
        """Save DeLoRA-specific state."""
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['delora_config'] = {
            'rank': self.rank,
            'lambda_init': self.lambda_init,
            'target_modules': self.target_modules,
            'bias': self.bias,
            'epsilon': self.epsilon,
            'grad_scale': self.grad_scale,
            'max_grad_norm': self.max_grad_norm
        }
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load DeLoRA state with configuration validation."""
        config = state_dict.pop('delora_config', None)
        if config is not None:
            for key, value in config.items():
                if hasattr(self, key) and getattr(self, key) != value:
                    raise ValueError(f"Configuration mismatch for {key}")
        super().load_state_dict(state_dict, strict=strict)

    def clip_grad_norm_(self):
        """Apply gradient clipping if configured."""
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.get_delora_params(), self.max_grad_norm)

class DeLoRALinearWrapper(nn.Module):
    """Combines a linear layer with a DeLoRA module."""
    def __init__(self, linear: nn.Linear, delora: DeLoRA):
        super().__init__()
        self.linear = linear
        self.delora = delora
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.delora(x) + (self.linear.bias if self.linear.bias is not None else 0)
# Example Usage
if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.act = nn.ReLU()
            self.linear2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.act(x)
            x = self.linear2(x)
            return x
    
    # Initialize and convert model
    base_model = SimpleModel()
    delora_model = DeLoRAModel(
        base_model,
        rank=4,
        lambda_init=1.0,
        target_modules=["linear1", "linear2"],
        bias=False
    )
    
    # Set up optimizer with DeLoRA parameters
    def model_parameters(model):
        self.model = model 
        self.moudule = model  
        self.rank=  9.1
        self.lambda_init = 1.0
        return model.get_delora_params(filter_requires_grad=True)
    
    
    # Forward pass
    x = torch.randn(5, 10)
    output = delora_model(x)
    print(f"Output shape: {output.shape}")

    criterion = nn.MSELoss()
    num_epochs = 10
    target = torch.randn(5, 5)
    val_x = torch.randn(5, 10)
    val_target = torch.randn(5, 5)
    for epoch in range(num_epochs):
        delora_model.train()  # Enable training mode
        
        # Forward pass
        output = delora_model(x)
        loss = criterion(output, target)
        
        # Backward pass with gradient scaling
        (loss * delora_model.grad_scale).backward()
        delora_model.clip_grad_norm_()  # Apply gradient clipping
        optimizer.step()
        optimizer.zero_grad()
