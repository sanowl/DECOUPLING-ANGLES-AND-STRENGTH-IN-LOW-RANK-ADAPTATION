import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

class DeLoRA(nn.Module):
    def __init__(
        self,
        pretrained_W: torch.Tensor,
        r: int,
        lambda_init: float = 1.0,
        bias: bool = False,
        epsilon: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super(DeLoRA, self).__init__()
        
        # Store pre-trained weight and its Frobenius norm
        self.pretrained_W = pretrained_W  # Shape: (out_features, in_features)
        self.r = r
        self.epsilon = epsilon
        self.norm_W = torch.norm(pretrained_W, p='fro')  # Scalar
        
        # Trainable parameters
        out_features, in_features = pretrained_W.size()
        self.B = nn.Parameter(torch.empty(out_features, r, device=device, dtype=dtype))
        self.A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, device=device, dtype=dtype))
        
        # Buffers for initial values (non-trainable)
        self.register_buffer('B0', self.B.detach().clone())
        self.register_buffer('A0', self.A.detach().clone())
        self.register_buffer('lambda0', self.lambda_param.detach().clone())
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
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
        """
        Forward pass of the DeLoRA layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Handle 1D input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        # Compute A x and A0 x
        A_x = self.A @ x.T    # Shape: (r, batch_size)
        A0_x = self.A0 @ x.T  # Shape: (r, batch_size)
        
        # Component-wise normalization
        norm_b = torch.norm(self.B, dim=0)  # Shape: (r,)
        norm_a = torch.norm(self.A, dim=1)  # Shape: (r,)
        xi = 1.0 / (norm_b * norm_a + self.epsilon)  # Shape: (r,)
        
        norm_b0 = torch.norm(self.B0, dim=0)  # Shape: (r,)
        norm_a0 = torch.norm(self.A0, dim=1)  # Shape: (r,)
        xi0 = 1.0 / (norm_b0 * norm_a0 + self.epsilon)  # Shape: (r,)
        
        # Compute dynamic and initial updates
        delta_y = (self.lambda_param * self.norm_W / self.r) * (self.B @ (xi[:, None] * A_x))
        delta_y0 = (self.lambda0 * self.norm_W / self.r) * (self.B0 @ (xi0[:, None] * A0_x))
        
        # Pre-trained output
        W_x = self.pretrained_W @ x.T  # Shape: (out_features, batch_size)
        
        # Combine components
        y = W_x + delta_y - delta_y0  # Shape: (out_features, batch_size)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(1)
        
        return y.T  # Shape: (batch_size, out_features)

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
    """
    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        lambda_init: float = 1.0,
        target_modules: Optional[List[str]] = None,
        bias: bool = False,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.rank = rank
        self.lambda_init = lambda_init
        self.target_modules = target_modules
        self.bias = bias
        self.epsilon = epsilon
        self.converted_modules = []
        
        self._apply_delora()
    
    def _apply_delora(self):
        """Replace target linear layers with DeLoRA versions."""
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
    def _split_name(name: str) -> tuple[str, str]:
        """Split module name into parent and child parts."""
        return name.rsplit('.', 1) if '.' in name else ('', name)
    
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
    
    def get_delora_params(self) -> List[torch.Tensor]:
        """Return trainable DeLoRA parameters for optimization."""
        params = []
        for _, module in self.model.named_modules():
            if isinstance(module, DeLoRALinearWrapper):
                params.extend(module.delora.parameters())
        return params

    def merge_weights(self):
        """Merge DeLoRA updates into the base model's weights."""
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
    optimizer = torch.optim.AdamW(delora_model.get_delora_params(), lr=1e-3)
    
    # Forward pass
    x = torch.randn(5, 10)
    output = delora_model(x)
    print(f"Output shape: {output.shape}")
    
    # Merge weights after training
    delora_model.merge_weights()