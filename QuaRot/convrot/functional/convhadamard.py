import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_hadamard_transform import hadamard_transform
import quarot


def _is_power_of_two(n: int) -> bool:
    return (n & (n - 1)) == 0 and n > 0

def _is_power_of_four(n: int) -> bool:
    """判断是否为 4 的幂（4, 16, 64, ...）。"""
    if n < 4:
        return False
    # 必须是 2 的幂，且指数为偶数
    return _is_power_of_two(n) and (n & 0x55555555) == n

def _make_hadamard_raw(n: int, dtype=torch.float32, device="cpu"):
    """构造不带缩放的 Sylvester Hadamard（元素为 ±1）。"""
    assert _is_power_of_two(n), "Hadamard size must be power of two"
    if n == 1:
        return torch.tensor([[1.0]], dtype=dtype, device=device)
    Hn2 = _make_hadamard_raw(n // 2, dtype=dtype, device=device)
    top = torch.cat([Hn2, Hn2], dim=1)
    bot = torch.cat([Hn2, -Hn2], dim=1)
    return torch.cat([top, bot], dim=0)

def _make_hadamard_normalized(n: int, dtype=torch.float32, device="cpu"):
    """只在最终结果上乘以 1/sqrt(n)。"""
    H = _make_hadamard_raw(n, dtype=dtype, device=device)
    return H * (1.0 / math.sqrt(n))

def _random_orthogonal_matrix(n, dtype=torch.float32, device="cpu"):
    """生成n阶随机正交矩阵"""
    random_matrix = torch.randn(n, n)
    
    # 使用QR分解获取正交矩阵Q
    q, r = torch.linalg.qr(random_matrix)
    
    # 确保行列式为正（可选）
    det = torch.linalg.det(q)
    if det < 0:
        q[:, 0] = -q[:, 0]
    
    return q.to(dtype).to(device)

def _make_hadamard_regular(n: int, dtype=torch.float16, device="cpu"):
    """
    构造缩放后的 Regular Hadamard（元素为 ±1/sqrt(n)）。
    仅支持 n = 4, 16, 64, 256 ... 4的幂次Sylvester类型。
    """
    if not _is_power_of_four(n):
        raise ValueError("n must be a power of 4 (4, 16, 64, ...)")

    # 定义 4x4 基础 H4
    H4 = torch.tensor([
        [ 1,  1,  1, -1],
        [ 1,  1, -1,  1],
        [ 1, -1,  1,  1],
        [-1,  1,  1,  1]
    ], dtype=dtype, device=device)

    H = H4
    current_order = 4
    # 迭代 Kronecker 构造
    while current_order < n:
        H = torch.kron(H4, H)
        current_order *= 4

    return H * (1.0 / math.sqrt(n))

class Regular(nn.Module):
    def __init__(self, size: int = 256):
        super().__init__()
        assert _is_power_of_two(size), "size must be power of two"
        self.size = int(size)
        print("[ConvHadamard] Create Regular Hadamard matrix with size =", size)
        H = _make_hadamard_regular(self.size, dtype=torch.float16, device="cuda")

        self.register_buffer("H_matrix", H)

    def forward(self, x: torch.Tensor, pad_to_block: bool = True) -> torch.Tensor:
        block = self.size
        orig_W = x.shape[-1]
        W = orig_W
        if W % block != 0:
            # print("Padding")
            if not pad_to_block:
                raise ValueError(f"last dim {W} not divisible by block={block}. set pad_to_block=True")
            pad = (block - (W % block)) % block
            x = F.pad(x, (0, pad))
            W = x.shape[-1]

        num_blocks = W // block
        leading = x.shape[:-1]

        y = x.reshape(-1, num_blocks, block)
        flat = y.reshape(-1, block)

        H = self.H_matrix.to(dtype=flat.dtype, device=flat.device)
        out_flat = flat @ H.T

        out = out_flat.view(-1, num_blocks, block).reshape(*leading, W)
        return out

    def get_rotation(self, device=None, dtype=None) -> torch.Tensor:
        """
        返回 Hadamard 正交旋转矩阵 Q (size x size)，已归一化。
        """
        return self.H_matrix.to(device=device or self.H_matrix.device,
                                dtype=dtype or self.H_matrix.dtype)
    

class Standard(nn.Module):
    def __init__(self, size: int = 256):
        super().__init__()
        assert _is_power_of_two(size), "size must be power of two"
        self.size = int(size)
        H = _make_hadamard_normalized(self.size, dtype=torch.float16, device="cuda")
        print("[ConvHadamard] Create Standard Hadamard matrix with size =", size)

        self.register_buffer("H_matrix", H)

    def forward(self, x: torch.Tensor, pad_to_block: bool = True) -> torch.Tensor:
        block = self.size
        orig_W = x.shape[-1]
        W = orig_W
        if W % block != 0:
            if not pad_to_block:
                raise ValueError(f"last dim {W} not divisible by block={block}. set pad_to_block=True")
            pad = (block - (W % block)) % block
            x = F.pad(x, (0, pad))
            W = x.shape[-1]

        num_blocks = W // block
        leading = x.shape[:-1]

        y = x.reshape(-1, num_blocks, block)
        flat = y.reshape(-1, block)

        H = self.H_matrix.to(dtype=flat.dtype, device=flat.device)
        out_flat = flat @ H.T

        out = out_flat.view(-1, num_blocks, block).reshape(*leading, W)
        return out

    def get_rotation(self, device=None, dtype=None) -> torch.Tensor:
        """
        返回 Hadamard 正交旋转矩阵 Q (size x size)，已归一化。
        """
        return self.H_matrix.to(device=device or self.H_matrix.device,
                                dtype=dtype or self.H_matrix.dtype)


class Random(nn.Module):
    def __init__(self, size: int = 256):
        super().__init__()
        assert _is_power_of_two(size), "size must be power of two"
        self.size = int(size)
        H = _random_orthogonal_matrix(self.size, dtype=torch.float16, device="cuda")
        print("[ConvHadamard] Create Random Hadamard matrix with size =", size)

        self.register_buffer("H_matrix", H)

    def forward(self, x: torch.Tensor, pad_to_block: bool = True) -> torch.Tensor:
        block = self.size
        orig_W = x.shape[-1]
        W = orig_W
        if W % block != 0:
            # print("Padding")
            if not pad_to_block:
                raise ValueError(f"last dim {W} not divisible by block={block}. set pad_to_block=True")
            pad = (block - (W % block)) % block
            x = F.pad(x, (0, pad))
            W = x.shape[-1]

        num_blocks = W // block
        leading = x.shape[:-1]

        y = x.reshape(-1, num_blocks, block)
        flat = y.reshape(-1, block)

        H = self.H_matrix.to(dtype=flat.dtype, device=flat.device)
        out_flat = flat @ H.T

        out = out_flat.view(-1, num_blocks, block).reshape(*leading, W)
        return out

    def get_rotation(self, device=None, dtype=None) -> torch.Tensor:
        """
        返回 Hadamard 正交旋转矩阵 Q (size x size)，已归一化。
        """
        return self.H_matrix.to(device=device or self.H_matrix.device,
                                dtype=dtype or self.H_matrix.dtype)

    
class FWHT(nn.Module):
    def __init__(self, hadamard_dim):
        super().__init__()
        had_rem_dim, self.rem_dim = quarot.functional.hadamard.get_hadK(hadamard_dim)
        if had_rem_dim is not None:
            self.register_buffer("had_rem_dim", had_rem_dim)
            self.had_rem_dim = self.had_rem_dim.to(torch.float16)
        else:
            self.had_rem_dim = None       
    
    def forward(self, x):
        x_dtype = x.dtype
        x = quarot.functional.matmul_hadU_cuda(x, self.had_rem_dim, self.rem_dim)
        x = x.to(x_dtype)
        return x
