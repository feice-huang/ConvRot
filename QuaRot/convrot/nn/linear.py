import torch
import quarot
import torch.nn as nn
from ..functional.convhadamard import Regular

class ConvLinear4bit(torch.nn.Module):
    _hadamard_modules = {} # 全局缓存旋转矩阵
    _n = 256 # 默认旋转矩阵大小

    def __init__(self, weight, rot_need, scale=None, bias=True, dtype=torch.float16, rot_size=None):
        super().__init__()
        self.in_features = weight.shape[-1]
        self.out_features = weight.shape[-2]
        self.dtype = dtype

        self.register_buffer('weight_scales', torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                     dtype=torch.uint8, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None

        self.rot_need = rot_need # 每个 Linear 层是否旋转
        if rot_size == None:
            rot_size = self._n
         # 缓存全局 ConvHadamard 实例和旋转矩阵
        if rot_size not in self.__class__._hadamard_modules:
            hadamard = Regular(rot_size)
            self.__class__._hadamard_modules[rot_size] = hadamard
        self.hadamard = self.__class__._hadamard_modules[rot_size]

        self.scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x):
        # 用 ConvHadamard.forward 替代 conv1d_matmul
        if self.rot_need:
            # pad_to_block=False 让输入长度必须是 block 整倍数，如果可能不整倍可改为 True
            x_rot = self.hadamard(x, pad_to_block=True)
        else:
            x_rot = x

        scales_x = (torch.max(torch.abs(x_rot), dim=-1)[0].unsqueeze(1) / 7).to(torch.float16)
        quantized_x = quarot.sym_quant(x_rot, scales_x)

        x = quarot.matmul(quantized_x, self.weight)

        if self.bias is not None:
            return quarot.sym_dequant(x, scales_x, self.scale) + self.bias
        else:
            return quarot.sym_dequant(x, scales_x, self.scale)

    @staticmethod
    def from_float(module: torch.nn.Linear, rot_need, rot_size):
        if rot_size == None:
            rot_size = ConvLinear4bit._n
        # 获取全局 ConvHadamard 和旋转矩阵
        if rot_size not in ConvLinear4bit._hadamard_modules:
            hadamard = Regular(rot_size)
            ConvLinear4bit._hadamard_modules[rot_size] = hadamard
        hadamard = ConvLinear4bit._hadamard_modules[rot_size]

        # 旋转权重
        if rot_need:
            weight_matrix = hadamard(module.weight, pad_to_block=True)
        else:
            weight_matrix = module.weight

        # 计算 weight scales
        weight_scales = (torch.max(torch.abs(weight_matrix), dim=-1)[0].unsqueeze(1)/7).to(
            dtype=torch.float16, 
            device=weight_matrix.device  # 与 weight_matrix 同设备
        )

        # 构造 ConvLinear4bit_Flux 实例
        int_module = ConvLinear4bit(weight_matrix, rot_need=rot_need, scale=weight_scales,
                                    bias=module.bias is not None, dtype=weight_matrix.dtype, rot_size=rot_size)

        # 量化权重并填充 buffer
        int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
        int_rounded_weight = (weight_matrix / weight_scales.cuda()).round()
        int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())

        if module.bias is not None:
            int_module.bias.copy_(module.bias)

        return int_module
    

