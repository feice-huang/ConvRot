import torch
import quarot
import torch.nn as nn
from ..functional.convhadamard import Regular, Standard, Random, FWHT


class Standard4bit(torch.nn.Module):
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

        self.rot_need = rot_need 
        if rot_size == None:
            rot_size = self._n
        # 缓存全局 ConvHadamard 实例和旋转矩阵
        if rot_size not in self.__class__._hadamard_modules:
            hadamard = Standard(rot_size)
            self.__class__._hadamard_modules[rot_size] = hadamard
        self.hadamard = self.__class__._hadamard_modules[rot_size]

        self.scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x):
        if self.rot_need:
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
            rot_size = Standard4bit._n
        # 获取全局 ConvHadamard 和旋转矩阵
        if rot_size not in Standard4bit._hadamard_modules:
            hadamard = Standard(rot_size)
            Standard4bit._hadamard_modules[rot_size] = hadamard
        hadamard = Standard4bit._hadamard_modules[rot_size]

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

        # 构造 Standard4bit_Flux 实例
        int_module = Standard4bit(weight_matrix, rot_need=rot_need, scale=weight_scales,
                                    bias=module.bias is not None, dtype=weight_matrix.dtype, rot_size=rot_size)

        # 量化权重并填充 buffer
        int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
        int_rounded_weight = (weight_matrix / weight_scales.cuda()).round()
        int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())

        if module.bias is not None:
            int_module.bias.copy_(module.bias)

        return int_module
    

class Regular4bit(torch.nn.Module):
    _hadamard_modules = {} # 全局缓存旋转矩阵
    _n = 256 # 默认旋转矩阵大小

    def __init__(self, weight, rot_need, scale=None, bias=True, dtype=torch.float16, rot_size=None):
        super().__init__()
        self.in_features = weight.shape[-1]
        self.out_features = weight.shape[-2]
        self.dtype = dtype

        self.register_buffer('weight_scales', torch.zeros((self.out_features, 1), requires_grad=False))
        # SubByte weight
        self.register_buffer('weight', torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                     dtype=torch.uint8, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None

        self.rot_need = rot_need
        if rot_size == None:
            rot_size = self._n
         # 缓存全局 ConvHadamard 实例和旋转矩阵
        if rot_size not in self.__class__._hadamard_modules:
            hadamard = Regular(rot_size)
            self.__class__._hadamard_modules[rot_size] = hadamard
        self.hadamard = self.__class__._hadamard_modules[rot_size]

        self.scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x):
        if self.rot_need:
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
            rot_size = Regular4bit._n
        # 获取全局 ConvHadamard 和旋转矩阵
        if rot_size not in Regular4bit._hadamard_modules:
            hadamard = Regular(rot_size)
            Regular4bit._hadamard_modules[rot_size] = hadamard
        hadamard = Regular4bit._hadamard_modules[rot_size]

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

        # 构造 Regular4bit_Flux 实例
        int_module = Regular4bit(weight_matrix, rot_need=rot_need, scale=weight_scales,
                                    bias=module.bias is not None, dtype=weight_matrix.dtype, rot_size=rot_size)

        # 量化权重并填充 buffer
        int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
        int_rounded_weight = (weight_matrix / weight_scales.cuda()).round()
        int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())

        if module.bias is not None:
            int_module.bias.copy_(module.bias)

        return int_module
    

class Random4bit(torch.nn.Module):
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
            hadamard = Random(rot_size)
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
        # rot_size = rot_size if rot_size else Random4bit_Flux._n
        if rot_size == None:
            rot_size = Random4bit._n
        # 获取全局 ConvHadamard 和旋转矩阵
        if rot_size not in Random4bit._hadamard_modules:
            hadamard = Random(rot_size)
            Random4bit._hadamard_modules[rot_size] = hadamard
        hadamard = Random4bit._hadamard_modules[rot_size]

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

        # 构造 Random4bit_Flux 实例
        int_module = Random4bit(weight_matrix, rot_need=rot_need, scale=weight_scales,
                                    bias=module.bias is not None, dtype=weight_matrix.dtype, rot_size=rot_size)

        # 量化权重并填充 buffer
        int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
        int_rounded_weight = (weight_matrix / weight_scales.cuda()).round()
        int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())

        if module.bias is not None:
            int_module.bias.copy_(module.bias)

        return int_module


class Standard8bit(torch.nn.Module):
    _hadamard_modules = {} 
    _n = 256 

    def __init__(self, weight, rot_need, scale=None, bias=True, dtype=torch.float16, rot_size=None):
        super().__init__()
        self.in_features = weight.shape[-1]
        self.out_features = weight.shape[-2]
        self.dtype = dtype

        self.register_buffer('weight_scales', torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', torch.zeros((self.out_features, self.in_features),
                                                   dtype=torch.int8, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.register_bias = None
            self.bias = None

        self.rot_need = rot_need
        if rot_size == None:
            rot_size = self._n
            
        if rot_size not in self.__class__._hadamard_modules:
            hadamard = Standard(rot_size)
            self.__class__._hadamard_modules[rot_size] = hadamard
        self.hadamard = self.__class__._hadamard_modules[rot_size]

        if scale is not None:
            self.weight_scales.copy_(scale)

    def forward(self, x):
        if self.rot_need:
            x_rot = self.hadamard(x, pad_to_block=True)
        else:
            x_rot = x

        max_val = torch.max(torch.abs(x_rot), dim=-1)[0].unsqueeze(-1)
        
        scales_x = (max_val / 127.0).to(x_rot.dtype)
        scales_x = torch.clamp(scales_x, min=1e-5)

        quantized_x = torch.clamp(torch.round(x_rot / scales_x), -127, 127).to(torch.int8)

        # Flatten for _int_mm
        x_shape = quantized_x.shape
        x_flat = quantized_x.view(-1, self.in_features)
        
        weight_t = self.weight.t()
        
        # INT8 GEMM
        out_int32 = torch._int_mm(x_flat, weight_t)

        out_int32 = out_int32.view(*x_shape[:-1], self.out_features)

        out = out_int32.to(torch.float32) * scales_x.to(torch.float32) * self.weight_scales.t().to(torch.float32)

        if self.bias is not None:
            out = out + self.bias.to(torch.float32)
            
        return out.to(self.dtype)

    @staticmethod
    def from_float(module: torch.nn.Linear, rot_need, rot_size):
        if rot_size == None:
            rot_size = Standard8bit._n
            
        if rot_size not in Standard8bit._hadamard_modules:
            hadamard = Standard(rot_size)
            Standard8bit._hadamard_modules[rot_size] = hadamard
        hadamard = Standard8bit._hadamard_modules[rot_size]

        if rot_need:
            weight_matrix = hadamard(module.weight, pad_to_block=True)
        else:
            weight_matrix = module.weight

        weight_scales = (torch.max(torch.abs(weight_matrix), dim=-1)[0].unsqueeze(-1) / 127.0).to(
            dtype=torch.float16, 
            device=weight_matrix.device
        )
        weight_scales = torch.clamp(weight_scales, min=1e-5)

        int_module = Standard8bit(
            weight_matrix, 
            rot_need=rot_need, 
            scale=weight_scales,
            bias=module.bias is not None, 
            dtype=weight_matrix.dtype, 
            rot_size=rot_size
        )

        int_rounded_weight = torch.clamp(
            torch.round(weight_matrix / weight_scales.to(weight_matrix.device)), 
            -127, 127
        )
        
        int_module.weight.copy_(int_rounded_weight.to(torch.int8))
        int_module.weight_scales.copy_(weight_scales)

        if module.bias is not None:
            int_module.bias.copy_(module.bias)

        return int_module
    

class Regular8bit(torch.nn.Module):
    _hadamard_modules = {} 
    _n = 256 

    def __init__(self, weight, rot_need, scale=None, bias=True, dtype=torch.float16, rot_size=None):
        super().__init__()
        self.in_features = weight.shape[-1]
        self.out_features = weight.shape[-2]
        self.dtype = dtype

        self.register_buffer('weight_scales', torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', torch.zeros((self.out_features, self.in_features),
                                                   dtype=torch.int8, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.register_bias = None
            self.bias = None

        self.rot_need = rot_need
        if rot_size == None:
            rot_size = self._n
            
        if rot_size not in self.__class__._hadamard_modules:
            hadamard = Regular(rot_size)
            self.__class__._hadamard_modules[rot_size] = hadamard
        self.hadamard = self.__class__._hadamard_modules[rot_size]

        if scale is not None:
            self.weight_scales.copy_(scale)

    def forward(self, x):
        if self.rot_need:
            x_rot = self.hadamard(x, pad_to_block=True)
        else:
            x_rot = x

        max_val = torch.max(torch.abs(x_rot), dim=-1)[0].unsqueeze(-1)
        
        scales_x = (max_val / 127.0).to(x_rot.dtype)
        scales_x = torch.clamp(scales_x, min=1e-5)

        quantized_x = torch.clamp(torch.round(x_rot / scales_x), -127, 127).to(torch.int8)

        # Flatten for _int_mm
        x_shape = quantized_x.shape
        x_flat = quantized_x.view(-1, self.in_features)
        
        weight_t = self.weight.t()
        
        # INT8 GEMM
        out_int32 = torch._int_mm(x_flat, weight_t)

        out_int32 = out_int32.view(*x_shape[:-1], self.out_features)
        
        out = out_int32.to(torch.float32) * scales_x.to(torch.float32) * self.weight_scales.t().to(torch.float32)

        if self.bias is not None:
            out = out + self.bias.to(torch.float32)
            
        return out.to(self.dtype)

    @staticmethod
    def from_float(module: torch.nn.Linear, rot_need, rot_size):
        if rot_size == None:
            rot_size = Regular8bit._n
            
        if rot_size not in Regular8bit._hadamard_modules:
            hadamard = Regular(rot_size)
            Regular8bit._hadamard_modules[rot_size] = hadamard
        hadamard = Regular8bit._hadamard_modules[rot_size]

        if rot_need:
            weight_matrix = hadamard(module.weight, pad_to_block=True)
        else:
            weight_matrix = module.weight

        weight_scales = (torch.max(torch.abs(weight_matrix), dim=-1)[0].unsqueeze(-1) / 127.0).to(
            dtype=torch.float16, 
            device=weight_matrix.device
        )
        weight_scales = torch.clamp(weight_scales, min=1e-5)

        int_module = Regular8bit(
            weight_matrix, 
            rot_need=rot_need, 
            scale=weight_scales,
            bias=module.bias is not None, 
            dtype=weight_matrix.dtype, 
            rot_size=rot_size
        )

        int_rounded_weight = torch.clamp(
            torch.round(weight_matrix / weight_scales.to(weight_matrix.device)), 
            -127, 127
        )
        
        int_module.weight.copy_(int_rounded_weight.to(torch.int8))
        int_module.weight_scales.copy_(weight_scales)

        if module.bias is not None:
            int_module.bias.copy_(module.bias)

        return int_module
    

class Random8bit(torch.nn.Module):
    _hadamard_modules = {} 
    _n = 256 

    def __init__(self, weight, rot_need, scale=None, bias=True, dtype=torch.float16, rot_size=None):
        super().__init__()
        self.in_features = weight.shape[-1]
        self.out_features = weight.shape[-2]
        self.dtype = dtype

        self.register_buffer('weight_scales', torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', torch.zeros((self.out_features, self.in_features),
                                                   dtype=torch.int8, requires_grad=False))
        
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.register_bias = None
            self.bias = None

        self.rot_need = rot_need
        if rot_size == None:
            rot_size = self._n
            
        if rot_size not in self.__class__._hadamard_modules:
            hadamard = Random(rot_size)
            self.__class__._hadamard_modules[rot_size] = hadamard
        self.hadamard = self.__class__._hadamard_modules[rot_size]

        if scale is not None:
            self.weight_scales.copy_(scale)

    def forward(self, x):
        if self.rot_need:
            x_rot = self.hadamard(x, pad_to_block=True)
        else:
            x_rot = x

        max_val = torch.max(torch.abs(x_rot), dim=-1)[0].unsqueeze(-1)
        
        scales_x = (max_val / 127.0).to(x_rot.dtype)
        scales_x = torch.clamp(scales_x, min=1e-5)

        quantized_x = torch.clamp(torch.round(x_rot / scales_x), -127, 127).to(torch.int8)

        # Flatten for _int_mm
        x_shape = quantized_x.shape
        x_flat = quantized_x.view(-1, self.in_features)
        
        weight_t = self.weight.t()
        
        # INT8 GEMM
        out_int32 = torch._int_mm(x_flat, weight_t)

        out_int32 = out_int32.view(*x_shape[:-1], self.out_features)
        
        out = out_int32.to(torch.float32) * scales_x.to(torch.float32) * self.weight_scales.t().to(torch.float32)

        if self.bias is not None:
            out = out + self.bias.to(torch.float32)
            
        return out.to(self.dtype)

    @staticmethod
    def from_float(module: torch.nn.Linear, rot_need, rot_size):
        if rot_size == None:
            rot_size = Random8bit._n
            
        if rot_size not in Random8bit._hadamard_modules:
            hadamard = Random(rot_size)
            Random8bit._hadamard_modules[rot_size] = hadamard
        hadamard = Random8bit._hadamard_modules[rot_size]

        if rot_need:
            weight_matrix = hadamard(module.weight, pad_to_block=True)
        else:
            weight_matrix = module.weight

        weight_scales = (torch.max(torch.abs(weight_matrix), dim=-1)[0].unsqueeze(-1) / 127.0).to(
            dtype=torch.float16, 
            device=weight_matrix.device
        )
        weight_scales = torch.clamp(weight_scales, min=1e-5)

        int_module = Random8bit(
            weight_matrix, 
            rot_need=rot_need, 
            scale=weight_scales,
            bias=module.bias is not None, 
            dtype=weight_matrix.dtype, 
            rot_size=rot_size
        )

        int_rounded_weight = torch.clamp(
            torch.round(weight_matrix / weight_scales.to(weight_matrix.device)), 
            -127, 127
        )
        
        int_module.weight.copy_(int_rounded_weight.to(torch.int8))
        int_module.weight_scales.copy_(weight_scales)

        if module.bias is not None:
            int_module.bias.copy_(module.bias)

        return int_module
    