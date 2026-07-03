#include <torch/extension.h>

torch::Tensor hadamard_rotate_cuda(
    const torch::Tensor& input,
    int64_t rot_size);

torch::Tensor rotated_linear_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t rot_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hadamard_rotate_cuda", &hadamard_rotate_cuda,
          "ConvRot regular Hadamard rotate (CUDA)");
    m.def("rotated_linear_cuda", &rotated_linear_cuda,
          "ConvRot rotated input + linear (CUDA)");
}
