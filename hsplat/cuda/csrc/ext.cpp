#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &hsplat::add_tensor, "Add two tensors");
    m.def("cgh_gaussians_naive", &hsplat::cgh_gaussians_naive_tensor, "Naively compute CGH from Gaussians (summation model)");
}
