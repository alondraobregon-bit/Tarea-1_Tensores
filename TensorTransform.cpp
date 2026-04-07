#include "TensorTransform.h"
#include "Tensor.h"


Tensor ReLU::apply(const Tensor& t) const {
    std::vector<double> values(t.elementos);

    for (size_t i = 0; i < t.elementos; i++) {
        double number = t.data[i];
        values[i] = (number > 0) ? number : 0.0;
    }
    return Tensor(t.shape, values);
}

Tensor Sigmoid::apply(const Tensor& t) const {
    std::vector<double> values(t.elementos);

    for (size_t i = 0; i < t.elementos; i++) {
        double x = t.data[i];
        values[i] = 1.0 / (1.0 + std::exp(-x));
    }

    return Tensor(t.shape, values);
}