#ifndef PROYECTP_TENSORTRANSFORM_H
#define PROYECTP_TENSORTRANSFORM_H

#include <vector>
#include <cmath>

class Tensor;

// Seccion 5.1: Interfaz de Transformación

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};


#endif //PROYECTP_TENSORTRANSFORM_H
