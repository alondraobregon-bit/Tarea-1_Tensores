#ifndef PROYECTP_TENSOR_H
#define PROYECTP_TENSOR_H

#include "TensorTransform.h"
#include <vector>
#include <stdexcept>
#include <ctime>

class Tensor {
private:
    std::vector<size_t> shape;
    size_t elementos = 1;
    double* data = nullptr;

    static size_t n_elementos(const std::vector<size_t>& shape);
public:
    // Constructores de la clase
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor(const std::vector<size_t>& shape, double* data_ptr, size_t total);

    // Tensores pre-definidos
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor random(const std::vector<size_t>& shape, double min, double max);
    static Tensor arange(double start, double end);

    // Metodo apply
    Tensor apply(const TensorTransform& transform) const;

    // Clases amigas, para el acceso a atributos privados
    friend class ReLU;
    friend class Sigmoid;

    //  Sobrecarga de operadores
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    // Metodos de redimensionamiento
    Tensor view(const std::vector<size_t>& new_shape) const;
    Tensor unsqueeze(size_t dim) const;

    static Tensor concat(const std::vector<Tensor>& tensores, size_t dim);

    //

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    // sobrecarga de operadores para la escritura y lectura del atributo data
    double& operator()(size_t i, size_t j);
    double operator()(size_t i, size_t j) const;

    ~Tensor();
};

#endif //PROYECTP_TENSOR_H
