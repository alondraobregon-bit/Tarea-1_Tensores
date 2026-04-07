#include <iostream>
#include "Tensor.h"
#include "TensorTransform.h"
using namespace std;

// seccion 9: funciones amigas permitidas
Tensor dot(const Tensor& a, const Tensor& b) {
    // validacion de tamaño de dimensiones
    if (a.elementos != b.elementos)
        throw std::invalid_argument("Dimensiones incompatibles para multiplicación punto de tensores");

    double resultado = 0.0;

    // calculo del producto punto
    for (size_t i = 0; i < a.elementos; i++) {
        resultado += a.data[i] * b.data[i];
    }
    return Tensor({1}, {resultado});
}

Tensor matmul(const Tensor& a, const Tensor& b) {

    // Validar que ambos tensores sean tensores 2D
    if (a.shape.size() != 2 || b.shape.size() != 2)
        throw std::invalid_argument("Solo multiplicacion cruz de tensores 2D");

    size_t row_a = a.shape[0];
    size_t col_a = a.shape[1];
    size_t col_b = b.shape[1];

    if (col_a != b.shape[0])
        throw std::invalid_argument("Multiplicacion cruz de tensores con dimensiones incompatibles");

    std::vector<double> valores(row_a * col_b, 0.0);

    // Multiplicación matricial
    for (size_t i = 0; i < row_a; i++) {
        for (size_t j = 0; j < col_b; j++) {
            for (size_t k = 0; k < col_a; k++) {
                valores[i * col_b + j] +=
                    a.data[i * col_a + k] *
                    b.data[k * col_b + j];
            }
        }
    }

    return Tensor({row_a, col_b}, std::move(valores));
}

// seccion 10: implementacion de red neuronal
int main() {

    Tensor tensor_input = Tensor::random({1000, 20, 20}, 0.0, 1.0);
    Tensor tensor_A = tensor_input.view({1000, 400});
    Tensor tensor_W1 = Tensor::random({400, 100}, -1.0, 1.0);
    Tensor tensor_b1 = Tensor::random({1, 100}, -1.0, 1.0);
    Tensor tensor_Z1 = matmul(tensor_A, tensor_W1);
    Tensor tensor_Z1_bias({1000, 100}, std::vector<double>(1000 * 100));

    for (size_t i = 0; i < 1000; i++) {
        for (size_t j = 0; j < 100; j++) {
            tensor_Z1_bias(i, j) = tensor_Z1(i, j) + tensor_b1(0, j);
        }
    }

    ReLU relu;
    Tensor tensor_A1 = tensor_Z1_bias.apply(relu);
    Tensor tensor_W2 = Tensor::random({100, 10}, -1.0, 1.0);
    Tensor tensor_b2 = Tensor::random({1, 10}, -1.0, 1.0);
    Tensor tensor_Z2 = matmul(tensor_A1, tensor_W2);
    Tensor tensor_Z2_bias({1000, 10}, std::vector<double>(1000 * 10));

    for (size_t i = 0; i < 1000; i++) {
        for (size_t j = 0; j < 10; j++) {
            tensor_Z2_bias(i, j) = tensor_Z2(i, j) + tensor_b2(0, j);
        }
    }

    Sigmoid sigmoid;
    Tensor tensor_output = tensor_Z2_bias.apply(sigmoid);

    std::cout << "Red neuronal ejecutada" << std::endl;

    return 0;
}