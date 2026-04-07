#include "Tensor.h"

// Seccion 3.1: Constructor principal
Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& values) {
    // Se valida las dimesiones del tensor
    if (shape.empty() || shape.size()>3)
        throw std::invalid_argument("Dimesiones invalidas. EL tensor debe ser de 1D, 2D O 3D");

    this->shape = shape;
    elementos = n_elementos(shape);

    // Manejo de errores
    if (values.size() != elementos)
        throw std::invalid_argument("El numero de elemento no coicide con las dimensiones de la matriz");

    // Se almacena los datos de manera dinamica
    data = new double[elementos];
    for (size_t i = 0; i < elementos; i++)
        data[i] = values[i];
}

// Función encargada de calcular el numero total de elementos
size_t Tensor::n_elementos(const std::vector<size_t>& shape) {
    size_t size = 1;
    for (size_t dim : shape)
        size *= dim;

    return size;
}

// Seccion 3.2: Creación de tensores pre-definidos
Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    // Se valida las dimensiones del tensor
    if (shape.empty() || shape.size()>3)
        throw std::invalid_argument("zeros: Dimesiones invalidas. EL tensor debe ser de 1D, 2D O 3D");

    // Se crea los valores del tensor zeros
    size_t total = n_elementos(shape);
    std::vector<double> values(total, 0.0);

    return Tensor(shape, values);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    // Se valida las dimensiones del tensor
    if (shape.empty() || shape.size()>3)
        throw std::invalid_argument("ones: Dimesiones invalidas. EL tensor debe ser de 1D, 2D O 3D");

    // Se crea los valores del tensor ones
    size_t total = n_elementos(shape);
    std::vector<double> values(total, 1.0);

    return Tensor(shape, values);
}

Tensor Tensor::random(const std::vector<size_t>& shape, double min, double max) {
    // Se valida las dimensiones del tensor
    if (shape.empty() || shape.size() > 3)
        throw std::invalid_argument("random: Dimesiones invalidas. EL tensor debe ser de 1D, 2D O 3D");

    // Se crea los valores random para el tensor
    size_t total = n_elementos(shape);
    std::vector<double> values(total);
    for (size_t i = 0; i < total; i++) {
        double r = static_cast<double>(rand()) / RAND_MAX;
        values[i] = min + r * (max - min);
    }

    return Tensor(shape, values);
}

Tensor Tensor::arange(double start, double end) {
    // Se valida de concordancia de limites
    if (start >= end)
        throw std::invalid_argument("arange: Limites invalidos. Start debe ser menor que End");

    // Se crea los valores secuanciales para el tensor
    std::vector<double> values;
    for (double i = start; i < end; i++) {
        values.push_back(i);
    }

    return Tensor({values.size()}, values);
}

// Seccion 4: Gestión de Memoria y Ciclo de Vida
Tensor::Tensor(const Tensor& other) {
    shape = other.shape;
    elementos = other.elementos;

    data = new double[elementos];
    for (size_t i = 0; i < elementos; i++)
        data[i] = other.data[i];
}

Tensor& Tensor::operator=(const Tensor& other) {
    // Valida si no es el mismo tensor
    if (this == &other) return *this;

    // Se libera la memoria
    delete[] data;

    shape = other.shape;
    elementos = other.elementos;

    // Se copia los datos en la memoria liberada
    data = new double[elementos];
    for (size_t i = 0; i < elementos; i++) {
        data[i] = other.data[i];
    }

    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept {
    shape = std::move(other.shape);
    elementos = other.elementos;
    data = other.data;

    // Dejar el antiguo tensor en estado nulo
    other.data = nullptr;
    other.elementos = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    // Valida si no es el mismo tensor
    if (this == &other) return *this;

    // Se libera la memoria
    delete[] data;

    // Se copia los datos en la memoria liberada
    shape = std::move(other.shape);
    elementos = other.elementos;
    data = other.data;

    // Dejar el antiguo tensor en estado nulo
    other.data = nullptr;
    other.elementos = 0;

    return *this;
}

Tensor::~Tensor() {
    // Se libera la memoria usada por el objeto dinamico
    delete[] data;
    data = nullptr;
}

// Seccion 5.2: Metodo de aplicación en clase Tensor

Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}

// Seccion 6: Sobrecarga de operadores

Tensor Tensor::operator+(const Tensor& other) const {
    // Validar que tengan la misma forma
    if (shape != other.shape)
        throw std::invalid_argument("Suma: Dimensiones incompatibles");

    std::vector<double> values(elementos);

    // Suma de elemento a elemento
    for (size_t i = 0; i < elementos; i++)
        values[i] = data[i] + other.data[i];

    return Tensor(shape, values);
}

Tensor Tensor::operator-(const Tensor& other) const {
    // Validar que tengan la misma forma
    if (shape != other.shape)
        throw std::invalid_argument("Resta: Dimensiones incompatibles");

    std::vector<double> values(elementos);

    // Resta de elemento a elemento
    for (size_t i = 0; i < elementos; i++) {
        values[i] = data[i] - other.data[i];
    }

    return Tensor(shape, values);
}

Tensor Tensor::operator*(const Tensor& other) const {
    // Validar que tengan la misma forma
    if (shape != other.shape)
        throw std::invalid_argument("MUltiplicacion: dimensiones incompatibles");

    std::vector<double> values(elementos);

    // Multiplicación elemento a elemento
    for (size_t i = 0; i < elementos; i++) {
        values[i] = data[i] * other.data[i];
    }

    return Tensor(shape, values);
}

Tensor Tensor::operator*(double scalar) const {
    std::vector<double> values(elementos);

    // Multiplicación elemento a escalar
    for (size_t i = 0; i < elementos; i++)
        values[i] = data[i] * scalar;

    return Tensor(shape, values);
}

// Seccion 7: Modificación de dimensiones

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    if (new_shape.empty() || new_shape.size() > 3)
        throw std::invalid_argument("view: Dimensiones invalidas, maximo 3D");

    size_t new_size = n_elementos(new_shape);

    if (new_size != elementos)
        throw std::invalid_argument("El numero de elemento no coicide con las dimensiones de la matriz");

    Tensor result(*this);
    result.shape = new_shape;

    return result;
}

Tensor Tensor::unsqueeze(size_t dim) const {
    // Validar dimensión y su limite
    if (dim > shape.size())
        throw std::invalid_argument("unsqueeze: dimension invalida");
    if (shape.size() + 1 > 3)
        throw std::invalid_argument("unsqueeze: maximo 3 dimensiones");

    // Crear y insertar la dimension extra
    std::vector<size_t> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1);

    Tensor result(*this);
    result.shape = new_shape;

    return result;
}

// Seccion 8: Concatenacion

Tensor Tensor::concat(const std::vector<Tensor>& tensores, size_t dim) {

    // validacion de dimensiones
    if (tensores.empty())
        throw std::invalid_argument("concat: lista vacia");

    const Tensor& base = tensores[0];

    if (dim >= base.shape.size())
        throw std::invalid_argument("concat: dimension invalida");

    // crear un objeto que alamnece el primer vector
    std::vector<size_t> nueva_forma = base.shape;
    size_t suma_dim = 0;

    // suma de dimensiones
    for (const Tensor& t : tensores) {
        if (t.shape.size() != base.shape.size())
            throw std::invalid_argument("concat: dimensiones incompatibles");
        for (size_t i = 0; i < t.shape.size(); i++) {
            if (i != dim && t.shape[i] != base.shape[i])
                throw std::invalid_argument("concat: dimensiones incompatibles");
        }
        suma_dim += t.shape[dim];
    }

    // calcular los nuevos parametros
    nueva_forma[dim] = suma_dim;
    size_t total = n_elementos(nueva_forma);

    std::vector<double> datos(total);
    size_t pos = 0;

    // Caso simple: concatenación en la primera dimensión
    if (dim == 0) {
        for (const Tensor& t : tensores) {

            // Copiar todos los datos del tensor
            std::copy(t.data, t.data + t.elementos, datos.begin() + pos);

            // Avanzar posición
            pos += t.elementos;
        }

    } else {
        // Caso general: concatenación en otras dimensiones

        // definir las filas como bloques de datos
        size_t bloque_interno = 1;
        for (size_t i = dim + 1; i < base.shape.size(); i++)
            bloque_interno *= base.shape[i];

        // hallar el número de bloques externos
        size_t bloque_externo = 1;
        for (size_t i = 0; i < dim; i++)
            bloque_externo *= base.shape[i];

        size_t escritura = 0;

        // Recorrer bloques
        for (size_t o = 0; o < bloque_externo; o++) {
            for (const Tensor& t : tensores) {
                size_t tam_bloque = t.shape[dim] * bloque_interno;
                size_t inicio = o * tam_bloque;

                // Copiar bloque completo
                std::copy(
                    t.data + inicio,
                    t.data + inicio + tam_bloque,
                    datos.begin() + escritura
                );

                escritura += tam_bloque;
            }
        }
    }

    // Crear el nuevo tensor usando move
    return Tensor(nueva_forma, std::move(datos));
}

// Seccion 9: red neuronal

double& Tensor::operator()(size_t i, size_t j) {
    return data[i * shape[1] + j];
}

double Tensor::operator()(size_t i, size_t j) const {
    return data[i * shape[1] + j];
}