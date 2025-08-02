#include "model/Trainer.cuh"
#include "utils/ModelUtils.hpp"
#include "utils/CudaUtils.cuh"
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
// #include "utils/stb_image.h"
#include "external/headers/stb/stb_image.h"

Tensor load_image_with_stb(const std::string &path)
{
    int width, height, channels;

    unsigned char *img_data = stbi_load(path.c_str(), &width, &height, &channels, 1); // 1 = grayscale
    if (!img_data)
    {
        throw std::runtime_error("No se pudo cargar la imagen: " + path);
    }

    const int target_size = 28;
    Tensor input({1, 1, target_size, target_size});

    float scale_x = static_cast<float>(width) / target_size;
    float scale_y = static_cast<float>(height) / target_size;

    for (int y = 0; y < target_size; ++y)
    {
        for (int x = 0; x < target_size; ++x)
        {
            int src_x = static_cast<int>(x * scale_x);
            int src_y = static_cast<int>(y * scale_y);
            int idx = src_y * width + src_x;

            float pixel = static_cast<float>(img_data[idx]) / 255.0f;
            pixel = (pixel - 0.1307f) / 0.3081f; // Normalización igual que entrenamiento

            input(0, 0, y, x) = pixel;
        }
    }

    stbi_image_free(img_data);
    return input;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Uso: " << argv[0] << " <ruta_a_imagen>" << std::endl;
        return 1;
    }
    try
    {
        const std::string model_name = "vit_mnist_test";
        const std::string weights_path = model_name + ".weights";
        const std::string config_path = model_name + ".json";

        // --- 1. Cargar conifguracion del ViT ---
        std::cout << "Cargando configuración desde: " << config_path << std::endl;
        ViTConfig loaded_config = ModelUtils::load_config("models/" + config_path);

        // --- 2. Crear y Cargar los pesos en el modelo ---
        std::cout << "Construyendo modelo con la arquitectura cargada..." << std::endl;
        VisionTransformer model(loaded_config);

        std::cout << "Cargando pesos desde: " << weights_path << std::endl;
        ModelUtils::load_weights(model, "models/" + weights_path);
        std::cout << "Pesos cargados correctamente.\n";

        // --- 3. Cargar imagen y convertir a tensor ---
        std::string image_path = argv[1];
        Tensor input = load_image_with_stb(image_path);

        // --- 4. Predicción ---
        Tensor logits = model.forward(input, false);
        Tensor probabilities = softmax_cuda(logits);

        int predicted_class = -1;
        float max_prob = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < probabilities.getShape()[1]; ++j)
        {
            if (probabilities(0, j) > max_prob)
            {
                max_prob = probabilities(0, j);
                predicted_class = j;
            }
        }

        std::cout << "\nResultado:\n";
        std::cout << "Imagen: " << image_path << "\n";
        std::cout << "Clase predicha: " << predicted_class << "\n";
        std::cout << "Confianza: " << max_prob * 100.0f << "%\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR CRÍTICO: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
