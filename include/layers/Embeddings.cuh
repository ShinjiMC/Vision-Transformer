#pragma once

#include "layers/Layer.cuh"
#include "layers/PatchEmbedding.cuh"
#include <memory>

/**
 * @class Embeddings
 * @brief Encapsula toda la lógica de preparación de la entrada para un ViT.
 *
 * Realiza tres operaciones clave:
 * 1. Utiliza una capa PatchEmbedding para convertir imágenes en embeddings de parches.
 * 2. Pre-añade un token de clasificación [CLS] entrenable a la secuencia de parches.
 * 3. Suma una codificación posicional entrenable a la secuencia combinada.
 */
class Embeddings : public Layer
{
public:
  Embeddings(size_t image_height, size_t image_width, size_t patch_size, size_t in_channels, size_t embedding_dim);

  Tensor forward(const Tensor &input, bool isTraining) override;

  Tensor backward(const Tensor &outputGradient) override;

  std::vector<Tensor *> getParameters() override;
  std::vector<Tensor *> getGradients() override;

  std::string getName() const override { return "Embeddings"; }

private:
  // Capa contenida para el parcheo
  std::unique_ptr<PatchEmbedding> patcher;

  // Parámetros entrenables propios de esta capa
  Tensor clsToken;           // Forma {1, 1, embedding_dim}
  Tensor positionalEncoding; // Forma {1, num_patches + 1, embedding_dim}

  // Gradientes correspondientes
  Tensor clsTokenGradient;
  Tensor positionalEncodingGradient;

  // Dimensiones guardadas para conveniencia
  size_t num_patches;
  size_t embedding_dim;
};