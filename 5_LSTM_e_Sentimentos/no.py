# no.py (Versão Final e Robusta)
import tensorflow as tf
from tensorflow.keras import layers

class No(layers.Layer):
    def __init__(self, name, layers_config, **kwargs):
        super(No, self).__init__(name=name, **kwargs)
        self.layers_config = layers_config
        self.internal_layers = []

        # Constrói as camadas internas a partir da "receita" (layers_config)
        for i, layer_info in enumerate(self.layers_config):
            if hasattr(layers, layer_info['type']):
                # Gera um nome único e seguro para cada camada interna
                layer_name = f"{name}_internal_{i}_{layer_info['type']}"
                self.internal_layers.append(
                    getattr(layers, layer_info['type'])(name=layer_name, **layer_info['config'])
                )
            else:
                raise ValueError(f"Tipo de camada desconhecido: {layer_info['type']}")

    def call(self, inputs):
        # O TensorFlow Keras já lida com a lista de tensores quando a camada é chamada
        # com múltiplos inputs, como a camada Add.
        tensor_atual = inputs

        # Aplica as camadas internas em sequência
        for layer in self.internal_layers:
            tensor_atual = layer(tensor_atual)

        return tensor_atual

    # O get_config é essencial para que o Keras saiba como salvar e carregar o modelo
    def get_config(self):
        config = super(No, self).get_config()
        config.update({"layers_config": self.layers_config})
        return config