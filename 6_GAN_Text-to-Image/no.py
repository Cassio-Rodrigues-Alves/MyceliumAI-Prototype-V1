# no.py
import tensorflow as tf
from tensorflow.keras import layers

class No(layers.Layer):
    """
    Representa um nó de processamento customizável na arquitetura MyceliumAI.
    Funciona como um container flexível que pode abrigar múltiplas camadas do Keras,
    processando tensores de entrada em sequência e agregando múltiplas entradas
    através de uma camada Add interna.
    """
    def __init__(self, name, layers_config, **kwargs):
        super(No, self).__init__(name=name, **kwargs)
        self.layers_config = layers_config
        self.internal_layers = []
        
        # A camada Add é criada uma única vez no construtor para garantir a estabilidade do grafo.
        self.add_layer = layers.Add(name=f"{self.name}_add")

        # Constrói as camadas internas a partir da "receita" (layers_config)
        for i, layer_info in enumerate(self.layers_config):
            if hasattr(layers, layer_info['type']):
                # Gera um nome único e seguro para cada camada interna para evitar conflitos
                layer_name = f"{name}_internal_{i}_{layer_info['type']}"
                self.internal_layers.append(
                    getattr(layers, layer_info['type'])(name=layer_name, **layer_info['config'])
                )
            else:
                raise ValueError(f"Tipo de camada desconhecido no Nó '{name}': {layer_info['type']}")

    def call(self, inputs):
        """
        Define o "forward pass" do Nó.
        """
        tensor_atual = inputs
        
        # Se a entrada for uma lista de tensores, primeiro os agrega com a camada Add.
        if isinstance(inputs, list) and len(inputs) > 1:
            tensor_atual = self.add_layer(inputs)
        
        # Aplica as camadas internas em sequência no tensor.
        for layer in self.internal_layers:
            tensor_atual = layer(tensor_atual)
            
        return tensor_atual

    # O método get_config é essencial para que o Keras saiba como salvar e carregar o modelo.
    def get_config(self):
        config = super(No, self).get_config()
        config.update({"layers_config": self.layers_config})
        return config