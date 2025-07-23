# caminho.py
import tensorflow as tf
from tensorflow.keras import layers

class Caminho(layers.Layer):
    """
    Representa uma conexão treinável (aresta) na arquitetura de grafo da MyceliumAI.
    Aplica uma transformação linear (y = x * w + b) elemento a elemento,
    permitindo que o "caminho" aprenda a escalar e deslocar as features que passam por ele.
    """
    def __init__(self, name, **kwargs):
        super(Caminho, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        """
        Cria os pesos (kernel) e o bias da camada.
        É chamado automaticamente pelo Keras na primeira vez que a camada é usada.
        """
        feature_dim = input_shape[-1]
        
        self.kernel = self.add_weight(
            shape=(feature_dim,),
            initializer="glorot_uniform", # Um bom inicializador padrão para os pesos
            trainable=True,
            name=f"{self.name}_kernel"
        )
        self.bias = self.add_weight(
            shape=(feature_dim,),
            initializer="zeros", # Um bom inicializador padrão para o bias
            trainable=True,
            name=f"{self.name}_bias"
        )
        super(Caminho, self).build(input_shape)

    def call(self, inputs):
        """
        Define o "forward pass" do Caminho.
        """
        # Realiza a operação y = x * w + b
        return (inputs * self.kernel) + self.bias

    # O get_config é simples, pois não temos parâmetros extras no construtor para salvar.
    def get_config(self):
        config = super(Caminho, self).get_config()
        return config