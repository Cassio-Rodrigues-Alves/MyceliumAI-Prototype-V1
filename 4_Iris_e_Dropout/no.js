// no.js

let tf;
try {
    tf = require('@tensorflow/tfjs');
} catch (error) {
    console.warn("no.js: Pacote '@tensorflow/tfjs' não encontrado. Usando mock.");
    // Mock para compatibilidade
    tf = {
        layers: {
            dense: (config) => ({ apply: (t) => t, getConfig: () => config }),
            dropout: (config) => ({ apply: (t) => t, getConfig: () => config }),
            activation: (config) => ({ apply: (t) => t, getConfig: () => config }),
            add: () => ({ apply: (tensors) => tensors[0] }),
        },
        tidy: (fn) => fn(),
    };
}

/**
 * Representa um nó de processamento.
 */
class No {
    id;
    layers;

    /**
     * Cria uma instância de Nó.
     * @param {string} id - Identificador único do nó.
     * @param {object[]} layersConfig - Um array de configurações de camada.
     * Ex: [{type: 'dense', config: {units: 32}}, {type: 'dropout', config: {rate: 0.5}}]
     */
    constructor(id, layersConfig) {
        if (!id || !Array.isArray(layersConfig) || layersConfig.length === 0) {
            throw new Error("Nó: 'id' e um array 'layersConfig' são obrigatórios.");
        }
        this.id = id;
        this.layers = [];

        // Cria as camadas do TensorFlow a partir da configuração
        for (const layerInfo of layersConfig) {
            if (tf.layers[layerInfo.type]) {
                this.layers.push(tf.layers[layerInfo.type](layerInfo.config));
            } else {
                throw new Error(`Tipo de camada desconhecido no Nó '${id}': ${layerInfo.type}`);
            }
        }
    }

    /**
     * O "forward pass" para o nó.
     * @param {tf.Tensor[]} listaTensoresEntrada - Um array de tensores.
     * @returns {tf.Tensor} O tensor de saída do nó.
     */
    call(listaTensoresEntrada) {
        if (!Array.isArray(listaTensoresEntrada) || listaTensoresEntrada.length === 0) {
            throw new Error(`Nó '${this.id}': A entrada deve ser um array de tensores não vazio.`);
        }

        return tf.tidy(() => {
            // Primeiro, agrega as entradas se houver mais de uma
            let tensorAtual = listaTensoresEntrada.length === 1
                ? listaTensoresEntrada[0]
                : tf.layers.add().apply(listaTensoresEntrada);

            // Aplica cada camada em sequência
            for (const layer of this.layers) {
                tensorAtual = layer.apply(tensorAtual);
            }

            return tensorAtual;
        });
    }
}

module.exports = No;