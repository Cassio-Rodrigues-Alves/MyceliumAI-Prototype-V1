// no.js

let tf;
try {
    tf = require('@tensorflow/tfjs');
} catch (error) {
    console.warn("no.js: Pacote '@tensorflow/tfjs' não encontrado. Usando mock.");
    tf = {
        layers: {
            dense: ({ units }) => ({ apply: (tensor) => ({ shape: [1, units] }), getWeights: () => [] }),
            add: () => ({ apply: (tensorList) => tensorList[0] }),
            activation: () => ({ apply: (tensor) => tensor }),
        },
        tidy: (fn) => fn(),
    };
}

/**
 * Representa um nó de processamento diferenciável na rede micelial.
 */
class No {
    /** @type {string} */
    id;
    /** @type {tf.layers.Layer} */
    camadaInterna;
    /** @type {tf.layers.Layer} */
    camadaAtivacaoFinal;

    /**
     * Cria uma instância de Nó (diferenciável).
     * @param {string} id - Identificador único do nó.
     * @param {object} opcoes - Opções para configurar a camada interna do nó.
     * @param {number} opcoes.units - O número de neurônios da camada interna.
     * @param {string} [opcoes.ativacaoCamada='relu'] - Função de ativação para a camada densa.
     * @param {string} [opcoes.ativacaoFinal='tanh'] - Função de ativação final para a saída do nó.
     */
    constructor(id, opcoes) {
        if (!id || !opcoes || !opcoes.units) {
            throw new Error("Nó: 'id' e um objeto de opções com 'units' são obrigatórios.");
        }
        this.id = id;

        this.camadaInterna = tf.layers.dense({
            units: opcoes.units,
            activation: opcoes.ativacaoCamada || 'relu',
        });

        const ativacao = opcoes.ativacaoFinal || 'tanh';
        this.camadaAtivacaoFinal = tf.layers.activation({
            activation: ativacao,
            name: `${id}_activation_output` 
        });
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
            const tensorAgregado = listaTensoresEntrada.length === 1
                ? listaTensoresEntrada[0]
                : tf.layers.add().apply(listaTensoresEntrada);

            const tensorModificado = this.camadaInterna.apply(tensorAgregado);
            const tensorSaida = this.camadaAtivacaoFinal.apply(tensorModificado);

            return tensorSaida;
        });
    }

    /**
     * Retorna uma lista de todos os pesos treináveis deste nó.
     * @returns {tf.Variable[]}
     */
    getWeights() {
        return this.camadaInterna.getWeights();
    }
}

module.exports = No;

if (require.main === module) {
    console.log("Teste isolado de 'no.js' desativado.");
}