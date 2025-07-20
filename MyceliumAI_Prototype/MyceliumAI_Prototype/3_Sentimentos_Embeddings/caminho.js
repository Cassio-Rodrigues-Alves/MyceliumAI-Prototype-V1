// caminho.js

let tf;
try {
    tf = require('@tensorflow/tfjs');
} catch (error) {
    console.warn("caminho.js: Pacote '@tensorflow/tfjs' não encontrado. Usando mock.");
    // Mock
    tf = {
        layers: {
            Layer: class { constructor(config) {} build(inputShape){} call(inputs){ return Array.isArray(inputs) ? inputs[0] : inputs; } getConfig(){return {};} },
        },
        initializers: {
            zeros: () => ({ className: 'Zeros' }),
            randomNormal: (config) => ({ className: 'RandomNormal', config }),
        },
        tidy: (fn) => fn(),
    };
}


class Caminho extends tf.layers.Layer {
    /** @type {string} */
    nome;
    /** @type {string} */
    origem;
    /** @type {string} */
    destino;
    /** @type {tf.Tensor} */
    kernel;
    /** @type {tf.Tensor} */
    bias;

    /**
     * @param {object} config Configuração da camada.
     * @param {string} config.nome Nome do caminho.
     * @param {string} config.origem Nó de origem.
     * @param {string} config.destino Nó de destino.
     */
    constructor(config) {
        super(config);
        if (!config || !config.nome || !config.origem || !config.destino) {
            throw new Error("Caminho (Layer): objeto 'config' com 'nome', 'origem' e 'destino' é obrigatório.");
        }
        this.nome = config.nome;
        this.origem = config.origem;
        this.destino = config.destino;
    }

    /**
     * Método build: Cria os pesos treináveis da camada.
     * @param {tf.Shape | tf.Shape[]} inputShape A forma da entrada. Ex: [null, 8].
     */
    build(inputShape) {
        const featureDim = inputShape[inputShape.length - 1];
        const weightShape = [featureDim];

        this.kernel = this.addWeight(
            `peso_${this.name || this.nome}`,
            weightShape,                       // Forma do peso
            'float32',                         // Tipo de dado
            tf.initializers.randomNormal({ stddev: 0.02 }), // Inicializador
            undefined,                         // Regularizador
            true                               // Treinável? Sim.
        );

        this.bias = this.addWeight(
            `bias_${this.name || this.nome}`,
            weightShape,
            'float32',
            tf.initializers.zeros(),           // Bias geralmente começa em zero
            undefined,
            true
        );
        super.build(inputShape);
    }

    /**
     * Método call: Define a lógica do forward pass.
     * @param {tf.Tensor | tf.Tensor[]} inputs O tensor de entrada.
     * @returns {tf.Tensor} O tensor de saída.
     */
    call(inputs) {
        return tf.tidy(() => {
            const inputTensor = Array.isArray(inputs) ? inputs[0] : inputs;
            const multiplicado = tf.mul(inputTensor, this.kernel.read());
            const adicionado = tf.add(multiplicado, this.bias.read());
            return adicionado;
        });
    }

    /**
     * Método getConfig: Necessário para salvar/carregar o modelo.
     */
    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            nome: this.nome,
            origem: this.origem,
            destino: this.destino,
        });
        return config;
    }

    static get className() {
        return 'Caminho';
    }
}

module.exports = Caminho;