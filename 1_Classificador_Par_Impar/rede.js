// rede.js
// Versão para a Fase 1: Classificador Par ou Ímpar.
// Contém o método buildModel para uma arquitetura linear simples.

let tf;
try {
    tf = require('@tensorflow/tfjs');
} catch (error) {
    console.warn("rede.js: Pacote '@tensorflow/tfjs' não encontrado. Usando mock.");
    // Mock para permitir que o script seja carregado sem o TensorFlow
    tf = {
        input: () => ({}),
        model: () => ({ compile: () => {}, fit: async () => {}, predict: () => {}, summary: () => console.log("Modelo mockado.") }),
        tensor2d: (d) => ({}),
        train: { adam: ()=>{} },
        layers: { add: ()=>({ apply: ()=>{} }) },
    };
}

// Importa os blocos de construção da sua arquitetura
const No = require('./no.js');
const Caminho = require('./caminho.js');

/**
 * A classe Rede atua como uma fábrica para construir um modelo TensorFlow.js
 * a partir de uma topologia de Nós e Caminhos.
 */
class Rede {
    /** @type {Map<string, No>} */
    nos;
    /** @type {Map<string, Caminho>} */
    caminhos;

    constructor() {
        this.nos = new Map();
        this.caminhos = new Map();
    }

    adicionarNo(no) {
        this.nos.set(no.id, no);
    }
    
    adicionarCaminho(caminho) {
        this.caminhos.set(caminho.nome, caminho);
    }

    /**
     * Constrói e retorna um tf.Model para o problema de PAR OU ÍMPAR.
     * Arquitetura: Entrada -> Oculto -> Saída
     * @param {number[]} inputShape - A forma da entrada do modelo, ex: [1].
     * @returns {tf.Model} O modelo TensorFlow.js pronto para compilar e treinar.
     */
    buildModel(inputShape) {
        if (!this.nos.has("Entrada") || !this.nos.has("Oculto") || !this.nos.has("Saida") ||
            !this.caminhos.has("C-Entrada_Oculto") || !this.caminhos.has("C-Oculto_Saida")) {
            throw new Error("Para construir o modelo par/ímpar, os nós 'Entrada', 'Oculto', 'Saida' e os caminhos correspondentes são necessários.");
        }

        // 1. Define a camada de entrada do modelo
        const inputTensor = tf.input({ shape: inputShape });

        // 2. Fluxo: Entrada -> Nó de Entrada
        const noEntrada = this.nos.get("Entrada");
        const saidaNoEntrada = noEntrada.call([inputTensor]);

        // 3. Fluxo: Saída do Nó de Entrada -> Caminho para o Nó Oculto
        const caminho_E_O = this.caminhos.get("C-Entrada_Oculto");
        const entradaNoOculto = caminho_E_O.apply(saidaNoEntrada);
        
        // 4. Fluxo: Entrada para Nó Oculto -> Nó Oculto -> Saída do Nó Oculto
        const noOculto = this.nos.get("Oculto");
        const saidaNoOculto = noOculto.call([entradaNoOculto]);

        // 5. Fluxo: Saída do Nó Oculto -> Caminho para o Nó de Saída
        const caminho_O_S = this.caminhos.get("C-Oculto_Saida");
        const entradaNoSaida = caminho_O_S.apply(saidaNoOculto);

        // 6. Fluxo: Entrada para Nó de Saída -> Nó de Saída -> Saída Final do Modelo
        const noSaida = this.nos.get("Saida");
        const saidaFinal = noSaida.call([entradaNoSaida]);
        
        // 7. Cria e retorna o modelo final
        return tf.model({
            inputs: inputTensor,
            outputs: saidaFinal
        });
    }
}

// Exporta a classe para que outros arquivos (como treinar_par_impar.js) possam usá-la.
module.exports = Rede;


// --- Teste de Construção do Modelo ---
// O código abaixo só é executado se você rodar "node rede.js" diretamente no terminal.
// Ele serve para verificar se a construção do modelo está funcionando corretamente.
async function testarConstrucaoDoModelo() {
    if (require.main !== module) return;

    console.log("### Teste de Construção: rede.js (Par/Ímpar) ###");

    // 1. Instancia a Rede
    const rede = new Rede();

    // 2. Define as dimensões para o teste
    const dimOculta = 8;
    const dimSaida = 1;

    // 3. Adiciona os Nós necessários para o buildModel
    rede.adicionarNo(new No("Entrada", { units: dimOculta, ativacaoCamada: 'relu' }));
    rede.adicionarNo(new No("Oculto",  { units: dimOculta, ativacaoCamada: 'relu' }));
    rede.adicionarNo(new No("Saida",   { units: dimSaida, ativacaoFinal: 'sigmoid' }));

    // 4. Adiciona os Caminhos necessários
    rede.adicionarCaminho(new Caminho({ nome: "C-Entrada_Oculto", origem: "Entrada", destino: "Oculto" }));
    rede.adicionarCaminho(new Caminho({ nome: "C-Oculto_Saida", origem: "Oculto", destino: "Saida" }));

    // 5. Constrói o modelo e exibe o resumo da arquitetura
    console.log("\nConstruindo modelo para teste de arquitetura...");
    const inputShape = [1]; // O modelo espera receber 1 número (na versão binária, seria [16])
    const model = rede.buildModel(inputShape);
    model.summary();

    console.log("\n✅ Construção do modelo para o classificador Par/Ímpar bem-sucedida.");
}

testarConstrucaoDoModelo().catch(console.error);