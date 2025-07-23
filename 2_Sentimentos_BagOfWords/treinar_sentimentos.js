// treinar_sentimentos_micelial.js
// Nesse arquivo, será definido o processo de treinamento e a forma da rede

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

// Importa nossas classes customizadas
const No = require('./no.js');
const Caminho = require('./caminho.js');

// A classe Rede
class Rede {
    constructor() {
        this.nos = new Map();
        this.caminhos = new Map();
    }
    adicionarNo(no) { this.nos.set(no.id, no); }
    adicionarCaminho(caminho) { this.caminhos.set(caminho.nome, caminho); }
    buildModel(inputShape) {
        const inputTensor = tf.input({ shape: inputShape });
        const noEntrada = this.nos.get("Entrada");
        const saidaNoEntrada = noEntrada.call([inputTensor]);
        const caminhoParaOcultoA = this.caminhos.get("C-E_A").apply(saidaNoEntrada);
        const caminhoParaOcultoB = this.caminhos.get("C-E_B").apply(saidaNoEntrada);
        const noOcultoA = this.nos.get("Oculto-A");
        const saidaOcultoA = noOcultoA.call([caminhoParaOcultoA]);
        const noOcultoB = this.nos.get("Oculto-B");
        const saidaOcultoB = noOcultoB.call([caminhoParaOcultoB]);
        const noAgregador = this.nos.get("Agregador");
        const saidaAgregador = noAgregador.call([saidaOcultoA, saidaOcultoB]);
        const noSaida = this.nos.get("Saida");
        const saidaFinal = noSaida.call([saidaAgregador]);
        return tf.model({ inputs: inputTensor, outputs: saidaFinal });
    }
}

async function main() {
    // --- 1. Carregamento e Preparação dos Dados ---
    console.log("Carregando dataset CORRIGIDO de emoções...");
    const datasetPath = path.join(__dirname, 'dataset_emocoes_corrigido.json');
    if (!fs.existsSync(datasetPath)) {
        console.error("\nERRO: Arquivo 'dataset_emocoes_corrigido.json' não encontrado.");
        process.exit(1);
    }
    const dataset = JSON.parse(fs.readFileSync(datasetPath, 'utf8'));

    // Pré processamento
    function preprocessTexto(texto) { return texto.toLowerCase().replace(/[.,!?:;()"']/g, ''); }
    const vocabulario = new Set();
    dataset.forEach(item => preprocessTexto(item.texto).split(/\s+/).forEach(palavra => vocabulario.add(palavra)));
    const vocabularioArray = Array.from(vocabulario);
    const vocabSize = vocabularioArray.length;
    const wordIndex = vocabularioArray.reduce((map, palavra, i) => ({...map, [palavra]: i}), {});
    console.log(`Vocabulário construído com ${vocabSize} palavras únicas.`);

    const sentimentosMap = {"alegria": [1,0,0,0,0,0,0,0,0],"tristeza": [0,1,0,0,0,0,0,0,0],"raiva": [0,0,1,0,0,0,0,0,0],"medo": [0,0,0,1,0,0,0,0,0],"surpresa": [0,0,0,0,1,0,0,0,0],"ansiedade": [0,0,0,0,0,1,0,0,0],"vergonha": [0,0,0,0,0,0,1,0,0],"amor": [0,0,0,0,0,0,0,1,0],"empolgação": [0,0,0,0,0,0,0,0,1]};
    const sentimentosArray = Object.keys(sentimentosMap);

    function vetorizarFrase(frase) {
        const vetor = new Array(vocabSize).fill(0);
        preprocessTexto(frase).split(/\s+/).forEach(palavra => { if(palavra in wordIndex) vetor[wordIndex[palavra]] = 1; });
        return vetor;
    }
    const entradas = dataset.map(item => vetorizarFrase(item.texto));
    const saidas = dataset.map(item => sentimentosMap[item.sentimento]);
    const xs = tf.tensor2d(entradas, [entradas.length, vocabSize]);
    const ys = tf.tensor2d(saidas, [saidas.length, sentimentosArray.length]);

    // --- 2. Construção da Arquitetura e Treinamento ---
    console.log("\n### Construindo e Treinando a Rede Micelial ###");
    const rede = new Rede();
    const dimSaida = sentimentosArray.length;

    rede.adicionarNo(new No("Entrada",   { units: 128, ativacaoCamada: 'relu', ativacaoFinal: 'relu' }));
    rede.adicionarNo(new No("Oculto-A",  { units: 64,  ativacaoCamada: 'relu', ativacaoFinal: 'relu' }));
    rede.adicionarNo(new No("Oculto-B",  { units: 64,  ativacaoCamada: 'relu', ativacaoFinal: 'relu' }));
    rede.adicionarNo(new No("Agregador", { units: 32,  ativacaoCamada: 'relu', ativacaoFinal: 'relu' }));
    rede.adicionarNo(new No("Saida",     { units: dimSaida, ativacaoCamada: 'linear', ativacaoFinal: 'softmax' }));
    rede.adicionarCaminho(new Caminho({ nome: "C-E_A", origem: "Entrada", destino: "Oculto-A" }));
    rede.adicionarCaminho(new Caminho({ nome: "C-E_B", origem: "Entrada", destino: "Oculto-B" }));

    const model = rede.buildModel([vocabSize]);
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    await model.fit(xs, ys, {
        epochs: 60, batchSize: 16, shuffle: true, validationSplit: 0.2,
        callbacks: [ tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 }) ]
    });
    console.log("Treinamento concluído.");

    // =========================================================
    //         **BLOCO PARA SALVAR O MODELO**
    // =========================================================
    console.log("\nSalvando o modelo treinado para uso futuro...");
    const savePath = path.join(__dirname, 'modelo_sentimentos_salvo');

    if (!fs.existsSync(savePath)) {
        fs.mkdirSync(savePath, { recursive: true });
    }

    let modelArtifacts;
    await model.save({
        save: async (artifacts) => {
            modelArtifacts = artifacts;
            return { modelArtifactsInfo: { /* dummy */ } };
        }
    });

    const modelJson = {
        "modelTopology": modelArtifacts.modelTopology,
        "weightsManifest": [{ "paths": ["./weights.bin"], "weights": modelArtifacts.weightSpecs }]
    };

    fs.writeFileSync(path.join(savePath, 'model.json'), JSON.stringify(modelJson, null, 2));
    fs.writeFileSync(path.join(savePath, 'weights.bin'), Buffer.from(modelArtifacts.weightData));
    
    // Salvar o vocabulário também! É crucial para usar o modelo depois.
    fs.writeFileSync(path.join(savePath, 'vocab.json'), JSON.stringify(vocabularioArray));

    console.log(`Modelo e vocabulário salvos com sucesso na pasta: ${savePath}`);
    // =========================================================
    
    // Limpeza de memória
    tf.dispose([xs, ys, model]);
}

main().catch(console.error);