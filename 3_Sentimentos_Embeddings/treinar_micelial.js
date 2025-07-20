// treinar_micelial.js

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

const No = require('./no.js');
const Caminho = require('./caminho.js');

// ======================================================================
// O "Painel de Controle" para o Terminal
// ======================================================================
class TrainingMonitorCallback extends tf.Callback {
    constructor(config) {
        super();
        this.totalEpochs = config.totalEpochs;
        this.totalSamples = config.totalSamples;
        this.batchSize = config.batchSize;
    }
    onEpochBegin(epoch, logs) {
        console.log(`\n--- Iniciando Época ${epoch + 1} de ${this.totalEpochs} ---`);
    }
    onBatchEnd(batch, logs) {
        const lossValue = logs.loss.dataSync()[0];
        const accValue = logs.acc.dataSync()[0];
        const samplesProcessed = Math.min((batch + 1) * this.batchSize, this.totalSamples);
        const progress = (samplesProcessed / this.totalSamples * 100).toFixed(0);
        process.stdout.write(`  [ Progresso: ${progress}% (${samplesProcessed}/${this.totalSamples}) ] Loss: ${lossValue.toFixed(4)}, Acc: ${accValue.toFixed(4)}\r`);
    }
    onEpochEnd(epoch, logs) {
        process.stdout.write('\r' + ' '.repeat(80) + '\r');
        const acc = logs.acc.dataSync()[0];
        const loss = logs.loss.dataSync()[0];
        let logString = `  Época ${epoch + 1} concluída -> Acc: ${acc.toFixed(4)}, Loss: ${loss.toFixed(4)}`;
        if (logs.val_acc && logs.val_loss) {
            const val_acc = logs.val_acc.dataSync()[0];
            const val_loss = logs.val_loss.dataSync()[0];
            logString += `, Val_Acc: ${val_acc.toFixed(4)}, Val_Loss: ${val_loss.toFixed(4)}`;
        }
        console.log(logString);
    }
    onTrainEnd(logs) {
        console.log(`\n\n### TREINAMENTO FINALIZADO! ### \x07`);
    }
}

// ======================================================================
// Função Principal
// ======================================================================
async function main() {
    // --- 1. PREPARAÇÃO DOS DADOS ---
    console.log("Carregando o dataset limpo...");
    const dataset = JSON.parse(fs.readFileSync(path.join(__dirname, 'dataset_para_treino.json'), 'utf8'));
    
    function preprocessTexto(texto) { return texto.toLowerCase().replace(/[.,!?:;()"']/g, '').trim(); }
    const vocabulario = new Set();
    dataset.forEach(item => preprocessTexto(item.texto).split(/\s+/).forEach(palavra => { if(palavra) vocabulario.add(palavra); }));
    const vocabSize = vocabulario.size + 2;
    const wordIndex = {};
    let i = 2; for (const palavra of vocabulario) { wordIndex[palavra] = i++; }
    console.log(`Vocabulário construído com ${vocabulario.size} palavras.`);
    
    const MAX_LEN = 20;
    function sequenciarFrase(frase) {
        let sequencia = preprocessTexto(frase).split(/\s+/).map(palavra => wordIndex[palavra] || 1);
        sequencia = sequencia.slice(0, MAX_LEN);
        while (sequencia.length < MAX_LEN) sequencia.unshift(0);
        return sequencia;
    }
    
    const sentimentosMap = {"alegria": 0, "tristeza": 1, "raiva": 2, "medo": 3, "surpresa": 4, "ansiedade": 5, "vergonha": 6, "amor": 7, "empolgação": 8};
    const numClasses = Object.keys(sentimentosMap).length;
    const entradas = dataset.map(item => sequenciarFrase(item.texto));
    const saidas = dataset.map(item => sentimentosMap[item.sentimento]);
    const xs = tf.tensor2d(entradas, [entradas.length, MAX_LEN]);
    const ys = tf.oneHot(tf.tensor1d(saidas, 'int32'), numClasses);
    const vocabularioParaSalvar = Array.from(vocabulario); 

    // --- 2. CONSTRUÇÃO DO MODELO "PLANO" ---
    console.log("\n### Construindo a arquitetura MyceliumAI (versão plana) ###");
    
    const input = tf.input({ shape: [MAX_LEN] });
    const embeddingLayer = tf.layers.embedding({ inputDim: vocabSize, outputDim: 50, inputLength: MAX_LEN });
    const poolingLayer = tf.layers.globalAveragePooling1d();
    const preprocessed = poolingLayer.apply(embeddingLayer.apply(input));
    const noEntrada = new No("Entrada", { units: 64, ativacaoCamada: 'relu', ativacaoFinal: 'relu' });
    const noOcultoA = new No("Oculto-A", { units: 32, ativacaoCamada: 'relu', ativacaoFinal: 'relu' });
    const noOcultoB = new No("Oculto-B", { units: 32, ativacaoCamada: 'relu', ativacaoFinal: 'relu' });
    const noAgregador = new No("Agregador", { units: 16, ativacaoCamada: 'relu', ativacaoFinal: 'relu' });
    const noSaida = new No("Saida", { units: numClasses, ativacaoCamada: 'linear', ativacaoFinal: 'softmax' });
    const caminho_E_A = new Caminho({ nome: "C-E_A", origem: "Entrada", destino: "Oculto-A" });
    const caminho_E_B = new Caminho({ nome: "C-E_B", origem: "Entrada", destino: "Oculto-B" });
    
    const saidaNoEntrada = noEntrada.call([preprocessed]);
    const entradaOcultoA = caminho_E_A.apply(saidaNoEntrada);
    const entradaOcultoB = caminho_E_B.apply(saidaNoEntrada);
    const saidaOcultoA = noOcultoA.call([entradaOcultoA]);
    const saidaOcultoB = noOcultoB.call([entradaOcultoB]);
    const saidaAgregador = noAgregador.call([saidaOcultoA, saidaOcultoB]);
    const output = noSaida.call([saidaAgregador]);
    
    const model = tf.model({ inputs: input, outputs: output });
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // --- 3. TREINAMENTO COM O MONITOR ---
    console.log("\nIniciando treinamento monitorado...");
    const epochs = 20;
    const batchSize = 32;
    const validationSplit = 0.2;
    const numTrainSamples = Math.floor(dataset.length * (1 - validationSplit));
    
    // Instancia o nosso monitor!
    const monitor = new TrainingMonitorCallback({ totalEpochs: epochs, totalSamples: numTrainSamples, batchSize: batchSize });

    await model.fit(xs, ys, {
        epochs, batchSize, shuffle: true, validationSplit,
        // Adiciona o monitor à lista de callbacks
        callbacks: [ monitor, tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 }) ]
    });
    
// --- 4. SALVAMENTO ---
    console.log("\nSalvando o modelo treinado para uso futuro (método manual)...");
    
    const savePath = path.join(__dirname, 'app_web');
    if (!fs.existsSync(savePath)) {
        fs.mkdirSync(savePath, { recursive: true });
    }

    // Captura os artefatos do modelo em memória
    let modelArtifacts;
    await model.save({
        save: async (artifacts) => {
            modelArtifacts = artifacts;
            return { modelArtifactsInfo: {} };
        }
    });

    // Cria o JSON do modelo com o manifesto de pesos
    const modelJson = {
        "modelTopology": modelArtifacts.modelTopology,
        "weightsManifest": [{ "paths": ["./weights.bin"], "weights": modelArtifacts.weightSpecs }]
    };

    // Escreve os arquivos no disco usando 'fs'
    fs.writeFileSync(path.join(savePath, 'model.json'), JSON.stringify(modelJson, null, 2));
    fs.writeFileSync(path.join(savePath, 'weights.bin'), Buffer.from(modelArtifacts.weightData));
    fs.writeFileSync(path.join(savePath, 'vocab.json'), JSON.stringify(vocabularioParaSalvar));

    console.log(`\n✅ Modelo e vocabulário salvos com sucesso na pasta: ${path.basename(savePath)}`);
    }

main().catch(console.error);