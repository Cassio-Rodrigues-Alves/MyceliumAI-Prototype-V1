// treinar_par_impar.js

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

const No = require('./no.js');
const Caminho = require('./caminho.js');
const Rede = require('./rede.js');

const TAMANHO_BINARIO = 16; 

function numeroParaBinario(numero) {
    let binarioStr = numero.toString(2);
    while (binarioStr.length < TAMANHO_BINARIO) {
        binarioStr = '0' + binarioStr;
    }
    return binarioStr.split('').map(bit => parseInt(bit, 10));
}

function gerarDadosParImparBinario(numAmostras = 2000) {
    const entradas = [];
    const saidas = [];

    for (let i = 0; i < numAmostras; i++) {
        const numero = Math.floor(Math.random() * 10000);
        entradas.push(numeroParaBinario(numero));
        const paridade = numero % 2;
        saidas.push([paridade]); 
    }
    
    const xs = tf.tensor2d(entradas, [numAmostras, TAMANHO_BINARIO]);
    const ys = tf.tensor2d(saidas, [numAmostras, 1]);

    return { xs, ys };
}

async function main() {
    console.log("### Teste de Viabilidade: Treinando e Salvando o Modelo (Método Final) ###");

    const rede = new Rede();
    rede.adicionarNo(new No("Entrada", { units: 16, ativacaoCamada: 'relu' }));
    rede.adicionarNo(new No("Oculto",  { units: 8, ativacaoCamada: 'relu' }));
    rede.adicionarNo(new No("Saida",   { units: 1, ativacaoCamada: 'linear', ativacaoFinal: 'sigmoid' }));
    rede.adicionarCaminho(new Caminho({ nome: "C-Entrada_Oculto", origem: "Entrada", destino: "Oculto" }));
    rede.adicionarCaminho(new Caminho({ nome: "C-Oculto_Saida", origem: "Oculto", destino: "Saida" }));

    console.log("\nConstruindo o modelo TensorFlow a partir da topologia...");
    const model = rede.buildModel([TAMANHO_BINARIO]);
    model.compile({
        optimizer: tf.train.adam(0.005),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    const { xs, ys } = gerarDadosParImparBinario(2000);
    console.log(`\nGeradas 2000 amostras para treino.`);
    console.log("Iniciando treinamento...");
    await model.fit(xs, ys, {
        epochs: 15,
        batchSize: 32,
        shuffle: true,
        validationSplit: 0.1,
        callbacks: { onEpochEnd: (epoch, logs) => console.log(`  Epoch ${epoch+1}, Acc: ${logs.acc.toFixed(4)}, Val_Acc: ${logs.val_acc.toFixed(4)}`) }
    });
    console.log("Treinamento concluído.");

    // ===== MÉTODO PARA SALVAR MANUALMENTE =====
    console.log("\nSalvando o modelo em memória e escrevendo no disco...");

    const savePath = path.join(__dirname, 'modelo_salvo');
    if (!fs.existsSync(savePath)) {
        fs.mkdirSync(savePath, { recursive: true });
    }

    let modelArtifacts;
    const saveHandler = {
        save: async (artifacts) => {
            modelArtifacts = artifacts;
            return { modelArtifactsInfo: { /* dummy */ } };
        }
    };
    await model.save(saveHandler);

    // Construir o objeto JSON completo com topologia e manifesto
    const modelJson = {
        "modelTopology": modelArtifacts.modelTopology,
        "weightsManifest": [
            {
                "paths": ["./model.weights.bin"], // Caminho relativo ao arquivo de pesos
                "weights": modelArtifacts.weightSpecs
            }
        ]
    };

    // Agora, escrevemos os artefatos corretos para os arquivos
    fs.writeFileSync(
        path.join(savePath, 'model.json'),
        JSON.stringify(modelJson, null, 2) // Salva o JSON completo
    );
    fs.writeFileSync(
        path.join(savePath, 'model.weights.bin'),
        Buffer.from(modelArtifacts.weightData)
    );

    console.log(`Modelo salvo com sucesso na pasta: ${savePath}`);
    // =========================================================

    tf.dispose([xs, ys]);
    console.log("\nScript finalizado.");
}

main().catch(console.error);