// comparar_arquiteturas.js
// MyceliumAI vs. Rede Neural Padrão no dataset Íris.

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

const No = require('./no.js');
const Caminho = require('./caminho.js');

// ======================================================================
// Classe da Rede Micelial
// ======================================================================
class Rede {
    constructor() { this.nos = new Map(); this.caminhos = new Map(); }
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
    // --- 1. PREPARAÇÃO DOS DADOS (Idêntico para ambos os modelos) ---
    console.log("Carregando e preparando o dataset Íris para o duelo...");
    const dataset = JSON.parse(fs.readFileSync(path.join(__dirname, 'iris_dataset.json'), 'utf8'));
    tf.util.shuffle(dataset);

    const speciesMap = { "setosa": 0, "versicolor": 1, "virginica": 2 };
    const numClasses = Object.keys(speciesMap).length;

    const tensores = tf.tidy(() => {
        const entradas = dataset.map(f => [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]);
        const saidas = dataset.map(f => speciesMap[f.species]);
        return { xs: tf.tensor2d(entradas), ys: tf.oneHot(tf.tensor1d(saidas, 'int32'), numClasses) };
    });

    const tamanhoTeste = Math.floor(dataset.length * 0.2);
    const tamanhoTreino = dataset.length - tamanhoTeste;
    const [xsTreino, xsTeste] = tf.split(tensores.xs, [tamanhoTreino, tamanhoTeste]);
    const [ysTreino, ysTeste] = tf.split(tensores.ys, [tamanhoTreino, tamanhoTeste]);

    // --- 2. CONSTRUÇÃO DOS COMPETIDORES ---
    
    // --- Competidor 1: MyceliumAI ---
    console.log("\n### Configurando a Rede Micelial ###");
    const redeMicelial = new Rede();
    redeMicelial.adicionarNo(new No("Entrada", [{ type: 'dense', config: { units: 10, activation: 'relu', inputShape: [4] } }]));
    redeMicelial.adicionarNo(new No("Oculto-A", [{ type: 'dense', config: { units: 10, activation: 'relu' } }]));
    redeMicelial.adicionarNo(new No("Oculto-B", [{ type: 'dense', config: { units: 10, activation: 'relu' } }]));
    redeMicelial.adicionarNo(new No("Agregador", [{ type: 'dense', config: { units: 10, activation: 'relu' } }]));
    redeMicelial.adicionarNo(new No("Saida", [{ type: 'dense', config: { units: numClasses, activation: 'softmax' } }]));
    redeMicelial.adicionarCaminho(new Caminho({ nome: "C-E_A", origem: "Entrada", destino: "Oculto-A" }));
    redeMicelial.adicionarCaminho(new Caminho({ nome: "C-E_B", origem: "Entrada", destino: "Oculto-B" }));
    const modelMicelial = redeMicelial.buildModel([4]);
    modelMicelial.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    console.log("Arquitetura Micelial:");
    modelMicelial.summary();

    // --- Competidor 2: Rede Neural Padrão ---
    console.log("\n### Configurando a Rede Neural Padrão (Sequencial) ###");
    const modelPadrao = tf.sequential();
    // Para ser uma comparação justa, vamos usar um número similar de camadas e neurônios.
    modelPadrao.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [4] }));
    modelPadrao.add(tf.layers.dense({ units: 20, activation: 'relu' })); // Camada oculta maior para compensar a falta de paralelismo
    modelPadrao.add(tf.layers.dense({ units: 10, activation: 'relu' }));
    modelPadrao.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
    modelPadrao.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    console.log("Arquitetura Padrão:");
    modelPadrao.summary();

    // --- 3. TREINAMENTO (Ambos usam os mesmos dados) ---
    console.log("\n--- INICIANDO TREINAMENTO DA MYCELIUMAI ---");
    await modelMicelial.fit(xsTreino, ysTreino, {
        epochs: 100, shuffle: true, validationSplit: 0.1, verbose: 0, // verbose: 0 para um log mais limpo
        callbacks: [ tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 }) ]
    });
    console.log("Treinamento da MyceliumAI concluído.");
    
    console.log("\n--- INICIANDO TREINAMENTO DA REDE PADRÃO ---");
    await modelPadrao.fit(xsTreino, ysTreino, {
        epochs: 100, shuffle: true, validationSplit: 0.1, verbose: 0,
        callbacks: [ tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 }) ]
    });
    console.log("Treinamento da Rede Padrão concluído.");

    // --- 4. O PLACAR FINAL ---
    console.log("\n### Avaliando os modelos no conjunto de teste ###");
    const resultadoMicelial = modelMicelial.evaluate(xsTeste, ysTeste, { verbose: 0 });
    const resultadoPadrao = modelPadrao.evaluate(xsTeste, ysTeste, { verbose: 0 });
    
    const acuraciaMicelial = resultadoMicelial[1].dataSync()[0];
    const acuraciaPadrao = resultadoPadrao[1].dataSync()[0];

    console.log("\n========================================");
    console.log("              PLACAR FINAL");
    console.log("========================================");
    console.log(`  ACURÁCIA MYCELIUMAI: ${(acuraciaMicelial * 100).toFixed(2)}%`);
    console.log(`  ACURÁCIA REDE PADRÃO:  ${(acuraciaPadrao * 100).toFixed(2)}%`);
    console.log("========================================");

    if (acuraciaMicelial > acuraciaPadrao) {
        console.log("\n🏆 Vitória da MyceliumAI! A arquitetura de grafo demonstrou uma performance superior.");
    } else if (acuraciaPadrao > acuraciaMicelial) {
        console.log("\n🏁 Vitória da Rede Padrão. A simplicidade venceu desta vez.");
    } else {
        console.log("\n🤝 Empate! Ambas as arquiteturas tiveram uma performance idêntica.");
    }

    tf.dispose([tensores.xs, tensores.ys, xsTreino, xsTeste, ysTreino, ysTeste, modelMicelial, modelPadrao, resultadoMicelial, resultadoPadrao]);
}

main().catch(console.error);