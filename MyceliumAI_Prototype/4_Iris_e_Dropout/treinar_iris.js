// treinar_iris.js
// Script completo: treina, avalia massivamente e salva o modelo MyceliumAI para o dataset Íris.

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

// Classes customizadas.
const No = require('./no.js');
const Caminho = require('./caminho.js');

// ======================================================================
// Classe da Rede Micelial (Arquitetura de grafo)
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

// ======================================================================
// Função Principal
// ======================================================================
async function main() {
    // --- 1. PREPARAÇÃO E DIVISÃO DOS DADOS ---
    console.log("Carregando e preparando o dataset Íris...");
    const datasetPath = path.join(__dirname, 'iris_dataset.json');
    if (!fs.existsSync(datasetPath)) {
        console.error(`ERRO: Arquivo 'iris_dataset.json' não encontrado. Execute 'gerar_dataset_iris.js' primeiro.`);
        process.exit(1);
    }
    const dataset = JSON.parse(fs.readFileSync(datasetPath, 'utf8'));
    tf.util.shuffle(dataset);

    const speciesMap = { "setosa": 0, "versicolor": 1, "virginica": 2 };
    const speciesArray = Object.keys(speciesMap);
    const numClasses = speciesArray.length;

    const tensores = tf.tidy(() => {
        const entradas = dataset.map(f => [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]);
        const saidas = dataset.map(f => speciesMap[f.species]);
        const xs = tf.tensor2d(entradas);
        const ys = tf.oneHot(tf.tensor1d(saidas, 'int32'), numClasses);
        return { xs, ys };
    });

    const tamanhoTeste = Math.floor(dataset.length * 0.2); // 20% para teste
    const tamanhoTreino = dataset.length - tamanhoTeste;

    const [xsTreino, xsTeste] = tf.split(tensores.xs, [tamanhoTreino, tamanhoTeste]);
    const [ysTreino, ysTeste] = tf.split(tensores.ys, [tamanhoTreino, tamanhoTeste]);

    console.log(`Dataset dividido em ${tamanhoTreino} amostras de treino e ${tamanhoTeste} amostras de teste.`);

    // --- 2. CONSTRUÇÃO DA ARQUITETURA ---
    console.log("\n### Configurando a Rede Micelial ###");
    const redeMicelial = new Rede();
    redeMicelial.adicionarNo(new No("Entrada", [{ type: 'dense', config: { units: 10, activation: 'relu' } }]));
    redeMicelial.adicionarNo(new No("Oculto-A", [{ type: 'dense', config: { units: 10, activation: 'relu' } }]));
    redeMicelial.adicionarNo(new No("Oculto-B", [{ type: 'dense', config: { units: 10, activation: 'relu' } }]));
    redeMicelial.adicionarNo(new No("Agregador", [{ type: 'dense', config: { units: 10, activation: 'relu' } }]));
    redeMicelial.adicionarNo(new No("Saida", [{ type: 'dense', config: { units: numClasses, activation: 'softmax' } }]));
    redeMicelial.adicionarCaminho(new Caminho({ nome: "C-E_A", origem: "Entrada", destino: "Oculto-A" }));
    redeMicelial.adicionarCaminho(new Caminho({ nome: "C-E_B", origem: "Entrada", destino: "Oculto-B" }));
    
    const model = redeMicelial.buildModel([4]); // 4 features de entrada
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // --- 3. TREINAMENTO (usando APENAS os dados de treino) ---
    console.log("\nIniciando treinamento...");
    await model.fit(xsTreino, ysTreino, {
        epochs: 100,
        shuffle: true,
        validationSplit: 0.1,
        callbacks: [ tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10, verbose: 1 }) ]
    });
    console.log("Treinamento concluído.");

    // --- 4. AVALIAÇÃO MASSIVA NO CONJUNTO DE TESTE ---
    console.log("\n### Avaliando o modelo no conjunto de teste que ele nunca viu ###");
    const resultado = model.evaluate(xsTeste, ysTeste);
    const acuraciaFinal = resultado[1].dataSync()[0];

    console.log("\n========================================");
    console.log(`  RESULTADO FINAL DO TESTE MASSIVO`);
    console.log("========================================");
    console.log(`  Acurácia (Accuracy): ${(acuraciaFinal * 100).toFixed(2)}%`);
    console.log("========================================");
    console.log(`\nIsso significa que o modelo acertou ${(acuraciaFinal * 100).toFixed(2)}% das ${tamanhoTeste} flores que ele nunca tinha visto antes.`);

    // --- 5. SALVAMENTO DO MODELO ---
    console.log("\nSalvando o modelo treinado...");
    const savePath = path.join(__dirname, 'modelo_iris_salvo');
    if (!fs.existsSync(savePath)) fs.mkdirSync(savePath, { recursive: true });

    let modelArtifacts;
    await model.save({ save: async (artifacts) => { modelArtifacts = artifacts; return { modelArtifactsInfo: {} }; }});
    const modelJson = { "modelTopology": modelArtifacts.modelTopology, "weightsManifest": [{ "paths": ["./weights.bin"], "weights": modelArtifacts.weightSpecs }] };
    
    fs.writeFileSync(path.join(savePath, 'model.json'), JSON.stringify(modelJson, null, 2));
    fs.writeFileSync(path.join(savePath, 'weights.bin'), Buffer.from(modelArtifacts.weightData));
    const metadata = { species: speciesArray };
    fs.writeFileSync(path.join(savePath, 'metadata.json'), JSON.stringify(metadata));

    console.log(`✅ Modelo e metadados salvos com sucesso na pasta: ${path.basename(savePath)}`);

    tf.dispose([tensores.xs, tensores.ys, xsTreino, xsTeste, ysTreino, ysTeste, model, resultado]);
}

main().catch(console.error);