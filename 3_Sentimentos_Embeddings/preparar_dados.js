// preparar_dados.js
const fs = require('fs');
const path = require('path');

console.log("Iniciando preparação do dataset...");

const arquivoOriginal = path.join(__dirname, 'dataset_sentimentos.json');
const arquivoFinal = path.join(__dirname, 'dataset_para_treino.json');

if (!fs.existsSync(arquivoOriginal)) {
    console.error(`ERRO: O dataset original '${path.basename(arquivoOriginal)}' não foi encontrado!`);
    process.exit(1);
}

const datasetBruto = JSON.parse(fs.readFileSync(arquivoOriginal, 'utf8'));
console.log(`Lendo ${datasetBruto.length} registros do arquivo original.`);

const mapaDeCorrecoes = { "surrpresa": "surpresa", "ira": "raiva", "orgulho": "alegria", "euforia": "alegria" };
const sentimentosValidos = new Set(["alegria", "tristeza", "raiva", "medo", "surpresa", "ansiedade", "vergonha", "amor", "empolgação"]);

let itensDescartados = 0;

const datasetProcessado = datasetBruto.map(item => {
    if (!item || typeof item.sentimento !== 'string') return null;
    let sentimentoCorrigido = item.sentimento;
    if (sentimentoCorrigido in mapaDeCorrecoes) {
        sentimentoCorrigido = mapaDeCorrecoes[sentimentoCorrigido];
    }
    return { ...item, sentimento: sentimentoCorrigido };
}).filter(item => {
    if (!item || !item.hasOwnProperty('texto') || typeof item.texto !== 'string' || item.texto.trim() === '' || !sentimentosValidos.has(item.sentimento)) {
        itensDescartados++;
        return false;
    }
    return true;
});

console.log(`\nProcessamento concluído.`);
console.log(`Itens válidos mantidos: ${datasetProcessado.length}`);
console.log(`Itens descartados: ${itensDescartados}`);

fs.writeFileSync(arquivoFinal, JSON.stringify(datasetProcessado, null, 2));
console.log(`\n✅ Dataset final salvo com sucesso em: ${arquivoFinal}`);