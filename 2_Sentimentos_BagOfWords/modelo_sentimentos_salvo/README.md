# MyceliumAI - Fase 2: Análise de Sentimentos com Bag of Words

Esta pasta contém a primeira aplicação da arquitetura **MyceliumAI** a um problema de Processamento de Linguagem Natural (NLP). O objetivo era ir além da classificação numérica e ensinar a rede a entender texto em português.

### O Desafio

O modelo foi treinado para classificar frases em 8 categorias de emoções, usando um dataset customizado. Esta fase introduziu vários desafios novos:

1.  **Pré-processamento de Texto:** Limpeza de pontuação e conversão para minúsculas.
2.  **Criação de Vocabulário:** Mapeamento de cada palavra única do dataset para um índice numérico.
3.  **Vetorização:** Conversão de frases de texto em vetores numéricos que o modelo pudesse entender.

### A Abordagem: "Bag of Words" (Saco de Palavras)

A técnica de vetorização utilizada aqui foi a **Bag of Words**. Neste método, cada frase é representada por um vetor gigante, do tamanho do vocabulário. A posição de cada palavra no vocabulário é marcada com `1` se ela aparece na frase, e `0` caso contrário.

**Exemplo:**
- Vocabulário: `["eu", "amo", "você", "bolo"]`
- Frase: `"eu amo bolo"`
- Vetor: `[1, 1, 0, 1]`

### Arquitetura Utilizada

O modelo usou uma arquitetura de grafo não-linear da **MyceliumAI**, onde a informação se dividia em caminhos paralelos e era reagregada antes da classificação final. Isso provou que a arquitetura era robusta o suficiente para lidar com dados de alta dimensionalidade como os vetores de Bag of Words.

### Limitações e Aprendizados

Embora funcional, a abordagem Bag of Words tem limitações significativas:
- **Ignora a Ordem:** As frases "eu amo você" e "você amo eu" teriam o mesmo vetor.
- **Ignora o Contexto:** Não captura o significado semântico. O modelo não sabe que "feliz" e "contente" são palavras similares.

Essas limitações foram a principal motivação para evoluir o projeto para a **Fase 3**, que introduziu a técnica de **Word Embeddings** para superar esses desafios.

### Estrutura dos Arquivos

-   `treinar_sentimentos.js`: O script Node.js que prepara os dados, constrói o grafo MyceliumAI, treina o modelo e o salva.
-   `index.html`: A interface web para testar o modelo treinado.
-   `no.js` / `caminho.js`: As classes fundamentais da arquitetura.
-   `dataset_emocoes_corrigido.json`: O dataset limpo usado para o treinamento.
-   `/modelo_sentimentos_salvo/`: A pasta gerada pelo script de treino, contendo o modelo e o vocabulário.