# Changelog - MyceliumAI

Todos as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), e este projeto adere ao [Versionamento Semântico](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-07-19

Foco em validação, regularização e análise comparativa da arquitetura MyceliumAI usando o dataset Íris como laboratório. Esta versão representa um aprofundamento no entendimento científico do modelo.

### Added
-   **Laboratório de Testes Íris:** Criação de uma nova suíte de testes para validar a arquitetura em um ambiente controlado e rápido.
-   **Script de Comparação de Arquiteturas:** Desenvolvimento do script `comparar_arquiteturas.js` para realizar um "duelo" justo entre a MyceliumAI e uma rede sequencial padrão.
-   **Implementação de Dropout:** Aprimoramento da classe `No.js` para se tornar um container flexível, permitindo a adição de camadas de `Dropout` para combater o overfitting.

### Changed
-   **Narrativa do Projeto:** A jornada do projeto agora inclui explicitamente a análise de resultados desfavoráveis (a vitória da rede sequencial no problema Íris) como um aprendizado crucial sobre o princípio da "Navalha de Occam" em Machine Learning.

### Fixed
-   **Diagnóstico de Overfitting:** Identificado e validado que a complexidade da MyceliumAI a torna mais propensa ao overfitting em datasets pequenos e simples, direcionando os esforços futuros para problemas mais complexos (Processamento de Linguagem Natural, Geração de Imagens e outros formatos de mídias complexas) e para o uso de técnicas de regularização.

---

## [1.0.0] - 2025-07-18

Esta é a primeira versão pública e estável do projeto, focada na arquitetura de grafo com Word Embeddings para análise de sentimentos.

### Added
-   **Estrutura de Repositório Final:** Projeto organizado em 3 fases evolutivas, com READMEs individuais para cada etapa.
-   **Fase 3: MyceliumAI com Word Embeddings:**
    -   Implementação da camada `tf.layers.embedding` para criar vetores de palavras que capturam significado semântico.
    -   Script de treino (`treinar_micelial_final.js`) que constrói a arquitetura de grafo sobre os embeddings.
    -   Aplicação web (`app_web/index.html`) para testar o modelo de sentimentos treinado.
-   **Pipeline de Dados Robusta:**
    -   Script `preparar_dados.js` que limpa, corrige e valida o dataset de sentimentos, gerando um arquivo `dataset_para_treino.json` garantido de ser limpo.
-   **Documentação Completa:**
    -   `README.md` principal detalhando a jornada completa do projeto.
    -   `CHANGELOG.md` (este arquivo) para rastrear o histórico de versões.
    -   `LICENSE` com a licença MIT.
    -   `.gitignore` para manter o repositório limpo.

### Changed
-   **Arquitetura do `No.js` e `Caminho.js`:** Refinamento das classes para serem totalmente compatíveis com o salvamento e carregamento de modelos do TensorFlow.js.

### Fixed
-   **Bugs de Carregamento no Navegador:** Resolvido o problema crônico de `Unknown Layer` ao garantir que o `model.json` e o `index.html` estivessem em perfeita sincronia, com as classes customizadas sendo corretamente definidas e registradas antes do carregamento do modelo.
-   **Inconsistências no Dataset:** Implementado um processo de limpeza automático que corrige erros de digitação e descarta dados inválidos, garantindo a estabilidade do treinamento.

---

## [0.2.0] - 2025-06-29

Versão intermediária que aplicou a arquitetura MyceliumAI a um problema de Processamento de Linguagem Natural (NLP) pela primeira vez.

### Added
-   **Fase 2: Análise de Sentimentos com Bag of Words:**
    -   Primeiro modelo treinado para classificar 8 emoções em português.
    -   Implementação de pré-processamento de texto, criação de vocabulário e vetorização "Bag of Words".
    -   Uso da arquitetura de grafo não-linear da MyceliumAI com os vetores de texto.

### Removed
-   Abandono da abordagem de modelo "congelado" ou "plano" em favor da solução correta de carregar as camadas customizadas.

---

## [0.1.0] - 2025-06-23

A primeira prova de conceito da arquitetura MyceliumAI.

### Added
-   **Fase 1: Classificador Par ou Ímpar:**
    -   Criação das classes fundamentais `No.js`, `Caminho.js` e `Rede.js`.
    -   Modelo treinado para classificar números com 100% de acurácia.
    -   Descoberta crucial sobre a importância da **Engenharia de Features** (usando representação binária).
-   **Início do Projeto:** Concepção da ideia de uma rede neural de grafo inspirada em micélios.
