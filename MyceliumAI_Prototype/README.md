# MyceliumAI-prototype
MyceliumAI: Uma arquitetura de rede neural de grafo, não-sequencial, construída do zero em TensorFlow.js.

# MyceliumAI: Uma Arquitetura de Rede Neural de Grafo Customizada

**Uma jornada do zero à IA, explorando arquiteturas não-sequenciais em TensorFlow.js para resolver problemas de classificação e análise de sentimentos.**

---

### Sobre o Projeto

Oi! Meu nome é Cássio, um desenvolvedor brasileiro de 16 anos que ama Inteligência Artificial. Este repositório documenta a minha jornada na criação de uma arquitetura de rede neural customizada, a **MyceliumAI**.

Inspirada na forma como os micélios de fungos criam redes interconectadas, a MyceliumAI abandona a estrutura sequencial tradicional das redes neurais em favor de um **grafo computacional** onde:
- **Nós (No.js):** Atuam como centros de processamento que agregam e transformam informações.
- **Caminhos (Caminho.js):** São conexões ponderadas e treináveis que formam a "rede" entre os nós, permitindo que a informação flua por múltiplas rotas.

Este projeto foi desenvolvido inteiramente em um computador de 2006 com um processador de 2011, como um desafio pessoal para provar que a engenhosidade na arquitetura e a qualidade dos dados são mais importantes que o poder bruto do hardware.

### A Evolução da MyceliumAI

O projeto evoluiu através de várias fases, cada uma com um objetivo e aprendizado diferente.

#### Fase 1: A Prova de Conceito (Classificador Par ou Ímpar)

- **Objetivo:** Validar se as classes `No`, `Caminho` e `Rede` conseguiam ser montadas em um modelo funcional e treinável.
- **Arquitetura:** Um grafo simples e linear **(Não muito diferente duma rede neural)**: `Entrada -> Caminho -> Oculto -> Caminho -> Saída`.
- **Desafio:** Ensinar a rede a distinguir números pares de ímpares.
- **Aprendizado Crucial:** O sucesso só foi alcançado após mudar a representação dos dados de números brutos para sua **representação binária**. Isso ensinou a lição mais importante: a **engenharia de features** é fundamental. Com essa mudança, o modelo atingiu 100% de acurácia.

#### Fase 2: O Baseline (Análise de Sentimentos com Rede Padrão)

- **Objetivo:** Estabelecer uma base de comparação usando uma rede neural sequencial padrão (`tf.sequential`) para o problema de análise de sentimentos com 9 emoções.
- **Arquitetura:** `Input -> Dense -> Dense -> Output (Softmax)`.
- **Desafio:** Lidar com dados de texto (NLP) pela primeira vez, incluindo pré-processamento, criação de vocabulário e vetorização usando "Bag of Words".
- **Aprendizado Crucial:** A técnica "Bag of Words" é funcional, mas limitada, pois ignora o contexto e a ordem das palavras. O modelo teve dificuldade em aprender nuances.

#### Fase 3: A MyceliumAI em Campo (Análise de Sentimentos com Grafo)

- **Objetivo:** Provar que a arquitetura MyceliumAI poderia superar a rede padrão em um problema real de NLP.
- **Arquitetura:** Um grafo não-linear: a informação entrava, se dividia em dois `Caminhos` paralelos, era processada por dois `Nós` ocultos diferentes, e depois reagregada em um `Nó` final antes da saída.
- **Desafio:** Depurar o fluxo de dados e gradientes em uma arquitetura mais complexa.
- **Aprendizado Crucial:** A arquitetura de grafo funcionou! Isso provou que o conceito era viável e abriu as portas para topologias mais complexas.

#### Fase 4: O Salto Quântico (MyceliumAI com Word Embeddings)

- **Objetivo:** Superar as limitações do "Bag of Words" e dar ao modelo uma compreensão mais profunda do significado das palavras.
- **Arquitetura:** A mesma arquitetura de grafo da Fase 3, mas com uma camada `Embedding` no início. Cada palavra agora é representada por um vetor denso de 50 dimensões, capturando seu significado semântico.
- **Desafio:** O custo computacional. O treinamento se tornou muito mais lento e intensivo, exigindo paciência e otimização.
- **Aprendizado Crucial:** Esta é a abordagem de ponta. Os embeddings permitiram ao modelo entender relações como "triste" ≈ "melancólico", resultando em uma IA muito mais inteligente e com maior capacidade de generalização. **Esta é a versão final e mais poderosa contida neste repositório (até o momento em que escrevo esse README.md).**

---

### Como Executar o Projeto Final (Fase 4)

Siga os passos para treinar e testar o Analisador de Sentimentos com a arquitetura MyceliumAI + Embeddings.

**Pré-requisitos:**
- [Node.js](https://nodejs.org/) (versão 20 ou superior)
- Um navegador web moderno

**Passo 1: Preparar os Dados**
Primeiro, vamos limpar e validar nosso dataset. O script abaixo lê o `dataset_emocoes_8_amostras.json`, corrige erros comuns e salva um arquivo limpo chamado `dataset_para_treino.json`.

```bash
node preparar_dados.js
```

**Passo 2: Treinar o Modelo**
Agora, execute o script de treinamento principal. Ele usará o dataset limpo para treinar a MyceliumAI, exibirá um painel de controle no terminal e, ao final, salvará o modelo treinado na pasta `modelo_sentimentos_salvo/`.

```bash
node treinar_micelial_final.js
```

**Passo 3: Testar na Interface Web**
Após o treinamento, você pode testar o modelo:
1.  Garanta que você tem a extensão **Live Server** no VS Code.
2.  Clique com o botão direito no arquivo `index_sentimentos.html`.
3.  Selecione "Open with Live Server".

A página carregará o modelo e o vocabulário salvos, e estará pronta para analisar suas frases!

### Estrutura dos Arquivos

-   `index_sentimentos.html`: A interface web para testar o modelo de sentimentos.
-   `preparar_dados.js`: Script utilitário para limpar o dataset original.
-   `treinar_micelial_final.js`: **O script principal.** Treina a arquitetura final e salva o modelo.
-   `no.js` / `caminho.js`: O coração da arquitetura MyceliumAI.
-   `dataset_emocoes_8_amostras.json`: O dataset original.
-   `dataset_para_treino.json`: O dataset limpo, gerado pelo `preparar_dados.js`.
-   `modelo_sentimentos_salvo/`: Pasta gerada pelo script de treino, contendo o modelo final (`model.json`, `weights.bin`) e o dicionário (`vocab.json`).

### Próximos Passos e Ideias

A MyceliumAI é uma plataforma para experimentação. Os próximos passos para evoluir este projeto podem incluir:
-   **Combater o Overfitting:** Implementar camadas de `Dropout` nos `Nós` para melhorar a generalização.
-   **Memória Sequencial:** Substituir a camada de `GlobalAveragePooling1D` por uma camada `LSTM` ou `GRU` para dar ao modelo uma melhor compreensão da ordem das palavras.
-   **Construção Dinâmica do Grafo:** Criar uma classe `Rede` ainda mais inteligente, que possa construir o `tf.Model` a partir de qualquer topologia de Nós e Caminhos, sem um `buildModel` pré-definido.

### Contato

- **Autor:** Cássio Rodrigues Alves
- **LinkedIn:** `(https://www.linkedin.com/in/c%C3%A1ssio-rodrigues-alves-640704371/)`
- **GitHub:** `(https://github.com/Cassio-Rodrigues-Alves)]`

### Licença

Este projeto está sob a **Licença MIT**. Sinta-se à vontade para usar, aprender e construir sobre ele.
