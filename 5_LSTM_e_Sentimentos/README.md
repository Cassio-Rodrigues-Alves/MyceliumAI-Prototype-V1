\# MyceliumAI: Uma Arquitetura de Rede Neural de Grafo



\*\*Uma arquitetura de IA para NLP, com camadas customizadas, construída do zero em TensorFlow.js e aprimorada em Python/Keras, que validou sua eficiência contra modelos padrão da indústria.\*\*



---



\### Sobre o Projeto



Olá! Eu sou Cássio, um desenvolvedor brasileiro de 16 anos apaixonado por Inteligência Artificial. Este repositório documenta a minha jornada na criação de uma arquitetura de rede neural customizada, a \*\*MyceliumAI\*\*.



Inspirada na forma como os micélios de fungos criam redes interconectadas, a MyceliumAI abandona a estrutura sequencial tradicional em favor de um \*\*grafo computacional\*\* onde a informação flui por rotas paralelas e se reencontra, permitindo um processamento mais rico e complexo.



O coração da arquitetura são duas classes customizadas:

\- \*\*`No.py`\*\*: Atua como um container flexível para camadas do Keras, funcionando como um centro de processamento que agrega e transforma informações.

\- \*\*`caminho.py`\*\*: Representa as conexões ponderadas e treináveis que formam a "rede" entre os nós.



Este projeto foi desenvolvido com recursos limitados, evoluindo de experimentos em JavaScript em um PC de 2011 para um laboratório de pesquisa acelerado por GPU no Google Colab, demonstrando um ciclo completo de engenharia de Machine Learning.



\### O Projeto Final: Duelo de Sentimentos



A versão mais avançada neste repositório é um \*\*laboratório de comparação\*\* para um Analisador de Sentimentos em português. O experimento coloca a MyceliumAI, equipada com `Dropout` e `LSTM`, em um duelo direto contra uma rede sequencial padrão otimizada.



O resultado foi um \*\*empate técnico\*\*, uma validação impressionante que prova que a arquitetura MyceliumAI é uma alternativa tão viável e poderosa quanto os modelos padrão da indústria para problemas de NLP.



\### Como Executar os Experimentos



Todo o fluxo de treinamento e comparação está consolidado em um único notebook do Google Colab para garantir a reprodutibilidade.



1\.  \*\*Abra o Laboratório:\*\* Faça o upload do notebook (`Duelo\_Final.ipynb`) para o Google Colab e habilite o ambiente com GPU.

2\.  \*\*Prepare os Dados:\*\* Faça o upload do dataset limpo (`dataset\_para\_treino.json`) para o ambiente do Colab.

3\.  \*\*Execute a Célula:\*\* A célula única contém todas as definições de classe, a preparação de dados e o fluxo completo de treinamento e avaliação de ambos os modelos.



\### Como Testar o Modelo Treinado (Uso Prático)



Após executar o notebook de treinamento, ele salvará o modelo campeão e as ferramentas necessárias. Você pode testá-lo interativamente no seu próprio computador.



\*\*Passo 1: Baixe os Artefatos do Colab\*\*

Após o treinamento, faça o download dos seguintes arquivos do seu ambiente Colab:

\-   `modelo\_sentimentos\_campeao.keras` (O cérebro da IA)

\-   `tokenizer.pickle` (O dicionário de palavras)

\-   `no.py` (A definição da classe `No`)

\-   `caminho.py` (A definição da classe `Caminho`)



\*\*Passo 2: Crie o Script de Teste\*\*

Crie um arquivo local chamado `testar\_modelo.py` e cole o código que está na pasta `4\_Laboratorio\_Colab/scripts\_de\_teste/`.



\*\*Passo 3: Execute e Interaja\*\*

Coloque todos os arquivos baixados na mesma pasta do `testar\_modelo.py` e, no seu terminal, execute:

```bash

python testar\_modelo.py

```

O script carregará o modelo e permitirá que você digite frases para ver a IA classificar os sentimentos em tempo real!



\### Estrutura do Repositório



\-   `/1\_Classificador\_Par\_Impar/`: A prova de conceito inicial.

\-   `/2\_Sentimentos\_BagOfWords/`: Primeira abordagem de NLP.

\-   `/3\_Sentimentos\_Embeddings/`: Evolução para Word Embeddings em Node.js.

\-   `/4\_Laboratorio\_Colab/`: \*\*A fase final e mais avançada.\*\* Contém o notebook do duelo e os scripts de teste.

\-   `CHANGELOG.md`: O histórico completo de desenvolvimento do projeto.

\-   `LICENSE`: Licença MIT.



\### Contato



\- \*\*Autor:\*\* Cássio Rodrigues Alves

\- \*\*GitHub:\*\* `https://github.com/Cassio-Rodrigues-Alves`

\- \*\*LinkedIn:\*\* `https://www.linkedin.com/in/c%C3%A1ssio-rodrigues-alves-640704371/`

