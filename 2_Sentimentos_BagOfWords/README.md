\# MyceliumAI - Fase 2: Análise de Sentimentos com Bag of Words



Esta pasta contém a primeira aplicação da arquitetura \*\*MyceliumAI\*\* a um problema de Processamento de Linguagem Natural (NLP). O objetivo era ambicioso: ir além da classificação numérica e ensinar a rede a entender texto em português.



\### O Desafio



O modelo foi treinado para classificar frases em 8 categorias de emoções, usando um dataset customizado. Esta fase introduziu vários desafios novos que foram cruciais para a evolução do projeto:



1\.  \*\*Pré-processamento de Texto:\*\* Desenvolver rotinas para limpar pontuação, converter para minúsculas e preparar o texto para a máquina.

2\.  \*\*Criação de Vocabulário:\*\* Mapear cada palavra única do dataset para um índice numérico.

3\.  \*\*Vetorização:\*\* Converter frases de texto em vetores numéricos que o modelo pudesse entender.



\### A Abordagem: "Bag of Words" (Saco de Palavras)



A técnica de vetorização utilizada aqui foi a \*\*Bag of Words\*\*. Neste método, cada frase é representada por um vetor gigante, do tamanho do vocabulário. A posição de cada palavra no vocabulário é marcada com `1` se ela aparece na frase, e `0` caso contrário.



\*\*Exemplo:\*\*

\- Vocabulário: `\["eu", "amo", "você", "bolo"]`

\- Frase: `"eu amo bolo"`

\- Vetor: `\[1, 1, 0, 1]`



\### Arquitetura e Validação



O modelo usou a arquitetura de grafo não-linear da \*\*MyceliumAI\*\*. O sucesso do treinamento provou que a arquitetura era robusta o suficiente para lidar com dados de alta dimensionalidade como os vetores de Bag of Words, validando seu potencial para problemas além da matemática pura.



\### Limitações e Aprendizados



Embora funcional, a abordagem Bag of Words tem limitações significativas que se tornaram claras durante os testes:

\- \*\*Ignora a Ordem das Palavras:\*\* As frases "eu não estou feliz" e "não estou eu feliz" teriam o mesmo vetor.

\- \*\*Ignora o Contexto Semântico:\*\* O modelo não tem como saber que "feliz" e "contente" são palavras com significados similares.



A identificação clara dessas limitações foi a principal motivação para evoluir o projeto para a \*\*Fase 3\*\*, que introduziu a técnica de \*\*Word Embeddings\*\* para superar esses desafios e dar à IA uma compreensão muito mais profunda da linguagem.

