# testar_gerador_local.py
# Carrega o Gerador treinado no Colab e o usa para criar imagens a partir de texto no seu PC.

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# --- Bloco 1: Carregando o Modelo e as Ferramentas ---

print("Carregando o Gerador e o Tokenizer...")

# ATENÇÃO: É crucial ter os arquivos .py das classes customizadas na mesma pasta.
# Embora o Gerador não as use diretamente, o Keras pode precisar delas para carregar metadados.
try:
    from no import No
    from caminho import Caminho
    CUSTOM_OBJECTS = {"No": No, "Caminho": Caminho}
except ImportError:
    print("Aviso: Arquivos 'no.py' e 'caminho.py' não encontrados. Continuando sem eles.")
    CUSTOM_OBJECTS = {}


# Encontra o arquivo de modelo mais recente na pasta
LATEST_EPOCH = 0
MODEL_FILE = ''
for f in os.listdir('.'):
    if "gerador_epoca_" in f:
        epoch_num = int(f.split('_')[-1].split('.')[0])
        if epoch_num > LATEST_EPOCH:
            LATEST_EPOCH = epoch_num
            MODEL_FILE = f

if not MODEL_FILE:
    print("ERRO: Nenhum arquivo de modelo .keras encontrado na pasta.")
    exit()

# Carrega o modelo Gerador
try:
    print(f"Carregando modelo do checkpoint: {MODEL_FILE}")
    gerador_modelo = load_model(MODEL_FILE, custom_objects=CUSTOM_OBJECTS)
    print("Modelo Gerador carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# Carrega o tokenizer (nosso dicionário)
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    print("ERRO: Arquivo 'tokenizer.pickle' não encontrado.")
    exit()

# --- Bloco 2: Configurações e Função de Geração ---

MAX_LEN = 6
TAMANHO_RUIDO = 100
mapa_classes = {
    0: "um avião", 1: "um automóvel", 2: "um pássaro", 3: "um gato", 4: "um cervo",
    5: "cachorro", 6: "sapo", 7: "cavalo", 8: "navio", 9: "caminhão"
}

def gerar_imagem(frase):
    sequencia = tokenizer.texts_to_sequences([frase])
    padded_sequencia = pad_sequences(sequencia, maxlen=MAX_LEN, padding='post')
    ruido = tf.random.normal([1, TAMANHO_RUIDO])
    imagem_gerada = gerador_modelo.predict([padded_sequencia, ruido], verbose=0)
    imagem_desnormalizada = (imagem_gerada * 127.5) + 127.5
    return imagem_desnormalizada[0].astype('uint8')

# --- Bloco 3: Loop Interativo de Teste ---

print("\n--- Gerador de Imagens MyceliumAI (Local) ---")
print("Digite um dos seguintes rótulos para gerar uma imagem:")
print(f"Opções: {', '.join(mapa_classes.values())}")
print("Ou digite 'sair' para terminar.")

while True:
    texto_usuario = input("\n> O que você quer que eu desenhe? ").lower()
    
    if texto_usuario == 'sair':
        break
        
    if texto_usuario not in [v.replace("um ","") for v in mapa_classes.values()]:
         # Adicionei uma pequena correção para aceitar 'gato' em vez de 'um gato'
        if "um " + texto_usuario in mapa_classes.values():
            texto_usuario = "um " + texto_usuario
        else:
            print("Desculpe, eu só sei desenhar as 10 classes do CIFAR-10 por enquanto.")
            continue

    print(f"Ok, tentando desenhar: '{texto_usuario}'...")
    imagem_final = gerar_imagem(texto_usuario)
    
    # Exibe a imagem gerada
    plt.figure(figsize=(4,4))
    plt.imshow(imagem_final)
    plt.title(f"Minha tentativa de desenhar: {texto_usuario}")
    plt.axis('off')
    plt.show()

print("\nCriado com sucesso!")