import json
import re
import string
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar modelo NLP do spaCy para remover stopwords
nlp = spacy.load("pt_core_news_sm")

# Carregar comandos do JSON
def carregar_comandos(arquivo_json):
    with open(arquivo_json, "r", encoding="utf-8") as f:
        return json.load(f)
    
comandos = carregar_comandos("database.json")

def preprocessar(texto):
    texto = texto.lower()  # Converte para minúsculas
    texto = re.sub(f"[{string.punctuation}]", "", texto)  # Remove pontuação
    doc = nlp(texto)
    tokens = [token.lemma_ for token in doc if not token.is_stop]  # Remove stopwords e lematiza
    return " ".join(tokens)

# Aplicar pré-processamento nos comandos conhecidos
comandos_processados = {preprocessar(k): v for k, v in comandos.items()}

# Treinamos o modelo com as frases conhecidas
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(comandos_processados.keys())

def salvar_comandos(arquivo_json, novos_comandos):
    """ Atualiza o arquivo JSON com novos comandos """
    try:
        with open(arquivo_json, "r", encoding="utf-8") as f:
            comandos = json.load(f)  # Carrega os comandos existentes
    except FileNotFoundError:
        comandos = {}  # Se o arquivo não existir, inicia um dicionário vazio

    comandos.update(novos_comandos)  # Adiciona os novos comandos ao dicionário
    with open(arquivo_json, "w", encoding="utf-8") as f:
        json.dump(comandos, f, ensure_ascii=False, indent=4)  # Salva no JSON formatado

def interpretar_comando(comando_usuario, limite_similaridade=0.3):
    comando_usuario = preprocessar(comando_usuario)  # Pré-processa a entrada
    entrada_vectorizada = vectorizer.transform([comando_usuario])

    # Calcula similaridade com os comandos conhecidos
    similaridades = cosine_similarity(entrada_vectorizada, X)
    
    indice_mais_proximo = similaridades.argmax()
    maior_similaridade = similaridades[0, indice_mais_proximo]

    # Verifica se a similaridade é maior que o limite mínimo
    if maior_similaridade < limite_similaridade:
        return f"Comando não reconhecido: similaridade = {maior_similaridade}"

    comando_mais_proximo = list(comandos_processados.keys())[indice_mais_proximo]
    return comandos_processados[comando_mais_proximo]


while True:
    entrada = input("Digite um comando (ou 'sair' para encerrar, 'novo' para cadastrar um comando): ").strip().lower()

    if entrada == "sair":
        print("Encerrando o programa...")
        break

    if entrada == "novo":
        novo_comando = input("Digite o novo comando: ").strip().lower()
        nova_acao = input("Digite a ação correspondente: ").strip().lower()
        
        salvar_comandos("database.json", {novo_comando: nova_acao})
        print(f"Novo comando '{novo_comando}' salvo com ação '{nova_acao}'!")
        
        # Recarrega os comandos para considerar a nova entrada
        comandos = carregar_comandos("database.json")
        comandos_processados = {preprocessar(k): v for k, v in comandos.items()}
        X = vectorizer.fit_transform(comandos_processados.keys())  # Atualiza os vetores TF-IDF
        continue

    acao = interpretar_comando(entrada)  # Chama a função que processa o comando
    print(f"Ação correspondente: {acao}\n\n")