import re
import string
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar modelo NLP do spaCy para remover stopwords
nlp = spacy.load("pt_core_news_md")

def preprocessar(texto):
    texto = texto.lower()  # Converte para minúsculas
    texto = re.sub(f"[{string.punctuation}]", "", texto)  # Remove pontuação
    doc = nlp(texto)
    tokens = [token.lemma_ for token in doc if not token.is_stop]  # Remove stopwords e lematiza
    return " ".join(tokens)

comandos = {
    "lançar uma despesa": "abrir_tela_lancamento",
    "cadastrar uma despesa": "abrir_tela_lancamento",
    "consultar saldo": "mostrar_saldo",
    "ver meu extrato": "mostrar_extrato"
}

# Aplicar pré-processamento nos comandos conhecidos
comandos_processados = {preprocessar(k): v for k, v in comandos.items()}

# Treinamos o modelo com as frases conhecidas
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(comandos_processados.keys())

def interpretar_comando(comando_usuario):
    comando_usuario = preprocessar(comando_usuario)  # Pré-processa a entrada
    entrada_vectorizada = vectorizer.transform([comando_usuario])

    # Calcula similaridade com os comandos conhecidos
    similaridades = cosine_similarity(entrada_vectorizada, X)
    
    indice_mais_proximo = similaridades.argmax()
    comando_mais_proximo = list(comandos_processados.keys())[indice_mais_proximo]

    return comandos_processados[comando_mais_proximo]

# Teste com uma frase parecida com as conhecidas
entrada_usuario = "Quero comprar uma casa"
acao = interpretar_comando(entrada_usuario)
print(f"Ação correspondente: {acao}")