import numpy as np
from sentence_transformers import SentenceTransformer, util

class LoadingModel:
    _instance = None
    _model = None
    
    def __new__(cls, filename):
        if cls._instance is None:
            cls._instance = super(LoadingModel, cls).__new__(cls)
            cls._instance._data = cls.loading_model(filename)
        return cls._instance
    
    def __init__(self, filename):
        self.filename = filename
    
    def get_data(self):
        return self._data

    @staticmethod
    def loading_file(arquivo, janela=3):
        """Lê o manual e divide em trechos maiores para manter o contexto"""
        with open(arquivo, "r", encoding="utf-8") as file:
            texto = file.read()

        # Dividir em parágrafos, mantendo trechos maiores
        paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
        
        # Criar "janelas deslizantes" de 3 parágrafos cada para manter contexto
        partes = []
        for i in range(len(paragrafos)):
            trecho = " ".join(paragrafos[i : i + janela])  # Junta 3 parágrafos
            partes.append(trecho)

        return partes

    @staticmethod
    def loading_model(filename):
        # Carregar modelo de embeddings
        if LoadingModel._model is None:
            LoadingModel._model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        manual = LoadingModel.loading_file(filename)
        embeddings_partes = LoadingModel.encode_data(manual)
        
        return embeddings_partes

    @staticmethod
    def match(pergunta):
        if LoadingModel._model is None:
            raise ValueError("O modelo ainda não foi carregado. Inicialize a classe primeiro.")
        
        embedding_pergunta = LoadingModel.encode_data(pergunta)
        embeddings_partes = LoadingModel._instance._data
        
        # Calcular similaridade de cosseno
        similaridades = util.pytorch_cos_sim(embedding_pergunta, embeddings_partes)[0]
        indice_mais_proximo = np.argmax(similaridades)

        # Se a similaridade for muito baixa, indicar que não encontrou resposta
        if similaridades[indice_mais_proximo] < 0.3:
            return "Não encontrei uma resposta no manual para essa pergunta."
        
        manual = LoadingModel.loading_file(LoadingModel._instance.filename)
        return manual[indice_mais_proximo]

    @staticmethod
    def encode_data(data):
        if LoadingModel._model is None:
            raise ValueError("O modelo ainda não foi carregado. Inicialize a classe primeiro.")

        return LoadingModel._model.encode(data, convert_to_tensor=True)