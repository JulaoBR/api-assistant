# api-assistantbot-assistant
# COMANDO PARA ATUALIZAR O REQUIREMENTS.TXT
pip freeze > requirements.txt

# INSTALAR BIBLIOTECAS A PARTIR DO REQUIREMENTS.TXT
pip install -r requirements.txt

# COMANDO PARA CRIAR UMA ENV
python3 -m venv venv

# COMANDO PARA INICIAR O AMBIENTE ENV
source venv/bin/activate

# MODELO PARA REMOVER stopwords
python -m spacy download pt_core_news_md || pt_core_news_sm