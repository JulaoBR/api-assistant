import os

from src.LoadingModel import LoadingModel

def encontrar_resposta(pergunta):
    """ Encontra o melhor trecho do manual baseado na pergunta do usuário """
    # Criar uma instância da classe
    loader = LoadingModel("manual.txt")
    
    return loader.match(pergunta)

def limpar_terminal():
    """Limpa o terminal"""
    os.system("cls" if os.name == "nt" else "clear")

# 🔄 Loop do menu
while True:
    limpar_terminal()
    print("📌 MENU:")
    print("1️⃣ Fazer uma pergunta")
    print("2️⃣ Treinar modelo (recarregar manual)")
    print("3️⃣ Sair")

    opcao = input("\nEscolha uma opção: ").strip()

    if opcao == "1":
        limpar_terminal()
        while True:
            pergunta = input("Digite sua pergunta (ou 'voltar' para o menu): ").strip()
            if pergunta.lower() == "voltar":
                break
            resposta = encontrar_resposta(pergunta)
            print("\n📖 Resposta do manual:\n", resposta, "\n")
            input("Pressione ENTER para continuar...")  # Aguarda antes de continuar

    elif opcao == "2":
        limpar_terminal()
        print("🔄 Recarregando manual e atualizando modelo...")
        LoadingModel.loading_model()
        print("✅ Modelo atualizado com sucesso!")
        input("\nPressione ENTER para voltar ao menu...")

    elif opcao == "3":
        limpar_terminal()
        print("👋 Saindo...")
        break

    else:
        print("⚠️ Opção inválida! Escolha novamente.")
        input("\nPressione ENTER para continuar...")