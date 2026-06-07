import os
from aider_files.requirements import requirements

def main():
    # Verificar se a pasta storage existe e cria-la se não existir
    if not os.path.exists('storage'):
        os.makedirs('storage')

    # Imprimir as dependências necessárias
    print("Requisitos:")
    for requirement in requirements:
        print(f"- {requirement}")

if __name__ == "__main__":
    main()
