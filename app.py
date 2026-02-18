import rag


def efetuar_pergunta():
    print('---\n')
    return input('# PERGUNTA (digite "sair" para encerrar): \n')


def imprimir_fontes(resposta: dict[str, list[str]]):
    print('## FONTES:')

    for fonte in resposta['fontes']:
        print(f'  - {fonte}')


def iniciar_chat():
    print('### BEM-VINDO AO ASSISTENTE JURÍDICO! FAÇA SUAS PERGUNTAS SOBRE O CDC E A LGPD. ###\n')

    while True:
        pergunta = efetuar_pergunta()

        if pergunta.strip().lower() == 'sair':
            print('### OBRIGADO POR USUAR O ASSISTENTE JURÍDICO! ATÉ LOGO! ###')
            break

        # Executa o método de RAG tradicional
        # resposta = rag.executar_prompt(pergunta)
        # print(f'\n# RESPOSTA TRADICIONAL:\n{resposta["resultado"]}\n')
        # Executa o método de Reranking manual
        resposta = rag.executa_prompt_reranking(pergunta)
        print(f'\n# RESPOSTA RERANKING:\n{resposta["resultado"]}\n')

        if resposta['fontes']:
            imprimir_fontes(resposta)
        else:
            print('Nenhuma fonte relevante encontrada para esta pergunta.')


if __name__ == "__main__":
    iniciar_chat()