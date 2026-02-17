# Carregar variáveis de ambiente do arquivo .env 
import dotenv

#  Manipular arquivos e diretórios
from pathlib import Path

# Biblioteca para trabalhar com leitura de documentos PDF e armezenamento de dados no banco de dados vetorial Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Biblioteca para criar embeddings usando o modelo de linguagem da OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Carregar as variáveis de ambiente do arquivo .env
config = dotenv.dotenv_values()
# Pasta onde o banco de dados vetorial do Chroma será armazenado
db_dir = Path(config['CHROMA_DB_PATH'])

# Criar o objeto de embeddings usando o modelo de linguagem da OpenAI
embedding = OpenAIEmbeddings(
    model=config['EMBEDDINGS_MODEL'],
    openai_api_key=config['OPENAI_API_KEY']
)


# Função para ler um arquivo PDF e retornar uma lista de documentos
def ler_pdf(caminho: str) -> list[Document]:
    loader = PyPDFLoader(caminho)
    return loader.load()


# Função para atribuir uma fonte/metadado a cada documento
def configurar_metadado(documento: Document, fonte: str) -> Document:
    # Configurar o metadado do documento para incluir a fonte
    documento.metadata['fonte'] = fonte
    return documento


# Função para dividir os documentos em pedaços menores usando um text splitter baseado em tamanho
def quebra_por_tamanho(documentos: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documentos)


# Função para dividir os documentos em pedaços menores usando um text splitter baseado em parágrafos
def quebra_por_paragrafo(documentos: list[Document]) -> list[Document]:
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documentos)


# Função para carregar os documentos dos arquivos PDF, configurar os metadados e retornar um dicionário com os documentos organizados por fonte
def carrega_documentos() -> dict[str, list[Document]]:
   documentos_cdc = [configurar_metadado(doc, 'CDC') for doc in ler_pdf(config['CDC_PATH'])]
   documentos_lgpd = [configurar_metadado(doc, 'LGPD') for doc in ler_pdf(config['LGPD_PATH'])]

   total_cdc = len(documentos_cdc)
   total_lgpd = len(documentos_lgpd)
   total_de_documentos = total_cdc + total_lgpd
   print(f'Documentos CDC: {total_cdc}. Documentos LGPD: {total_lgpd}. Total de documentos: {total_de_documentos}.')

   return {
         'CDC': documentos_cdc,
         'LGPD': documentos_lgpd
   }


# Função para criar os chunks dos documentos usando as funções de quebra por tamanho e por parágrafo, e retornar um dicionário com os chunks organizados por fonte
def criar_chunks(documentos: dict[str, list[Document]]) -> dict[str, list[Document]]:
    chunks_cdc = quebra_por_tamanho(documentos['CDC'])
    chunks_lgpd = quebra_por_paragrafo(documentos['LGPD'])

    total_chunks_cdc = len(chunks_cdc)
    total_chunks_lgpd = len(chunks_lgpd)
    total_de_chunks = total_chunks_cdc + total_chunks_lgpd
    print(f'Chunks CDC: {total_chunks_cdc}. Chunks LGPD: {total_chunks_lgpd}. Total de chunks: {total_de_chunks}.')

    return {
        'CDC': chunks_cdc,
        'LGPD': chunks_lgpd
    }


# Função para carregar o banco de dados vetorial do Chroma, 
# verificando se o diretório do banco de dados existe e, caso contrário, criando o banco de dados a partir dos documentos
def carregar_banco_vetorial() -> Chroma:
    # Se o diretório do banco de dados vetorial existe, carrega os dados
    if db_dir.exists():
        print(f'Banco de dados vetorial já existe. Carregando ...\n')
        return Chroma(embedding_function=embedding, persist_directory=str(db_dir), collection_name='documentos_juridicos')
    
    # Se o diretório do banco de dados vetorial não existe, cria o banco de dados a partir dos documentos
    print(f'Banco de dados vetorial não encontrado. Criando banco de dados a partir dos documentos ...\n')
    
    documentos = carrega_documentos()
    
    chunks = criar_chunks(documentos)
    todos_os_chunks = chunks['CDC'] + chunks['LGPD']

    return Chroma.from_documents(todos_os_chunks, embedding, persist_directory=str(db_dir), collection_name='documentos_juridicos')
    

if __name__ == "__main__":
    # Carregar o banco de dados vetorial do Chroma
    banco  = carregar_banco_vetorial()
    
    # Exemplo de consulta para recuperar os documentos mais relevantes com base em uma pergunta
    documentos_recuperados = banco.similarity_search("Em que casos o consentimento é obrigatório?", k=5)
    for i, doc in enumerate(documentos_recuperados, start=1):
        print(f"Documento {i}:")
        print(f"Metadados/Fonte: {doc.metadata['fonte']}")
        print(f"Conteúdo: {doc.page_content}")  # Imprime os primeiros 200 caracteres do conteúdo
        print("---\n")