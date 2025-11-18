from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

CAMINHO_DB = "db"

prompt_template = ""

def perguntar():
    pergunta = input("Escreva sua pergunta: ")

    # carregar o banco de dados
    funcao_embedding = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    # comparar a pergunta do usuário (embedding) com o meu banco de dados
    resultados = db.similarity_search_with_relevance_scores(pergunta)
    if len(resultados) == 0 or resultados[0][1] < 0.7:
        print("Não conseguiu encontrar alguma informação relevante na base")
        return 
    
    textos_resultado = []
    for resultado in resultados:
        texto = resultado.page_content
        textos_resultado.append(texto)
    
    base_conhecimento = "\n\n----\n\n".join(textos_resultado)
    prompt = ChatPromptTemplate(prompt_template)
    prompt = prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})
    
    modelo = OllamaLLM(model="llama3.2")
    texto_resposta = modelo.invoke(prompt)
    print("Resposta da IA:", texto_resposta)

perguntar()