import os
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
langchain_token = os.environ["LANGSMITH_API"] 
DATA_PATH = "data/"

vectordb_file_path = "faiss_index"


model_id = "meta-llama/Llama-3.2-3B-Instruct"


model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


llm = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512,
)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def create_vector_database(): 
    chunks = split_documents(load_documents())
    vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



from langchain_core.runnables import Runnable

class Llama3PromptRunnable(Runnable):
    def __init__(self, system=""):
        super().__init__()
        self.system = system

    def invoke(self, inputs: dict, config=None) -> str:
        question = inputs["question"]
        context = inputs["context"]
        # Create the system prompt if provided
        system_prompt = ""
        if self.system != "":
            system_prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n\n{self.system}\n\n"
                f"context: {context}\n\n"
                f"<|eot_id|>\n\n"
            )
            prompt = (
                f"<|begin_of_text|>{system_prompt}"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{question}\n\n"
                f"<|eot_id|>\n\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n" # header - assistant
            )

        # Return the formatted prompt
        return prompt

llama_prompt = Llama3PromptRunnable(system="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.")


from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings,allow_dangerous_deserialization=True)
    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} #output {"context":"XX", "question": "YY"}
        | llama_prompt #take input from previous runable, {"context":"XX", "question": "YY"}, and return prompt str
        | llm #take input from previous runable, prompt str, to generate the answer
        | StrOutputParser() #take input from previous runable, answer, to parser output
    )

    return rag_chain




if __name__ == '__main__':
    create_vector_database()
    chain = get_qa_chain()
    print(chain.invoke("how to win this game?"))
