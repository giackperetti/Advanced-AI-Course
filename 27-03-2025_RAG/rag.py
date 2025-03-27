from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os


class RAGSystem:
    def __init__(self, model_url: str, model_name: str, doc_paths: list, embed_url: str, embed_model: str, persist_dir: str):
        """Inizializza il sistema RAG con il modello, i documenti e il database vettoriale."""
        self.model_url = model_url
        self.model_name = model_name
        self.doc_paths = doc_paths
        self.embed_url = embed_url
        self.embed_model = embed_model
        self.persist_dir = persist_dir

        self.model = self.load_model()
        self.documents = self.load_documents()
        self.vectorstore = self.create_vectorstore()
        self.retriever = self.create_retriever()
        self.rag_chain = self.create_rag_chain()

    def load_model(self):
        """Carica il modello LLM."""
        return Ollama(base_url=self.model_url, model=self.model_name, verbose=False)

    def load_documents(self):
        """Carica e suddivide i documenti in chunks."""
        documents = []
        for path in self.doc_paths:
            loader = TextLoader(path)
            documents.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)

    def create_vectorstore(self):
        """Crea o carica il database vettoriale."""
        embeddings = OllamaEmbeddings(base_url=self.embed_url, model=self.embed_model)
        if not os.path.exists(self.persist_dir):
            return Chroma.from_documents(documents=self.documents, embedding=embeddings, persist_directory=self.persist_dir)
        return Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)

    def create_retriever(self):
        """Crea il retriever per la ricerca nei documenti."""
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    def create_rag_chain(self):
        """Crea la catena RAG per rispondere alle domande."""
        template = """Rispondi e conversa solamente in italiano.
        Rispondi alla domanda prediligendo le informazioni fornite nel contesto.
        Se le informazioni nel contesto non sono sufficienti per rispondere o la domanda risulta scollegata dal contesto, rispondi con la tua conoscenza acquisita durante il tuo training.

        Contesto:
        {context}

        Domanda: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        """Formatta i documenti recuperati in una stringa."""
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str) -> str:
        """Interroga il sistema RAG con una domanda."""
        return self.rag_chain.invoke(question)

def main():
    rag = RAGSystem(
        model_url="https://huge-ape-apparent.ngrok-free.app",
        model_name="deepseek-r1:8b",
        doc_paths=["./dati/ReEric.txt"],
        embed_url="https://huge-ape-apparent.ngrok-free.app",
        embed_model="nomic-embed-text",
        persist_dir="db_vectoriale",
    )

    while True:
        question = input("\nFai una domanda (o scrivi '/exit' per uscire): ")
        if question.lower() == "/exit":
            break
        print("\nRisposta:", rag.query(question))

if __name__ == "__main__":
    main()