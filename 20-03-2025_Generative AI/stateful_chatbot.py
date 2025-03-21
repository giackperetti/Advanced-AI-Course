from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

class StatefulChatbot:
    def __init__(self, base_url="https://huge-ape-apparent.ngrok-free.app", model="gemma3:4b"):
        # Inizializziamo il modello LLM usando Ollama
        self.llm = Ollama(model=model, base_url=base_url)
        # Definiamo il template per il prompt
        template = """Sei un assistente AI amichevole e disponibile.
        Rispondi alle domande con fare amichevole e stralunato.

        {contesto}

        Domanda: {input}

        Risposta:"""
        # Creiamo il prompt template che verrà usato per ogni interazione
        self.prompt = PromptTemplate(input_variables=["input", "contesto"], template=template)
        # Inizializziamo la memoria come lista di dizionari
        self.memory = []

    def chat(self, user_input: str) -> str:
        """
        Gestisce un'interazione con l'utente mantenendo la memoria della conversazione.
        Args:
        user_input (str): Il messaggio dell'utente
        Returns:
        str: La risposta del chatbot
        """
        try:
            # Costruzione del contesto dalla memoria
            contesto = ""
            for entry in self.memory:
                contesto += f"Utente: {entry['user']}\\nAssistente: {entry['bot']}\\n"

            # Formattiamo il prompt con l'input attuale e il contesto
            formatted_prompt = self.prompt.format(input=user_input, contesto=contesto)

            # Generiamo la risposta usando l'LLM
            response = self.llm.invoke(formatted_prompt)

            # Aggiorniamo la memoria con l'interazione corrente
            self.memory.append({'user': user_input, 'bot': response.strip()})

            return response.strip()
        except Exception as e:
            return f"Mi dispiace, si è verificato un errore: {str(e)}"

    def clear_memory(self):
        """Pulisce la memoria della conversazione."""
        self.memory = []

    def get_memory(self):
        """Ritorna la memoria corrente della conversazione."""
        return self.memory

def main():
    print("Inizializzazione del chatbot stateful con memoria...")
    chatbot = StatefulChatbot()
    print("\nChatbot pronto! La conversazione sarà memorizzata.")
    print("Scrivi 'exit' per uscire o 'clear' per resettare la memoria.")
    while True:
        user_input = input("\nTu: ")
        if user_input.lower() == "/exit":
            print("\nArrivederci!")
            break
        elif user_input.lower() == "/clear":
            chatbot.clear_memory()
            print("\nMemoria resettata!")
            continue
        response = chatbot.chat(user_input)
        print(f"\n{chatbot.llm.model}: {response}")

if __name__ == "__main__":
    main()
