from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

class StatelessChatbot:
    def __init__(self, base_url="https://huge-ape-apparent.ngrok-free.app", model="gemma3:4b"):
        # Inizializziamo il modello LLM usando Ollama
        # Questo sarà il "cervello" del chatbot
        self.llm = Ollama(model=model, base_url=base_url)
        # Definiamo il template per il prompt
        template = """Sei un assistente AI amichevole e disponibile.
        Rispondi alle domande con fare amichevole e stralunato.

        Domanda: {input}

        Risposta:"""
        # Creiamo il prompt template che verrà usato per ogni interazione
        self.prompt = PromptTemplate(input_variables=["input", "contesto"], template=template)

    def chat(self, user_input: str) -> str:
        """
        Gestisce una singola interazione con l'utente.
        Ogni chiamata è completamente indipendente dalle precedenti.
        Args:
        user_input (str): Il messaggio dell'utente
        Returns:
        str: La risposta del chatbot
        """
        try:
            # Formattiamo il prompt con l'input dell'utente
            formatted_prompt = self.prompt.format(input=user_input, contesto=self.contesto)
            # Generiamo la risposta usando l'LLM
            response = self.llm.invoke(formatted_prompt)
            return response.strip()
        except Exception as e:
            return f"Mi dispiace, si è verificato un errore: {str(e)}"

def main():
    # Creiamo un'istanza del chatbot
    print("Inizializzazione del chatbot stateless...")
    chatbot = StatelessChatbot()
    print("\nChatbot pronto! Ogni messaggio sarà trattato in modo indipendente.")
    print("Scrivi 'exit' per uscire.")
    while True:
        user_input = input("\nTu: ")
        if user_input.lower() == "exit":
            print("\nArrivederci!")
            break
        response = chatbot.chat(user_input)
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    main()