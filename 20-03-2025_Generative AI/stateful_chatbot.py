from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate


class StatefulChatbot:
    def __init__(
        self,
        base_url="huge-ape-apparent.ngrok-free.app",
        model="gemma3:4b",
    ):
        # Inizializziamo il modello LLM usando Ollama
        # Questo sarà il nostro "cervello" del chatbot
        self.llm = Ollama(model=model, base_url=base_url)

        # Creiamo una lista vuota per memorizzare la cronologia delle conversazioni
        # Ogni elemento sarà una tupla (mittente, messaggio)
        self.conversation_history = []

        # Definiamo il template per il prompt che include la cronologia delle conversazioni
        template = """Sei un assistente AI amichevole e disponibile.

        Ecco la cronologia della nostra conversazione:

        {conversation_history}

        Rispondi alla seguente domanda in modo utile e conciso, tenendo in considerazione

        la conversazione precedente.

        Domanda: {input}

        Risposta:"""

        # Creiamo il prompt template con le nuove variabili di input
        self.prompt = PromptTemplate(
            input_variables=["conversation_history", "input"], template=template
        )

    def format_conversation_history(self):
        """
        Formatta la cronologia della conversazione in modo leggibile per il modello.
        Returns:
            str: La cronologia formattata
        """
        # Se non c'è cronologia, restituiamo una stringa vuota
        if not self.conversation_history:

            return "Nessuna conversazione precedente."

        # Altrimenti, formattiamo ogni scambio
        formatted_history = ""
        for sender, message in self.conversation_history:
            formatted_history += f"{sender}: {message}\n"

        return formatted_history

    def chat(self, user_input: str) -> str:
        """
        Gestisce una singola interazione con l'utente, tenendo conto della cronologia.
        Args:
            user_input (str): Il messaggio dell'utente
        Returns:
            str: La risposta del chatbot
        """
        try:
            # Aggiungiamo il messaggio dell'utente alla cronologia
            self.conversation_history.append(("Utente", user_input))

            # Formattiamo la cronologia della conversazione
            formatted_history = self.format_conversation_history()

            # Formattiamo il prompt con l'input dell'utente e la cronologia
            formatted_prompt = self.prompt.format(
                input=user_input, conversation_history=formatted_history
            )

            # Generiamo la risposta usando l'LLM
            response = self.llm.invoke(formatted_prompt)
            response = response.strip()

            # Aggiungiamo la risposta del chatbot alla cronologia
            self.conversation_history.append(("Chatbot", response))
            return response
        except Exception as e:
            error_message = f"Mi dispiace, si è verificato un errore: {str(e)}"

            # Aggiungiamo anche il messaggio di errore alla cronologia
            self.conversation_history.append(("Chatbot", error_message))

            return error_message

    def clear_history(self):
        """
        Cancella la cronologia della conversazione.
        Utile per iniziare una nuova conversazione.
        """
        self.conversation_history = []
        return "Cronologia della conversazione cancellata."


def main():
    # Creiamo un'istanza del chatbot con memoria
    print("Inizializzazione del chatbot con memoria...")
    chatbot = StatefulChatbot()

    print("\nChatbot pronto! Il chatbot ricorderà le conversazioni precedenti.")
    print("Scrivi '/exit' per uscire.")
    print("Scrivi '/clear' per cancellare la cronologia della conversazione.")

    while True:
        user_input = input("\nTu: ")
        command = user_input.lower()
        if command == "/exit":
            print("\nArrivederci!")
            break
        elif command == "/clear":
            message = chatbot.clear_history()
            print(f"\nChatbot: {message}")
        else:
            response = chatbot.chat(user_input)
            print(f"\nChatbot: {response}")


if __name__ == "__main__":
    main()
