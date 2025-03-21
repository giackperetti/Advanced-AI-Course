def is_palindrome(word: str) -> bool:
    return word == word[::-1]

if __name__ == '__main__':
    word = input("Inserisci una parola: ")
    print("La parola è palindroma" if (is_palindrome(word)) else "La parola non è palindroma")