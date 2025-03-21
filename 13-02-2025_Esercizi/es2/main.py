import operazioni


def main():
    numeri = list(
        map(int, input("Inserisci numeri separati da una virgola: ").split(","))
    )
    somma_pari, prodotto_dispari = operazioni.somma_pari_prodotto_dispari(*numeri)

    print(f"Somma dei numeri pari: {somma_pari}")
    print(
        f"Prodotto dei numeri dispari: {prodotto_dispari}"
        if (prodotto_dispari != 0)
        else "Non ci sono numeri dispari da moltiplicare"
    )


if __name__ == "__main__":
    main()
