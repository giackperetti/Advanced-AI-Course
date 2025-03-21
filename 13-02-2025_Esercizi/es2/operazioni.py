def somma_pari_prodotto_dispari(*numeri):
    somma_pari = sum(n for n in numeri if n % 2 == 0)
    prodotto_dispari = 1
    almeno_un_dispari = False

    for n in numeri:
        if n % 2 != 0:
            prodotto_dispari *= n
            almeno_un_dispari = True

    if not almeno_un_dispari:
        prodotto_dispari = 0

    return somma_pari, prodotto_dispari
