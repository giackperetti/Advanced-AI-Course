import cmath

try:
    a, b, c = map(
        float,
        input(
            "Inserisci i coefficienti di una equazione di secondo grado separati da una virgola\nax^2 + bx + c: "
        ).split(","),
    )

    if a == 0:
        print("L'equazione non è di secondo grado perché a = 0.")
    else:
        delta = (b**2) - (4 * a * c)
        x1 = (-b + cmath.sqrt(delta)) / (2 * a)
        x2 = (-b - cmath.sqrt(delta)) / (2 * a)
        print(f"Le soluzioni reali: x1 = {x1}, x2 = {x2}")
except ValueError:
    print("Errore: assicurati di inserire tre numeri separati da una virgola.")
