listaIN = [2, -4, 5, 6, 5, 5, 2]
listaOUT_controllo=[2,-4,5,6]

listaOUT = list(set(listaIN))

print(f"La lista output è: {listaOUT}")
print("la lista risultante è corretta" if(listaOUT == listaOUT_controllo) else f"Expected: {listaOUT_controllo}, output: {listaOUT}")
