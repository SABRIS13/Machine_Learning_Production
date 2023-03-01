import argparse

# Crear argumento para parsear
parser = argparse.ArgumentParser(description="Script para sumar dos numeros")

# Suma de dos números
parser.add_argument("num1", type=float, help="Primer numero")
parser.add_argument("num2", type=float, help="Segundo numero")

# Parsear argumentos
args = parser.parse_args()

# Sumar los dos números
resultado = args.num1 + args.num2
print(resultado)