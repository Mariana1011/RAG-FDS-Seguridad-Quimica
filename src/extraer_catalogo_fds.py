import csv
import re
import zipfile
from pathlib import Path

import pymupdf

BASE = Path(__file__).resolve().parents[1]
CORPUS = BASE / "corpus"
SALIDA = BASE / "resultados" / "catalogo_fds_preliminar.csv"

def extraer(texto, patrones):
    for patron in patrones:
        resultado = re.search(patron, texto, re.I | re.M)
        if resultado:
            return " ".join(resultado.group(1).split()).strip()
    return ""

patrones_producto = [
    r"Nombre del Producto:\s*\n?\s*([^\n]+)",
    r"1\.1\s+Identificador SGA del producto:\s*\n?\s*([^\n]+)",
]

patrones_fabricante = [
    r"Empresa:\s*\n?\s*([^\n]+)",
    r"1\.3\s+Datos sobre el proveedor:\s*\n?\s*([^\n]+)",
]

patrones_version = [
    r"Versi[oó]n:\s*([^\n]+)",
    r"Versi[oó]n\s+n[uú]mero:\s*([^\n]+)",
]

with (CORPUS / "inventario.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    inventario = list(csv.DictReader(f))

registros = []

with zipfile.ZipFile(CORPUS / "FDS EVALUADAS.zip") as z:
    pdfs = {
        Path(nombre).name.casefold(): nombre
        for nombre in z.namelist()
        if nombre.lower().endswith(".pdf")
    }

    for fila in inventario:
        numero = int(fila["fds_num"])
        nombre = Path(fila["fuente_pdf"]).name.casefold()

        with pymupdf.open(
            stream=z.read(pdfs[nombre]),
            filetype="pdf"
        ) as pdf:
            texto = pdf[0].get_text("text")

        registros.append({
            "fds_num": numero,
            "documento": fila["fuente_pdf"],
            "producto_candidato": extraer(texto, patrones_producto),
            "fabricante_candidato": extraer(texto, patrones_fabricante),
            "version_candidata": extraer(texto, patrones_version),
            "pagina_evidencia": 1,
            "estado": "PENDIENTE DE VERIFICACION VISUAL",
            "observaciones": ""
        })

SALIDA.parent.mkdir(parents=True, exist_ok=True)

with SALIDA.open("w", encoding="utf-8", newline="") as f:
    escritor = csv.DictWriter(
        f, fieldnames=registros[0].keys()
    )
    escritor.writeheader()
    escritor.writerows(registros)

print("=== COBERTURA PRELIMINAR ===")
print("FDS procesadas:", len(registros))

for campo in [
    "producto_candidato",
    "fabricante_candidato",
    "version_candidata"
]:
    cantidad = sum(bool(r[campo]) for r in registros)
    print(f"{campo}: {cantidad}/101")

print("\n=== EJEMPLOS ===")
for numero in [1, 2, 59, 90]:
    r = next(x for x in registros if x["fds_num"] == numero)
    print(f"\nFDS {numero}")
    print("Producto:", r["producto_candidato"])
    print("Fabricante:", r["fabricante_candidato"])
    print("Versión:", r["version_candidata"])

print("\nCATÁLOGO GENERADO:", SALIDA)
