import csv
import re
import unicodedata
import zipfile
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

import pymupdf

BASE = Path(__file__).resolve().parents[1]
CORPUS = BASE / "corpus"
SALIDA = BASE / "resultados" / "auditoria_pdf_markdown_101.csv"

def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(
        c for c in texto if not unicodedata.combining(c)
    )
    return " ".join(re.findall(r"[a-z0-9]+", texto))

def mejor_coincidencia(bloque, markdown):
    palabras = normalizar(bloque).split()
    texto = normalizar(markdown).split()

    if not palabras or not texto:
        return 0.0, ""

    longitud = len(palabras)
    mejor = (0.0, "")

    paso = max(1, longitud // 8)

    for i in range(0, len(texto), paso):
        segmento = texto[i:i + longitud + 12]
        if not segmento:
            continue

        similitud = SequenceMatcher(
            None, palabras, segmento,
            autojunk=False
        ).ratio()

        if similitud > mejor[0]:
            mejor = (similitud, " ".join(segmento))

    return mejor

with (CORPUS / "inventario.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    inventario = list(csv.DictReader(f))

resultados = []

with zipfile.ZipFile(CORPUS / "FDS EVALUADAS.zip") as z:
    nombres = {
        Path(n).name.casefold(): n
        for n in z.namelist()
        if n.lower().endswith(".pdf")
    }

    for fila in inventario:
        numero = int(fila["fds_num"])
        pdf_nombre = nombres[
            Path(fila["fuente_pdf"]).name.casefold()
        ]

        with pymupdf.open(
            stream=z.read(pdf_nombre), filetype="pdf"
        ) as pdf:
            bloques = []

            for pagina_num, pagina in enumerate(pdf, 1):
                for b in pagina.get_text("blocks"):
                    texto = b[4].strip()
                    if len(normalizar(texto).split()) >= 20:
                        bloques.append((pagina_num, texto))

            if bloques:
                pagina_num, bloque = max(
                    bloques,
                    key=lambda x: len(normalizar(x[1]).split())
                )
                bloque = " ".join(bloque.split())[:1000]
            else:
                pagina_num, bloque = None, ""

        markdown = (
            CORPUS / fila["md"]
        ).read_text(encoding="utf-8")

        puntuacion, fragmento = mejor_coincidencia(
            bloque, markdown
        )

        estado = (
            "COINCIDENCIA AUTOMATICA ALTA"
            if puntuacion >= 0.80
            else "REQUIERE REVISION"
        )

        resultados.append({
            "fds_num": numero,
            "documento": fila["fuente_pdf"],
            "pagina_pdf": pagina_num,
            "bloque_pdf": bloque,
            "fragmento_markdown_aproximado": fragmento[:1200],
            "similitud": round(puntuacion, 4),
            "estado_automatico": estado,
            "verificacion_manual": "PENDIENTE",
            "observaciones": ""
        })

SALIDA.parent.mkdir(parents=True, exist_ok=True)

with SALIDA.open(
    "w", encoding="utf-8", newline=""
) as f:
    escritor = csv.DictWriter(
        f, fieldnames=resultados[0].keys()
    )
    escritor.writeheader()
    escritor.writerows(resultados)

print("=== RESULTADOS PRELIMINARES ===")
for estado, cantidad in Counter(
    r["estado_automatico"] for r in resultados
).items():
    print(f"{estado}: {cantidad}")

print("\n=== CASOS PARA REVISAR ===")
for r in resultados:
    if r["estado_automatico"] == "REQUIERE REVISION":
        print(
            f"FDS {r['fds_num']}: "
            f"similitud={r['similitud']}"
        )

print("\nCSV GENERADO:", SALIDA)
