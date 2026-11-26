import csv
import re
import zipfile
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

import pymupdf

BASE = Path(__file__).resolve().parents[1]
CORPUS = BASE / "corpus"
SALIDA = BASE / "resultados" / "auditoria_tecnica_101.csv"

def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(
        c for c in texto if not unicodedata.combining(c)
    )
    return " ".join(re.findall(r"[a-z0-9]+", texto))

PATRONES = [
    r"secci[oó]n\s+[234589]\b",
    r"\bprimeros auxilios\b",
    r"\bcomposici[oó]n\b",
    r"\bpropiedades f[ií]sicas\b",
    r"\bmedidas de seguridad\b",
    r"\bcas\s*[:\-]?\s*\d",
    r"\btemperatura\b",
    r"\bdensidad\b",
    r"\bpunto de inflamaci[oó]n\b",
    r"\btoxicidad\b",
]

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
        nombre = nombres[Path(fila["fuente_pdf"]).name.casefold()]

        with pymupdf.open(
            stream=z.read(nombre), filetype="pdf"
        ) as pdf:
            candidatos = []

            for pagina_num, pagina in enumerate(pdf, 1):
                for bloque in pagina.get_text("blocks"):
                    texto = " ".join(bloque[4].split())

                    if len(texto) < 100:
                        continue

                    coincidencias = sum(
                        bool(re.search(p, texto, re.I))
                        for p in PATRONES
                    )

                    if coincidencias:
                        candidatos.append(
                            (coincidencias, pagina_num, texto)
                        )

            if candidatos:
                _, pagina_num, bloque = max(
                    candidatos,
                    key=lambda x: (x[0], len(x[2]))
                )
                seleccion = "TÉCNICO"
            else:
                pagina_num, bloque = 1, " ".join(
                    pdf[0].get_text().split()
                )[:1000]
                seleccion = "RESPALDO"

        md = (CORPUS / fila["md"]).read_text(
            encoding="utf-8"
        )

        bloque_norm = normalizar(bloque)
        md_norm = normalizar(md)

        palabras = bloque_norm.split()
        corpus_palabras = md_norm.split()

        mejor = 0.0
        ventana_mejor = ""

        tam = len(palabras)
        if tam:
            paso = max(1, tam // 8)
            for inicio in range(0, len(corpus_palabras), paso):
                ventana = corpus_palabras[inicio:inicio + tam + 12]
                puntuacion = SequenceMatcher(
                    None, palabras, ventana,
                    autojunk=False
                ).ratio()
                if puntuacion > mejor:
                    mejor = puntuacion
                    ventana_mejor = " ".join(ventana)

        resultados.append({
            "fds_num": numero,
            "documento": fila["fuente_pdf"],
            "pagina_pdf": pagina_num,
            "tipo_bloque": seleccion,
            "bloque_pdf": bloque[:2000],
            "fragmento_md": ventana_mejor[:2000],
            "similitud": round(mejor, 4),
            "estado_automatico": (
                "COINCIDENCIA ALTA"
                if mejor >= 0.8
                else "REQUIERE REVISIÓN"
            ),
            "verificacion_manual": "PENDIENTE",
            "observaciones": ""
        })

with SALIDA.open(
    "w", encoding="utf-8", newline=""
) as f:
    escritor = csv.DictWriter(
        f, fieldnames=resultados[0].keys()
    )
    escritor.writeheader()
    escritor.writerows(resultados)

print("=== AUDITORÍA TÉCNICA ===")
print("Documentos:", len(resultados))

for estado, cantidad in Counter(
    r["estado_automatico"] for r in resultados
).items():
    print(f"{estado}: {cantidad}")

print("\n=== MENOR SIMILITUD ===")
for r in sorted(
    resultados,
    key=lambda x: x["similitud"]
)[:10]:
    print(
        f"FDS {r['fds_num']}: "
        f"{r['similitud']:.4f} | "
        f"{r['tipo_bloque']}"
    )

print("\nEVIDENCIA:", SALIDA)
