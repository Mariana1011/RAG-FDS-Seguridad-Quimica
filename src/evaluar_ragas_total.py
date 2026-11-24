import argparse
import asyncio
import json
import math
from pathlib import Path

import instructor
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecisionWithReference,
    ContextRecall,
)

BASE = Path("resultados")
SALIDA = BASE / "ragas_evaluacion_completa.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--desde", type=int, required=True)
parser.add_argument("--hasta", type=int, required=True)
args = parser.parse_args()

datos = json.loads(
    (BASE / "dataset_ragas_preparado.json")
    .read_text(encoding="utf-8")
)

def leer_jsonl(ruta):
    if not ruta.exists():
        return []
    return [
        json.loads(linea)
        for linea in ruta.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]

# Reutilizar los seis resultados válidos del piloto P03.
if not SALIDA.exists():
    pilotos = (
        leer_jsonl(BASE / "piloto_faithfulness_qwen.jsonl")
        + leer_jsonl(BASE / "piloto_ragas_qwen_resultados.jsonl")
    )

    claves = set()
    with SALIDA.open("w", encoding="utf-8") as archivo:
        for r in pilotos:
            if r.get("id_pregunta") != "P03":
                continue
            if "puntuacion" not in r:
                continue

            nombre_metrica = r.get("metrica", "faithfulness")
            clave = (r["id_pregunta"], r["k"], nombre_metrica)
            if clave in claves:
                continue

            claves.add(clave)
            archivo.write(json.dumps({
                "id_pregunta": r["id_pregunta"],
                "k": r["k"],
                "metrica": nombre_metrica,
                "puntuacion": r["puntuacion"],
                "modelo_evaluador": "qwen2.5:7b",
                "origen": "piloto"
            }, ensure_ascii=False) + "\n")

existentes = leer_jsonl(SALIDA)
completados = {
    (r["id_pregunta"], r["k"], r["metrica"])
    for r in existentes
    if isinstance(r.get("puntuacion"), (int, float))
    and math.isfinite(r["puntuacion"])
}

cliente = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    timeout=300.0,
    max_retries=0,
)

juez = llm_factory(
    "qwen2.5:7b",
    provider="openai",
    client=cliente,
    mode=instructor.Mode.JSON,
    temperature=0,
)

metricas = {
    "faithfulness": Faithfulness(llm=juez),
    "context_precision": ContextPrecisionWithReference(llm=juez),
    "context_recall": ContextRecall(llm=juez),
}

async def main():
    casos = sorted(
        (
            r for r in datos
            if r.get("reference")
            and args.desde <= int(r["id_pregunta"][1:]) <= args.hasta
        ),
        key=lambda r: (r["id_pregunta"], r["k"])
    )

    for caso in casos:
        identificador = caso["id_pregunta"]
        k = caso["k"]

        for nombre, metrica in metricas.items():
            clave = (identificador, k, nombre)

            if clave in completados:
                print(f"OMITIDO {identificador} k={k} {nombre}", flush=True)
                continue

            print(f"EVALUANDO {identificador} k={k} {nombre}", flush=True)

            registro = {
                "id_pregunta": identificador,
                "k": k,
                "metrica": nombre,
                "modelo_evaluador": "qwen2.5:7b",
            }

            try:
                parametros = {
                    "user_input": caso["user_input"],
                    "retrieved_contexts": caso["retrieved_contexts"],
                }
                if nombre == "faithfulness":
                    parametros["response"] = caso["response"]
                else:
                    parametros["reference"] = caso["reference"]

                resultado = await metrica.ascore(**parametros)
                valor = float(resultado.value)

                if not math.isfinite(valor):
                    raise ValueError("Puntuación no finita")

                registro["puntuacion"] = valor
                completados.add(clave)
                print(f"RESULTADO: {valor:.4f}", flush=True)

            except Exception as e:
                registro["error"] = (
                    f"{type(e).__name__}: {str(e)[:300]}"
                )
                print(f"ERROR: {registro['error']}", flush=True)

            with SALIDA.open("a", encoding="utf-8") as archivo:
                archivo.write(
                    json.dumps(registro, ensure_ascii=False) + "\n"
                )
                archivo.flush()

    validos = {
        (r["id_pregunta"], r["k"], r["metrica"])
        for r in leer_jsonl(SALIDA)
        if "puntuacion" in r
    }
    print(f"\nMEDICIONES GUARDADAS: {len(validos)} / 60", flush=True)
    print("Archivo:", SALIDA, flush=True)

asyncio.run(main())
