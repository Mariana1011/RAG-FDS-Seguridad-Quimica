"""RAG FDS general. Recuperación FAISS+TF-IDF, citas auditables y control de redundancia.
No modifica corpus ni indices. Dependencias: faiss, joblib, numpy, requests,
scipy, sentence_transformers. Ejecutar desde RAG_FDS con el .venv activo.
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import faiss
import joblib
import numpy as np
import requests
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parents[1]
INDICES = BASE / "indices"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
GENERATOR = "llama3.2:3b"
ABSTENCION = "No encuentro información suficiente en las FDS recuperadas."
STOP = set("a al algo algunas algunos ante bajo como con contra cual cuales cuando de del desde donde el ella en entre era es esta estas este estos fue ha hay la las le lo los mas me mi no nos o para por que se ser si sin sobre son su sus te tu un una unas unos y ya cual cuales que como dice indica informacion datos segun".split())
CAS_RE = re.compile(r"(?<!\d)\d{2,7}-\d{2}-\d(?!\d)")
FDS_RE = re.compile(r"\b(?:fds|ficha(?:\s+de\s+seguridad)?)\s*(?:n[uú]m(?:ero)?\.?\s*)?(\d{1,3})\b", re.I)
H_RE = re.compile(r"\bH\d{3}(?:\+H\d{3})*\b", re.I)
CITA_RE = re.compile(r"\[FUENTE\s+(\w+)\]", re.I)


def norm(value):
    s = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(c for c in s if not unicodedata.combining(c))


def tokens(value):
    return [x for x in re.findall(r"[\w]+(?:-[\w]+)*", norm(value)) if len(x) > 2 and x not in STOP]


# Términos generales de consulta documental, NO sustancias particulares.
COMPOSICION_TERMS = ("concentracion", "porcentaje", "composicion", "cas", "ingrediente", "componentes")
PELIGRO_TERMS = ("clasificacion", "inflamab", "peligro", "toxic", "h22", "h31", "h33")


def tema_especifico(pregunta):
    """Extrae un tema explícito sin mantener un catálogo de sustancias.

    Por ejemplo: «para el xileno», «del acetato de etilo» o «sobre el xileno».
    Si no es inequívoco devuelve None en lugar de inventar una entidad.
    """
    p = norm(pregunta)
    pats = (
        r"\bpara\s+(?:el|la|los|las)\s+([\w-]+(?:\s+(?:de|del|\w+)){0,3})[?.,;:]?$",
        r"\bsobre\s+(?:el|la|los|las)\s+([\w-]+(?:\s+(?:de|del|\w+)){0,3})[?.,;:]?$",
    )
    for pat in pats:
        m = re.search(pat, p)
        if m:
            partes = tokens(m.group(1))
            if partes:
                return partes
    return None


def puntuar_evidencia(pregunta, texto, tema=None):
    """Bonificación transversal por prueba documental, no por nombre químico.

    Exige una sustancia explícita cerca de CAS/concentración/clasificación si
    la pregunta solicita asociaciones; evita premiar datos de otras filas.
    """
    q, t = norm(pregunta), norm(texto)
    composicion = any(w in q for w in COMPOSICION_TERMS)
    peligro = any(w in q for w in PELIGRO_TERMS)
    if not (composicion or peligro):
        return 0.0
    # Una tabla reconstruida puede incluir ruido de auditoría: no premiarlo.
    if tema:
        ocurrencias = [m.start() for m in re.finditer(
            r"\b" + r"\s+".join(map(re.escape, tema)) + r"\b", t
        )]
        if not ocurrencias:
            return -0.018
    else:
        ocurrencias = []
    bono = 0.0
    # Descriptores documentales; los encabezados 2/3 no se presumen correctos.
    if composicion and ("cas" in t or "composicion" in t) and ("%" in t or "concentracion" in t):
        bono += 0.007
    if peligro and (re.search(r"\bh\d{3}\b", t) or "categoria" in t):
        bono += 0.006
    if ocurrencias:
        mejor_asociacion = 0.0
        for pos in ocurrencias:
            ventana = t[max(0, pos - 35):pos + 235]
            tiene_cas = bool(CAS_RE.search(ventana))
            tiene_conc = bool(re.search(r"(?:\d[\d.,]*\s*[-–]\s*[<≤>=]?\s*\d|\d[\d.,]*\s*%)", ventana))
            tiene_peligro = bool(re.search(r"\bh\d{3}\b|liq\.?\s*infl|categoria", ventana))
            puntos = (0.016 if composicion and tiene_cas and tiene_conc else 0.0)
            puntos += (0.016 if peligro and tiene_peligro else 0.0)
            # Premio máximo cuando se asocian explícitamente tema, CAS,
            # concentración y clasificación en una ventana acotada.
            mejor_asociacion = max(mejor_asociacion, puntos)
        bono += mejor_asociacion
    return min(bono, 0.052)


def abstencion_reconocida(respuesta):
    """Una abstención no requiere cita y no es fallo de formato."""
    t = norm(respuesta)
    return "no encuentro informacion suficiente" in t or "no hay informacion suficiente" in t


def fusionar_rrf(rankings, k0=60):
    output = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            output[int(idx)] += 1 / (k0 + rank)
    return output


def identificadores_cas(texto):
    return set(CAS_RE.findall(texto))


def duplicado_documental(nuevo, seleccionado):
    """Conservador: evita transcripción+tabla de una misma página si comparten CAS.
    No equipara páginas diferentes, ni suprime distintos CAS, ni borra archivos.
    """
    if nuevo.get("fds_num") != seleccionado.get("fds_num") or nuevo.get("pagina_pdf") != seleccionado.get("pagina_pdf"):
        return False
    a = norm(nuevo.get("texto", ""))
    b = norm(seleccionado.get("texto", ""))
    if re.sub(r"\W+", "", a)[:300] == re.sub(r"\W+", "", b)[:300]:
        return True
    cas_a, cas_b = identificadores_cas(a), identificadores_cas(b)
    tabla_a = "tabla reconstruida" in a or "| ---" in a
    tabla_b = "tabla reconstruida" in b or "| ---" in b
    if tabla_a != tabla_b and cas_a and cas_b and cas_a.intersection(cas_b):
        return True
    # Fragmentos suficientemente parecidos de una misma página.
    ta, tb = set(tokens(a)), set(tokens(b))
    if len(ta) >= 15 and len(tb) >= 15 and len(ta & tb) / min(len(ta), len(tb)) >= 0.88:
        return True
    return False


def extraer_fds(pregunta):
    """Reconoce 'FDS 51 y 102', 'FDS 51, 60 y 102' y referencias completas."""
    encontrados = []
    continuacion = re.compile(
        r"\s*(?:,|\by\b|\be\b)\s*(?:(?:la|el|las|los)\s+)?(?:FDS\s+)?(\d{1,3})\b",
        re.I,
    )
    for match in FDS_RE.finditer(pregunta):
        encontrados.append(int(match.group(1)))
        posicion = match.end()
        while True:
            siguiente = continuacion.match(pregunta, posicion)
            if not siguiente:
                break
            encontrados.append(int(siguiente.group(1)))
            posicion = siguiente.end()
    return list(dict.fromkeys(encontrados))


def relevancia_evidencia(pregunta, texto):
    """Prioriza el tipo de dato solicitado, sin codificar sustancias ni productos.

    Usa señales observables del fragmento, NO infiere clasificación química.
    No presume que seccion_numero haya sido asignada correctamente.
    """
    q, t = norm(pregunta), norm(texto)
    pide_composicion = any(x in q for x in ('concentracion', 'porcentaje', 'intervalo', 'numero cas', 'composicion', 'ingrediente'))
    pide_categoria = any(x in q for x in ('categoria', 'clasificacion de inflamabilidad', 'liquido inflamable'))
    pide_inflamabilidad = 'inflamab' in q
    pide_version = 'version' in q or 'fecha de emision' in q
    puntuacion = 0.0

    # Una fila de componentes, con sustancia-CAS-concentración en la misma tabla.
    tiene_tabla = '|' in t and ('cas' in t or 'nombre de ingrediente' in t)
    intervalo = bool(re.search(r'\b\d+[.,]?\d*\s*[-–]\s*<\s*\d+|\b\d+\s*[-–]\s*<\d+\s*%', t))
    if pide_composicion:
        if 'seccion 3' in t or 'composicion' in t[:160]:
            puntuacion += 0.045
        if tiene_tabla and intervalo:
            puntuacion += 0.100
        elif intervalo and CAS_RE.search(texto):
            puntuacion += 0.055
        if 'noec' in t or 'tlv-twa' in t or 'tlv-stel' in t:
            puntuacion -= 0.065

    # Clasificación del PRODUCTO: distintas de las de ingredientes de sección 3.
    if pide_categoria and pide_inflamabilidad:
        clasificacion_product = bool(re.search(
            r'liquidos?\s+inflamables?\s*[-–:]?\s*(?:-\s*)?categoria\s*[1-4]', t
        ))
        if clasificacion_product:
            puntuacion += 0.15
        if 'seccion 2' in t[:140] or 'identificacion de los peligros' in t[:180]:
            puntuacion += 0.065
        if ('composicion' in t[:160] or tiene_tabla) and not clasificacion_product:
            puntuacion -= 0.04

    if pide_version and ('version' in t and ('fecha de emision' in t or 'emision' in t)):
        puntuacion += 0.06
    return puntuacion


class FDSRAG:
    def __init__(self):
        try:
            self.emb = SentenceTransformer(str(BASE / "modelos" / "minilm_multilingual"), device="cpu", local_files_only=True)
            self.modo_recuperacion = "Híbrido FAISS + TF-IDF"
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Modelo de embeddings no disponible: {exc}")
            print("Activando recuperación TF-IDF de respaldo.")
            self.emb = None
            self.modo_recuperacion = "Solo TF-IDF (respaldo)"
        self.index = faiss.read_index(str(INDICES / "fds_faiss.index"))
        with (INDICES / "metadatos.jsonl").open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        self.vectorizer = joblib.load(INDICES / "tfidf_vectorizer.joblib")
        self.matrix = load_npz(INDICES / "tfidf_matrix.npz")
        if not (self.index.ntotal == len(self.chunks) == self.matrix.shape[0]):
            raise ValueError("FAISS, TF-IDF y metadatos no coinciden")
        self.textos_normalizados = [norm(c.get("texto", "")) for c in self.chunks]
        self.docs_normalizados = [norm(" ".join(str(c.get(k) or "") for k in ("documento", "producto", "fabricante", "version"))) for c in self.chunks]

    @staticmethod
    def objetivo(pregunta):
        m = re.search(r"\bsobre\s+(?:el|la|los|las|un|una)?\s*([^?.,;:]+)", norm(pregunta))
        return [t for t in tokens(m.group(1)) if t not in {"fds", "ficha", "seguridad"}] if m else []

    def buscar(self, pregunta, k=3, pool=150, metodo="hibrido"):
        if not 1 <= k <= 20:
            raise ValueError("k debe estar entre 1 y 20")
        objetivos_fds = extraer_fds(pregunta)
        numero_fds = objetivos_fds[0] if len(objetivos_fds) == 1 else None
        n = len(self.chunks) if objetivos_fds else min(len(self.chunks), max(pool, k * 20))
        terms = list(dict.fromkeys(tokens(pregunta)))
        objetivo = self.objetivo(pregunta)
        tema = tema_especifico(pregunta)
        cas = [norm(x) for x in CAS_RE.findall(pregunta)]
        hs = [norm(x) for x in H_RE.findall(pregunta)]

        if metodo not in ("hibrido", "tfidf"):
            raise ValueError("Método de recuperación inválido")
        if metodo == "hibrido" and self.emb is None:
            raise RuntimeError("MiniLM no está disponible para recuperación híbrida")
        sem = []
        if metodo == "hibrido":
            qv = self.emb.encode([pregunta], normalize_embeddings=True, convert_to_numpy=True)
            _, fi = self.index.search(np.ascontiguousarray(qv, dtype=np.float32), n)
            sem = [int(x) for x in fi[0] if x >= 0]
        qs = self.vectorizer.transform([pregunta])
        sims = (self.matrix @ qs.T).toarray().ravel()
        nonzero = np.flatnonzero(sims > 0)
        lex = sorted(nonzero, key=lambda i: (-sims[i], i))[:n]
        fused = fusionar_rrf([sem, lex])

        orden = []
        for idx, base_score in fused.items():
            c = self.chunks[idx]
            txt = self.textos_normalizados[idx]
            if objetivos_fds and int(c.get("fds_num") or -1) not in objetivos_fds:
                continue
            if cas and not any(x in txt or x in self.docs_normalizados[idx] for x in cas):
                continue
            if len(txt.strip()) < 85:
                continue
            covered = sum(t in txt or t in self.docs_normalizados[idx] for t in terms)
            coverage = covered / max(len(terms), 1)
            proximity = 0.0
            if len(terms) >= 2:
                for t in terms[:8]:
                    at = txt.find(t)
                    if at >= 0:
                        ventana = txt[max(0, at - 100):at + 220]
                        proximity = max(proximity, sum(term in ventana for term in terms) / len(terms))
            score = base_score + 0.006 * coverage + 0.004 * proximity
            score += puntuar_evidencia(pregunta, c.get("texto", ""), tema)
            score += relevancia_evidencia(pregunta, c.get("texto", ""))
            if hs and any(h in txt for h in hs):
                score += 0.005
            if objetivo:
                presente = all(re.search(r"\b" + re.escape(t) + r"\b", txt) for t in objetivo)
                if presente:
                    score += 0.015
                    if numero_fds is not None and CAS_RE.search(c.get("texto", "")):
                        score += 0.012
                    if numero_fds is not None and H_RE.search(c.get("texto", "")):
                        score += 0.007
                else:
                    score -= 0.020
            orden.append((score, idx))
        orden.sort(key=lambda x: (-x[0], x[1]))

        # Para consultas que mencionan varias FDS, reservar evidencia por documento.
        # La alternancia permite comparar top-k=3 y top-k=6 sin cambiar el recuperador.
        if len(objetivos_fds) > 1:
            grupos = {
                fds: [(score, idx) for score, idx in orden
                      if int(self.chunks[idx].get("fds_num") or -1) == fds]
                for fds in objetivos_fds
            }
            orden_intercalado = []
            while any(grupos.values()):
                for fds in objetivos_fds:
                    if grupos[fds]:
                        orden_intercalado.append(grupos[fds].pop(0))
            orden = orden_intercalado

        seleccion = []
        for score, idx in orden:
            candidato = self.chunks[idx]
            if any(duplicado_documental(candidato, item["chunk"]) for item in seleccion):
                continue
            seleccion.append({"chunk": candidato, "score": round(float(score), 6)})
            if len(seleccion) == k:
                break
        return seleccion

    @staticmethod
    def pasaje(texto, pregunta, max_chars=490):
        """Escoge un tramo literal; no modifica datos químicos."""
        if len(texto) <= max_chars:
            return texto.strip()
        terms = tokens(pregunta)
        if not terms:
            return texto[:max_chars].strip()
        lowered = norm(texto)
        best_at, best_score = 0, -1
        for pos in range(0, len(texto), max(100, max_chars // 3)):
            window = lowered[pos:pos + max_chars]
            score = sum(t in window for t in terms)
            if score > best_score:
                best_at, best_score = pos, score
        return texto[best_at:best_at + max_chars].strip()

    def responder(self, pregunta, k=3, modelo=GENERATOR, metodo="hibrido"):
        recuperados = self.buscar(pregunta, k=k, metodo=metodo)
        fuentes, contextos = [], []
        for i, resultado in enumerate(recuperados, 1):
            chunk = resultado["chunk"]
            texto = self.pasaje(chunk.get("texto", ""), pregunta, max_chars=490)
            contextos.append(texto)
            fuentes.append({
                "id": i, "chunk_id": chunk.get("chunk_id"), "fds": chunk.get("fds_num"),
                "documento": chunk.get("documento"), "pagina": chunk.get("pagina_pdf"),
                "seccion_reportada": chunk.get("seccion_numero"),
                "producto_verificado": chunk.get("producto"),
                "fabricante_verificado": chunk.get("fabricante"),
                "version_verificada": chunk.get("version"),
                "texto_enviado": texto, "score": resultado["score"]
            })

        base = {"user_input": pregunta, "retrieved_contexts": contextos, "fuentes": fuentes, "k": k,
                "configuracion": {"modelo": modelo, "num_ctx": 4096, "num_predict": 230, "max_chars_por_fragmento": 490}}
        if not fuentes:
            return {**base, "response": ABSTENCION, "alertas_validacion": ["Sin fragmentos recuperados"]}

        contexto = "\n\n".join(
            f"[FUENTE {s['id']}] FDS {s['fds']}, página {s['pagina']}, "
            f"sección registrada {s['seccion_reportada'] if s['seccion_reportada'] is not None else 'sin verificar'}\n"
            f"{s['texto_enviado']}" for s in fuentes
        )
        prompt = (
            "Eres un asistente de consulta documental sobre fichas de seguridad. "
            "Responde en español, SOLO con información que figure literalmente en las fuentes. "
            "Cada afirmación técnica debe terminar con un identificador exacto [FUENTE 1], [FUENTE 2], etc. "
            "No afirmes que una FDS carece de información: solo puedes decir que los FRAGMENTOS recuperados no la muestran. "
            "Distingue explícitamente la sustancia del producto o mezcla; no extrapoles clasificaciones entre ambos. "
            "Si existe duda sobre una asociación CAS-sustancia-clasificación, abstente de afirmar esa asociación. "
            "IARC Grupo 3 significa NO CLASIFICABLE respecto de carcinogenicidad en humanos, NO cancerígeno. "
            "No inventes páginas, productos, fabricantes, versiones, valores o citas. "
            f"Si no hay soporte suficiente contesta exactamente: {ABSTENCION}\n\n"
            f"FUENTES:\n{contexto}\n\nPREGUNTA: {pregunta}\nRESPUESTA:"
        )
        try:
            http = requests.post(OLLAMA_URL, json={
                "model": modelo, "prompt": prompt, "stream": False,
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 230},
                "keep_alive": "10m"
            }, timeout=600)
            if not http.ok:
                return {**base, "error": f"Ollama HTTP {http.status_code}: {http.text[:500]}"}
            respuesta = http.json().get("response", "").strip() or ABSTENCION
        except (requests.RequestException, ValueError) as exc:
            return {**base, "error": str(exc)}

        alertas = []
        citas_validas = {str(x["id"]) for x in fuentes}
        citas = CITA_RE.findall(respuesta)
        invalidas = [x for x in citas if x not in citas_validas]
        if invalidas:
            alertas.append("Marcadores de cita inexistentes: " + ", ".join(invalidas))
            respuesta = CITA_RE.sub(lambda m: m.group(0) if m.group(1) in citas_validas else "[CITA NO VERIFICADA]", respuesta)
        if not abstencion_reconocida(respuesta) and not any(x in citas_validas for x in citas):
            alertas.append("Respuesta sin citas a fuentes: requiere revisión")
        if re.search(r"\b(?:no proporciona|no contiene|carece de|no existe informacion)\b", norm(respuesta)):
            alertas.append("Afirmación de ausencia documental: verificar contra FDS completa")
        if re.search(r"\b(?:cancerigen[oa]|carcinogen[oa])\b", norm(respuesta)) and re.search(r"iarc.{0,50}(?:\(3\)|grupo\s*3)", norm(contexto), re.S):
            alertas.append("Posible confusión IARC Grupo 3; requiere verificación humana")
        # Lista trazable de referencias disponibles; no certifica automáticamente afirmaciones.
        return {**base, "response": respuesta,
                "referencias_recuperadas": [f"[FUENTE {f['id']}] FDS {f['fds']}, pág. {f['pagina']}, chunk {f['chunk_id']}" for f in fuentes],
                "alertas_validacion": alertas, "requiere_revision": bool(alertas)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pregunta", nargs="?", default="¿Cuáles son los peligros del xileno?")
    parser.add_argument("--k", type=int, choices=range(1, 21), metavar="1..20", default=3)
    parser.add_argument("--solo-busqueda", action="store_true")
    args = parser.parse_args()
    rag = FDSRAG()
    if args.solo_busqueda:
        salida = [{"fds": r["chunk"].get("fds_num"), "pagina": r["chunk"].get("pagina_pdf"),
                   "seccion_reportada": r["chunk"].get("seccion_numero"), "chunk_id": r["chunk"].get("chunk_id"),
                   "texto": r["chunk"].get("texto", "")[:600]} for r in rag.buscar(args.pregunta, k=args.k)]
    else:
        salida = rag.responder(args.pregunta, k=args.k)
    print(json.dumps(salida, ensure_ascii=False, indent=2))
