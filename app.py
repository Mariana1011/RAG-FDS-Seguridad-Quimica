import gradio as gr
from src.rag_general import FDSRAG

rag = FDSRAG()

def consultar(pregunta, k, metodo):
    etiqueta = ("Híbrido FAISS + TF-IDF"
                if metodo == "hibrido" else "Solo TF-IDF")
    pregunta = (pregunta or "").strip()
    if not pregunta:
        return "Escribe una pregunta sobre las FDS.", "", etiqueta

    try:
        resultado = rag.responder(pregunta, k=int(k), metodo=metodo)

        if "error" in resultado:
            return f"Error al generar la respuesta: {resultado['error']}", "", etiqueta

        respuesta = resultado.get("response", "Sin respuesta.")
        fuentes = resultado.get("fuentes", [])

        detalles = []
        for f in fuentes:
            detalles.append(
                f"### Fuente {f['id']}\n"
                f"**FDS:** {f.get('fds')}\n\n"
                f"**Producto:** {f.get('producto_verificado') or 'No verificado'}\n\n"
                f"**Fabricante:** {f.get('fabricante_verificado') or 'No verificado'}\n\n"
                f"**Página:** {f.get('pagina') or 'No verificada'}\n\n"
                f"**Sección:** {f.get('seccion_reportada') or 'No verificada'}\n\n"
                f"**Fragmento recuperado:**\n\n"
                f"> {f.get('texto_enviado', '').replace(chr(10), chr(10) + '> ')}"
            )

        alertas = resultado.get("alertas_validacion", [])
        if alertas:
            respuesta += "\n\n**Advertencias de validación:**\n"
            respuesta += "\n".join(f"- {a}" for a in alertas)

        return respuesta, "\n\n---\n\n".join(detalles), etiqueta

    except Exception as exc:
        return f"Error durante la consulta: {type(exc).__name__}: {exc}", "", etiqueta


with gr.Blocks(title="RAG FDS - Seguridad Química") as demo:
    gr.Markdown(
        "# Sistema RAG para Fichas de Datos de Seguridad\n"
        "Consulta documental sobre 101 FDS utilizando recuperación "
        "de fragmentos y generación de respuestas con Ollama."
    )

    pregunta = gr.Textbox(
        label="Pregunta",
        placeholder="Ejemplo: ¿Qué medidas de primeros auxilios indican las FDS?",
        lines=3
    )

    k = gr.Radio(
        choices=[3, 6],
        value=3,
        label="Fragmentos recuperados (top-k)"
    )

    metodo = gr.Radio(
        choices=[
            ("Híbrido FAISS + TF-IDF", "hibrido"),
            ("Solo TF-IDF", "tfidf")
        ],
        value="hibrido",
        label="Método de recuperación"
    )

    boton = gr.Button("Consultar", variant="primary")

    respuesta = gr.Markdown(label="Respuesta")
    modo = gr.Textbox(
        label="Modo de recuperación utilizado",
        value="Híbrido FAISS + TF-IDF",
        interactive=False
    )

    metodo.change(
        fn=lambda valor: (
            "Híbrido FAISS + TF-IDF"
            if valor == "hibrido" else "Solo TF-IDF"
        ),
        inputs=metodo,
        outputs=modo
    )

    gr.Markdown("## Fuentes documentales recuperadas")
    fuentes = gr.Markdown()

    boton.click(
        fn=consultar,
        inputs=[pregunta, k, metodo],
        outputs=[respuesta, fuentes, modo],
        concurrency_limit=1
    )

    gr.Markdown(
        "**Métodos disponibles:** híbrido FAISS + TF-IDF y solo TF-IDF. "
        "La evaluación RAGAS original corresponde al método híbrido. "
        "Las respuestas deben contrastarse con las FDS originales."
    )

demo.queue(default_concurrency_limit=1)
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=False
)
