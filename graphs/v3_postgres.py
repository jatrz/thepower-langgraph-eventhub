"""Grafo V3 de EventHub con memoria persistente en PostgreSQL.

A diferencia de InMemorySaver, PostgresSaver conserva el hilo después de cerrar
el proceso. Esta versión no usa memoria en RAM ni borra el historial al arrancar.
"""

from __future__ import annotations

import os
import re
import sys

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import OpenAIError

from graphs.v2_chatbot import (
    ENV_VISIBLE,
    ErrorConfiguracion,
    RUTA_POLITICAS,
    cargar_entorno,
    construir_prompt_sistema,
    crear_modelo,
    crear_nodo_chatbot,
    interpretar_error_api,
    leer_configuracion,
    leer_politicas,
    ruta_visible,
)
from graphs.v3_memory import (
    NODO_CHATBOT,
    imprimir_estado,
    imprimir_turno,
    preguntar,
)

THREAD_ID = "eventhub_postgres_demo"
MENSAJE_PRESENTACION = "Me llamo Ana y mi ciudad favorita es Madrid"
MENSAJE_MUSICA = "Trabajo en el sector de la música y me gustan los conciertos"
MENSAJE_RECUERDO = "¿Cómo me llamo, cuál es mi ciudad favorita y qué tipo de música me gusta?"
_CREDENCIALES = re.compile(r"://[^/\s@]+@")


def instrucciones() -> None:
    print("Uso:")
    print("  python -m graphs.v3_postgres guardar")
    print("  python -m graphs.v3_postgres recuperar")


def leer_postgres_uri() -> str:
    """Lee POSTGRES_URI sin imprimirla."""
    cargar_entorno()
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise ErrorConfiguracion(
            "POSTGRES_URI no está configurada. "
            f"Añádela en {ENV_VISIBLE}. El valor no se muestra."
        )
    return uri


def sanear(texto: str, uri: str = "") -> str:
    """Quita la cadena de conexión y cualquier usuario o contraseña visibles."""
    if uri:
        texto = texto.replace(uri, "[conexion-oculta]")
    return _CREDENCIALES.sub("://[credenciales-ocultas]@", " ".join(texto.split()))


def compilar_grafo(modelo, prompt_sistema: str, checkpointer: PostgresSaver) -> CompiledStateGraph:
    """El mismo grafo de V3, compilado con PostgresSaver en lugar de memoria RAM."""
    grafo = StateGraph(MessagesState)
    grafo.add_node(NODO_CHATBOT, crear_nodo_chatbot(modelo, prompt_sistema))
    grafo.add_edge(START, NODO_CHATBOT)
    grafo.add_edge(NODO_CHATBOT, END)
    return grafo.compile(checkpointer=checkpointer)


def guardar(graph: CompiledStateGraph) -> int:
    """Añade dos turnos al hilo fijo y comprueba que quedan persistidos."""
    print(f"Guardando en el hilo {THREAD_ID}. No se borra el historial anterior.")
    print(f"Políticas cargadas desde: {ruta_visible(RUTA_POLITICAS)}")
    print()
    for texto in (MENSAJE_PRESENTACION, MENSAJE_MUSICA):
        respuesta = preguntar(graph, THREAD_ID, texto)
        imprimir_turno(THREAD_ID, texto, respuesta)

    mensajes = imprimir_estado(graph, THREAD_ID)
    if len(mensajes) < 4:
        raise RuntimeError(
            "El hilo debe conservar al menos cuatro mensajes. "
            f"graph.get_state ha devuelto {len(mensajes)}."
        )
    return 0


def recuperar(graph: CompiledStateGraph) -> int:
    """Abre el mismo hilo ya guardado y pregunta usando solo ese estado."""
    print(f"Recuperando el hilo {THREAD_ID} desde PostgreSQL.")
    print(f"Políticas cargadas desde: {ruta_visible(RUTA_POLITICAS)}")
    print()
    mensajes = imprimir_estado(graph, THREAD_ID)
    if not mensajes:
        print(
            "No hay historial persistido en este hilo. "
            "Ejecuta primero: python -m graphs.v3_postgres guardar",
            file=sys.stderr,
        )
        return 1

    respuesta = preguntar(graph, THREAD_ID, MENSAJE_RECUERDO)
    imprimir_turno(THREAD_ID, MENSAJE_RECUERDO, respuesta)
    texto = respuesta.casefold()
    recuerda = all(dato in texto for dato in ("ana", "madrid")) and (
        "concierto" in texto or "música" in texto or "musica" in texto
    )
    if not recuerda:
        print(
            "La respuesta no refleja los datos persistidos del hilo. "
            "No se ha inventado un recuerdo.",
            file=sys.stderr,
        )
        return 1
    print("La respuesta utiliza los datos recuperados del estado persistido.")
    return 0


def ejecutar(modo: str) -> int:
    """Mantiene abierta la conexión de PostgresSaver durante todo el modo."""
    uri = leer_postgres_uri()
    api_key, modelo = leer_configuracion()
    prompt = construir_prompt_sistema(leer_politicas())
    modelo_chat = crear_modelo(api_key, modelo)
    try:
        with PostgresSaver.from_conn_string(uri) as checkpointer:
            checkpointer.setup()
            graph = compilar_grafo(modelo_chat, prompt, checkpointer)
            if modo == "guardar":
                return guardar(graph)
            return recuperar(graph)
    except psycopg.Error as exc:
        print(
            "No se ha podido usar PostgreSQL.\n"
            f"{type(exc).__name__}: {sanear(str(exc), uri)}",
            file=sys.stderr,
        )
        return 1
    except OpenAIError as exc:
        error = interpretar_error_api(exc, api_key=api_key, modelo=modelo)
        print(
            f"No se ha podido completar la demostración.\n{type(exc).__name__}: {sanear(str(error), uri)}",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    argumentos = list(sys.argv[1:] if argv is None else argv)
    if len(argumentos) != 1 or argumentos[0] not in {"guardar", "recuperar"}:
        instrucciones()
        return 0 if not argumentos else 1
    try:
        return ejecutar(argumentos[0])
    except ErrorConfiguracion as exc:
        print(f"No se puede iniciar la memoria PostgreSQL.\n{sanear(str(exc))}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"No se ha podido completar la demostración.\n{type(exc).__name__}: {sanear(str(exc))}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
