"""Grafo V3 de EventHub: el mismo chatbot de V2, con memoria temporal por hilo.

InMemorySaver conserva el historial solo mientras este proceso sigue en marcha.
Al cerrar y volver a ejecutar el programa, los hilos empiezan vacíos.
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import OpenAIError

from graphs.v2_chatbot import (
    ErrorConfiguracion,
    RUTA_POLITICAS,
    construir_prompt_sistema,
    crear_modelo,
    crear_nodo_chatbot,
    interpretar_error_api,
    leer_configuracion,
    leer_politicas,
    ruta_visible,
    texto_de_mensaje,
)

NODO_CHATBOT = "chatbot"
HILO_1 = "conversacion_1"
HILO_2 = "conversacion_2"

MENSAJE_PRESENTACION = "Me llamo Ana y mi ciudad favorita es Madrid"
MENSAJE_RECUERDO = "¿Cómo me llamo y cuál es mi ciudad favorita?"
MENSAJE_OTRO_HILO = "¿Sabes cómo me llamo y cuál es mi ciudad favorita?"


def config_hilo(thread_id: str) -> dict:
    """Misma configuración en todas las llamadas de un hilo."""
    return {"configurable": {"thread_id": thread_id}}


def compilar_grafo(modelo, prompt_sistema: str) -> CompiledStateGraph:
    """Grafo START → chatbot → END, con MessagesState y checkpointer en memoria."""
    grafo = StateGraph(MessagesState)
    grafo.add_node(NODO_CHATBOT, crear_nodo_chatbot(modelo, prompt_sistema))
    grafo.add_edge(START, NODO_CHATBOT)
    grafo.add_edge(NODO_CHATBOT, END)
    return grafo.compile(checkpointer=InMemorySaver())


def preguntar(graph: CompiledStateGraph, thread_id: str, texto: str) -> str:
    """Envía solo el mensaje nuevo. El checkpointer recupera el historial del hilo."""
    resultado = graph.invoke(
        {"messages": [HumanMessage(content=texto)]},
        config_hilo(thread_id),
    )
    return texto_de_mensaje(resultado["messages"][-1])


def mensajes_del_hilo(graph: CompiledStateGraph, thread_id: str) -> list[BaseMessage]:
    """Lee los mensajes guardados con graph.get_state(config)."""
    estado = graph.get_state(config_hilo(thread_id))
    return list(estado.values.get("messages", []))


def imprimir_turno(thread_id: str, usuario: str, respuesta: str) -> None:
    print(f"[{thread_id}] Usuario: {usuario}")
    print(f"[{thread_id}] EventHub: {respuesta}")
    print()


def imprimir_estado(graph: CompiledStateGraph, thread_id: str) -> list[BaseMessage]:
    """Muestra el checkpoint del hilo y el número de mensajes."""
    mensajes = mensajes_del_hilo(graph, thread_id)
    print(f"Estado de {thread_id} (graph.get_state)")
    if not mensajes:
        print("No hay mensajes guardados.")
    for numero, mensaje in enumerate(mensajes, start=1):
        rol = "Asistente" if isinstance(mensaje, AIMessage) else "Usuario"
        print(f"{numero}. {rol}: {texto_de_mensaje(mensaje)}")
    print(f"Total de mensajes: {len(mensajes)}")
    print()
    return mensajes


def comprobar_aislamiento(mensajes_hilo_1: list[BaseMessage], mensajes_hilo_2: list[BaseMessage]) -> None:
    """Falla de forma visible si la memoria no cumple lo pedido en la práctica."""
    humanos_1 = [m for m in mensajes_hilo_1 if isinstance(m, HumanMessage)]
    asistente_1 = [m for m in mensajes_hilo_1 if isinstance(m, AIMessage)]
    if len(mensajes_hilo_1) < 4 or len(humanos_1) < 2 or len(asistente_1) < 2:
        raise RuntimeError(
            "La conversación 1 debe conservar al menos cuatro mensajes: "
            "dos del usuario y dos del asistente. "
            f"Hay {len(mensajes_hilo_1)} en total."
        )

    texto_hilo_2 = "\n".join(texto_de_mensaje(mensaje) for mensaje in mensajes_hilo_2)
    if MENSAJE_PRESENTACION in texto_hilo_2 or "Ana" in texto_hilo_2 or "Madrid" in texto_hilo_2:
        raise RuntimeError(
            "La conversación 2 no debe conocer los datos de la conversación 1."
        )


def demostrar(graph: CompiledStateGraph) -> None:
    """Dos hilos distintos: uno recuerda los datos y el otro empieza de cero."""
    print("Demostración V3. La memoria dura solo mientras este proceso sigue abierto.")
    print(f"Políticas cargadas desde: {ruta_visible(RUTA_POLITICAS)}")
    print()

    respuesta_presentacion = preguntar(graph, HILO_1, MENSAJE_PRESENTACION)
    imprimir_turno(HILO_1, MENSAJE_PRESENTACION, respuesta_presentacion)

    respuesta_recuerdo = preguntar(graph, HILO_1, MENSAJE_RECUERDO)
    imprimir_turno(HILO_1, MENSAJE_RECUERDO, respuesta_recuerdo)

    respuesta_otro_hilo = preguntar(graph, HILO_2, MENSAJE_OTRO_HILO)
    imprimir_turno(HILO_2, MENSAJE_OTRO_HILO, respuesta_otro_hilo)

    mensajes_1 = imprimir_estado(graph, HILO_1)
    mensajes_2 = imprimir_estado(graph, HILO_2)
    comprobar_aislamiento(mensajes_1, mensajes_2)
    print("La conversación 1 conserva el historial y la conversación 2 no lo comparte.")


def main() -> int:
    """Carga la misma configuración que V2, compila con memoria y ejecuta la demo."""
    api_key = ""
    modelo = "OPENAI_MODEL"
    try:
        api_key, modelo = leer_configuracion()
        politicas = leer_politicas()
        graph = compilar_grafo(crear_modelo(api_key, modelo), construir_prompt_sistema(politicas))
        demostrar(graph)
    except ErrorConfiguracion as exc:
        print(f"No se puede iniciar V3.\n{exc}", file=sys.stderr)
        return 1
    except OpenAIError as exc:
        error = interpretar_error_api(exc, api_key=api_key, modelo=modelo)
        print(
            f"No se ha podido completar la demostración.\n{type(exc).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"No se ha podido completar la demostración.\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
