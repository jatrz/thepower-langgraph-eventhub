"""Grafo V1 de EventHub: clasificación por reglas y respuestas fijas, sin LLM."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

# Intenciones que el nodo clasificar puede asignar.
INTENCION_EVENTOS = "eventos"
INTENCION_PEDIDO = "pedido"
INTENCION_POLITICAS = "politicas"
INTENCION_OTRO = "otro"

NODO_CLASIFICAR = "clasificar"
NODO_EVENTOS = "responder_eventos"
NODO_PEDIDO = "responder_pedido"
NODO_GENERAL = "responder_general"

NodoRespuesta = Literal["responder_eventos", "responder_pedido", "responder_general"]

# Prioridad de clasificación: políticas, pedido, eventos y, si no hay coincidencia, otro.
# Se usan formas sin acentos para que «política» y «politica» cuenten igual.
_REGLAS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        INTENCION_POLITICAS,
        (
            "politica",
            "politicas",
            "devolucion",
            "devoluciones",
            "reembolso",
            "reembolsos",
            "cancelacion",
            "cancelaciones",
        ),
    ),
    (
        INTENCION_PEDIDO,
        ("pedido", "pedidos", "compra", "compras", "ticket", "tickets"),
    ),
    (
        INTENCION_EVENTOS,
        (
            "evento",
            "eventos",
            "concierto",
            "conciertos",
            "cartel",
            "cartelera",
            "agenda",
            "festival",
        ),
    ),
)

_PATRON_PEDIDO = re.compile(r"\bped-\d+\b")

RESPUESTA_EVENTOS = (
    "Puedo ayudarte con la cartelera de EventHub. "
    "Indica ciudad, fecha o tipo de evento y te orientaré sobre la programación."
)
RESPUESTA_PEDIDO = (
    "Puedo ayudarte a consultar un pedido de EventHub. "
    "Indica el identificador (por ejemplo, PED-1001) o el correo asociado."
)
RESPUESTA_POLITICAS = (
    "Puedo orientarte sobre las políticas de EventHub: "
    "devoluciones, cancelaciones, cambios y reembolsos."
)
RESPUESTA_OTRO = (
    "Puedo ayudarte con eventos, pedidos o políticas de EventHub. "
    "Reformula la consulta indicando uno de esos temas."
)


class EstadoEventHub(TypedDict, total=False):
    """Estado compartido del grafo V1.

    ``total=False`` permite invocar el grafo solo con ``mensaje``;
    ``intencion`` y ``respuesta`` los rellenan los nodos.
    """

    mensaje: str
    intencion: str
    respuesta: str


def _normalizar(texto: str) -> str:
    """Pasa el texto a minúsculas, sin acentos, conservando guiones de identificadores."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    sin_acentos = "".join(
        caracter
        for caracter in descompuesto
        if unicodedata.category(caracter) != "Mn"
    )
    limpio = re.sub(r"[^a-z0-9-]+", " ", sin_acentos)
    return re.sub(r"\s+", " ", limpio).strip()


def _contiene_termino(texto_normalizado: str, termino: str) -> bool:
    """Comprueba el término como palabra completa para evitar coincidencias parciales."""
    return re.search(rf"\b{re.escape(termino)}\b", texto_normalizado) is not None


def clasificar(state: EstadoEventHub) -> dict[str, str]:
    """Detecta la intención del mensaje con reglas sencillas."""
    texto = _normalizar(state.get("mensaje", ""))

    # Un identificador PED-xxxx es una señal directa de consulta de pedido.
    if _PATRON_PEDIDO.search(texto):
        return {"intencion": INTENCION_PEDIDO}

    for intencion, terminos in _REGLAS:
        if any(_contiene_termino(texto, termino) for termino in terminos):
            return {"intencion": intencion}

    return {"intencion": INTENCION_OTRO}


def enrutar(state: EstadoEventHub) -> NodoRespuesta:
    """Elige el nodo de respuesta según la intención ya clasificada."""
    intencion = state.get("intencion", INTENCION_OTRO)
    if intencion == INTENCION_EVENTOS:
        return NODO_EVENTOS
    if intencion == INTENCION_PEDIDO:
        return NODO_PEDIDO
    # «politicas» y «otro» comparten el nodo general; la respuesta fija distingue el caso.
    return NODO_GENERAL


def responder_eventos(state: EstadoEventHub) -> dict[str, str]:
    """Respuesta fija para consultas de cartelera."""
    del state
    return {"respuesta": RESPUESTA_EVENTOS}


def responder_pedido(state: EstadoEventHub) -> dict[str, str]:
    """Respuesta fija para consultas de pedidos."""
    del state
    return {"respuesta": RESPUESTA_PEDIDO}


def responder_general(state: EstadoEventHub) -> dict[str, str]:
    """Respuesta fija para políticas y para cualquier otra consulta."""
    if state.get("intencion") == INTENCION_POLITICAS:
        return {"respuesta": RESPUESTA_POLITICAS}
    return {"respuesta": RESPUESTA_OTRO}


def construir_grafo() -> StateGraph:
    """Construye el grafo sin compilarlo, útil para inspeccionar la topología."""
    grafo = StateGraph(EstadoEventHub)
    grafo.add_node(NODO_CLASIFICAR, clasificar)
    grafo.add_node(NODO_EVENTOS, responder_eventos)
    grafo.add_node(NODO_PEDIDO, responder_pedido)
    grafo.add_node(NODO_GENERAL, responder_general)

    grafo.add_edge(START, NODO_CLASIFICAR)
    grafo.add_conditional_edges(
        NODO_CLASIFICAR,
        enrutar,
        {
            NODO_EVENTOS: NODO_EVENTOS,
            NODO_PEDIDO: NODO_PEDIDO,
            NODO_GENERAL: NODO_GENERAL,
        },
    )
    grafo.add_edge(NODO_EVENTOS, END)
    grafo.add_edge(NODO_PEDIDO, END)
    grafo.add_edge(NODO_GENERAL, END)
    return grafo


def compilar_grafo():
    """Compila el grafo V1 listo para invocarse."""
    return construir_grafo().compile()


# Grafo compilado que consume el resto de la práctica.
graph = compilar_grafo()


def ruta_diagrama() -> Path:
    """Ruta del diagrama Mermaid relativa a la raíz del proyecto."""
    return Path(__file__).resolve().parents[1] / "diagrams" / "v1_graph.mmd"


def guardar_diagrama(destino: Path | None = None) -> Path:
    """Genera el diagrama con ``graph.get_graph().draw_mermaid()`` y lo guarda."""
    ruta = destino or ruta_diagrama()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    return ruta


# Casos de demostración al ejecutar el módulo. No forman parte del grafo.
_PRUEBAS_DEMO: tuple[tuple[str, str, NodoRespuesta], ...] = (
    ("Quiero ver los conciertos disponibles", INTENCION_EVENTOS, NODO_EVENTOS),
    ("¿Dónde está mi pedido EH-1003?", INTENCION_PEDIDO, NODO_PEDIDO),
    ("Quiero conocer la política de devoluciones", INTENCION_POLITICAS, NODO_GENERAL),
    ("Hola, ¿qué puedes hacer?", INTENCION_OTRO, NODO_GENERAL),
)


def _ejecutar_mensaje(mensaje: str) -> tuple[EstadoEventHub, str]:
    """Invoca el grafo y devuelve el estado final junto al nodo de respuesta visitado."""
    estado: EstadoEventHub = {"mensaje": mensaje}
    nodo_respuesta = ""
    for evento in graph.stream(estado, stream_mode="updates"):
        for nombre, actualizacion in evento.items():
            estado = {**estado, **actualizacion}
            if nombre.startswith("responder_"):
                nodo_respuesta = nombre
    return estado, nodo_respuesta


def ejecutar_demostracion() -> int:
    """Imprime las cuatro pruebas de la V1 y devuelve cuántas han sido correctas."""
    superadas = 0
    total = len(_PRUEBAS_DEMO)

    for numero, (mensaje, intencion_esperada, nodo_esperado) in enumerate(_PRUEBAS_DEMO, start=1):
        estado, nodo = _ejecutar_mensaje(mensaje)
        intencion = estado.get("intencion", "")
        respuesta = estado.get("respuesta", "")
        correcto = intencion == intencion_esperada and nodo == nodo_esperado
        superadas += int(correcto)

        print(f"Prueba {numero}")
        print(f"Mensaje enviado: {mensaje}")
        print(f"Intención detectada: {intencion}")
        print(f"Nodo de respuesta ejecutado: {nodo}")
        print(f"Respuesta obtenida: {respuesta}")
        print(f"Resultado: {'correcto' if correcto else 'incorrecto'}")
        print()

    print(f"{superadas}/{total} pruebas superadas")
    return superadas


if __name__ == "__main__":
    ruta = guardar_diagrama()
    ejecutar_demostracion()
    print(f"Diagrama generado: {ruta}")
