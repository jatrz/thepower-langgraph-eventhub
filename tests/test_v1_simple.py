"""Pruebas del grafo V1: una por cada intención, sin LLM."""

import unittest

from graphs.v1_simple import (
    INTENCION_EVENTOS,
    INTENCION_OTRO,
    INTENCION_PEDIDO,
    INTENCION_POLITICAS,
    RESPUESTA_EVENTOS,
    RESPUESTA_OTRO,
    RESPUESTA_PEDIDO,
    RESPUESTA_POLITICAS,
    graph,
)


class TestGrafoV1(unittest.TestCase):
    def test_intencion_eventos(self) -> None:
        resultado = graph.invoke({"mensaje": "¿Qué conciertos hay en la cartelera?"})

        self.assertEqual(resultado["intencion"], INTENCION_EVENTOS)
        self.assertEqual(resultado["respuesta"], RESPUESTA_EVENTOS)

    def test_intencion_pedido(self) -> None:
        resultado = graph.invoke({"mensaje": "Quiero consultar el estado de mi pedido PED-1001"})

        self.assertEqual(resultado["intencion"], INTENCION_PEDIDO)
        self.assertEqual(resultado["respuesta"], RESPUESTA_PEDIDO)

    def test_intencion_politicas(self) -> None:
        resultado = graph.invoke({"mensaje": "¿Cuál es la política de devoluciones?"})

        self.assertEqual(resultado["intencion"], INTENCION_POLITICAS)
        self.assertEqual(resultado["respuesta"], RESPUESTA_POLITICAS)

    def test_intencion_otro(self) -> None:
        resultado = graph.invoke({"mensaje": "Hola, ¿qué tiempo hace hoy?"})

        self.assertEqual(resultado["intencion"], INTENCION_OTRO)
        self.assertEqual(resultado["respuesta"], RESPUESTA_OTRO)


if __name__ == "__main__":
    unittest.main()
