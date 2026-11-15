# ThePower LangGraph EventHub

Asistente de práctica para consultar eventos, pedidos y políticas de EventHub. El proyecto reconstruye ese asistente con LangGraph, de un grafo por reglas hasta un chatbot con modelo de lenguaje, memoria y herramientas.

No está conectado a una plataforma real de venta de entradas. Los eventos, pedidos y políticas son datos locales de demostración. PostgreSQL, cuando se usa, solo persiste el estado de la conversación de LangGraph.

## Objetivo de la actividad

Reconstruir de forma progresiva un asistente virtual EventHub:

1. Clasificar intenciones con reglas y enrutar el grafo.
2. Sustituir el estado propio por un historial de mensajes y un modelo de lenguaje.
3. Conservar la conversación por `thread_id`, primero en memoria y después en PostgreSQL.
4. Dejar que el modelo consulte herramientas reales antes de responder.

Cada versión es un módulo independiente. No hay una aplicación única que las ejecute a la vez.

## Arquitectura

| Componente | Qué hace en este proyecto |
|---|---|
| Estado | V1 usa un `TypedDict` con `mensaje`, `intencion` y `respuesta`. A partir de V2 el estado es `MessagesState`. |
| Nodos | Funciones de Python registradas en un `StateGraph`. V1 clasifica y responde con textos fijos. V2, V3 y V4 comparten un nodo `chatbot`. V4 añade el nodo `herramientas`. |
| Grafos | Cada versión compila su propio grafo. El flujo condicional está en el código, no en un orquestador externo. |
| Modelo | `ChatOpenAI`, configurado con `OPENAI_API_KEY` y `OPENAI_MODEL`. V1 no llama a ningún modelo. |
| Herramientas | Solo V4. Reutiliza `buscar_eventos` y `consultar_pedido` del proyecto anterior y las ejecuta con `ToolNode`. |
| Datos | `data/events.json`, `data/orders.json` y `data/policies.md`. Las políticas entran en el prompt desde V2. Los JSON los leen las herramientas de V4. |
| Checkpointers | V3 usa `InMemorySaver`. La variante PostgreSQL usa `PostgresSaver`. V1, V2 y V4 no tienen checkpointer. |

```mermaid
flowchart TD
    usuario[Terminal]
    env[".env"]
    datos["data/events.json\ndata/orders.json\ndata/policies.md"]
    v1["V1\nreglas y rutas condicionales"]
    v2["V2\nMessagesState y ChatOpenAI"]
    v3["V3\nInMemorySaver"]
    v3pg["V3 PostgreSQL\nPostgresSaver"]
    v4["V4\nToolNode"]
    modelo["API del modelo\nOPENAI_MODEL"]
    postgres[("PostgreSQL\nPOSTGRES_URI")]
    herramientas["buscar_eventos\nconsultar_pedido"]

    usuario --> v1
    usuario --> v2
    usuario --> v3
    usuario --> v3pg
    usuario --> v4
    env --> v2
    env --> v3
    env --> v3pg
    env --> v4
    datos --> v2
    datos --> v3
    datos --> v3pg
    v2 --> modelo
    v3 --> modelo
    v3pg --> modelo
    v4 --> modelo
    v3pg --> postgres
    v4 --> herramientas
    herramientas --> datos
```

## Evolución del proyecto

### V1: grafo simple

`graphs/v1_simple.py` no usa modelo ni API. El estado es un `TypedDict`. El nodo `clasificar` detecta `eventos`, `pedido`, `politicas` u `otro` con reglas. `add_conditional_edges` envía el flujo a `responder_eventos`, `responder_pedido` o `responder_general`, y de ahí a `END`. `politicas` y `otro` comparten el nodo general.

### V2: chatbot

`graphs/v2_chatbot.py` sustituye el estado por `MessagesState`. El nodo `chatbot` lee `data/policies.md`, construye un prompt de sistema en español y llama a `ChatOpenAI`. El grafo es `START` → `chatbot` → `END`. El bucle de terminal conserva el historial pasando los mensajes anteriores en cada invocación. No hay checkpointer. Termina con `salir` o `exit`.

### V3: memoria en el proceso

`graphs/v3_memory.py` reutiliza el nodo, el prompt y el modelo de V2, y compila el grafo con `InMemorySaver`. Cada conversación usa su `thread_id` (`conversacion_1` y `conversacion_2`). Las llamadas del mismo hilo reenvían solo el mensaje nuevo; el checkpointer recupera el resto. `graph.get_state(config)` lee el historial. Esa memoria desaparece al cerrar el proceso.

### V4: herramientas

`graphs/v4_tools.py` registra `buscar_eventos` y `consultar_pedido` en el modelo y las ejecuta con `ToolNode`. Si el último `AIMessage` trae `tool_calls`, el flujo va a `herramientas` y vuelve a `chatbot`. Si no hay llamadas, termina. `consultar_pedido` exige identificador y email coincidente. `buscar_eventos` filtra por ciudad y categoría. El límite de 12 pasos se aplica con `recursion_limit` al hacer `graph.stream()`; no es un nodo del grafo. Si se supera, se captura `GraphRecursionError` y se muestra un mensaje de parada.

Las implementaciones de las herramientas no están en este repositorio. V4 las carga desde `../thepower-langchain-eventhub/tools/` y las apunta a los JSON de este proyecto.

## Extra: PostgreSQL

`graphs/v3_postgres.py` es una variante de V3. No modifica el grafo de memoria en RAM y no usa `InMemorySaver`.

`PostgresSaver.from_conn_string(POSTGRES_URI)` abre la conexión durante el modo elegido y la cierra al terminar. `checkpointer.setup()` crea las tablas del checkpointer. Los dos modos usan el hilo `eventhub_postgres_demo` y no borran el historial.

| | `InMemorySaver` | `PostgresSaver` |
|---|---|---|
| Dónde vive el historial | En el proceso de Python | En PostgreSQL |
| Tras cerrar el programa | Se pierde | Se puede recuperar con el mismo `thread_id` |
| Módulo | `graphs/v3_memory.py` | `graphs/v3_postgres.py` |

PostgreSQL no sustituye a `orders.json` ni a `events.json`. Solo guarda checkpoints de LangGraph.

## Estructura de carpetas

```text
.
├── data/
│   ├── events.json
│   ├── orders.json
│   └── policies.md
├── diagrams/
│   ├── v1_graph.mmd
│   └── v4_graph.mmd
├── evidencias/
│   ├── v1_diagrama.png
│   ├── v1_pruebas.png
│   ├── v2_conversacion.png
│   ├── v3_memoria.png
│   ├── v3_postgres_guardar.png
│   ├── v3_postgres_recuperar.png
│   ├── v4_conversacion.png
│   └── v4_diagrama.png
├── graphs/
│   ├── v1_simple.py
│   ├── v2_chatbot.py
│   ├── v3_memory.py
│   ├── v3_postgres.py
│   └── v4_tools.py
├── tests/
│   └── test_v1_simple.py
├── tools/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

- `data/`: catálogo, pedidos y políticas ficticias. No los genera el código.
- `diagrams/`: diagramas Mermaid de V1 y V4.
- `evidencias/`: capturas de las pruebas ya realizadas.
- `graphs/`: una versión por archivo. No hay un grafo que las combine.
- `tests/`: cuatro pruebas de unittest para las intenciones de V1. No hay pruebas automáticas de V2, V3 ni V4.
- `tools/`: existe y está vacía. Las herramientas que ejecuta V4 viven en el proyecto LangChain anterior.
- `.env`: configuración local. No debe publicarse. No está en el ejemplo de arriba porque `.gitignore` lo excluye.

## Requisitos previos

- WSL2 con Ubuntu.
- Python 3. La revisión de este entorno se hizo con Python 3.14.4. No se han probado otras versiones.
- `venv`, incluido en la instalación habitual de Python.
- Git, solo cuando se vaya a publicar el repositorio. Este directorio todavía no es un repositorio Git.
- Una clave y un identificador de modelo aceptados por la API configurada en `OPENAI_MODEL`. V1 no los necesita.
- PostgreSQL accesible desde Ubuntu, solo para `graphs/v3_postgres.py`.
- El proyecto `thepower-langchain-eventhub` como directorio hermano, solo para V4.

## Instalación

Desde la raíz del proyecto, en la terminal de Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para la variante PostgreSQL hacen falta paquetes que el entorno actual tiene instalados, pero que `requirements.txt` todavía no declara:

```bash
python -m pip install "langgraph-checkpoint-postgres==3.2.1" "psycopg==3.3.9"
```

`psycopg-pool` se instala como dependencia de `langgraph-checkpoint-postgres`.

## Dependencias

`requirements.txt` fija:

- `langchain==1.4.9`
- `langchain-openai==1.7.4`
- `openai==3.46.0`
- `python-dotenv==1.2.4`
- `langgraph==1.2.22`

Importaciones adicionales usadas por el código y no listadas en ese archivo:

- `langgraph-checkpoint-postgres==3.2.1`
- `psycopg==3.3.9`

## Configuración

Copia la plantilla y edítala. No subas el archivo real.

```bash
cp .env.example .env
```

Variables que lee el código:

| Variable | Dónde se usa | Qué poner |
|---|---|---|
| `OPENAI_API_KEY` | V2, V3, V4 y V3 PostgreSQL | Clave de la API. El programa no la imprime. |
| `OPENAI_MODEL` | Las mismas versiones | Identificador de modelo que acepte esa API. El código no elige otro si este falla. |
| `POSTGRES_URI` | Solo `graphs/v3_postgres.py` | URI de PostgreSQL. El programa no la imprime. |

V1 no lee `.env`.

Ejemplo ficticio, no válido para conectar:

```text
OPENAI_API_KEY=pon_aqui_tu_api_key
OPENAI_MODEL=identificador-de-tu-modelo
POSTGRES_URI=postgresql://usuario:contrasena@localhost:5432/eventhub
```

Para PostgreSQL, la base debe existir y el usuario de la URI debe poder crear las tablas del checkpointer. El programa llama a `checkpointer.setup()`, pero no crea la base de datos.

## Ejecución

Activa antes el entorno:

```bash
source .venv/bin/activate
```

Todos los comandos se lanzan desde la raíz del proyecto.

### V1

```bash
python -m graphs.v1_simple
python -m unittest tests.test_v1_simple -v
```

El primer comando imprime cuatro casos y regenera `diagrams/v1_graph.mmd`.

### V2

```bash
python -m graphs.v2_chatbot
```

Escribe mensajes y termina con `salir` o `exit`.

### V3 en memoria

```bash
python -m graphs.v3_memory
```

Ejecuta sola la demostración de `conversacion_1` y `conversacion_2`.

### V4

```bash
python -m graphs.v4_tools
```

Requiere que exista `../thepower-langchain-eventhub/tools/`.

### V3 con PostgreSQL

```bash
python -m graphs.v3_postgres guardar
python -m graphs.v3_postgres recuperar
```

El segundo comando debe lanzarse en otro proceso, después de que el primero haya terminado. Así se comprueba que el historial no dependía de la memoria del intérprete.

Sin argumento, el módulo muestra estas dos formas de uso.

## Ejemplos de uso

V1, por reglas:

```text
Usuario: ¿Qué conciertos hay en la cartelera?
Intención: eventos
Nodo: responder_eventos
```

V2 y V3 conservan el turno anterior porque el estado es una lista de mensajes. Una comprobación habitual es decir el nombre y, en el siguiente turno, preguntar cómo te llamas.

V3 separa los hilos. Lo dicho en `conversacion_1` no aparece en `conversacion_2`.

V4, con los datos actuales:

```text
Usuario: Mi pedido es PED-1004 y el email es demo.madrid@correo-ficticio.test. ¿Es para un concierto en Madrid? Si es así, dime otros conciertos en Madrid ese mes.
```

El grafo consulta primero el pedido. Si el email coincide, el pedido devuelve `EVT-009`. Después busca conciertos en Madrid y puede citar los de junio que no son el del pedido: Jazz en el Retiro y Rock en La Latina. No debe incluir el concierto de julio.

`consultar_pedido` no muestra datos si falta el email o no coincide con el registrado. No inventa un correo.

La variante PostgreSQL guarda estos dos mensajes en `eventhub_postgres_demo`:

```text
Me llamo Ana y mi ciudad favorita es Madrid
Trabajo en el sector de la música y me gustan los conciertos
```

`recuperar` lee ese estado y después pregunta cómo te llamas, cuál es tu ciudad favorita y qué tipo de música te gusta. La respuesta tiene que salir de los mensajes persistidos. El segundo mensaje dice que te gustan los conciertos; no indica un género concreto.

## Datos de demostración

### `data/events.json`

Doce eventos ficticios. Cada uno tiene `id`, `nombre`, `ciudad`, `categoria`, `fecha`, `precio` y `disponibilidad`.

- `EVT-001` a `EVT-005`: conciertos en Valencia, entre junio y octubre de 2026.
- `EVT-006`: Feria del Libro de Otoño, Madrid, cultura, 18 de octubre de 2026.
- `EVT-007`: Maratón Nocturna del Puerto, Barcelona, deporte.
- `EVT-008`: Taller de Cocina Mediterránea, Sevilla, gastronomía.
- `EVT-009`: Noche Indie en Malasaña, Madrid, concierto, 12 de junio de 2026.
- `EVT-010`: Jazz en el Retiro, Madrid, concierto, 21 de junio de 2026.
- `EVT-011`: Rock en La Latina, Madrid, concierto, 27 de junio de 2026.
- `EVT-012`: Clásicos en el Auditorio, Madrid, concierto, 18 de julio de 2026.

### `data/orders.json`

Cuatro pedidos ficticios. Los correos son de demostración y están en el propio JSON.

- `PED-1001`: confirmado, `EVT-001` y `EVT-003`.
- `PED-1002`: pendiente, `EVT-006`.
- `PED-1003`: cancelado, `EVT-008`.
- `PED-1004`: confirmado, `EVT-009`, correo `demo.madrid@correo-ficticio.test`.

No existe ningún pedido `EH-1003`.

### `data/policies.md`

Normas ficticias de devoluciones, cancelaciones, cambios, reembolsos y plazos. V2 y las versiones posteriores las copian al prompt. No se consultan con una herramienta.

## Diagramas y evidencias

Los diagramas Mermaid están en `diagrams/`:

- `diagrams/v1_graph.mmd`: generado con `graph.get_graph().draw_mermaid()`.
- `diagrams/v4_graph.mmd`: mismos nodos y aristas que el grafo compilado de V4. Las aristas discontinuas son la decisión sobre `tool_calls`.

No hay diagrama de V2 ni de V3 porque su grafo es solo `START` → `chatbot` → `END`.

Las capturas están en `evidencias/`:

- `v1_diagrama.png` y `v1_pruebas.png`
- `v2_conversacion.png`
- `v3_memoria.png`
- `v3_postgres_guardar.png` y `v3_postgres_recuperar.png`
- `v4_conversacion.png` y `v4_diagrama.png`

Para ver un `.mmd`, usa la vista previa de Mermaid en VS Code o pega el contenido en [https://mermaid.live](https://mermaid.live).

## Limitaciones

- Los eventos, pedidos y políticas son ficheros locales. No hay una tienda real ni una base de datos de pedidos.
- PostgreSQL persiste checkpoints de LangGraph. No es el origen de los pedidos ni de la cartelera.
- V2 no puede consultar disponibilidad ni el estado de un pedido: el prompt indica que no tiene herramientas.
- V3 en memoria no sobrevive a otro proceso.
- V4 no funciona si falta el directorio hermano `thepower-langchain-eventhub`.
- `buscar_eventos` no filtra por mes ni por identificador. El mes lo deduce el modelo a partir de las fechas devueltas.
- `consultar_pedido` no devuelve el correo registrado y no muestra el pedido si el email no coincide.
- El identificador de modelo no se sustituye si la API lo rechaza.
- No hay interfaz web, API REST, autenticación ni despliegue.

## Solución de problemas

**`ModuleNotFoundError: langgraph` o `langchain_openai`.** El entorno virtual no está activo o falta la instalación. Ejecuta `source .venv/bin/activate` y `python -m pip install -r requirements.txt`.

**`No module named pytest`.** Las pruebas de V1 usan unittest, no pytest:

```bash
python -m unittest tests.test_v1_simple -v
```

**`OPENAI_API_KEY no está configurada`.** `.env` no está en la raíz o todavía contiene el valor de ejemplo. El programa no muestra la clave.

**La API rechaza `OPENAI_MODEL`.** El identificador no es válido para esa cuenta. Cámbialo en `.env`. El código no elige un modelo alternativo.

**`POSTGRES_URI no está configurada`.** Añade la variable en `.env`. Hace falta solo para `graphs/v3_postgres.py`.

**Error de conexión a PostgreSQL.** Comprueba que el servidor esté en marcha dentro de Ubuntu, que la base exista y que la URI sea alcanzable desde WSL. El mensaje del programa oculta usuario y contraseña. Si el fallo es de autenticación, revisa la URI sin pegarla en issues ni en capturas.

**`No module named langgraph.checkpoint.postgres` o `psycopg`.** Instala los paquetes adicionales indicados en la sección de instalación.

**V4 no encuentra las herramientas.** Tiene que existir `../thepower-langchain-eventhub/tools/events_tool.py` y `orders_tool.py`. Este repositorio no incluye su código.

**El pedido no se consulta.** El mensaje debe incluir un `id_pedido` que exista y el email asociado. Un email vacío produce «Faltan datos» y no llega a buscar eventos.

## Mejoras futuras

No están implementadas. Serían extensiones posibles:

- Declarar en `requirements.txt` los paquetes de PostgreSQL.
- Copiar las herramientas a `tools/` para que el repositorio sea autónomo.
- Añadir pruebas automáticas de V2, V3 y V4 que no dependan de una llamada real a la API.
- Exponer el grafo como API REST.
- Añadir una interfaz web.
- Consultar pedidos y eventos en una base de datos de negocio, distinta del checkpointer.
- Añadir autenticación.
- Incorporar nuevas herramientas, por ejemplo la reserva de entradas que existe en el proyecto LangChain anterior y aquí no se usa.
- Preparar un despliegue.

## Licencia y autoría

Licencia: pendiente de definir. Este repositorio no incluye un archivo `LICENSE`.

Autoría: pendiente de definir. No se ha añadido un nombre, una empresa ni un enlace que no conste en el proyecto.
