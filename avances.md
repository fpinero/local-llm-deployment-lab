# Avances

Cuaderno de bitácora del lab. Orden cronológico inverso (lo más reciente arriba). Las entradas existentes no se modifican.

## 2026-10-22: fase 0, prueba de control con Qwen3-4B desde el disco externo (modo guiado)

- Primer arranque con `-m /Volumes/Ext1Tb/models/Qwen3-4B-Q4_K_M.gguf`: funciona sin descarga ni caché.
- Pregunta del lavadero, 5 intentos: 0/5 (el intento 5 no se pudo juzgar porque `ask.py` solo mostraba el principio de la respuesta). De 1.390 a 2.224 tokens y de 35 a 59 s por respuesta. Un intento inventa que 150 m son "un cuarto de milla".
- Comparación final: 4B 0/6, 8B entre 2 y 3 de 10 (Q4_K_M y Q8_0 juntos). Coherente con la teoría, sin ser una prueba firme por el tamaño de la muestra.
- `scripts/ask.py` ahora muestra el principio y el final de las respuestas largas. Probado contra un servidor falso local.
- Lección de disco: el llama-server del Q8_0 seguía abierto tras el `rm -rf` y retenía 8,7 GB de un archivo borrado (detectado con `lsof`). Al pararlo, el disco pasó de 32 a 41 GiB libres.
- El razonamiento del Qwen3-4B queda medido con esta prueba: entre 1.390 y 2.224 tokens por respuesta, frente a unos 60 tokens de respuesta visible.

## 2026-10-22: fase 0, evaluación de razonamiento y almacenamiento externo (modo guiado)

- Creado `scripts/ask.py`: repite una pregunta N veces con el muestreo recomendado por Qwen (temperatura 0,6, top_p 0,95, top_k 20) y muestra cada respuesta para juzgarla a mano. Probado contra un servidor falso local.
- Pregunta trampa del lavadero (la respuesta correcta es ir en coche), 5 intentos con razonamiento activado:
  - Qwen3-8B Q4_K_M: 2/5 aciertos, de 693 a 3.498 tokens y de 31 a 156 s por respuesta.
  - Qwen3-8B Q8_0: de 0/5 a 1/5 (el intento 5 admite que conducir es la única opción, pero lo llama ineficiente).
- Lecciones: una respuesta no demuestra nada; razonar más ayuda pero no garantiza; con 5 intentos Q4_K_M y Q8_0 son indistinguibles.
- El disco interno llegó al 98% (12 GiB libres). El 4B y el 8B Q4_K_M se copiaron con `cp -L` a `/Volumes/Ext1Tb/models` (hash SHA-256 verificado) y se borraron las cachés de Hugging Face y de Ollama. A partir de ahora los modelos se cargan con `-m` desde el disco externo.
- Limpieza adicional de Docker hecha por el usuario (`docker builder prune -a` y `docker image prune -a --filter until=168h`): `Docker.raw` pasó de 53 GB a 12 GB. Disco interno: 32 GiB libres. Parte del espacio sigue retenido por una instantánea local de Time Machine.
- Detalle completo en `docs/phase-0-local.md`, paso 10 y sección "Model storage".

## 2026-10-22: fase 0, Ollama frente a llama.cpp (modo guiado)

- Instalado Ollama 0.40.8 con Homebrew y descargado el mismo GGUF (`hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`, mismo hash SHA-256 que la caché de llama.cpp).
- Hallazgo: Ollama 0.40 arranca internamente su propio `llama-server` (`-c 40960 -np 1 --context-shift --keep 4 -b 512`). El contexto queda en 40.960 porque los parámetros publicados por Qwen tienen prioridad sobre `OLLAMA_CONTEXT_LENGTH`.
- `ollama ps`: 8,8 GB en 100% GPU (2,5 GB de pesos y ~6 GB de caché KV). Presupuesto de GPU medido por Ollama: 11,8 GiB.
- Velocidad idéntica a llama-server: decode 40 tok/s y prefill del prompt largo 148,3 tok/s.
- Prueba de cabeceras (en local, sin túnel, porque Ollama no tiene autenticación): Ollama envía las cabeceras a los 204,97 s, junto con el primer token, incluso en streaming. Tras un Quick Tunnel daría 524. Es la causa concreta de los cortes de los tutoriales del vídeo.
- El muestreo cambia según la API: `ollama run` usa los valores recomendados por Qwen (temperatura 0,6) y el endpoint OpenAI usa temperatura 1,0.
- Correcciones registradas: Ollama 0.40.8 sí devuelve `timings` en la API OpenAI, y el contexto no siguió `OLLAMA_CONTEXT_LENGTH`.
- Prueba de razonamiento: con la pregunta trampa del lavadero, Qwen3-4B recomienda ir andando tras 531 tokens de razonamiento.
- `scripts/bench.py`: nueva opción `--reasoning-effort` (Ollama la necesita para desactivar el thinking).
- Conclusión: Ollama no penaliza la velocidad sino el control. Para servir modelos en el lab se usará llama-server.
- Detalle completo en `docs/phase-0-local.md`, paso 9.

## 2026-10-22: fase 0, servidor local expuesto con Cloudflare Quick Tunnel (modo guiado)

- Qwen3-4B Q4_K_M expuesto con `cloudflared tunnel --url http://localhost:8080` (cloudflared 2026.10.0, una conexión QUIC a Madrid).
- Verificado: sin API key la URL pública responde 401.
- Prompt corto por el túnel con streaming: funciona y los tokens llegan progresivamente (decode 41 tok/s). El TTFT de 1,6 s se debió sobre todo al servidor recién arrancado (prefill de 45 tok/s en la primera petición).
- Prompt largo (30.363 tokens) con streaming: llega bien tras 205 s de prefill, sin 524.
- El mismo prompt sin streaming y sin caché: HTTP 524. cloudflared registra `context canceled` y llama-server cancela la tarea a los ~124 s (74% del prefill).
- Conclusión: el límite de 100 s afecta al tiempo hasta que empieza la respuesta. llama-server envía las cabeceras al instante en streaming, así que el corte solo aparece sin streaming. Queda por comprobar el comportamiento de Ollama (fase 1).
- Corrección registrada: se esperaba que llama-server no detectara el corte en modo no streaming. Sí lo detecta, pero con unos 20-25 s de retraso.
- `scripts/bench.py`: nuevas opciones `--no-stream` y `--no-cache`, línea `headers received` y tiempo transcurrido en los errores HTTP. Probado contra servidores falsos locales.
- Detalle completo en `docs/phase-0-local.md`, paso 8.

## 2026-10-21: fase 0, Qwen3-8B en Q4_K_M y Q8_0 (modo guiado)

- Primera sesión en modo guiado: el usuario ejecuta los comandos en el panel Terminal y Claude explica la herramienta, los parámetros y la salida.
- Qwen3-8B Q4_K_M (5,03 GB, contexto 16k): decode 23,6 tok/s. Prompt largo de 14.190 tokens: TTFT 91,7 s y decode 17,2 tok/s.
- Qwen3-8B Q8_0 (8,71 GB, contexto 4k para que quepa en la memoria de la GPU): decode 20,1 tok/s.
- `llama-bench` (pp512 / tg128): Q4_K_M 224 / 24,5 tok/s y Q8_0 245 / 20,4 tok/s.
- Lecciones: el decode no escala de forma lineal con el tamaño (hay un coste fijo por token), Q8_0 aprovecha el bus mucho mejor que Q4_K_M en el M1 Pro (~87% frente a ~60%), cada formato gana en una fase (prefill Q8, decode Q4) y la caché KV se lee en cada token generado (16 ms extra a 14k de contexto).
- Probada la web de chat de llama-server con API key y razonamiento activo, y observada la reutilización de caché por LCP en una conversación.
- Detalle completo en `docs/phase-0-local.md`, paso 7.

## 2026-10-21: fase 0, Qwen3-4B en local con llama.cpp

- Instalado `llama.cpp` 0.7.0 con Homebrew (backend Metal).
- API key local generada en `.secrets/llama-api-key` (ignorado por Git). Verificado: sin key el servidor responde 401.
- `llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M -c 32768` descarga y carga el modelo en unos 2 minutos. La caché está en `~/.cache/huggingface/hub/`.
- Creado `scripts/bench.py` (solo librería estándar) para medir TTFT, prefill y decode contra cualquier endpoint compatible con OpenAI. Incluye los prompts corto y largo (~30.000 tokens).
- Resultados (M1 Pro 16 GB, Q4_K_M, sin thinking):
  - Prompt corto: TTFT 0,45 s, decode 39 tok/s.
  - Prompt largo (30.363 tokens): TTFT 208 s, prefill medio 146 tok/s (de 517 tok/s al inicio a unos 70 tok/s al final), decode 17 tok/s.
  - Prompt largo repetido: TTFT 0,17 s gracias a la caché de prefijo.
  - `llama-bench`: pp512 414 tok/s y tg128 42 tok/s con Flash Attention. A 16k de profundidad, Flash Attention multiplica por 2,5 el decode (24 frente a 9,4 tok/s).
- Conclusión: en este Mac un prompt de agente de 30k tokens superaría el corte de 100 s del Quick Tunnel. La caché de prefijo es lo que hace usables a los agentes tras el primer mensaje.
- Guía paso a paso y tablas completas en `docs/phase-0-local.md`.

## 2026-10-20: definición del laboratorio

- Analizada la transcripción del vídeo de Jorge Marfil (16/10/2026) sobre el "servidor de IA gratis de 32 GB" en Kaggle.
- Contrastados los datos clave con fuentes primarias:
  - README de kaggle-tpu-lab (695 estrellas a fecha de hoy, MIT, TPU v5e-8, unas 20 h de TPU a la semana, sesiones de algo menos de 9 h, API key, endpoints compatibles con OpenAI y Anthropic).
  - Documentación de Cloudflare Quick Tunnels: solo para pruebas, sin garantía de disponibilidad, 200 peticiones simultáneas y sin soporte de SSE.
  - Modelos en Hugging Face: `Qwen/Qwen3.8-27B` (oficial, Apache-2.0), `unsloth/Qwen3.8-27B-GGUF` (Qwen no publica GGUF oficial de este modelo), `Qwen/Qwen3-4B-GGUF` y `Qwen/Qwen3-8B-GGUF` (oficiales).
- Hardware local: Apple M1 Pro con 16 GB, suficiente para modelos de 4B y 8B cuantizados (fase 0).
- Creados `README.md` (descripción del lab, conceptos, herramientas, fases, reglas y primeros pasos), `TODO.md` y este archivo.
- Pendiente de verificar en el lab: las cifras del vídeo y del repo son hipótesis hasta medirlas.
