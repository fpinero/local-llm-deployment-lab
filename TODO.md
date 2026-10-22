# TODO

Cola de trabajo del lab. Solo tareas pendientes. Al completar y verificar una tarea, se elimina de aquí y se registra en `avances.md`.

## Preparación

- [ ] Crear cuenta de Kaggle, verificar el teléfono y anotar la cuota semanal real de GPU y TPU (Settings).
- [ ] Comprobar si la cuenta necesita verificación de identidad (Persona) para usar TPU.
- [ ] Crear un pequeño conjunto de preguntas de evaluación (empezando por la trampa del lavadero) para comparar modelos en todas las fases.

## Fase 0: local en el Mac

- [ ] (Opcional) Comprobar si la primera petición tras 10 minutos de inactividad tiene un prefill más lento (hipótesis: la GPU baja su frecuencia en reposo).
- [ ] (Opcional) Medir en Ollama el efecto de `OLLAMA_KV_CACHE_TYPE=q8_0` en memoria y velocidad.
- [ ] (Opcional) Convertir y cuantizar un modelo pequeño con `convert_hf_to_gguf.py` y `llama-quantize`.
- [ ] (Opcional) Probar el túnel desde fuera de casa (móvil con datos) para medir la latencia real de red.

## Fase 1: Kaggle 2x T4

Guía en `docs/phase-1-kaggle-t4.md`.

- [ ] (Opcional) Crear un Dataset privado de Kaggle con el GGUF para adjuntarlo sin descargar (la descarga cuesta ~4 min por sesión).
- [ ] (Opcional) Medir `--split-mode row` para completar la comparación con `layer` y `tensor`.
- [ ] (Opcional) Subir `-c` por encima de 32k y medir cuánta caché KV cabe en la VRAM libre (~12 GiB).
- [ ] (Opcional) Pregunta del lavadero al 27B con `ask.py` (5 intentos) para compararlo con el 4B y el 8B.
- [ ] (Opcional) Medir el umbral exacto (~30 s) en el que el túnel libera las cabeceras, con prompts de entre 20 y 40 s de prefill.
- [ ] (Opcional, si la cuota lo permite) Servir el mismo GGUF con Ollama y confirmar en las T4 que retiene las cabeceras hasta el primer token (medido en el Mac con la 0.40.8) y que llama-server no.
- [ ] (Opcional) vLLM con tensor parallel 2 en un modelo que quepa en fp16.

## Fase 2: Kaggle TPU v5e-8

- [ ] Leer el código de kaggle-tpu-lab (`qwen38-27b/kernel/` y `launch.py`).
- [ ] Primera ejecución por la vía notebook.
- [ ] Segunda ejecución por terminal con `launch.py serve --keepalive-min 90`.
- [ ] Ejecutar los benchmarks y contrastar con las cifras del repo (130 tok/s, 10.300 tok/s, 22 min).
- [ ] Conectar un agente de programación para una tarea corta y no sensible.

## Fase 3: comparación

- [ ] Ejecutar los mismos prompts contra un proveedor alojado (OpenRouter).
- [ ] Rellenar la tabla de resultados y calcular el coste efectivo por millón de tokens.
- [ ] Redactar las conclusiones en `avances.md`.

## Publicación

- [ ] Inicializar el repositorio Git y publicarlo en GitHub.
