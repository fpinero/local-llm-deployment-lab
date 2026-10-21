# Avances

Cuaderno de bitácora del lab. Orden cronológico inverso (lo más reciente arriba). Las entradas existentes no se modifican.

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
