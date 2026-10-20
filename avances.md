# Avances

Cuaderno de bitácora del lab. Orden cronológico inverso (lo más reciente arriba). Las entradas existentes no se modifican.

## 2026-10-20: definición del laboratorio

- Analizada la transcripción del vídeo de Jorge Marfil (16/10/2026) sobre el "servidor de IA gratis de 32 GB" en Kaggle.
- Contrastados los datos clave con fuentes primarias:
  - README de kaggle-tpu-lab (695 estrellas a fecha de hoy, MIT, TPU v5e-8, unas 20 h de TPU a la semana, sesiones de algo menos de 9 h, API key, endpoints compatibles con OpenAI y Anthropic).
  - Documentación de Cloudflare Quick Tunnels: solo para pruebas, sin garantía de disponibilidad, 200 peticiones simultáneas y sin soporte de SSE.
  - Modelos en Hugging Face: `Qwen/Qwen3.8-27B` (oficial, Apache-2.0), `unsloth/Qwen3.8-27B-GGUF` (Qwen no publica GGUF oficial de este modelo), `Qwen/Qwen3-4B-GGUF` y `Qwen/Qwen3-8B-GGUF` (oficiales).
- Hardware local: Apple M1 Pro con 16 GB, suficiente para modelos de 4B y 8B cuantizados (fase 0).
- Creados `README.md` (descripción del lab, conceptos, herramientas, fases, reglas y primeros pasos), `TODO.md` y este archivo.
- Pendiente de verificar en el lab: las cifras del vídeo y del repo son hipótesis hasta medirlas.
