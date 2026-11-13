# Tokens de los reportes de Vision

## Causa y corrección

`vision_session_images` guarda `gemini_call_id`, pero no guarda columnas de tokens
ni de latencia. Esas métricas están en `vision_analysis_calls`. El reporte sumaba
campos inexistentes de las imágenes y los convertía en cero. Los eventos canónicos
de consumo sí tenían los valores de Gemini, incluido el razonamiento facturable.

El repositorio ahora carga las métricas mediante la relación existente
`gemini_call_id`. Cada imagen usa su llamada seleccionada, sin sumar los intentos
anteriores. La respuesta de una imagen reutilizada también devuelve sus métricas
guardadas; si faltan, devuelve valores desconocidos (`null`), no ceros inventados.

Los reportes nuevos suman imágenes y consolidación. Al leer un mapa en caché se
recalculan los totales con las llamadas guardadas y la última consolidación exitosa
anterior a `built_at`. Se conservan los hallazgos del caché, sin repetir inferencias,
crear eventos de consumo ni reescribir filas. Si falla esa lectura de métricas,
el error se propaga y no provoca una consolidación pagada como fallback.

No se requiere migración. No se hace un backfill de los snapshots ya copiados
en `inspection_ai_runs` de la API; los reportes nuevos reciben los totales correctos
y el endpoint de reportes de Vision corrige sus respuestas en caché al leerlas.

## Verificación

- 88 tests pytest: PASS. Cobertura de imágenes nuevas/reutilizadas, relación de
  métricas, agregación, caché con métricas antiguas, ausencia de métricas y fallo
  de base de datos sin inferencia pagada adicional.
- Verificación de solo lectura con el código de la rama sobre nuestra inspección
  de prueba `bbc494c3-24ab-48bd-9c08-23783fffc040`, sesión Vision
  `de58bf9a-db59-4ac0-af74-7889c8f29e3c`, en el proyecto de producción confirmado
  desde la configuración desplegada (`otppptvxawpqvhgrbydm`).
- Caché anterior: 0 entrada, 0 salida, 0 ms. Respuesta corregida: **3201 tokens de
  entrada, 926 de salida y 9802 ms**, iguales a las llamadas guardadas de las dos
  imágenes. Sin escritura de producción ni llamada nueva a Gemini.
- Reutilización de imagen: 1614 tokens de entrada y 430 de salida, conservados.

El runtime de Vision había resuelto Google GenAI Python 3.3.0; la API usa Google
GenAI JS 1.48.0. Los tokens registrados demuestran que el parser funcionaba en
esa ejecución. La causa comprobada está en la lectura/presentación del reporte;
esta corrección no cambia el modelo, ADC ni las dependencias.
