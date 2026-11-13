# Agent Platform con ADC

Cambio preparado desde `main`. Modelo seleccionado: `gemini-3-flash-preview`,
conservando el modelo actual. Preflight ADC aprobado el 12 de noviembre de 2026
(Perú); el usuario autorizó el merge y despliegue por CI/CD tras las pruebas.

El cliente de Gemini usa ADC cuando existe `VERTEX_AI_PROJECT`. En ese modo no se
pasa `GEMINI_API_KEY`, aunque haya una clave antigua en el entorno. La ubicación
predeterminada es `global` y la versión es `v1`. Se pueden configurar mediante
`VERTEX_AI_LOCATION` y `GEMINI_API_VERSION`.

Sin proyecto, el modo Developer API sigue requiriendo `GEMINI_API_KEY` para
desarrollo local y `cloudbuild.dev.yaml`. Si faltan ambos, el servicio rechaza la
configuración al arrancar. Análisis y consolidación comparten el mismo cliente.

`cloudbuild.yaml` selecciona ADC y deja de inyectar la clave de Gemini. Mantiene
`VISION_API_KEY` para autenticar callers del servicio y los demás secretos.
No elimina secretos de Secret Manager ni cambia permisos IAM.

Antes de activar en producción:

1. Verificar `gemini-3-flash-preview` con la API y este servicio mediante ADC.
   La selección está resuelta y `_GEMINI_MODEL` sigue configurable.
2. Verificar facturación Cloud, API `aiplatform.googleapis.com` y permiso de
   inferencia `aiplatform.endpoints.predict` para la identidad de Cloud Run.
   La identidad actual es `crashguard-api@seguros-489903.iam.gserviceaccount.com`.
3. Probar con esa identidad el análisis de imágenes y la consolidación: salida
   JSON, cajas, severidad, uso de tokens y latencia. Coordinar también las pruebas
   de fotos, VIN y video en la API.

Cloud Run obtiene credenciales de su identidad adjunta. No cargar un JSON ni
configurar `GOOGLE_APPLICATION_CREDENTIALS` en Cloud Run. Para uso local con ADC:
`gcloud auth application-default login`, más las variables anteriores.

Los triggers de producción se ejecutan con pushes a `main`. Tras el merge,
verificar el build del SHA fusionado, la revisión Ready, el tráfico al 100%,
las variables ADC y la respuesta de `/openapi.json`.

## Evidencia del preflight

[Cloud Build 9f80f6dd](https://console.cloud.google.com/cloud-build/builds;region=us-east4/9f80f6dd-bf09-4985-ba02-922f5c92e244?project=731075321554)
terminó en `SUCCESS`. Compiló el Dockerfile del commit `2f15b16`, que resolvió
`google-genai` 3.3.0, y ejecutó `_gemini_client()` y los adapters de análisis y
consolidación con la cuenta de servicio de producción, sin `GEMINI_API_KEY`.
Se comprobó la identidad mediante metadata y el conteo de tokens con ADC.

El análisis de una foto pública sin daños produjo JSON válido y 0 daños en 4.1 s
(1611 tokens de entrada, 443 de salida). La consolidación de dos hallazgos
sintéticos duplicados produjo un daño canónico en 3.9 s (586 de entrada, 809 de
salida). Los 78 tests locales también pasaron. Estas pruebas verifican integración;
no son una evaluación de exactitud con un conjunto de daños etiquetados.

La cuenta recibió el rol `projects/seguros-489903/roles/crashguardGeminiInference`,
que contiene únicamente `aiplatform.endpoints.predict`. No se otorgó suplantación
al operador. Se retiró el acceso temporal al archivo fuente del preflight.
La prueba no modificó registros de clientes.

Referencias: [ADC e identidad de Cloud Run](https://docs.cloud.google.com/run/docs/securing/service-identity),
[permisos](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/access-control),
[facturación Cloud y prepago](https://docs.cloud.google.com/billing/docs/how-to/split-charging-cycle).
