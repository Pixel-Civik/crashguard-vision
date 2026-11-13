# Agent Platform con ADC

Cambio preparado desde `main`; activación y modelo pendientes de revisión.

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

1. Elegir el modelo junto con la API. `_GEMINI_MODEL` sigue configurable y mantiene
   el valor actual solo como línea base. No se ha aprobado un reemplazo.
2. Verificar facturación Cloud, API `aiplatform.googleapis.com` y permiso de
   inferencia `aiplatform.endpoints.predict` para la identidad de Cloud Run.
   La identidad actual es `crashguard-api@seguros-489903.iam.gserviceaccount.com`.
3. Probar con esa identidad el análisis de imágenes y la consolidación: salida
   JSON, cajas, severidad, uso de tokens y latencia. Coordinar también las pruebas
   de fotos, VIN y video en la API.

Cloud Run obtiene credenciales de su identidad adjunta. No cargar un JSON ni
configurar `GOOGLE_APPLICATION_CREDENTIALS` en Cloud Run. Para uso local con ADC:
`gcloud auth application-default login`, más las variables anteriores.

Los triggers actuales de producción se ejecutan con pushes a `main`. Mantener
la PR sin fusionar hasta autorizar el despliegue. El cambio no activa inferencia
en producción mientras permanezca en una rama de revisión.

Referencias: [ADC e identidad de Cloud Run](https://docs.cloud.google.com/run/docs/securing/service-identity),
[permisos](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/access-control),
[facturación Cloud y prepago](https://docs.cloud.google.com/billing/docs/how-to/split-charging-cycle).
