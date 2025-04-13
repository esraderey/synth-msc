# Guía para Contribuir al Proyecto MSC (`synth-msc`)

¡Muchas gracias por tu interés en contribuir al Marco de Síntesis Colectiva! Tu ayuda es bienvenida, ya sea reportando errores, sugiriendo ideas, mejorando la documentación o aportando código. A continuación, encontrarás las pautas para colaborar de manera efectiva.

## ¿Cómo Puedo Contribuir?

Buscamos ayuda en diversas áreas:
* Refinamiento de la lógica de los agentes (`EvaluatorAgent`, `ProposerAgent`, `CombinerAgent`).
* Implementación de nuevos tipos de agentes Sintetizadores.
* Mejora de la simulación (métricas, persistencia, configuración).
* Añadir visualizaciones más avanzadas.
* Mejorar la documentación del código y del proyecto.
* Añadir pruebas unitarias y de integración.
* Discusión teórica sobre los principios de MSC.

## Reportar Errores (Bugs)

Si encuentras un error en la simulación o la documentación:
1.  Revisa primero la [pestaña de Issues](https://github.com/esraderey/synth-msc/issues) para asegurarte de que no haya sido reportado ya.
2.  Si no existe, abre un nuevo "Issue".
3.  Proporciona un **título claro y descriptivo**.
4.  Incluye una **descripción detallada** del problema y los **pasos exactos para reproducirlo**, si es posible.
5.  Adjunta cualquier información relevante: versión del código (commit hash), sistema operativo, logs de error, capturas de pantalla, etc.

## Sugerir Mejoras o Nuevas Características

¿Tienes una idea para mejorar MSC?
1.  Abre un nuevo "Issue" describiendo tu propuesta.
2.  Explica claramente la **mejora o característica** que sugieres y **por qué** crees que sería valiosa para el proyecto.
3.  Si tienes ideas sobre la implementación, ¡compártelas! También puedes iniciar una conversación en la [pestaña de Discussions](https://github.com/esraderey/synth-msc/discussions) (si está activada) para ideas más abiertas.

## Flujo de Trabajo para Contribuir Código

1.  **Haz un Fork:** Crea una copia (fork) de este repositorio (`esraderey/synth-msc`) en tu propia cuenta de GitHub.
2.  **Clona tu Fork:** Clona tu copia localmente: `git clone https://github.com/TU_USUARIO/synth-msc.git`.
3.  **Crea una Rama:** Navega a tu repositorio local y crea una rama nueva y descriptiva para tus cambios: `git checkout -b TIPO/descripcion-corta` (ej. `feature/new-refuter-agent`, `fix/evaluator-logic-bug`, `docs/improve-readme`).
4.  **Realiza tus Cambios:** Escribe tu código o documentación.
5.  **Asegura la Calidad:**
    * Formatea tu código Python usando [**Black**](https://black.readthedocs.io/en/stable/). (Puedes instalarlo con `pip install black` y ejecutar `black .` en la raíz del proyecto).
    * Escribe docstrings claras siguiendo el estilo [**PEP 257**](https://peps.python.org/pep-0257/).
    * Añade comentarios donde sea necesario.
    * *Idealmente (a futuro)*: Añade pruebas para tus cambios y asegúrate de que pasen (`pytest` o similar).
6.  **Confirma tus Cambios (Commit):** Haz commit de tus cambios con mensajes claros y descriptivos: `git commit -m "Feat: Add RefuterAgent class"`
7.  **Sube tus Cambios (Push):** Sube tu rama a *tu fork* en GitHub: `git push origin feature/nueva-funcionalidad`
8.  **Abre un Pull Request (PR):** Ve a la página de tu fork en GitHub y haz clic en "Compare & pull request".
    * Asegúrate de que la base sea `esraderey/synth-msc` en la rama `main`.
    * Escribe un título y una descripción claros para tu PR, explicando *qué* cambiaste y *por qué*. Si tu PR resuelve un Issue existente, menciónalo en la descripción (ej. "Closes #12").
    * Envía el Pull Request.

Esperaremos a revisar tu PR, darte feedback si es necesario, y fusionarlo si todo está correcto.

## Configurar el Entorno de Desarrollo

1.  Clona el repositorio (tu fork o el principal si solo quieres ejecutarlo): `git clone https://github.com/esraderey/synth-msc.git`.
2.  Navega a la carpeta: `cd synth-msc`.
3.  (Recomendado) Crea y activa un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate # En Windows: venv\Scripts\activate
    ```
4.  Instala las dependencias: `pip install -r requirements.txt`.
5.  ¡Listo para ejecutar `python msc_simulation.py` o empezar a codificar!

## Código de Conducta

Esperamos que todos los contribuyentes sigan nuestro [**Código de Conducta**](CODE_OF_CONDUCT.md) (*pendiente de crear*) para mantener una comunidad abierta, respetuosa y colaborativa.

## Contacto y Discusión

Si tienes preguntas generales o necesitas ayuda para empezar a contribuir:
* Usa la [pestaña de Issues](https://github.com/esraderey/synth-msc/issues).
* Participa en [Discussions](https://github.com/esraderey/synth-msc/discussions) (si se activa).

¡Agradecemos sinceramente tu tiempo y esfuerzo para mejorar MSC!