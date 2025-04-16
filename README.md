# Marco de Síntesis Colectiva (MSC) / Collective Synthesis Framework (MSC)

[![Estado Construcción](https://img.shields.io/badge/build-passing-brightgreen)] [![Licencia](https://img.shields.io/badge/license-MIT-blue.svg)][![License](https://img.shields.io/badge/License-BUSL--1.1-lightgrey)](LICENSE) [[Contribuciones Bienvenidas](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md) **Un marco para la emergencia de soluciones complejas y conocimiento estructurado a través de inteligencia colectiva sintética.**

## Resumen

El Marco de Síntesis Colectiva (MSC) es un proyecto de investigación y desarrollo que explora cómo la **inteligencia colectiva puede ir más allá de la clasificación o el consenso para realizar activamente la síntesis de conocimiento y la construcción de soluciones complejas**.

Proponemos un sistema donde agentes autónomos especializados (**Sintetizadores**) interactúan en un grafo dinámico (**Grafo de Síntesis**) que representa componentes de conocimiento y fragmentos de solución. A través de operaciones locales de **propuesta, evaluación, combinación y refutación**, guiadas por principios de **utilidad estimada y confianza**, el sistema busca generar estructuras emergentes (subgrafos) que representen soluciones coherentes, innovadoras y de alta calidad a problemas específicos.

**Visión:** Crear sistemas de IA capaces de colaborar en la resolución de problemas complejos, el descubrimiento científico, el diseño de ingeniería y la innovación creativa de una manera más análoga a la colaboración humana experta, pero a la escala y velocidad de la computación.

## Visión a Largo Plazo

Hemos esbozado una visión ambiciosa y detallada a largo plazo para MSC como una red descentralizada con un fuerte enfoque en la sostenibilidad y el impacto social, integrando conceptos de blockchain, DAO y el token $SYNTH.

➡️ Puedes leer la visión completa aquí: [**Documento de Visión (VISION.md)**](VISION.md)

## Autores

* **esraderey** - Conceptualización, Dirección, Arquitectura y Desarrollo.
* **Asistencia IA (Gemini / Synthia)** – Herramienta utilizada bajo dirección humana para acelerar la generación de código base y documentación. **No tiene coautoría ni derechos sobre este proyecto.**

## Estado Actual del Proyecto

Actualmente, el proyecto MSC se encuentra principalmente en la **Fase 1: Fundamentación Teórica y Simulación Avanzada**. Estamos refinando los modelos matemáticos, desarrollando simulaciones para validar las dinámicas centrales y explorando la teoría económica interna (ver Roadmap).

El código en este repositorio incluye una **implementación base de simulación en Python** diseñada para ilustrar los conceptos fundamentales (configurable vía `config.yaml` y argumentos CLI).

## Conceptos Clave

* **Grafo de Síntesis (G'):** Grafo dirigido dinámico (nodos=`V'`, aristas=`E'`).
* **Componente de Conocimiento (Nodo V'):** Unidad de info/solución con estado `sj`.
* **Relación de Síntesis (Arista E'):** Conexión dirigida con utilidad `uij`.
* **Estado del Nodo (`sj`):** Confianza/calidad estimada del nodo.
* **Utilidad/Compatibilidad (`uij`):** Calidad/fuerza estimada de la relación.
* **Sintetizador (Agente):** Realiza Operaciones de Síntesis (Proponer, Evaluar, Combinar, etc.).
* **Métrica de Calidad de Solución (Φ):** Evalúa la calidad de soluciones emergentes.

## Modelo Económico Interno (Ψ y Ω)

(En diseño/simulación) Basado en:
* **Reputación (Ψ - Psi):** Puntuación intransferible basada en calidad de contribuciones.
* **Recurso Computacional/Atencional (Ω - Omega):** Recurso consumible para acciones, ganado por contribuciones valiosas.

## Primeros Pasos (Simulación Base)

**Requisitos:**
* Python 3.7+
* PyYAML (`pip install -r requirements.txt`)

**Instalación y Ejecución:**

1.  Clonar: `git clone https://github.com/esraderey/synth-msc.git && cd synth-msc`
2.  (Opcional) Entorno virtual: `python -m venv venv && source venv/bin/activate` (o `venv\Scripts\activate` en Win)
3.  Instalar dependencias: `pip install -r requirements.txt`
4.  Ejecutar (ejemplos):
    * `python msc_simulation.py` (usa defaults internos)
    * `python msc_simulation.py --config config.yaml` (usa archivo de configuración)
    * `python msc_simulation.py --simulation_steps 50` (sobrescribe un parámetro)

## Contribuciones

¡Las contribuciones son bienvenidas! Consulta `CONTRIBUTING.md` (*pendiente*) y el `CODE_OF_CONDUCT.md` (*pendiente*). Reporta errores o ideas en [Issues](https://github.com/esraderey/synth-msc/issues).

## Roadmap del Proyecto

1.  ✅ **Fase 1:** Fundamentación Teórica y Simulación Avanzada.
2.   ✅ **Fase 2 (Parcial):** Núcleo (Simulador Base + Configuración Externa).
3.   EN PROGRESO **Fase 3:** Economía (Ψ, Ω), Especialización Agentes, Mejoras Evaluación/Visualización.
4.   FUTURO **Fase 4:** Expansión Dominio, Escalabilidad.
5.   FUTURO (VISIÓN) **Fase 5:** Descentralización (MSC Network), Ecosistema Abierto. (Ver [VISION.md](VISION.md))

## Documentación Adicional

* **Visión Detallada:** [VISION.md](VISION.md)
* **Whitepaper:** (*Próximamente*)
* **Documentación de API:** (*Próximamente*)

## Licencia

Este proyecto se distribuye bajo la **Business Source License 1.1 (BUSL-1.1)**.
El uso, modificación y distribución están permitidos para fines no comerciales, de investigación y experimentación según los términos especificados.
**Todo uso comercial antes del 16 de Abril de 2029 requiere permiso explícito del autor.** Después de esa fecha, el proyecto estará disponible bajo la Licencia MIT.

Ver el archivo [`LICENSE`](LICENSE) para los términos completos.

## Cita 
Si utilizas este trabajo en tu investigación, por favor cita:esraderey, & Synth . (2025). Marco de Síntesis Colectiva (MSC): Un Framework para Inteligencia Colectiva Sintética.