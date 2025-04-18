import logging
import argparse
from msc_simulation import SimulationRunner, load_config

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Ejecutar simulación MSC con visualización TAECViz+")
    parser.add_argument('--config', type=str, default='config.yaml', help='Archivo de configuración')
    parser.add_argument('--viz-port', type=int, default=8080, help='Puerto para el visualizador')
    parser.add_argument('--no-viz', action='store_true', help='Desactivar visualizador')
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Agregar configuración de visualización
    config['viz_port'] = args.viz_port
    config['viz_active'] = not args.no_viz
    
    # Crear y ejecutar simulación
    simulation = SimulationRunner(config)
    
    try:
        print("🚀 Iniciando simulación MSC...")
        simulation.start()
        print("✅ Simulación iniciada correctamente.")
        
        if config['viz_active']:
            print(f"🔍 Visualizador disponible en: http://localhost:{config['viz_port']}")
            print("Presione Ctrl+C para detener...")
        
        # Mantener el programa en ejecución
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("⏹️ Deteniendo simulación...")
        simulation.stop()
        print("👋 Simulación detenida.")

if __name__ == "__main__":
    main()