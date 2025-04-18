import logging
import argparse
import os
from msc_simulation import SimulationRunner, load_config

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Ejecutar simulación MSC con visualizador TAECViz+")
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración YAML')
    parser.add_argument('--viz-port', type=int, default=8080, help='Puerto para el visualizador')
    args = parser.parse_args()
    
    # Cargar configuración
    if args.config and os.path.exists(args.config):
        config = load_config(args)
    else:
        # Configuración por defecto
        config = {
            'simulation_steps': None,  # Ejecutar indefinidamente
            'num_proposers': 3,
            'num_evaluators': 6,
            'num_combiners': 2,
            'step_delay': 0.5,
            'initial_omega': 100.0,
            'omega_regeneration_rate': 0.1,
            'num_bridging_agents': 2,
            'num_synthesizer_agents': 3, 
            'num_pattern_miner_agents': 2,
            'gnn_update_frequency': 10,
            'viz_port': args.viz_port,
            'viz_active': True
        }
    
    # Crear y ejecutar la simulación
    print("\n🚀 Iniciando simulación MSC con TAECViz+...")
    simulation = SimulationRunner(config)
    
    try:
        simulation.start()
        print(f"\n✅ Simulación iniciada correctamente!")
        print(f"🔍 Visualizador disponible en: http://localhost:{config.get('viz_port', 8080)}")
        print("\nPresiona Ctrl+C para detener la simulación...\n")
        
        # Mantener el programa en ejecución
        while True:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo simulación...")
        simulation.stop()
        print("👋 Simulación detenida.")

if __name__ == "__main__":
    main()