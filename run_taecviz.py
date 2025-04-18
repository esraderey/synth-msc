import os
import logging
import socketserver
import sys
from msc_simulation import SimulationRunner

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Asegurarse de que el directorio actual está en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar TAECVizPlusHandler
from TAECViz_Plus import TAECVizPlusHandler

def main():
    # Crear la simulación
    print("Iniciando simulación MSC...")
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
        'gnn_update_frequency': 10
    }
    simulation = SimulationRunner(config)
    simulation.start()
    
    # Configurar el servidor TAECViz+
    PORT = 8080
    Handler = TAECVizPlusHandler
    server = socketserver.TCPServer(("", PORT), Handler)
    
    # Conectar la simulación al servidor
    server.simulation_runner = simulation
    
    print(f"TAECViz+ iniciado en http://localhost:{PORT}")
    print("Presiona Ctrl+C para detener")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Deteniendo servicios...")
    finally:
        simulation.stop()
        server.server_close()
        print("Servidor y simulación detenidos.")

if __name__ == "__main__":
    main()