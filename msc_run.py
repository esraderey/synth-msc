#!/usr/bin/env python3
"""
MSC Framework - Script Principal Integrado
Este script integra todos los componentes del framework MSC
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path

# Agregar el directorio actual al path de Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar el módulo de importaciones centralizadas
try:
    from msc_imports import *
    from msc_imports import check_imports
except ImportError as e:
    print(f"Error importando msc_imports: {e}")
    print("Asegúrate de ejecutar este script desde el directorio del proyecto")
    sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MSCIntegratedRunner:
    """Ejecutor integrado del framework MSC"""
    
    def __init__(self):
        self.components = {}
        self.running = False
        
    def check_components(self):
        """Verifica qué componentes están disponibles"""
        logger.info("=== Verificando componentes del MSC Framework ===")
        
        status = check_imports()
        available = []
        unavailable = []
        
        for component, loaded in status.items():
            if loaded:
                available.append(component)
                logger.info(f"✅ {component}: Disponible")
            else:
                unavailable.append(component)
                logger.warning(f"❌ {component}: No disponible")
        
        return available, unavailable
    
    async def start_core_simulation(self, config):
        """Inicia la simulación principal del MSC"""
        if 'MSCFramework' not in globals():
            logger.error("MSCFramework no está disponible")
            return
        
        logger.info("Iniciando simulación MSC Core...")
        try:
            from msc_simulation import main as msc_main
            await msc_main()
        except Exception as e:
            logger.error(f"Error en simulación: {e}")
    
    async def start_taecviz(self, port=8080):
        """Inicia el servidor TAECViz"""
        if TAECVizServer is None:
            logger.error("TAECViz no está disponible")
            return
        
        logger.info(f"Iniciando TAECViz en puerto {port}...")
        server = TAECVizServer()
        await server.start(port)
    
    async def start_blockchain(self):
        """Inicia el componente blockchain SCED"""
        if SCEDBlockchain is None:
            logger.error("SCED Blockchain no está disponible")
            return
        
        logger.info("Iniciando SCED Blockchain...")
        blockchain = SCEDBlockchain()
        # Aquí puedes agregar lógica adicional
        return blockchain
    
    async def start_digital_entities(self):
        """Inicia el ecosistema de entidades digitales"""
        if DigitalEntityEcosystem is None:
            logger.error("Digital Entities no está disponible")
            return
        
        logger.info("Iniciando ecosistema de entidades digitales...")
        ecosystem = DigitalEntityEcosystem()
        # Generar algunas entidades iniciales
        if EntityGenerator:
            generator = EntityGenerator()
            for i in range(5):
                entity = generator.generate_random_entity()
                ecosystem.add_entity(entity)
        
        return ecosystem
    
    async def start_chaos_module(self):
        """Inicia el módulo de caos"""
        if TAECChaosModule is None:
            logger.error("TAEC Chaos Module no está disponible")
            return
        
        logger.info("Iniciando módulo de caos...")
        chaos = TAECChaosModule()
        chaos.initialize()
        return chaos
    
    async def start_integrated_system(self, components_to_start):
        """Inicia el sistema integrado con los componentes especificados"""
        tasks = []
        
        if 'core' in components_to_start:
            tasks.append(self.start_core_simulation({}))
        
        if 'viz' in components_to_start:
            tasks.append(self.start_taecviz())
        
        if 'blockchain' in components_to_start:
            blockchain = await self.start_blockchain()
            self.components['blockchain'] = blockchain
        
        if 'entities' in components_to_start:
            ecosystem = await self.start_digital_entities()
            self.components['ecosystem'] = ecosystem
        
        if 'chaos' in components_to_start:
            chaos = await self.start_chaos_module()
            self.components['chaos'] = chaos
        
        # Ejecutar tareas concurrentemente
        if tasks:
            await asyncio.gather(*tasks)
    
    def run_interactive_mode(self):
        """Ejecuta el modo interactivo"""
        print("\n=== MSC Framework - Modo Interactivo ===")
        print("\nComandos disponibles:")
        print("  status    - Ver estado de componentes")
        print("  start     - Iniciar componente")
        print("  stop      - Detener componente")
        print("  info      - Información del sistema")
        print("  help      - Mostrar esta ayuda")
        print("  exit      - Salir")
        
        while True:
            try:
                command = input("\nmsc> ").strip().lower()
                
                if command == 'exit':
                    break
                elif command == 'status':
                    self.check_components()
                elif command == 'help':
                    self.run_interactive_mode()  # Mostrar ayuda de nuevo
                elif command == 'info':
                    print(f"\nComponentes activos: {len(self.components)}")
                    for name, component in self.components.items():
                        print(f"  - {name}: {type(component).__name__}")
                elif command.startswith('start'):
                    parts = command.split()
                    if len(parts) > 1:
                        component = parts[1]
                        print(f"Iniciando {component}...")
                        # Aquí podrías agregar lógica para iniciar componentes específicos
                    else:
                        print("Uso: start <componente>")
                else:
                    print(f"Comando desconocido: {command}")
                    
            except KeyboardInterrupt:
                print("\n\nSaliendo...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="MSC Framework - Sistema Integrado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python msc_run.py --check           # Verificar componentes disponibles
  python msc_run.py --all             # Iniciar todos los componentes
  python msc_run.py --core --viz      # Iniciar core y visualización
  python msc_run.py --interactive     # Modo interactivo
        """
    )
    
    # Argumentos generales
    parser.add_argument('--check', action='store_true',
                       help='Verificar componentes disponibles')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Ejecutar en modo interactivo')
    
    # Componentes individuales
    parser.add_argument('--core', action='store_true',
                       help='Iniciar simulación MSC core')
    parser.add_argument('--viz', action='store_true',
                       help='Iniciar servidor TAECViz')
    parser.add_argument('--blockchain', action='store_true',
                       help='Iniciar blockchain SCED')
    parser.add_argument('--entities', action='store_true',
                       help='Iniciar entidades digitales')
    parser.add_argument('--chaos', action='store_true',
                       help='Iniciar módulo de caos')
    parser.add_argument('--all', action='store_true',
                       help='Iniciar todos los componentes disponibles')
    
    # Opciones adicionales
    parser.add_argument('--port', type=int, default=8080,
                       help='Puerto para TAECViz (default: 8080)')
    parser.add_argument('--config', type=str,
                       help='Archivo de configuración')
    
    args = parser.parse_args()
    
    # Crear runner
    runner = MSCIntegratedRunner()
    
    # Verificar componentes
    if args.check:
        available, unavailable = runner.check_components()
        print(f"\nComponentes disponibles: {len(available)}")
        print(f"Componentes no disponibles: {len(unavailable)}")
        return
    
    # Modo interactivo
    if args.interactive:
        runner.run_interactive_mode()
        return
    
    # Determinar qué componentes iniciar
    components_to_start = []
    if args.all:
        components_to_start = ['core', 'viz', 'blockchain', 'entities', 'chaos']
    else:
        if args.core:
            components_to_start.append('core')
        if args.viz:
            components_to_start.append('viz')
        if args.blockchain:
            components_to_start.append('blockchain')
        if args.entities:
            components_to_start.append('entities')
        if args.chaos:
            components_to_start.append('chaos')
    
    if not components_to_start and not args.check:
        print("No se especificaron componentes para iniciar.")
        print("Usa --help para ver las opciones disponibles.")
        return
    
    # Ejecutar sistema
    try:
        logger.info(f"Iniciando componentes: {components_to_start}")
        asyncio.run(runner.start_integrated_system(components_to_start))
    except KeyboardInterrupt:
        logger.info("\nSistema detenido por el usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
