"""
Casos de prueba específicos para el MSC Framework
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Callable, Optional

from simtest import TestCase, logger

class SimulationTest(TestCase):
    """Caso de prueba específico para simulaciones"""
    
    def __init__(self, name: str, description: str = "", 
                 config: Dict[str, Any] = None,
                 expected_results: Dict[str, Any] = None,
                 success_criteria: Callable[[Dict[str, Any]], bool] = None):
        super().__init__(name, description)
        self.config = config or {}
        self.expected_results = expected_results or {}
        self.success_criteria = success_criteria
        self.simulation = None
        
    def setup(self) -> None:
        """Configura la simulación"""
        try:
            from msc_simulation import SimulationRunner
            
            # Añadir valores por defecto para pruebas
            test_config = {
                "test_mode": True,
                "max_steps": 10,
                "node_count": 20,
                "agent_count": 3
            }
            # Sobrescribir con configuración específica
            test_config.update(self.config)
            
            # Inicializar el simulador directamente con el diccionario de configuración
            # Ya que parece que SimulationRunner acepta directamente un diccionario 
            # en lugar de requerir un objeto Configuration
            self.simulation = SimulationRunner(test_config)
            
            # Registrar configuración como resultado
            self.add_result("config", test_config)
            
        except Exception as e:
            logger.error(f"Error configurando simulación: {e}")
            raise
    
    def execute(self) -> bool:
        """Ejecuta la simulación y verifica los resultados"""
        if not self.simulation:
            self.error = "Simulación no inicializada"
            return False
            
        try:
            # Ejecutar simulación - verificamos si tiene el método run con blocking
            if hasattr(self.simulation, 'run') and callable(self.simulation.run):
                # Intentamos con blocking primero
                try:
                    self.simulation.run(blocking=True)
                except TypeError:
                    # Si no acepta blocking, intenta sin él
                    self.simulation.run()
                    # Esperar un momento para que la simulación avance
                    time.sleep(1)
            
            # Obtener resultados - Usar get_status si está disponible
            if hasattr(self.simulation, 'get_status') and callable(self.simulation.get_status):
                status = self.simulation.get_status()
                self.add_result("status", status)
            
            # Si hay método para obtener métricas, usarlo
            if hasattr(self.simulation, 'get_metrics'):
                metrics = self.simulation.get_metrics()
                self.add_result("metrics", metrics)
            
            # Si se proporcionó un criterio de éxito personalizado, usarlo
            if self.success_criteria:
                return self.success_criteria(status)
                
            # De lo contrario, comparar con resultados esperados
            for key, expected in self.expected_results.items():
                if key not in status:
                    self.add_result("missing_key", key)
                    return False
                
                actual = status[key]
                
                # Si es un número, permitir un margen de error
                if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                    if abs(expected - actual) > abs(expected * 0.1):  # 10% de margen
                        self.add_result("value_mismatch", {
                            "key": key,
                            "expected": expected,
                            "actual": actual
                        })
                        return False
                # De lo contrario, igualdad exacta
                elif expected != actual:
                    self.add_result("value_mismatch", {
                        "key": key,
                        "expected": expected,
                        "actual": actual
                    })
                    return False
                    
            return True
            
        except Exception as e:
            self.error = f"Error ejecutando simulación: {str(e)}"
            logger.error(self.error)
            return False
    
    def teardown(self) -> None:
        """Limpia los recursos de la simulación"""
        if hasattr(self, 'simulation') and self.simulation:
            # Guardar gráfico de la red como artefacto
            try:
                os.makedirs("artifacts", exist_ok=True)
                artifact_path = f"artifacts/{self.name}_graph.png"
                if hasattr(self.simulation, 'export_graph_visualization'):
                    self.simulation.export_graph_visualization(artifact_path)
                    self.add_artifact("graph", artifact_path)
            except Exception as e:
                logger.warning(f"No se pudo guardar el gráfico: {e}")
            
            # Limpiar la simulación
            if hasattr(self.simulation, 'cleanup'):
                self.simulation.cleanup()

class VisualizationTest(TestCase):
    """Caso de prueba específico para visualización"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.server = None
    
    def setup(self) -> None:
        """Configura el servidor de visualización"""
        try:
            from msc_simulation import SimulationRunner, Configuration
            
            # Verificar si el módulo de visualización está disponible
            try:
                from taecviz import TAECVizServer
            except ImportError:
                self.error = "Módulo taecviz no disponible"
                raise ImportError(self.error)
            
            # Crear un simulador básico
            config = Configuration({
                "test_mode": True,
                "max_steps": 5,
                "node_count": 10,
                "agent_count": 2
            })
            sim = SimulationRunner(config)
            
            # Inicializar el servidor
            self.server = TAECVizServer(sim, port=0)  # Puerto 0 = automático
            
        except Exception as e:
            logger.error(f"Error configurando visualización: {e}")
            raise
    
    def execute(self) -> bool:
        """Ejecuta la prueba del servidor de visualización"""
        if not self.server:
            self.error = "Servidor de visualización no inicializado"
            return False
            
        try:
            import requests
            
            # Iniciar servidor
            self.server.start()
            port = self.server.port
            self.add_result("port", port)
            
            # Verificar que el servidor responde
            time.sleep(1)  # Dar tiempo para iniciar
            url = f"http://localhost:{port}/api/status"
            response = requests.get(url, timeout=5)
            
            self.add_result("status_code", response.status_code)
            self.add_result("response", response.json())
            
            return response.status_code == 200
            
        except Exception as e:
            self.error = f"Error en prueba de visualización: {str(e)}"
            logger.error(self.error)
            return False
    
    def teardown(self) -> None:
        """Detiene el servidor"""
        if self.server and hasattr(self.server, 'stop'):
            try:
                self.server.stop()
            except Exception as e:
                logger.warning(f"Error al detener servidor: {e}")