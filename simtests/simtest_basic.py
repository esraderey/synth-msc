"""
Pruebas básicas para el framework MSC
"""
import sys
from pathlib import Path

# Asegurar que podemos importar
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simtest import SimTest, TestSuite, TestCase
from sim_testcases import SimulationTest, VisualizationTest

# Añadir una prueba simple que solo verifica la importación
class ImportTest(TestCase):
    """Prueba que verifica que los módulos principales se pueden importar"""
    
    def execute(self) -> bool:
        try:
            import msc_simulation
            self.add_result("msc_simulation_loaded", True)
            
            # Verificar que SimulationRunner existe
            has_simulation_runner = hasattr(msc_simulation, 'SimulationRunner')
            self.add_result("has_simulation_runner", has_simulation_runner)
            
            # Intentar importar taecviz si existe
            try:
                import taecviz
                self.add_result("taecviz_loaded", True)
            except ImportError:
                self.add_result("taecviz_loaded", False)
                
            # La prueba es exitosa si al menos se pudo importar msc_simulation
            return True
            
        except Exception as e:
            self.error = f"Error importando módulos: {str(e)}"
            return False

def register_tests(simtest: SimTest):
    """Registra las pruebas en el sistema SimTest"""
    
    # Suite de pruebas básicas
    basic_suite = TestSuite(
        name="basic",
        description="Pruebas básicas de funcionalidad del framework"
    )
    
    # Prueba de inicialización
    basic_suite.add_test(SimulationTest(
        name="init_simulation",
        description="Verifica que la simulación se inicializa correctamente",
        config={
            "node_count": 10,
            "edge_probability": 0.3,
            "agent_count": 2,
            "max_steps": 5
        },
        expected_results={
            "node_count": 10,
            "agent_count": 2
        }
    ))
    
    # Registrar la suite en SimTest
    simtest.add_suite(basic_suite)