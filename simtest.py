#!/usr/bin/env python
"""
SimTest: Sistema especializado de pruebas para frameworks de simulación multi-agente
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Union

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("simtest")

class TestCase:
    """Clase base para casos de prueba de simulación"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.results = {}
        self.passed = False
        self.error = None
        self.duration = 0
        self.artifacts = {}
        
    def setup(self) -> None:
        """Configuración inicial del caso de prueba"""
        pass
        
    def run(self) -> bool:
        """Ejecuta el caso de prueba y devuelve True si pasa"""
        start_time = time.time()
        try:
            self.setup()
            self.passed = self.execute()
            return self.passed
        except Exception as e:
            self.error = str(e)
            self.passed = False
            logger.error(f"Error en prueba {self.name}: {e}")
            return False
        finally:
            self.duration = time.time() - start_time
            self.teardown()
            
    def execute(self) -> bool:
        """Método principal a implementar en subclases"""
        raise NotImplementedError("Las subclases deben implementar execute()")
    
    def teardown(self) -> None:
        """Limpieza después de la ejecución"""
        pass
    
    def add_result(self, key: str, value: Any) -> None:
        """Agrega un resultado a la prueba"""
        self.results[key] = value
    
    def add_artifact(self, name: str, path: str) -> None:
        """Registra un artefacto generado por la prueba"""
        self.artifacts[name] = path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte los resultados de la prueba a un diccionario"""
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "duration": round(self.duration, 3),
            "error": self.error,
            "results": self.results,
            "artifacts": self.artifacts
        }

class TestSuite:
    """Conjunto de pruebas relacionadas"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tests: List[TestCase] = []
        self.start_time = None
        self.end_time = None
        self.duration = 0
        
    def add_test(self, test: TestCase) -> None:
        """Añade una prueba al conjunto"""
        self.tests.append(test)
    
    def run(self, parallel: bool = False) -> Dict[str, Any]:
        """Ejecuta todas las pruebas del conjunto"""
        self.start_time = time.time()
        results = []
        
        if parallel and len(self.tests) > 1:
            import multiprocessing
            with multiprocessing.Pool() as pool:
                # Usar map para ejecutar en paralelo
                def run_and_return(test):
                    test.run()
                    return test.to_dict()
                
                results = pool.map(run_and_return, self.tests)
        else:
            for test in self.tests:
                test.run()
                results.append(test.to_dict())
                
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
                
        # Calcular estadísticas
        passed = sum(1 for r in results if r["passed"])
        
        return {
            "name": self.name,
            "description": self.description,
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "success_rate": round(passed / len(results) * 100, 1) if results else 0,
            "duration": round(self.duration, 2),
            "timestamp": datetime.now().isoformat(),
            "tests": results
        }

class SimTest:
    """Controlador principal del sistema de pruebas"""
    
    def __init__(self):
        self.suites: Dict[str, TestSuite] = {}
        self.output_dir = Path("simtest_results")
        
    def add_suite(self, suite: TestSuite) -> None:
        """Añade una suite de pruebas"""
        self.suites[suite.name] = suite
    
    def run_suite(self, name: str, parallel: bool = False) -> Dict[str, Any]:
        """Ejecuta una suite específica de pruebas"""
        if name not in self.suites:
            raise ValueError(f"Suite de pruebas '{name}' no encontrada")
            
        suite = self.suites[name]
        return suite.run(parallel)
    
    def run_all(self, parallel: bool = False) -> Dict[str, Dict[str, Any]]:
        """Ejecuta todas las suites de pruebas"""
        results = {}
        for name, suite in self.suites.items():
            results[name] = suite.run(parallel)
        return results
    
    def save_results(self, results: Dict[str, Any], format: str = "json") -> str:
        """Guarda los resultados de las pruebas"""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            output_file = self.output_dir / f"results_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Formato de salida '{format}' no soportado")
            
        return str(output_file)
    
    def load_test_modules(self, directory: str = "simtests") -> None:
        """Carga dinámicamente módulos de prueba desde un directorio"""
        test_dir = Path(directory)
        if not test_dir.exists():
            test_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio de pruebas {directory} creado")
        
        # Asegurarse de que el directorio está en el path
        abs_path = test_dir.resolve()
        if str(abs_path.parent) not in sys.path:
            sys.path.insert(0, str(abs_path.parent))
        
        # Buscar módulos de prueba
        modules_loaded = 0
        for file in test_dir.glob("simtest_*.py"):
            module_name = file.stem
            try:
                if directory == "simtests":
                    module = __import__(f"simtests.{module_name}", fromlist=['register_tests'])
                else:
                    sys.path.insert(0, str(test_dir))
                    module = __import__(module_name, fromlist=['register_tests'])
                    
                if hasattr(module, "register_tests"):
                    logger.info(f"Cargando pruebas desde {module_name}")
                    module.register_tests(self)
                    modules_loaded += 1
                else:
                    logger.warning(f"Módulo {module_name} no tiene función register_tests()")
            except Exception as e:
                logger.error(f"Error al cargar el módulo {module_name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        if modules_loaded == 0:
            logger.warning(f"No se encontraron módulos de prueba en {directory}")

def main():
    """Función principal para ejecución desde línea de comandos"""
    parser = argparse.ArgumentParser(description="SimTest - Sistema de pruebas para MSC Framework")
    parser.add_argument("--tests-dir", help="Directorio de pruebas", default="simtests")
    parser.add_argument("--suite", help="Suite específica a ejecutar")
    parser.add_argument("--parallel", help="Ejecutar pruebas en paralelo", action="store_true")
    parser.add_argument("--output", help="Directorio de salida", default="simtest_results")
    parser.add_argument("--list", help="Listar pruebas disponibles", action="store_true")
    
    args = parser.parse_args()
    
    # Crear instancia de SimTest
    simtest = SimTest()
    simtest.output_dir = Path(args.output)
    
    # Cargar pruebas
    logger.info(f"Cargando pruebas desde {args.tests_dir}")
    simtest.load_test_modules(args.tests_dir)
    
    # Verificar si hay pruebas
    if not simtest.suites:
        logger.warning("No se encontraron pruebas")
        print("\nNo hay pruebas disponibles. Cree archivos con formato 'simtest_*.py' en el directorio 'simtests'.")
        print("Cada archivo debe tener una función 'register_tests(simtest)' que registre las suites de prueba.")
        return 1
    
    # Mostrar pruebas disponibles si se solicita
    if args.list:
        print("\nPruebas disponibles:")
        for suite_name, suite in simtest.suites.items():
            print(f"\nSuite: {suite_name} - {suite.description}")
            for i, test in enumerate(suite.tests):
                print(f"  {i+1}. {test.name}: {test.description}")
        return 0
        
    try:
        # Ejecutar pruebas
        if args.suite:
            if args.suite not in simtest.suites:
                print(f"Error: La suite '{args.suite}' no existe")
                print("Suites disponibles:", ", ".join(simtest.suites.keys()))
                return 1
                
            results = {args.suite: simtest.run_suite(args.suite, args.parallel)}
        else:
            results = simtest.run_all(args.parallel)
            
        # Guardar resultados
        output_file = simtest.save_results(results)
        print(f"Resultados guardados en {output_file}")
        
        # Mostrar resumen
        total_tests = sum(suite["total"] for suite in results.values())
        total_passed = sum(suite["passed"] for suite in results.values())
        
        print("\n" + "="*80)
        print(f"RESUMEN DE PRUEBAS:")
        print(f"Total de pruebas: {total_tests}")
        print(f"Pruebas exitosas: {total_passed}")
        print(f"Pruebas fallidas: {total_tests - total_passed}")
        print(f"Tasa de éxito: {round(total_passed / total_tests * 100, 1)}%" if total_tests else "N/A")
        print("="*80)
        
        # Devolver código de salida según resultados
        return 0 if total_passed == total_tests else 1
        
    except Exception as e:
        logger.error(f"Error en la ejecución de SimTest: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 2

if __name__ == "__main__":
    sys.exit(main())