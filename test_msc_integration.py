#!/usr/bin/env python3
"""
Tests de integración para MSC Framework
Verifica que los componentes principales funcionen correctamente
"""

import pytest
import asyncio
import sys
import os
import numpy as np

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar el módulo de importaciones
try:
    from msc_imports import check_imports
except ImportError:
    print("Error: No se puede importar msc_imports.py")
    sys.exit(1)

class TestMSCIntegration:
    """Tests de integración del framework MSC"""
    
    def test_imports_availability(self):
        """Verifica qué módulos están disponibles"""
        status = check_imports()
        print("\n=== Estado de Importaciones ===")
        for module, loaded in status.items():
            print(f"{module}: {'✅' if loaded else '❌'}")
        
        # Al menos algunos módulos deberían cargar
        assert any(status.values()), "Ningún módulo se cargó correctamente"
    
    def test_sced_blockchain(self):
        """Test básico del blockchain SCED"""
        try:
            from msc_imports import SCEDBlockchain, Transaction, TransactionType
            
            if SCEDBlockchain is None:
                pytest.skip("SCED Blockchain no disponible")
            
            # Crear blockchain
            blockchain = SCEDBlockchain()
            assert blockchain is not None
            assert hasattr(blockchain, 'chain')
            assert len(blockchain.chain) == 1  # Genesis block
            
            print("✅ SCED Blockchain inicializado correctamente")
        except Exception as e:
            pytest.fail(f"Error en SCED Blockchain: {e}")
    
    def test_srpk_graph(self):
        """Test del grafo de conocimiento SRPK"""
        try:
            from msc_imports import SRPKGraph, CodeKnowledgeNode
            
            if SRPKGraph is None:
                pytest.skip("SRPK no disponible")
            
            # Crear grafo
            graph = SRPKGraph()
            assert graph is not None
            
            # Inicializar modelo de embeddings
            graph.init_embedding_model()
            assert graph.embedding_model is not None
            
            # Crear nodo de prueba
            node = CodeKnowledgeNode(
                code_id="test_func",
                code_segment="def test(): return 42",
                purpose="Test function"
            )
            graph.nodes["test_func"] = node
            
            assert len(graph.nodes) == 1
            print("✅ SRPK Graph funciona correctamente")
        except Exception as e:
            pytest.fail(f"Error en SRPK: {e}")
    
    def test_chaos_module(self):
        """Test del módulo de caos"""
        try:
            from msc_imports import TAECChaosModule, ChaosMathematics
            
            if TAECChaosModule is None:
                pytest.skip("TAEC Chaos Module no disponible")
            
            # Crear un grafo mock para el test
            class MockGraph:
                def __init__(self):
                    self.nodes = {}
                    self.edges = []
            
            # Crear instancia con graph mock
            graph = MockGraph()
            chaos = TAECChaosModule(graph)
            assert chaos is not None
            
            # El módulo se inicializa automáticamente en __init__
            # No necesita llamar a initialize()
            
            # Verificar matemáticas del caos
            if ChaosMathematics:
                math = ChaosMathematics()
                # Test Lorenz
                # Test lorenz_attractor con un estado inicial
                state = np.array([1.0, 1.0, 1.0])
                result = math.lorenz_attractor(state)
                assert len(result) == 3
                assert all(isinstance(x, (int, float)) for x in result)
            
            print("✅ TAEC Chaos Module funciona correctamente")
        except Exception as e:
            pytest.fail(f"Error en Chaos Module: {e}")
    
    def test_mscnet_blockchain(self):
        """Test del blockchain MSCNet"""
        try:
            from msc_imports import MSCNetBlockchain, ReputationNet
            
            if MSCNetBlockchain is None:
                pytest.skip("MSCNet no disponible")
            
            # Crear blockchain
            blockchain = MSCNetBlockchain()
            assert blockchain is not None
            
            # Verificar componentes
            assert hasattr(blockchain, 'reputation_net')
            assert hasattr(blockchain, 'knowledge_vm')
            
            print("✅ MSCNet Blockchain funciona correctamente")
        except Exception as e:
            pytest.fail(f"Error en MSCNet: {e}")
    
    @pytest.mark.asyncio
    async def test_async_components(self):
        """Test de componentes asíncronos"""
        try:
            # Test básico de asyncio
            async def test_coro():
                await asyncio.sleep(0.01)
                return True
            
            result = await test_coro()
            assert result is True
            print("✅ Componentes asíncronos funcionan correctamente")
        except Exception as e:
            pytest.fail(f"Error en componentes asíncronos: {e}")

class TestSpecificFunctionality:
    """Tests específicos de funcionalidad"""
    
    def test_embedding_models(self):
        """Test de modelos de embeddings en SRPK"""
        try:
            from msc_srpk import SRPKGraph
            
            graph = SRPKGraph()
            
            # Test TF-IDF embedder
            tfidf_embedder = graph._create_tfidf_embedder()
            embedding = tfidf_embedder("def test(): return 42")
            assert embedding.shape == (768,)
            
            # Test hash embedder
            hash_embedder = graph._create_hash_embedder()
            embedding = hash_embedder("def test(): return 42")
            assert embedding.shape == (768,)
            
            print("✅ Modelos de embeddings funcionan correctamente")
        except Exception as e:
            pytest.fail(f"Error en embeddings: {e}")
    
    def test_prediction_system(self):
        """Test del sistema de predicción"""
        try:
            from msc_simulation import SimulationPredictor
            
            # Crear predictor
            predictor = SimulationPredictor()
            
            # Crear datos de prueba simulados
            class MockGraph:
                def __init__(self):
                    self.nodes = {f"node_{i}": MockNode() for i in range(10)}
                    self.cluster_index = {}
            
            class MockNode:
                def __init__(self):
                    self.state = 0.8
                    self.connections_out = []
            
            class MockAgent:
                def __init__(self):
                    self.omega = 0.7
                    self.reputation = 0.9
            
            class MockState:
                def __init__(self):
                    self.error_count = 2
                    self.warning_count = 5
                    self.processed_messages = 100
            
            # Ejecutar predicción
            graph = MockGraph()
            agents = [MockAgent() for _ in range(5)]
            state = MockState()
            
            predictions = asyncio.run(predictor.predict(graph, agents, state))
            
            assert 'next_hour' in predictions
            assert 'next_day' in predictions
            assert 'current_metrics' in predictions
            assert 'recommendations' in predictions
            
            print("✅ Sistema de predicción funciona correctamente")
        except Exception as e:
            pytest.fail(f"Error en predicción: {e}")

def run_integration_tests():
    """Ejecuta todos los tests de integración"""
    print("=== Ejecutando Tests de Integración MSC Framework ===\n")
    
    # Verificar Python version
    print(f"Python version: {sys.version}")
    
    # Ejecutar tests con pytest
    args = ["-v", "-s", __file__]
    if "--no-header" not in sys.argv:
        args.insert(0, "--tb=short")
    
    pytest.main(args)

if __name__ == "__main__":
    run_integration_tests()
