#!/usr/bin/env python3
"""
MSC Framework - Módulo de importaciones centralizadas
Este módulo facilita la importación de componentes entre los diferentes archivos del framework
"""

import logging
import importlib.util
import sys
import os

logger = logging.getLogger(__name__)

def import_module_from_file(module_name: str, file_path: str):
    """Importa un módulo desde un archivo con espacios en el nombre"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"No se pudo crear spec para {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(f"No se pudo obtener loader para {file_path}")
            
        if hasattr(spec.loader, 'exec_module'):
            spec.loader.exec_module(module)
        else:
            # Fallback para loaders que no tienen exec_module
            module = spec.loader.load_module(module_name)
        return module
    except Exception as e:
        logger.error(f"Error importando {module_name} desde {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

# === IMPORTACIONES CENTRALIZADAS ===

# MSC Core
try:
    from msc_simulation import *
except ImportError:
    logger.warning("No se pudo importar msc_simulation")

# SCED Blockchain
sced_module = import_module_from_file("sced_v3", "sced v3.py")
if sced_module:
    SCEDBlockchain = getattr(sced_module, 'SCEDBlockchain', None)
    Transaction = getattr(sced_module, 'Transaction', None)
    TransactionType = getattr(sced_module, 'TransactionType', None)
    ExtendedEpistemicVector = getattr(sced_module, 'ExtendedEpistemicVector', None)
    ConsensusLevel = getattr(sced_module, 'ConsensusLevel', None)
    ValidationStrength = getattr(sced_module, 'ValidationStrength', None)
    SCEDCryptoEngine = getattr(sced_module, 'SCEDCryptoEngine', None)
    ZKPSystem = getattr(sced_module, 'ZKPSystem', None)
    SmartContract = getattr(sced_module, 'SmartContract', None)
    PostQuantumCrypto = getattr(sced_module, 'PostQuantumCrypto', None)

# Digital Entities
entities_module = import_module_from_file("MSC_Digital_Entities", "MSC_Digital_Entities_Extension v5.0.py")
if entities_module:
    DigitalEntity = getattr(entities_module, 'DigitalEntity', None)
    EntityType = getattr(entities_module, 'EntityType', None)
    EntityPersonality = getattr(entities_module, 'EntityPersonality', None)
    EntityMemory = getattr(entities_module, 'EntityMemory', None)
    DigitalEntityEcosystem = getattr(entities_module, 'DigitalEntityEcosystem', None)
    EntityGenerator = getattr(entities_module, 'EntityGenerator', None)

# TAEC Digital Entities
taec_entities_module = import_module_from_file("TAEC_Msc_Digital_Entities", "TAEC_Msc_Digital_Enties.py")
if taec_entities_module:
    MSCLTokenType = getattr(taec_entities_module, 'MSCLTokenType', None)
    MSCLToken = getattr(taec_entities_module, 'MSCLToken', None)
    MSCLLexer = getattr(taec_entities_module, 'MSCLLexer', None)
    MSCLParser = getattr(taec_entities_module, 'MSCLParser', None)
    MSCLCompiler = getattr(taec_entities_module, 'MSCLCompiler', None)
    MSCLCodeGenerator = getattr(taec_entities_module, 'MSCLCodeGenerator', None)
    SemanticAnalyzer = getattr(taec_entities_module, 'SemanticAnalyzer', None)
    QuantumState = getattr(taec_entities_module, 'QuantumState', None)
    QuantumMemoryCell = getattr(taec_entities_module, 'QuantumMemoryCell', None)
    QuantumVirtualMemory = getattr(taec_entities_module, 'QuantumVirtualMemory', None)
    CodeEvolutionEngine = getattr(taec_entities_module, 'CodeEvolutionEngine', None)
    BehaviorCompiler = getattr(taec_entities_module, 'BehaviorCompiler', None)

# TAEC v3
taec_v3_module = import_module_from_file("TAEC_v3", "Taec V 3.0.py")
if taec_v3_module:
    TAECSystem = getattr(taec_v3_module, 'TAECSystem', None)
    TAECCodeGenerator = getattr(taec_v3_module, 'TAECCodeGenerator', None)

# Performance Extensions
perf_module = import_module_from_file("MSC_Performance", "MSC Performance & Advanced Features Extension v6.0.py")
if perf_module:
    DistributedMSCFramework = getattr(perf_module, 'DistributedMSCFramework', None)
    GPUAcceleratedMSC = getattr(perf_module, 'GPUAcceleratedMSC', None)
    FederatedLearningCoordinator = getattr(perf_module, 'FederatedLearningCoordinator', None)

# TAECViz
viz_module = import_module_from_file("TAECViz", "Taecviz v.2.0 .py")
if viz_module:
    TAECVizServer = getattr(viz_module, 'TAECVizServer', None)

# Chaos Module
try:
    from taec_chaos_module import TAECChaosModule, ChaosMathematics, ChaoticMSCLCompiler
except ImportError:
    logger.warning("No se pudo importar taec_chaos_module")
    TAECChaosModule = None
    ChaosMathematics = None
    ChaoticMSCLCompiler = None

# Optimization Twin
try:
    from otaec_optimization_twin import OTAEC, OTAECVirtualMachine, OTAECTerminal
except ImportError:
    logger.warning("No se pudo importar otaec_optimization_twin")
    OTAEC = None
    OTAECVirtualMachine = None
    OTAECTerminal = None

# Virtual World
try:
    from osced_virtual_world import OSCED, OSCEDConfig, VirtualWorld
except ImportError:
    logger.warning("No se pudo importar osced_virtual_world")
    OSCED = None
    OSCEDConfig = None
    VirtualWorld = None

# Chaos Evolution Framework
try:
    from chaos_evolution_framework import ChaosEvolutionFramework, ChaosMathematics as ChaosFrameworkMath
except ImportError:
    logger.warning("No se pudo importar chaos_evolution_framework")
    ChaosEvolutionFramework = None
    ChaosFrameworkMath = None

# SRPK
try:
    from msc_srpk import SRPKGraph, CodeKnowledgeNode
except ImportError:
    logger.warning("No se pudo importar msc_srpk")
    SRPKGraph = None
    CodeKnowledgeNode = None

# MSCNet Blockchain
try:
    from mscnet_blockchain import MSCNetBlockchain, ReputationNet, KnowledgeVM
except ImportError:
    logger.warning("No se pudo importar mscnet_blockchain")
    MSCNetBlockchain = None
    ReputationNet = None
    KnowledgeVM = None

# Función helper para verificar importaciones
def check_imports():
    """Verifica qué módulos se importaron correctamente"""
    # Para msc_simulation, verificar si se importó alguna clase específica
    msc_simulation_loaded = False
    try:
        from msc_simulation import KnowledgeGraph, MSCAgent
        msc_simulation_loaded = True
    except:
        pass
    
    imports_status = {
        "msc_simulation": msc_simulation_loaded,
        "sced_v3": sced_module is not None,
        "digital_entities": entities_module is not None,
        "taec_entities": taec_entities_module is not None,
        "taec_v3": taec_v3_module is not None,
        "performance": perf_module is not None,
        "taecviz": viz_module is not None,
        "chaos_module": TAECChaosModule is not None,
        "otaec": OTAEC is not None,
        "osced": OSCED is not None,
        "chaos_framework": ChaosEvolutionFramework is not None,
        "srpk": SRPKGraph is not None,
        "mscnet": MSCNetBlockchain is not None
    }
    
    return imports_status

if __name__ == "__main__":
    # Verificar estado de importaciones
    print("Iniciando verificación de importaciones...")
    status = check_imports()
    print("\n=== Estado de Importaciones MSC Framework ===")
    for module, loaded in status.items():
        status_str = "✅ Cargado" if loaded else "❌ No disponible"
        print(f"{module:<20} {status_str}")
    
    # Contar resultados
    loaded_count = sum(1 for loaded in status.values() if loaded)
    total_count = len(status)
    print(f"\nTotal: {loaded_count}/{total_count} módulos cargados correctamente")
