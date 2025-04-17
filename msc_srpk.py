import inspect
import ast
import re
import logging
import hashlib
import json
import os
import time
import torch
from collections import defaultdict

class CodeKnowledgeNode:
    """Nodo que representa conocimiento sobre una parte del código."""
    def __init__(self, code_id, code_segment, dependencies=None, purpose=None, test_cases=None):
        self.code_id = code_id
        self.code_segment = code_segment
        self.code_hash = hashlib.sha256(code_segment.encode()).hexdigest()
        self.dependencies = dependencies or []
        self.purpose = purpose or "No documented purpose"
        self.test_cases = test_cases or []
        self.creation_timestamp = time.time()
        self.update_timestamp = time.time()
        self.functional_state = 1.0  # 0.0 to 1.0 indicating functional integrity
        self.change_history = []

    def update_code(self, new_code_segment, reason=""):
        """Actualiza el código preservando el historial."""
        old_hash = self.code_hash
        self.change_history.append({
            "old_hash": old_hash,
            "old_code": self.code_segment,
            "timestamp": time.time(),
            "reason": reason
        })
        self.code_segment = new_code_segment
        self.code_hash = hashlib.sha256(new_code_segment.encode()).hexdigest()
        self.update_timestamp = time.time()
        return old_hash != self.code_hash

    def add_dependency(self, dependency_id):
        """Agrega una dependencia a otro fragmento de código."""
        if dependency_id not in self.dependencies:
            self.dependencies.append(dependency_id)
            return True
        return False

    def remove_dependency(self, dependency_id):
        """Elimina una dependencia."""
        if dependency_id in self.dependencies:
            self.dependencies.remove(dependency_id)
            return True
        return False

    def add_test_case(self, input_data, expected_output, description=""):
        """Agrega un caso de prueba para este fragmento de código."""
        test_case = {
            "input": input_data,
            "expected_output": expected_output,
            "description": description,
            "timestamp": time.time()
        }
        self.test_cases.append(test_case)
        return len(self.test_cases) - 1  # Return index of new test case

    def to_json(self):
        """Convierte el nodo a formato JSON."""
        return {
            "code_id": self.code_id,
            "code_hash": self.code_hash,
            "dependencies": self.dependencies,
            "purpose": self.purpose,
            "creation_timestamp": self.creation_timestamp,
            "update_timestamp": self.update_timestamp,
            "functional_state": self.functional_state,
            "test_cases": self.test_cases,
            "change_history_count": len(self.change_history)
        }


class SRPKGraph:
    """Grafo de conocimiento sobre el código del MSC Framework."""
    def __init__(self):
        self.nodes = {}  # code_id -> CodeKnowledgeNode
        self.function_map = {}  # function_name -> code_id
        self.class_map = {}  # class_name -> code_id
        self.module_map = {}  # module_name -> [code_id]
        self.impact_map = {}  # code_id -> [dependent_code_id]
        self.embedding_model = None  # Para comparaciones semánticas
        self.code_embeddings = {}  # code_id -> embedding vector

    def init_embedding_model(self, model_path=None):
        """Inicializa el modelo de embeddings para análisis de código."""
        try:
            if model_path and os.path.exists(model_path):
                self.embedding_model = torch.load(model_path)
                logging.info(f"Loaded code embedding model from {model_path}")
            else:
                # Simplified dummy model for demonstration
                self.embedding_model = lambda x: torch.randn(768)  # Simulated embedding
                logging.info("Using dummy code embedding model")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            self.embedding_model = lambda x: torch.randn(768)

    def analyze_code_file(self, file_path):
        """Analiza un archivo de código y extrae nodos de conocimiento."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file
            tree = ast.parse(content)
            
            # Extract file-level node
            file_id = f"file:{os.path.basename(file_path)}"
            file_node = CodeKnowledgeNode(
                code_id=file_id,
                code_segment=content,
                purpose=f"File: {file_path}"
            )
            self.nodes[file_id] = file_node
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_code = ast.get_source_segment(content, node)
                    class_id = f"class:{node.name}"
                    class_node = CodeKnowledgeNode(
                        code_id=class_id,
                        code_segment=class_code,
                        dependencies=[file_id],
                        purpose=f"Class: {node.name}"
                    )
                    self.nodes[class_id] = class_node
                    self.class_map[node.name] = class_id
                    
                    # Add this class to the module map
                    module_name = os.path.basename(file_path).replace('.py', '')
                    if module_name not in self.module_map:
                        self.module_map[module_name] = []
                    self.module_map[module_name].append(class_id)
                    
                    # Update impact map
                    if file_id not in self.impact_map:
                        self.impact_map[file_id] = []
                    self.impact_map[file_id].append(class_id)
                
                elif isinstance(node, ast.FunctionDef):
                    # Find parent class if any
                    parent_class = None
                    for ancestor in ast.walk(tree):
                        if isinstance(ancestor, ast.ClassDef) and node in ancestor.body:
                            parent_class = ancestor.name
                            break
                    
                    func_code = ast.get_source_segment(content, node)
                    func_id = f"func:{node.name}" if not parent_class else f"method:{parent_class}.{node.name}"
                    func_dependencies = [file_id]
                    if parent_class:
                        func_dependencies.append(f"class:{parent_class}")
                    
                    func_node = CodeKnowledgeNode(
                        code_id=func_id,
                        code_segment=func_code,
                        dependencies=func_dependencies,
                        purpose=f"Function: {node.name}" if not parent_class else f"Method: {parent_class}.{node.name}"
                    )
                    self.nodes[func_id] = func_node
                    self.function_map[node.name] = func_id
                    
                    # Add this function to the module map
                    module_name = os.path.basename(file_path).replace('.py', '')
                    if module_name not in self.module_map:
                        self.module_map[module_name] = []
                    self.module_map[module_name].append(func_id)
                    
                    # Update impact map
                    for dep in func_dependencies:
                        if dep not in self.impact_map:
                            self.impact_map[dep] = []
                        self.impact_map[dep].append(func_id)
            
            # Generate embeddings for all nodes
            self._generate_embeddings()
            
            # Analyze dependencies between functions by looking at function calls
            self._analyze_function_dependencies()
            
            return len(self.nodes)
        
        except Exception as e:
            logging.error(f"Error analyzing code file {file_path}: {e}")
            return 0

    def _generate_embeddings(self):
        """Genera embeddings para todos los nodos de código."""
        if not self.embedding_model:
            self.init_embedding_model()
        
        for code_id, node in self.nodes.items():
            try:
                embedding = self.embedding_model(node.code_segment)
                self.code_embeddings[code_id] = embedding
            except Exception as e:
                logging.error(f"Error generating embedding for {code_id}: {e}")
                self.code_embeddings[code_id] = torch.zeros(768)  # Fallback

    def _analyze_function_dependencies(self):
        """Analiza las dependencias entre funciones basado en llamadas."""
        function_names = set(self.function_map.keys())
        
        for func_id, func_node in self.nodes.items():
            if not func_id.startswith("func:") and not func_id.startswith("method:"):
                continue
            
            # Look for function calls in the code
            for func_name in function_names:
                # Simple regex pattern to detect function calls
                pattern = r'\b' + re.escape(func_name) + r'\s*\('
                if re.search(pattern, func_node.code_segment):
                    callee_id = self.function_map[func_name]
                    if callee_id != func_id:  # Don't add self-dependency
                        func_node.add_dependency(callee_id)
                        
                        # Update impact map
                        if callee_id not in self.impact_map:
                            self.impact_map[callee_id] = []
                        if func_id not in self.impact_map[callee_id]:
                            self.impact_map[callee_id].append(func_id)

    def find_similar_code(self, code_segment, threshold=0.8):
        """Encuentra fragmentos de código similares basado en embeddings."""
        if not self.embedding_model:
            self.init_embedding_model()
        
        query_embedding = self.embedding_model(code_segment)
        results = []
        
        for code_id, embedding in self.code_embeddings.items():
            similarity = self._compute_similarity(query_embedding, embedding)
            if similarity >= threshold:
                results.append((code_id, similarity, self.nodes[code_id]))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _compute_similarity(self, emb1, emb2):
        """Calcula similitud coseno entre dos embeddings."""
        if isinstance(emb1, torch.Tensor) and isinstance(emb2, torch.Tensor):
            return torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()
        return 0.0

    def estimate_change_impact(self, code_id):
        """Estima el impacto de un cambio en un fragmento de código."""
        impact = {
            "direct": [],
            "indirect": [],
            "risk_level": 0.0
        }
        
        # Check if code_id exists
        if code_id not in self.nodes:
            return impact
        
        # Get direct dependencies (what this code impacts)
        if code_id in self.impact_map:
            impact["direct"] = self.impact_map[code_id].copy()
        
        # Get indirect dependencies (second-level impacts)
        for dep_id in impact["direct"]:
            if dep_id in self.impact_map:
                for indirect_dep in self.impact_map[dep_id]:
                    if indirect_dep not in impact["direct"] and indirect_dep != code_id:
                        impact["indirect"].append(indirect_dep)
        
        # Calculate risk level based on number and type of dependencies
        impact["risk_level"] = min(1.0, (len(impact["direct"]) * 0.1 + len(impact["indirect"]) * 0.01))
        
        return impact

    def generate_update_plan(self, code_id, new_code):
        """Genera un plan de actualización para un cambio de código."""
        old_node = self.nodes.get(code_id)
        if not old_node:
            return {"error": "Code ID not found"}
        
        # Compare old and new code
        old_code = old_node.code_segment
        if old_code == new_code:
            return {"status": "No changes detected"}
        
        # Estimate impact
        impact = self.estimate_change_impact(code_id)
        
        # Generate update plan
        update_plan = {
            "code_id": code_id,
            "impact": impact,
            "tests_required": [test for test in old_node.test_cases],
            "dependent_code": [],
            "update_steps": [
                {
                    "step": 1,
                    "action": "backup_code",
                    "target": code_id
                },
                {
                    "step": 2,
                    "action": "update_code",
                    "target": code_id,
                    "new_code": new_code
                }
            ]
        }
        
        # Add steps for each impacted component that needs testing
        step_counter = 3
        for dep_id in impact["direct"]:
            update_plan["dependent_code"].append({
                "code_id": dep_id,
                "purpose": self.nodes[dep_id].purpose if dep_id in self.nodes else "Unknown"
            })
            update_plan["update_steps"].append({
                "step": step_counter,
                "action": "test_dependent",
                "target": dep_id
            })
            step_counter += 1
        
        return update_plan

    def apply_update(self, update_plan):
        """Aplica un plan de actualización."""
        results = {
            "success": False,
            "steps_completed": 0,
            "errors": [],
            "tests_passed": 0,
            "tests_failed": 0
        }
        
        if "error" in update_plan:
            results["errors"].append(update_plan["error"])
            return results
        
        code_id = update_plan["code_id"]
        if code_id not in self.nodes:
            results["errors"].append(f"Code ID {code_id} not found")
            return results
        
        # Execute update steps
        for step in update_plan["update_steps"]:
            try:
                if step["action"] == "backup_code":
                    # Backup is automatically handled by the change history
                    pass
                
                elif step["action"] == "update_code":
                    self.nodes[code_id].update_code(
                        step["new_code"], 
                        reason="Planned update via SRPK"
                    )
                
                elif step["action"] == "test_dependent":
                    dep_id = step["target"]
                    if dep_id in self.nodes:
                        test_results = self._run_tests_for_node(dep_id)
                        results["tests_passed"] += test_results["passed"]
                        results["tests_failed"] += test_results["failed"]
                        if test_results["failed"] > 0:
                            results["errors"].append(f"Tests failed for {dep_id}")
                
                results["steps_completed"] += 1
            
            except Exception as e:
                results["errors"].append(f"Error in step {step['step']}: {str(e)}")
                break
        
        # Update success flag
        results["success"] = (
            results["steps_completed"] == len(update_plan["update_steps"]) and 
            len(results["errors"]) == 0 and
            results["tests_failed"] == 0
        )
        
        # If failed, try to rollback
        if not results["success"] and results["steps_completed"] > 1:
            try:
                # Get the most recent backup from history
                history = self.nodes[code_id].change_history
                if history:
                    latest_backup = history[-1]
                    self.nodes[code_id].code_segment = latest_backup["old_code"]
                    self.nodes[code_id].code_hash = latest_backup["old_hash"]
                    results["errors"].append("Update failed, rolled back to previous version")
            except Exception as e:
                results["errors"].append(f"Rollback failed: {str(e)}")
        
        # Regenerate embedding if successful
        if results["success"]:
            try:
                if self.embedding_model:
                    self.code_embeddings[code_id] = self.embedding_model(self.nodes[code_id].code_segment)
            except Exception as e:
                logging.error(f"Error updating embedding for {code_id}: {e}")
        
        return results

    def _run_tests_for_node(self, code_id):
        """Ejecuta las pruebas para un nodo específico."""
        results = {"passed": 0, "failed": 0, "details": []}
        
        if code_id not in self.nodes:
            return results
        
        node = self.nodes[code_id]
        for test_case in node.test_cases:
            # In a real implementation, this would execute the code with the input
            # and compare against expected output. Here we simulate test results.
            test_passed = True  # Simplified - always pass in this demo
            
            if test_passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append({
                "test_description": test_case["description"],
                "result": "passed" if test_passed else "failed"
            })
        
        return results

    def save_state(self, file_path):
        """Guarda el estado del SRPK en un archivo JSON."""
        try:
            state = {
                "nodes": {k: v.to_json() for k, v in self.nodes.items()},
                "function_map": self.function_map,
                "class_map": self.class_map,
                "module_map": self.module_map,
                "impact_map": self.impact_map,
                "timestamp": time.time()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            # Embeddings need to be saved separately as tensors
            embeddings_path = file_path.replace('.json', '_embeddings.pt')
            torch.save(self.code_embeddings, embeddings_path)
            
            logging.info(f"SRPK state saved to {file_path}")
            return True
        
        except Exception as e:
            logging.error(f"Error saving SRPK state: {e}")
            return False

    def load_state(self, file_path):
        """Carga el estado del SRPK desde un archivo JSON."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Rebuild the node objects
            self.nodes = {}
            for code_id, node_data in state["nodes"].items():
                node = CodeKnowledgeNode(
                    code_id=code_id,
                    code_segment="",  # Will be filled from full_state
                    dependencies=node_data.get("dependencies", []),
                    purpose=node_data.get("purpose", "")
                )
                node.code_hash = node_data.get("code_hash", "")
                node.creation_timestamp = node_data.get("creation_timestamp", 0)
                node.update_timestamp = node_data.get("update_timestamp", 0)
                node.functional_state = node_data.get("functional_state", 1.0)
                node.test_cases = node_data.get("test_cases", [])
                self.nodes[code_id] = node
            
            self.function_map = state.get("function_map", {})
            self.class_map = state.get("class_map", {})
            self.module_map = state.get("module_map", {})
            self.impact_map = state.get("impact_map", {})
            
            # Load embeddings
            embeddings_path = file_path.replace('.json', '_embeddings.pt')
            if os.path.exists(embeddings_path):
                self.code_embeddings = torch.load(embeddings_path)
            else:
                self.code_embeddings = {}
            
            logging.info(f"SRPK state loaded from {file_path}")
            return True
        
        except Exception as e:
            logging.error(f"Error loading SRPK state: {e}")
            return False

    def generate_documentation(self, output_path):
        """Genera documentación para todo el código registrado."""
        try:
            doc = ["# MSC Framework Documentation", "Generated by SRPK\n\n"]
            
            # Group by modules
            modules = defaultdict(list)
            for module_name, code_ids in self.module_map.items():
                for code_id in code_ids:
                    if code_id in self.nodes:
                        modules[module_name].append(self.nodes[code_id])
            
            # Generate documentation for each module
            for module_name, nodes in modules.items():
                doc.append(f"## Module: {module_name}\n")
                
                # Group by classes and functions
                classes = {}
                functions = []
                
                for node in nodes:
                    if node.code_id.startswith("class:"):
                        class_name = node.code_id.replace("class:", "")
                        classes[class_name] = {
                            "node": node,
                            "methods": []
                        }
                    elif node.code_id.startswith("func:"):
                        functions.append(node)
                
                # Add methods to their classes
                for node in nodes:
                    if node.code_id.startswith("method:"):
                        parts = node.code_id.replace("method:", "").split(".")
                        if len(parts) == 2:
                            class_name, method_name = parts
                            if class_name in classes:
                                classes[class_name]["methods"].append(node)
                
                # Document classes
                for class_name, class_data in classes.items():
                    doc.append(f"### Class: {class_name}\n")
                    doc.append(f"**Purpose:** {class_data['node'].purpose}\n")
                    doc.append("**Methods:**\n")
                    
                    for method in class_data["methods"]:
                        method_name = method.code_id.split(".")[-1]
                        doc.append(f"- `{method_name}`: {method.purpose}")
                    
                    doc.append("\n")
                
                # Document standalone functions
                if functions:
                    doc.append("### Standalone Functions\n")
                    for func in functions:
                        func_name = func.code_id.replace("func:", "")
                        doc.append(f"- `{func_name}`: {func.purpose}\n")
                
                doc.append("\n---\n")
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(doc))
            
            logging.info(f"Documentation generated at {output_path}")
            return True
        
        except Exception as e:
            logging.error(f"Error generating documentation: {e}")
            return False

# --- Implementación de la CLI para el SRPK ---

class SRPKManager:
    """Interfaz para gestionar el SRPK."""
    def __init__(self, state_file=None):
        self.srpk = SRPKGraph()
        if state_file and os.path.exists(state_file):
            self.srpk.load_state(state_file)
        self.state_file = state_file or "srpk_state.json"
    
    def analyze_project(self, project_path):
        """Analiza todo un proyecto de Python."""
        file_count = 0
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    logging.info(f"Analyzing {file_path}...")
                    self.srpk.analyze_code_file(file_path)
                    file_count += 1
        
        # Save the state
        self.srpk.save_state(self.state_file)
        return file_count
    
    def find_similar_code(self, code_snippet, threshold=0.8):
        """Encuentra código similar al fragmento proporcionado."""
        return self.srpk.find_similar_code(code_snippet, threshold)
    
    def estimate_impact(self, code_id):
        """Estima el impacto de cambiar un fragmento de código."""
        return self.srpk.estimate_change_impact(code_id)
    
    def plan_update(self, code_id, new_code):
        """Planifica una actualización de código."""
        return self.srpk.generate_update_plan(code_id, new_code)
    
    def apply_update(self, update_plan):
        """Aplica un plan de actualización."""
        results = self.srpk.apply_update(update_plan)
        if results["success"]:
            self.srpk.save_state(self.state_file)
        return results
    
    def generate_docs(self, output_path):
        """Genera documentación para el proyecto."""
        return self.srpk.generate_documentation(output_path)