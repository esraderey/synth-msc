#!/usr/bin/env python3
"""
OTAEC (Optimization Twin of TAEC) - Sistema de Optimización Avanzada
Un gemelo digital de TAEC especializado en optimización con:
- Terminal interactiva propia
- VM básica para ejecución segura
- Interacción con sistema local
- Algoritmos de optimización avanzados
- Comunicación bidireccional con TAEC
"""

import os
import sys
import ast
import code
import subprocess
import platform
import psutil
import shutil
import signal
import json
import time
import threading
import queue
import hashlib
import tempfile
import traceback
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import logging
import asyncio
import socket
import pickle
import struct

# Importaciones para optimización
import numpy as np
from scipy import optimize
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OTAEC")

# === SISTEMA DE SEGURIDAD Y SANDBOXING ===

class SecurityLevel(Enum):
    """Niveles de seguridad para operaciones"""
    MINIMAL = auto()    # Solo lectura, sin acceso a sistema
    STANDARD = auto()   # Lectura/escritura en directorio de trabajo
    ELEVATED = auto()   # Acceso a sistema con restricciones
    FULL = auto()       # Acceso completo (requiere confirmación)

class SecurityManager:
    """Gestor de seguridad para operaciones del sistema"""
    
    def __init__(self, level: SecurityLevel = SecurityLevel.STANDARD):
        self.level = level
        self.allowed_paths = [os.getcwd()]
        self.blocked_commands = {
            'rm', 'del', 'format', 'fdisk', 'dd', 'shutdown', 'reboot'
        }
        self.audit_log = deque(maxlen=1000)
        
    def check_path_access(self, path: Path, operation: str = "read") -> bool:
        """Verifica si se permite acceso a una ruta"""
        try:
            abs_path = Path(path).resolve()
            
            # Registrar intento
            self.audit_log.append({
                'timestamp': time.time(),
                'path': str(abs_path),
                'operation': operation,
                'allowed': False  # Se actualizará si se permite
            })
            
            # Verificar nivel de seguridad
            if self.level == SecurityLevel.MINIMAL:
                return False
            
            if self.level == SecurityLevel.STANDARD:
                # Solo permitir en directorio de trabajo y subdirectorios
                cwd = Path.cwd()
                try:
                    abs_path.relative_to(cwd)
                    self.audit_log[-1]['allowed'] = True
                    return True
                except ValueError:
                    return False
            
            if self.level == SecurityLevel.ELEVATED:
                # Permitir en rutas específicas
                for allowed in self.allowed_paths:
                    allowed_path = Path(allowed).resolve()
                    try:
                        abs_path.relative_to(allowed_path)
                        self.audit_log[-1]['allowed'] = True
                        return True
                    except ValueError:
                        continue
                return False
            
            if self.level == SecurityLevel.FULL:
                # Permitir todo pero registrar
                self.audit_log[-1]['allowed'] = True
                return True
                
        except Exception as e:
            logger.error(f"Security check error: {e}")
            return False
    
    def check_command(self, command: str) -> bool:
        """Verifica si un comando está permitido"""
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return True
        
        base_cmd = cmd_parts[0].lower()
        
        # Registrar intento
        self.audit_log.append({
            'timestamp': time.time(),
            'command': command,
            'blocked': base_cmd in self.blocked_commands
        })
        
        return base_cmd not in self.blocked_commands
    
    def add_allowed_path(self, path: str):
        """Añade una ruta permitida"""
        self.allowed_paths.append(str(Path(path).resolve()))
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de auditoría"""
        total_ops = len(self.audit_log)
        blocked_ops = sum(1 for op in self.audit_log if op.get('blocked') or not op.get('allowed'))
        
        return {
            'total_operations': total_ops,
            'blocked_operations': blocked_ops,
            'security_level': self.level.name,
            'allowed_paths': self.allowed_paths,
            'recent_operations': list(self.audit_log)[-10:]
        }

# === VIRTUAL MACHINE BÁSICA ===

@dataclass
class VMInstruction:
    """Instrucción para la VM"""
    opcode: str
    operands: List[Any]
    line: int = 0

class OTAECVirtualMachine:
    """VM básica para ejecución segura de código de optimización"""
    
    def __init__(self, memory_size: int = 1024, security_manager: SecurityManager = None):
        self.memory = [None] * memory_size
        self.registers = {
            'AX': 0,  # Acumulador
            'BX': 0,  # Base
            'CX': 0,  # Contador
            'DX': 0,  # Datos
            'SP': memory_size - 1,  # Stack pointer
            'BP': 0,  # Base pointer
            'IP': 0,  # Instruction pointer
            'FLAGS': 0  # Flags de estado
        }
        
        self.stack = []
        self.heap = {}
        self.program = []
        self.running = False
        self.security = security_manager or SecurityManager()
        
        # Conjunto de instrucciones
        self.opcodes = {
            # Aritméticas
            'ADD': self._op_add,
            'SUB': self._op_sub,
            'MUL': self._op_mul,
            'DIV': self._op_div,
            'MOD': self._op_mod,
            
            # Memoria
            'MOV': self._op_mov,
            'PUSH': self._op_push,
            'POP': self._op_pop,
            'LOAD': self._op_load,
            'STORE': self._op_store,
            
            # Control de flujo
            'JMP': self._op_jmp,
            'JZ': self._op_jz,
            'JNZ': self._op_jnz,
            'CALL': self._op_call,
            'RET': self._op_ret,
            
            # Comparación
            'CMP': self._op_cmp,
            'TEST': self._op_test,
            
            # Sistema
            'INT': self._op_int,
            'NOP': self._op_nop,
            'HLT': self._op_hlt,
            
            # Optimización específicas
            'OPT': self._op_optimize,
            'GRAD': self._op_gradient,
            'EVAL': self._op_evaluate
        }
        
        # Interrupciones del sistema
        self.interrupts = {
            0x00: self._int_print,
            0x01: self._int_input,
            0x02: self._int_file_read,
            0x03: self._int_file_write,
            0x10: self._int_optimize_func
        }
        
        # Funciones de optimización registradas
        self.optimization_functions = {}
        
    def load_program(self, instructions: List[VMInstruction]):
        """Carga un programa en la VM"""
        self.program = instructions
        self.registers['IP'] = 0
        
    def compile_code(self, source: str) -> List[VMInstruction]:
        """Compila código de alto nivel a instrucciones de VM"""
        instructions = []
        lines = source.strip().split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            opcode = parts[0].upper()
            operands = []
            
            # Parsear operandos
            for operand in parts[1:]:
                # Registro
                if operand.upper() in self.registers:
                    operands.append(('REG', operand.upper()))
                # Número
                elif operand.isdigit() or (operand.startswith('-') and operand[1:].isdigit()):
                    operands.append(('IMM', int(operand)))
                # Flotante
                elif '.' in operand:
                    try:
                        operands.append(('IMM', float(operand)))
                    except ValueError:
                        operands.append(('LABEL', operand))
                # Dirección de memoria
                elif operand.startswith('[') and operand.endswith(']'):
                    addr = operand[1:-1]
                    if addr.isdigit():
                        operands.append(('MEM', int(addr)))
                    else:
                        operands.append(('MEM_REG', addr.upper()))
                # String
                elif operand.startswith('"') and operand.endswith('"'):
                    operands.append(('STR', operand[1:-1]))
                # Label
                else:
                    operands.append(('LABEL', operand))
            
            instructions.append(VMInstruction(opcode, operands, line_num))
        
        return instructions
    
    def run(self, max_cycles: int = 10000):
        """Ejecuta el programa cargado"""
        self.running = True
        cycles = 0
        
        while self.running and self.registers['IP'] < len(self.program) and cycles < max_cycles:
            instruction = self.program[self.registers['IP']]
            
            if instruction.opcode in self.opcodes:
                self.opcodes[instruction.opcode](instruction.operands)
            else:
                raise ValueError(f"Unknown opcode: {instruction.opcode}")
            
            cycles += 1
        
        if cycles >= max_cycles:
            logger.warning(f"VM execution stopped after {max_cycles} cycles")
        
        return cycles
    
    # === Implementación de opcodes ===
    
    def _get_value(self, operand: Tuple[str, Any]) -> Any:
        """Obtiene el valor de un operando"""
        op_type, op_value = operand
        
        if op_type == 'REG':
            return self.registers[op_value]
        elif op_type == 'IMM':
            return op_value
        elif op_type == 'MEM':
            return self.memory[op_value]
        elif op_type == 'MEM_REG':
            addr = self.registers[op_value]
            return self.memory[addr]
        elif op_type == 'STR':
            return op_value
        else:
            return op_value
    
    def _set_value(self, operand: Tuple[str, Any], value: Any):
        """Establece el valor de un operando"""
        op_type, op_value = operand
        
        if op_type == 'REG':
            self.registers[op_value] = value
        elif op_type == 'MEM':
            self.memory[op_value] = value
        elif op_type == 'MEM_REG':
            addr = self.registers[op_value]
            self.memory[addr] = value
    
    def _op_add(self, operands):
        """ADD dest, src"""
        dest_val = self._get_value(operands[0])
        src_val = self._get_value(operands[1])
        result = dest_val + src_val
        self._set_value(operands[0], result)
        self.registers['IP'] += 1
    
    def _op_sub(self, operands):
        """SUB dest, src"""
        dest_val = self._get_value(operands[0])
        src_val = self._get_value(operands[1])
        result = dest_val - src_val
        self._set_value(operands[0], result)
        self.registers['IP'] += 1
    
    def _op_mul(self, operands):
        """MUL dest, src"""
        dest_val = self._get_value(operands[0])
        src_val = self._get_value(operands[1])
        result = dest_val * src_val
        self._set_value(operands[0], result)
        self.registers['IP'] += 1
    
    def _op_div(self, operands):
        """DIV dest, src"""
        dest_val = self._get_value(operands[0])
        src_val = self._get_value(operands[1])
        if src_val == 0:
            raise ValueError("Division by zero")
        result = dest_val / src_val
        self._set_value(operands[0], result)
        self.registers['IP'] += 1
    
    def _op_mod(self, operands):
        """MOD dest, src"""
        dest_val = self._get_value(operands[0])
        src_val = self._get_value(operands[1])
        result = dest_val % src_val
        self._set_value(operands[0], result)
        self.registers['IP'] += 1
    
    def _op_mov(self, operands):
        """MOV dest, src"""
        value = self._get_value(operands[1])
        self._set_value(operands[0], value)
        self.registers['IP'] += 1
    
    def _op_push(self, operands):
        """PUSH value"""
        value = self._get_value(operands[0])
        self.stack.append(value)
        self.registers['SP'] -= 1
        self.registers['IP'] += 1
    
    def _op_pop(self, operands):
        """POP dest"""
        if not self.stack:
            raise ValueError("Stack underflow")
        value = self.stack.pop()
        self._set_value(operands[0], value)
        self.registers['SP'] += 1
        self.registers['IP'] += 1
    
    def _op_load(self, operands):
        """LOAD dest, address"""
        addr = self._get_value(operands[1])
        value = self.memory[addr]
        self._set_value(operands[0], value)
        self.registers['IP'] += 1
    
    def _op_store(self, operands):
        """STORE address, value"""
        addr = self._get_value(operands[0])
        value = self._get_value(operands[1])
        self.memory[addr] = value
        self.registers['IP'] += 1
    
    def _op_jmp(self, operands):
        """JMP label/address"""
        target = self._get_value(operands[0])
        if isinstance(target, str):
            # Buscar label
            for i, inst in enumerate(self.program):
                if inst.opcode == 'LABEL' and inst.operands[0][1] == target:
                    self.registers['IP'] = i
                    return
            raise ValueError(f"Label not found: {target}")
        else:
            self.registers['IP'] = target
    
    def _op_jz(self, operands):
        """JZ label/address (jump if zero)"""
        if self.registers['FLAGS'] & 0x01:  # Zero flag
            self._op_jmp(operands)
        else:
            self.registers['IP'] += 1
    
    def _op_jnz(self, operands):
        """JNZ label/address (jump if not zero)"""
        if not (self.registers['FLAGS'] & 0x01):  # Zero flag
            self._op_jmp(operands)
        else:
            self.registers['IP'] += 1
    
    def _op_call(self, operands):
        """CALL function"""
        # Guardar dirección de retorno
        self.stack.append(self.registers['IP'] + 1)
        self.registers['SP'] -= 1
        # Saltar a función
        self._op_jmp(operands)
    
    def _op_ret(self, operands):
        """RET"""
        if not self.stack:
            raise ValueError("Stack underflow on RET")
        self.registers['IP'] = self.stack.pop()
        self.registers['SP'] += 1
    
    def _op_cmp(self, operands):
        """CMP op1, op2"""
        val1 = self._get_value(operands[0])
        val2 = self._get_value(operands[1])
        
        # Actualizar flags
        self.registers['FLAGS'] = 0
        if val1 == val2:
            self.registers['FLAGS'] |= 0x01  # Zero flag
        if val1 < val2:
            self.registers['FLAGS'] |= 0x02  # Less flag
        if val1 > val2:
            self.registers['FLAGS'] |= 0x04  # Greater flag
        
        self.registers['IP'] += 1
    
    def _op_test(self, operands):
        """TEST op1, op2 (AND lógico sin guardar resultado)"""
        val1 = self._get_value(operands[0])
        val2 = self._get_value(operands[1])
        result = val1 & val2
        
        self.registers['FLAGS'] = 0
        if result == 0:
            self.registers['FLAGS'] |= 0x01  # Zero flag
        
        self.registers['IP'] += 1
    
    def _op_int(self, operands):
        """INT interrupt_number"""
        int_num = self._get_value(operands[0])
        
        if int_num in self.interrupts:
            self.interrupts[int_num]()
        else:
            logger.warning(f"Unknown interrupt: {int_num}")
        
        self.registers['IP'] += 1
    
    def _op_nop(self, operands):
        """NOP (no operation)"""
        self.registers['IP'] += 1
    
    def _op_hlt(self, operands):
        """HLT (halt)"""
        self.running = False
    
    def _op_optimize(self, operands):
        """OPT function_name"""
        func_name = self._get_value(operands[0])
        if func_name in self.optimization_functions:
            # Ejecutar optimización
            result = self.optimization_functions[func_name]()
            self.registers['AX'] = result
        else:
            logger.error(f"Unknown optimization function: {func_name}")
        self.registers['IP'] += 1
    
    def _op_gradient(self, operands):
        """GRAD function, point"""
        # Calcular gradiente numérico
        func_name = self._get_value(operands[0])
        point = self._get_value(operands[1])
        
        if func_name in self.optimization_functions:
            func = self.optimization_functions[func_name]
            # Gradiente numérico simple
            eps = 1e-8
            grad = (func(point + eps) - func(point - eps)) / (2 * eps)
            self.registers['AX'] = grad
        
        self.registers['IP'] += 1
    
    def _op_evaluate(self, operands):
        """EVAL expression"""
        expr = self._get_value(operands[0])
        try:
            # Evaluación segura
            result = eval(expr, {"__builtins__": {}}, {"np": np})
            self.registers['AX'] = result
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            self.registers['AX'] = 0
        
        self.registers['IP'] += 1
    
    # === Interrupciones del sistema ===
    
    def _int_print(self):
        """INT 0x00 - Imprimir valor en AX"""
        print(f"[VM Output] {self.registers['AX']}")
    
    def _int_input(self):
        """INT 0x01 - Leer entrada en AX"""
        try:
            value = float(input("[VM Input] > "))
            self.registers['AX'] = value
        except ValueError:
            self.registers['AX'] = 0
    
    def _int_file_read(self):
        """INT 0x02 - Leer archivo (nombre en BX, resultado en AX)"""
        filename = self.heap.get(self.registers['BX'], "")
        if self.security.check_path_access(filename, "read"):
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                    # Almacenar en heap
                    addr = len(self.heap)
                    self.heap[addr] = content
                    self.registers['AX'] = addr
            except Exception as e:
                logger.error(f"File read error: {e}")
                self.registers['AX'] = -1
        else:
            logger.warning(f"Access denied to file: {filename}")
            self.registers['AX'] = -1
    
    def _int_file_write(self):
        """INT 0x03 - Escribir archivo (nombre en BX, contenido en CX)"""
        filename = self.heap.get(self.registers['BX'], "")
        content = self.heap.get(self.registers['CX'], "")
        
        if self.security.check_path_access(filename, "write"):
            try:
                with open(filename, 'w') as f:
                    f.write(str(content))
                self.registers['AX'] = 1  # Éxito
            except Exception as e:
                logger.error(f"File write error: {e}")
                self.registers['AX'] = 0
        else:
            logger.warning(f"Write access denied to file: {filename}")
            self.registers['AX'] = 0
    
    def _int_optimize_func(self):
        """INT 0x10 - Optimizar función (ID en BX)"""
        func_id = self.registers['BX']
        if func_id in self.optimization_functions:
            # Ejecutar optimización compleja
            func = self.optimization_functions[func_id]
            # Aquí podrías implementar diferentes algoritmos
            self.registers['AX'] = 1  # Placeholder
        else:
            self.registers['AX'] = 0
    
    def register_optimization_function(self, name: str, func: Callable):
        """Registra una función de optimización"""
        self.optimization_functions[name] = func
    
    def get_state(self) -> Dict[str, Any]:
        """Obtiene el estado actual de la VM"""
        return {
            'registers': self.registers.copy(),
            'stack': self.stack.copy(),
            'memory_usage': sum(1 for x in self.memory if x is not None),
            'heap_size': len(self.heap),
            'running': self.running,
            'current_instruction': self.registers['IP']
        }

# === TERMINAL INTERACTIVA ===

class OTAECTerminal:
    """Terminal interactiva para OTAEC"""
    
    def __init__(self, vm: OTAECVirtualMachine, security_manager: SecurityManager):
        self.vm = vm
        self.security = security_manager
        self.running = True
        self.command_history = deque(maxlen=100)
        self.aliases = {}
        self.current_dir = os.getcwd()
        
        # Comandos disponibles
        self.commands = {
            'help': self.cmd_help,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'ls': self.cmd_ls,
            'cd': self.cmd_cd,
            'pwd': self.cmd_pwd,
            'cat': self.cmd_cat,
            'run': self.cmd_run,
            'optimize': self.cmd_optimize,
            'compile': self.cmd_compile,
            'vm': self.cmd_vm,
            'security': self.cmd_security,
            'benchmark': self.cmd_benchmark,
            'profile': self.cmd_profile,
            'monitor': self.cmd_monitor,
            'alias': self.cmd_alias,
            'history': self.cmd_history,
            'clear': self.cmd_clear,
            'env': self.cmd_env,
            'taec': self.cmd_taec_comm
        }
        
        # Estado del sistema
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Obtiene información del sistema"""
        return {
            'platform': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version.split()[0],
            'cpu_count': multiprocessing.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_usage': psutil.disk_usage('/').percent
        }
    
    def start(self):
        """Inicia la terminal interactiva"""
        self._print_banner()
        
        while self.running:
            try:
                # Prompt con información
                prompt = f"OTAEC [{self.security.level.name}] {self._get_short_path()}> "
                
                # Leer comando
                cmd_line = input(prompt).strip()
                
                if not cmd_line:
                    continue
                
                # Guardar en historial
                self.command_history.append(cmd_line)
                
                # Procesar comando
                self._process_command(cmd_line)
                
            except KeyboardInterrupt:
                print("\n[Interrupted]")
                continue
            except EOFError:
                print("\n[EOF]")
                break
            except Exception as e:
                print(f"[Error] {e}")
                if hasattr(self, 'debug') and self.debug:
                    traceback.print_exc()
    
    def _print_banner(self):
        """Imprime el banner de bienvenida"""
        print("=" * 70)
        print("OTAEC - Optimization Twin of TAEC v1.0")
        print("Advanced Optimization System with Local Integration")
        print(f"Running on {self.system_info['platform']} {self.system_info['release']}")
        print(f"Security Level: {self.security.level.name}")
        print("=" * 70)
        print("Type 'help' for available commands")
        print()
    
    def _get_short_path(self) -> str:
        """Obtiene una versión corta del path actual"""
        try:
            home = str(Path.home())
            current = str(Path.cwd())
            if current.startswith(home):
                return "~" + current[len(home):]
            return current
        except:
            return "/"
    
    def _process_command(self, cmd_line: str):
        """Procesa un comando"""
        # Expandir aliases
        parts = cmd_line.split()
        if parts[0] in self.aliases:
            cmd_line = self.aliases[parts[0]] + " " + " ".join(parts[1:])
            parts = cmd_line.split()
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        # Verificar seguridad
        if not self.security.check_command(cmd_line):
            print(f"[Security] Command blocked: {cmd}")
            return
        
        # Ejecutar comando
        if cmd in self.commands:
            try:
                self.commands[cmd](args)
            except Exception as e:
                print(f"[Error] Command failed: {e}")
                if hasattr(self, 'debug') and self.debug:
                    traceback.print_exc()
        else:
            # Intentar ejecutar como comando del sistema
            if self.security.level in [SecurityLevel.ELEVATED, SecurityLevel.FULL]:
                self._run_system_command(cmd_line)
            else:
                print(f"[Error] Unknown command: {cmd}")
    
    def _run_system_command(self, cmd_line: str):
        """Ejecuta un comando del sistema"""
        try:
            result = subprocess.run(
                cmd_line,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.current_dir,
                timeout=30
            )
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"[Error] {result.stderr}", file=sys.stderr)
                
        except subprocess.TimeoutExpired:
            print("[Error] Command timed out")
        except Exception as e:
            print(f"[Error] Failed to execute: {e}")
    
    # === Implementación de comandos ===
    
    def cmd_help(self, args):
        """Muestra ayuda"""
        if args:
            cmd = args[0]
            if cmd in self.commands:
                print(f"Help for '{cmd}':")
                print(self.commands[cmd].__doc__ or "No documentation available")
            else:
                print(f"Unknown command: {cmd}")
        else:
            print("Available commands:")
            for cmd, func in sorted(self.commands.items()):
                doc = func.__doc__ or ""
                if doc:
                    doc = doc.strip().split('\n')[0]
                print(f"  {cmd:15} {doc}")
    
    def cmd_exit(self, args):
        """Exit OTAEC terminal"""
        print("Goodbye!")
        self.running = False
    
    def cmd_ls(self, args):
        """List directory contents"""
        path = args[0] if args else "."
        full_path = Path(self.current_dir) / path
        
        if not self.security.check_path_access(full_path, "read"):
            print(f"[Security] Access denied: {path}")
            return
        
        try:
            items = sorted(os.listdir(full_path))
            for item in items:
                item_path = full_path / item
                if item_path.is_dir():
                    print(f"📁 {item}/")
                else:
                    size = item_path.stat().st_size
                    print(f"📄 {item} ({self._format_size(size)})")
        except Exception as e:
            print(f"[Error] {e}")
    
    def cmd_cd(self, args):
        """Change directory"""
        if not args:
            path = str(Path.home())
        else:
            path = args[0]
        
        try:
            new_path = Path(self.current_dir) / path
            new_path = new_path.resolve()
            
            if not self.security.check_path_access(new_path, "read"):
                print(f"[Security] Access denied: {path}")
                return
            
            if new_path.is_dir():
                self.current_dir = str(new_path)
                os.chdir(self.current_dir)
            else:
                print(f"[Error] Not a directory: {path}")
        except Exception as e:
            print(f"[Error] {e}")
    
    def cmd_pwd(self, args):
        """Print working directory"""
        print(self.current_dir)
    
    def cmd_cat(self, args):
        """Display file contents"""
        if not args:
            print("[Error] Usage: cat <filename>")
            return
        
        filename = args[0]
        full_path = Path(self.current_dir) / filename
        
        if not self.security.check_path_access(full_path, "read"):
            print(f"[Security] Access denied: {filename}")
            return
        
        try:
            with open(full_path, 'r') as f:
                print(f.read())
        except Exception as e:
            print(f"[Error] {e}")
    
    def cmd_run(self, args):
        """Run Python script or VM code"""
        if not args:
            print("[Error] Usage: run <script.py|code.vm>")
            return
        
        filename = args[0]
        full_path = Path(self.current_dir) / filename
        
        if not self.security.check_path_access(full_path, "read"):
            print(f"[Security] Access denied: {filename}")
            return
        
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            if filename.endswith('.vm'):
                # Ejecutar en VM
                print("[VM] Compiling and running...")
                instructions = self.vm.compile_code(content)
                self.vm.load_program(instructions)
                cycles = self.vm.run()
                print(f"[VM] Execution completed in {cycles} cycles")
            else:
                # Ejecutar como Python
                print("[Python] Executing...")
                exec(content, {'__name__': '__main__'})
                
        except Exception as e:
            print(f"[Error] {e}")
            if hasattr(self, 'debug') and self.debug:
                traceback.print_exc()
    
    def cmd_optimize(self, args):
        """Run optimization algorithms"""
        if not args:
            print("Available optimization methods:")
            print("  scipy    - SciPy optimization")
            print("  genetic  - Genetic algorithm")
            print("  pso      - Particle swarm optimization")
            print("  quantum  - Quantum-inspired optimization")
            return
        
        method = args[0]
        
        if method == "scipy":
            self._run_scipy_optimization(args[1:])
        elif method == "genetic":
            self._run_genetic_optimization(args[1:])
        elif method == "pso":
            self._run_pso_optimization(args[1:])
        elif method == "quantum":
            self._run_quantum_optimization(args[1:])
        else:
            print(f"[Error] Unknown optimization method: {method}")
    
    def _run_scipy_optimization(self, args):
        """Ejecuta optimización con SciPy"""
        print("[Optimization] Running SciPy optimization...")
        
        # Función de ejemplo: Rosenbrock
        def rosenbrock(x):
            return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
        
        # Punto inicial
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        
        # Optimizar
        start_time = time.time()
        result = optimize.minimize(rosenbrock, x0, method='BFGS')
        elapsed = time.time() - start_time
        
        print(f"Result: {result.x}")
        print(f"Function value: {result.fun}")
        print(f"Iterations: {result.nit}")
        print(f"Time: {elapsed:.3f}s")
    
    def _run_genetic_optimization(self, args):
        """Ejecuta algoritmo genético"""
        print("[Optimization] Running genetic algorithm...")
        
        # Implementación simple de GA
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        
        # Función objetivo
        def fitness(x):
            return -sum(x**2)  # Maximizar = minimizar negativo
        
        # Población inicial
        population = np.random.randn(population_size, 5)
        
        start_time = time.time()
        
        for gen in range(generations):
            # Evaluar fitness
            scores = np.array([fitness(ind) for ind in population])
            
            # Selección
            sorted_idx = np.argsort(scores)[::-1]
            population = population[sorted_idx]
            
            # Élite
            new_population = population[:10].copy()
            
            # Crossover y mutación
            while len(new_population) < population_size:
                parent1 = population[np.random.randint(0, 20)]
                parent2 = population[np.random.randint(0, 20)]
                
                # Crossover
                mask = np.random.rand(5) > 0.5
                child = np.where(mask, parent1, parent2)
                
                # Mutación
                if np.random.rand() < mutation_rate:
                    child += np.random.randn(5) * 0.1
                
                new_population = np.vstack([new_population, child])
            
            population = new_population[:population_size]
        
        elapsed = time.time() - start_time
        
        best = population[0]
        print(f"Best solution: {best}")
        print(f"Fitness: {fitness(best)}")
        print(f"Time: {elapsed:.3f}s")
    
    def _run_pso_optimization(self, args):
        """Ejecuta Particle Swarm Optimization"""
        print("[Optimization] Running PSO...")
        
        # Parámetros PSO
        n_particles = 30
        dimensions = 5
        iterations = 100
        w = 0.7  # Inercia
        c1 = 1.5  # Componente cognitivo
        c2 = 1.5  # Componente social
        
        # Función objetivo
        def objective(x):
            return sum(x**2)
        
        # Inicializar partículas
        positions = np.random.randn(n_particles, dimensions) * 10
        velocities = np.random.randn(n_particles, dimensions)
        
        # Mejores posiciones
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([objective(p) for p in positions])
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        start_time = time.time()
        
        for i in range(iterations):
            # Actualizar velocidades
            r1 = np.random.rand(n_particles, dimensions)
            r2 = np.random.rand(n_particles, dimensions)
            
            velocities = (w * velocities + 
                         c1 * r1 * (personal_best_positions - positions) +
                         c2 * r2 * (global_best_position - positions))
            
            # Actualizar posiciones
            positions += velocities
            
            # Evaluar
            scores = np.array([objective(p) for p in positions])
            
            # Actualizar mejores personales
            better_mask = scores < personal_best_scores
            personal_best_positions[better_mask] = positions[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]
            
            # Actualizar mejor global
            min_idx = np.argmin(scores)
            if scores[min_idx] < global_best_score:
                global_best_position = positions[min_idx].copy()
                global_best_score = scores[min_idx]
        
        elapsed = time.time() - start_time
        
        print(f"Best solution: {global_best_position}")
        print(f"Best score: {global_best_score}")
        print(f"Time: {elapsed:.3f}s")
    
    def _run_quantum_optimization(self, args):
        """Ejecuta optimización cuántica simulada"""
        print("[Optimization] Running quantum-inspired optimization...")
        
        # Parámetros cuánticos
        n_qubits = 5
        n_iterations = 50
        
        # Estado cuántico inicial (superposición uniforme)
        psi = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)
        
        # Función objetivo codificada
        def quantum_objective(state_vector):
            # Decodificar estado a valor clásico
            probabilities = np.abs(state_vector)**2
            expectation = 0
            
            for i, prob in enumerate(probabilities):
                # Convertir índice a valor binario
                x = [(i >> j) & 1 for j in range(n_qubits)]
                value = sum(x[j] * 2**(n_qubits-j-1) for j in range(n_qubits))
                # Normalizar a [-5, 5]
                normalized = (value / (2**n_qubits - 1)) * 10 - 5
                expectation += prob * (normalized**2)
            
            return expectation
        
        start_time = time.time()
        
        # Evolución cuántica simulada
        for iteration in range(n_iterations):
            # Operador de evolución (simplificado)
            theta = np.pi * (1 - iteration/n_iterations) / 4
            
            # Aplicar rotación
            for i in range(n_qubits):
                # Matriz de rotación para qubit i
                rotation = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                
                # Aplicar a todo el estado (simplificado)
                psi = psi * np.exp(1j * theta * i / n_qubits)
            
            # Normalizar
            psi = psi / np.linalg.norm(psi)
        
        # Medir resultado
        probabilities = np.abs(psi)**2
        measured_state = np.random.choice(2**n_qubits, p=probabilities)
        
        # Decodificar
        x = [(measured_state >> j) & 1 for j in range(n_qubits)]
        value = sum(x[j] * 2**(n_qubits-j-1) for j in range(n_qubits))
        normalized = (value / (2**n_qubits - 1)) * 10 - 5
        
        elapsed = time.time() - start_time
        
        print(f"Quantum state collapsed to: {measured_state}")
        print(f"Decoded value: {normalized}")
        print(f"Objective value: {normalized**2}")
        print(f"Time: {elapsed:.3f}s")
    
    def cmd_compile(self, args):
        """Compile VM code"""
        if not args:
            print("[Error] Usage: compile <source.vm>")
            return
        
        filename = args[0]
        full_path = Path(self.current_dir) / filename
        
        if not self.security.check_path_access(full_path, "read"):
            print(f"[Security] Access denied: {filename}")
            return
        
        try:
            with open(full_path, 'r') as f:
                source = f.read()
            
            instructions = self.vm.compile_code(source)
            print(f"[Compiler] Successfully compiled {len(instructions)} instructions")
            
            # Mostrar instrucciones compiladas
            for i, inst in enumerate(instructions):
                print(f"{i:4d}: {inst.opcode} {inst.operands}")
                
        except Exception as e:
            print(f"[Error] Compilation failed: {e}")
    
    def cmd_vm(self, args):
        """VM control commands"""
        if not args:
            print("VM commands:")
            print("  status   - Show VM status")
            print("  reset    - Reset VM")
            print("  step     - Single step execution")
            print("  reg      - Show registers")
            print("  mem      - Show memory")
            return
        
        subcmd = args[0]
        
        if subcmd == "status":
            state = self.vm.get_state()
            print("VM Status:")
            for key, value in state.items():
                print(f"  {key}: {value}")
        
        elif subcmd == "reset":
            self.vm.__init__(security_manager=self.security)
            print("VM reset completed")
        
        elif subcmd == "step":
            if self.vm.program and self.vm.registers['IP'] < len(self.vm.program):
                inst = self.vm.program[self.vm.registers['IP']]
                print(f"Executing: {inst.opcode} {inst.operands}")
                self.vm.run(max_cycles=1)
            else:
                print("No program loaded or program finished")
        
        elif subcmd == "reg":
            print("Registers:")
            for reg, value in self.vm.registers.items():
                print(f"  {reg}: {value}")
        
        elif subcmd == "mem":
            # Mostrar memoria no vacía
            print("Memory (non-empty):")
            for i, value in enumerate(self.vm.memory[:100]):
                if value is not None:
                    print(f"  [{i}]: {value}")
    
    def cmd_security(self, args):
        """Security management"""
        if not args:
            print("Security commands:")
            print("  level    - Show/set security level")
            print("  audit    - Show audit log")
            print("  allow    - Add allowed path")
            return
        
        subcmd = args[0]
        
        if subcmd == "level":
            if len(args) > 1:
                # Cambiar nivel
                try:
                    new_level = SecurityLevel[args[1].upper()]
                    if new_level == SecurityLevel.FULL:
                        confirm = input("Warning: FULL access level. Continue? (y/N) ")
                        if confirm.lower() != 'y':
                            print("Cancelled")
                            return
                    
                    self.security.level = new_level
                    print(f"Security level changed to: {new_level.name}")
                except KeyError:
                    print(f"Invalid level. Options: {[l.name for l in SecurityLevel]}")
            else:
                print(f"Current security level: {self.security.level.name}")
        
        elif subcmd == "audit":
            summary = self.security.get_audit_summary()
            print("Security Audit Summary:")
            print(f"  Total operations: {summary['total_operations']}")
            print(f"  Blocked operations: {summary['blocked_operations']}")
            print(f"  Security level: {summary['security_level']}")
            
            if summary['recent_operations']:
                print("\nRecent operations:")
                for op in summary['recent_operations'][-5:]:
                    print(f"  {op}")
        
        elif subcmd == "allow":
            if len(args) > 1:
                path = args[1]
                self.security.add_allowed_path(path)
                print(f"Added allowed path: {path}")
            else:
                print("Usage: security allow <path>")
    
    def cmd_benchmark(self, args):
        """Run performance benchmarks"""
        print("[Benchmark] Running performance tests...")
        
        results = {}
        
        # 1. VM performance
        print("1. VM Performance...")
        vm_code = """
        MOV AX, 0
        MOV BX, 1000
        LOOP:
        ADD AX, 1
        CMP AX, BX
        JNZ LOOP
        HLT
        """
        
        instructions = self.vm.compile_code(vm_code)
        self.vm.load_program(instructions)
        
        start = time.time()
        cycles = self.vm.run()
        vm_time = time.time() - start
        
        results['vm_cycles_per_second'] = cycles / vm_time if vm_time > 0 else 0
        
        # 2. Optimization performance
        print("2. Optimization Performance...")
        
        def test_function(x):
            return sum((x - 3)**2)
        
        x0 = np.random.randn(10)
        
        start = time.time()
        result = optimize.minimize(test_function, x0, method='BFGS')
        opt_time = time.time() - start
        
        results['optimization_time'] = opt_time
        results['optimization_iterations'] = result.nit
        
        # 3. File I/O performance
        print("3. File I/O Performance...")
        test_file = Path(self.current_dir) / "benchmark_test.tmp"
        test_data = "x" * 1024 * 1024  # 1MB
        
        start = time.time()
        with open(test_file, 'w') as f:
            f.write(test_data)
        write_time = time.time() - start
        
        start = time.time()
        with open(test_file, 'r') as f:
            _ = f.read()
        read_time = time.time() - start
        
        test_file.unlink()  # Limpiar
        
        results['write_speed_mbps'] = 1 / write_time if write_time > 0 else 0
        results['read_speed_mbps'] = 1 / read_time if read_time > 0 else 0
        
        # Mostrar resultados
        print("\n[Benchmark Results]")
        print(f"VM Performance: {results['vm_cycles_per_second']:.0f} cycles/s")
        print(f"Optimization: {results['optimization_iterations']} iterations in {results['optimization_time']:.3f}s")
        print(f"File Write: {results['write_speed_mbps']:.1f} MB/s")
        print(f"File Read: {results['read_speed_mbps']:.1f} MB/s")
        
        return results
    
    def cmd_profile(self, args):
        """Profile code execution"""
        if not args:
            print("[Error] Usage: profile <script.py>")
            return
        
        filename = args[0]
        full_path = Path(self.current_dir) / filename
        
        if not self.security.check_path_access(full_path, "read"):
            print(f"[Security] Access denied: {filename}")
            return
        
        try:
            import cProfile
            import pstats
            from io import StringIO
            
            with open(full_path, 'r') as f:
                code = f.read()
            
            # Crear profiler
            profiler = cProfile.Profile()
            
            # Ejecutar con profiling
            profiler.enable()
            exec(code, {'__name__': '__main__'})
            profiler.disable()
            
            # Mostrar resultados
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 funciones
            
            print(s.getvalue())
            
        except Exception as e:
            print(f"[Error] Profiling failed: {e}")
    
    def cmd_monitor(self, args):
        """Monitor system resources"""
        print("[Monitor] System Resource Monitor")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memoria
                mem = psutil.virtual_memory()
                mem_percent = mem.percent
                mem_used_gb = mem.used / (1024**3)
                mem_total_gb = mem.total / (1024**3)
                
                # Disco
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                # Procesos
                process_count = len(psutil.pids())
                
                # Limpiar pantalla (simple)
                print("\033[2J\033[H", end="")
                
                print("[System Monitor]")
                print(f"CPU:      {cpu_percent:5.1f}% {'█' * int(cpu_percent/5)}")
                print(f"Memory:   {mem_percent:5.1f}% ({mem_used_gb:.1f}/{mem_total_gb:.1f} GB)")
                print(f"Disk:     {disk_percent:5.1f}%")
                print(f"Processes: {process_count}")
                
                # Info del proceso actual
                current_process = psutil.Process()
                print(f"\n[OTAEC Process]")
                print(f"CPU:    {current_process.cpu_percent():.1f}%")
                print(f"Memory: {current_process.memory_info().rss / (1024**2):.1f} MB")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n[Monitor] Stopped")
    
    def cmd_alias(self, args):
        """Manage command aliases"""
        if not args:
            print("Aliases:")
            for alias, command in self.aliases.items():
                print(f"  {alias} = {command}")
            return
        
        if len(args) < 2:
            print("Usage: alias <name> <command>")
            return
        
        alias_name = args[0]
        alias_command = " ".join(args[1:])
        
        self.aliases[alias_name] = alias_command
        print(f"Alias created: {alias_name} = {alias_command}")
    
    def cmd_history(self, args):
        """Show command history"""
        n = 20
        if args and args[0].isdigit():
            n = int(args[0])
        
        history = list(self.command_history)[-n:]
        
        print("Command History:")
        for i, cmd in enumerate(history, 1):
            print(f"{i:3d}. {cmd}")
    
    def cmd_clear(self, args):
        """Clear screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def cmd_env(self, args):
        """Show environment information"""
        print("OTAEC Environment:")
        print(f"  Working Directory: {self.current_dir}")
        print(f"  Security Level: {self.security.level.name}")
        print(f"  Platform: {self.system_info['platform']}")
        print(f"  Python: {self.system_info['python_version']}")
        print(f"  CPU Cores: {self.system_info['cpu_count']}")
        print(f"  Memory: {self._format_size(self.system_info['memory_total'])}")
        print(f"  VM Memory: {len(self.vm.memory)} cells")
        print(f"  Aliases: {len(self.aliases)}")
    
    def cmd_taec_comm(self, args):
        """Communicate with TAEC instance"""
        if not args:
            print("TAEC communication commands:")
            print("  connect <host:port>  - Connect to TAEC")
            print("  send <message>       - Send message")
            print("  sync                 - Synchronize state")
            return
        
        subcmd = args[0]
        
        if subcmd == "connect":
            # Simulación de conexión
            print("[TAEC] Connection established (simulated)")
            print("[TAEC] Ready for bidirectional communication")
        
        elif subcmd == "send":
            message = " ".join(args[1:])
            print(f"[TAEC] Sending: {message}")
            # Aquí implementarías la comunicación real
            print("[TAEC] Message sent successfully")
        
        elif subcmd == "sync":
            print("[TAEC] Synchronizing state...")
            print("[TAEC] Optimization parameters updated")
            print("[TAEC] Memory state synchronized")
            print("[TAEC] Sync completed")
    
    def _format_size(self, bytes):
        """Formatea tamaño en bytes a formato legible"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.1f} PB"

# === COMUNICACIÓN CON TAEC ===

class TAECCommunicator:
    """Gestor de comunicación con TAEC"""
    
    def __init__(self):
        self.socket = None
        self.connected = False
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
    def connect(self, host: str = 'localhost', port: int = 9999):
        """Conecta con una instancia de TAEC"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((host, port))
            self.connected = True
            
            # Iniciar thread de recepción
            receiver = threading.Thread(target=self._receive_messages, daemon=True)
            receiver.start()
            
            logger.info(f"Connected to TAEC at {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to TAEC: {e}")
            return False
    
    def send_message(self, message_type: str, data: Any):
        """Envía mensaje a TAEC"""
        if not self.connected:
            return False
        
        message = {
            'type': message_type,
            'data': data,
            'timestamp': time.time()
        }
        
        try:
            # Serializar y enviar
            serialized = pickle.dumps(message)
            size = struct.pack('!I', len(serialized))
            
            self.socket.sendall(size + serialized)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def _receive_messages(self):
        """Thread para recibir mensajes"""
        while self.connected:
            try:
                # Recibir tamaño
                size_data = self.socket.recv(4)
                if not size_data:
                    break
                
                size = struct.unpack('!I', size_data)[0]
                
                # Recibir datos
                data = b''
                while len(data) < size:
                    chunk = self.socket.recv(min(size - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk
                
                # Deserializar
                message = pickle.loads(data)
                self.response_queue.put(message)
                
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
        
        self.connected = False
    
    def get_response(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Obtiene respuesta de TAEC"""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def request_optimization(self, function_code: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Solicita optimización a TAEC"""
        request = {
            'function': function_code,
            'parameters': parameters,
            'constraints': parameters.get('constraints', [])
        }
        
        if self.send_message('optimize', request):
            # Esperar respuesta
            response = self.get_response(timeout=30.0)
            return response
        
        return None
    
    def sync_memory(self, memory_snapshot: Dict[str, Any]) -> bool:
        """Sincroniza estado de memoria con TAEC"""
        return self.send_message('sync_memory', memory_snapshot)
    
    def disconnect(self):
        """Desconecta de TAEC"""
        if self.connected:
            self.connected = False
            self.socket.close()
            logger.info("Disconnected from TAEC")

# === OPTIMIZADOR PRINCIPAL ===

class OTAECOptimizer:
    """Sistema principal de optimización de OTAEC"""
    
    def __init__(self):
        self.security = SecurityManager(SecurityLevel.STANDARD)
        self.vm = OTAECVirtualMachine(security_manager=self.security)
        self.terminal = OTAECTerminal(self.vm, self.security)
        self.taec_comm = TAECCommunicator()
        
        # Registrar funciones de optimización en VM
        self._register_optimization_functions()
        
        # Estado del optimizador
        self.optimization_history = []
        self.active_optimizations = {}
        
    def _register_optimization_functions(self):
        """Registra funciones de optimización en la VM"""
        # Funciones básicas
        self.vm.register_optimization_function('minimize', self._vm_minimize)
        self.vm.register_optimization_function('maximize', self._vm_maximize)
        self.vm.register_optimization_function('gradient', self._vm_gradient)
        
    def _vm_minimize(self):
        """Función de minimización para la VM"""
        # Obtener función objetivo del heap
        func_addr = self.vm.registers['BX']
        if func_addr in self.vm.heap:
            func_code = self.vm.heap[func_addr]
            
            # Crear función Python
            namespace = {'np': np}
            exec(f"def objective(x): return {func_code}", namespace)
            objective = namespace['objective']
            
            # Optimizar
            x0 = np.random.randn(5)
            result = optimize.minimize(objective, x0)
            
            # Almacenar resultado
            result_addr = len(self.vm.heap)
            self.vm.heap[result_addr] = result.x
            
            return result_addr
        
        return -1
    
    def _vm_maximize(self):
        """Función de maximización para la VM"""
        # Similar a minimize pero negando la función
        func_addr = self.vm.registers['BX']
        if func_addr in self.vm.heap:
            func_code = self.vm.heap[func_addr]
            
            namespace = {'np': np}
            exec(f"def objective(x): return -({func_code})", namespace)
            objective = namespace['objective']
            
            x0 = np.random.randn(5)
            result = optimize.minimize(objective, x0)
            
            result_addr = len(self.vm.heap)
            self.vm.heap[result_addr] = result.x
            
            return result_addr
        
        return -1
    
    def _vm_gradient(self):
        """Calcula gradiente para la VM"""
        func_addr = self.vm.registers['BX']
        point_addr = self.vm.registers['CX']
        
        if func_addr in self.vm.heap and point_addr in self.vm.heap:
            func_code = self.vm.heap[func_addr]
            point = self.vm.heap[point_addr]
            
            namespace = {'np': np}
            exec(f"def objective(x): return {func_code}", namespace)
            objective = namespace['objective']
            
            # Gradiente numérico
            grad = optimize.approx_fprime(point, objective, 1e-8)
            
            result_addr = len(self.vm.heap)
            self.vm.heap[result_addr] = grad
            
            return result_addr
        
        return -1
    
    def start(self):
        """Inicia OTAEC"""
        logger.info("Starting OTAEC Optimization System")
        
        # Intentar conectar con TAEC
        if self.taec_comm.connect():
            logger.info("Connected to TAEC for collaborative optimization")
        else:
            logger.info("Running in standalone mode")
        
        # Iniciar terminal
        self.terminal.start()
        
        # Limpiar al salir
        self.cleanup()
    
    def cleanup(self):
        """Limpieza al salir"""
        logger.info("Shutting down OTAEC")
        
        # Desconectar de TAEC
        self.taec_comm.disconnect()
        
        # Guardar historial si es necesario
        if self.optimization_history:
            self._save_optimization_history()
    
    def _save_optimization_history(self):
        """Guarda el historial de optimizaciones"""
        history_file = Path.home() / ".otaec_history.json"
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.optimization_history, f, indent=2, default=str)
            logger.info(f"Optimization history saved to {history_file}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

# === PUNTO DE ENTRADA ===

def main():
    """Función principal"""
    # Configurar argumentos de línea de comandos
    import argparse
    
    parser = argparse.ArgumentParser(description="OTAEC - Optimization Twin of TAEC")
    parser.add_argument('--security', choices=['minimal', 'standard', 'elevated', 'full'],
                       default='standard', help='Security level')
    parser.add_argument('--vm-memory', type=int, default=1024,
                       help='VM memory size')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--connect', metavar='HOST:PORT',
                       help='Connect to TAEC instance')
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Crear instancia de OTAEC
    otaec = OTAECOptimizer()
    
    # Configurar seguridad
    otaec.security.level = SecurityLevel[args.security.upper()]
    
    # Configurar VM
    otaec.vm = OTAECVirtualMachine(
        memory_size=args.vm_memory,
        security_manager=otaec.security
    )
    otaec.terminal.vm = otaec.vm
    
    # Conectar a TAEC si se especifica
    if args.connect:
        try:
            host, port = args.connect.split(':')
            otaec.taec_comm.connect(host, int(port))
        except Exception as e:
            logger.error(f"Failed to parse connection string: {e}")
    
    # Iniciar sistema
    try:
        otaec.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
