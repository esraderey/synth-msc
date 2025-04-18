#!/usr/bin/env python
"""
Utilidad para ver los resultados de las pruebas
"""
import os
import sys
import argparse
from pathlib import Path
import webbrowser

def main():
    parser = argparse.ArgumentParser(description="Ver resultados de pruebas")
    parser.add_argument("--results-dir", help="Directorio de resultados",
                       default="test_results/latest/results")
    parser.add_argument("--stdout", help="Ver salida estándar", action="store_true")
    parser.add_argument("--stderr", help="Ver salida de error", action="store_true")
    parser.add_argument("--coverage", help="Abrir informe de cobertura", action="store_true")
    parser.add_argument("--summary", help="Mostrar resumen", action="store_true", default=True)
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: El directorio de resultados {results_dir} no existe.")
        if Path("test_results").exists():
            print("\nResultados disponibles:")
            for d in Path("test_results").iterdir():
                if d.is_dir() and d.name != "latest":
                    print(f"  - {d}")
        return 1
    
    if args.summary or (not args.stdout and not args.stderr and not args.coverage):
        summary_file = results_dir / "test_summary.txt"
        if summary_file.exists():
            print("\n" + "="*80)
            print(summary_file.read_text())
            print("="*80 + "\n")
        else:
            print("No se encontró archivo de resumen.")
    
    if args.stdout:
        stdout_file = results_dir / "test_stdout.txt"
        if stdout_file.exists():
            print("\n" + "="*80)
            print("SALIDA ESTÁNDAR:")
            print("="*80)
            print(stdout_file.read_text())
        else:
            print("No se encontró archivo de salida estándar.")
    
    if args.stderr:
        stderr_file = results_dir / "test_stderr.txt"
        if stderr_file.exists():
            print("\n" + "="*80)
            print("SALIDA DE ERROR:")
            print("="*80)
            print(stderr_file.read_text())
        else:
            print("No se encontró archivo de salida de error.")
    
    if args.coverage:
        coverage_dir = results_dir / "coverage"
        if coverage_dir.exists():
            coverage_file = coverage_dir / "index.html"
            if coverage_file.exists():
                print(f"Abriendo informe de cobertura: {coverage_file}")
                webbrowser.open(f"file://{coverage_file.absolute()}")
            else:
                print("No se encontró informe de cobertura.")
        else:
            print("No se encontró directorio de cobertura.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())