#!/usr/bin/env python3
"""
Quick example: Generate scalability analysis from existing reports
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Find existing benchmark reports
    reports = list(Path('.').glob('rf_benchmark_*.json'))
    
    if not reports:
        print("No benchmark reports found in current directory.")
        print("Run ./run_scalability.sh first to generate reports.")
        return
    
    print(f"Found {len(reports)} benchmark report(s):")
    for i, report in enumerate(reports, 1):
        print(f"  {i}. {report}")
    
    print("\n" + "="*60)
    print("EXAMPLE ANALYSES YOU CAN RUN:")
    print("="*60)
    
    if len(reports) >= 1:
        report1 = reports[0]
        print(f"\n1. Single processor analysis:")
        print(f"   python3 scalability_analysis.py {report1}")
        print(f"\n   This will generate:")
        print(f"   - Speedup graph (speedup vs threads)")
        print(f"   - Efficiency graph (efficiency vs threads)")
        print(f"   - Time vs estimators graph")
        print(f"   - Text report with statistics")
    
    if len(reports) >= 2:
        report1 = reports[0]
        report2 = reports[1]
        print(f"\n2. Compare two processors:")
        print(f"   python3 compare_scalability.py {report1} {report2}")
        print(f"\n   This will generate:")
        print(f"   - Speedup comparison")
        print(f"   - Efficiency comparison")
        print(f"   - Time comparison")
        print(f"   - Combined comparison dashboard")
    
    print("\n" + "="*60)
    print("RUNNING NEW EXPERIMENTS:")
    print("="*60)
    print("\nTo run new scalability experiments:")
    print("  ./run_scalability.sh")
    print("\nThis will:")
    print("  - Run thread scaling experiment (1, 2, 4, 8, 16, 32 threads)")
    print("  - Run estimator scaling experiment (25, 50, 100, 200, 400 trees)")
    print("  - Automatically generate all visualizations")
    
    print("\n" + "="*60)
    print("CUSTOM EXPERIMENTS:")
    print("="*60)
    print("\nTo customize experiments, edit config files:")
    print("  - config_scalability_threads.json (thread scaling)")
    print("  - config_scalability_estimators.json (estimator scaling)")
    print("\nThen run manually:")
    print("  ./build/rf_benchmark ../dataset/creditcard.csv <config_file>")
    
    print("\n" + "="*60)
    
    # Offer to run analysis on first report
    if reports and len(sys.argv) == 1:
        print("\nWould you like to analyze the first report now? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            print(f"\nRunning analysis on {reports[0]}...")
            subprocess.run(['python3', 'scalability_analysis.py', str(reports[0])])


if __name__ == '__main__':
    main()
