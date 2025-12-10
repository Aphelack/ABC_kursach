#!/usr/bin/env python3
"""
Compare scalability across multiple processors
Usage: python compare_scalability.py report1.json report2.json [--output-dir DIR]
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

class MultiProcessorComparison:
    def __init__(self, reports: List[Dict]):
        self.reports = reports
        self.cpu_names = [r['system_info']['cpu_model'] for r in reports]
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62828']
        
    def extract_thread_data(self, report: Dict) -> Dict:
        """Extract thread scaling data from report"""
        results = report['results']
        
        # Group by n_estimators
        by_estimators = {}
        for r in results:
            key = r['n_estimators']
            if key not in by_estimators:
                by_estimators[key] = []
            by_estimators[key].append(r)
        
        # Find series with most thread variations
        best_series = max(by_estimators.values(), key=len)
        best_series = sorted(best_series, key=lambda x: x['n_threads'])
        
        if len(best_series) < 2:
            return None
        
        # Calculate speedup
        logical_cores = report['system_info']['logical_cores']
        threads = []
        times = []
        speedups = []
        efficiencies = []
        
        for item in best_series:
            p = item['n_threads'] if item['n_threads'] > 0 else logical_cores
            t = item['avg_time']
            threads.append(p)
            times.append(t)
        
        t1 = times[0]
        for p, t in zip(threads, times):
            s = t1 / t
            e = s / p
            speedups.append(s)
            efficiencies.append(e)
        
        return {
            'threads': threads,
            'times': times,
            'speedups': speedups,
            'efficiencies': efficiencies
        }
    
    def extract_estimator_data(self, report: Dict) -> Dict:
        """Extract estimator scaling data from report"""
        results = report['results']
        
        # Group by threads
        by_threads = {}
        for r in results:
            key = r['n_threads']
            if key not in by_threads:
                by_threads[key] = []
            by_threads[key].append(r)
        
        # Prefer -1 (all cores) or maximum
        if -1 in by_threads:
            series = by_threads[-1]
        else:
            max_threads = max(by_threads.keys())
            series = by_threads[max_threads]
        
        series = sorted(series, key=lambda x: x['n_estimators'])
        
        if len(series) < 2:
            return None
        
        return {
            'n_estimators': [r['n_estimators'] for r in series],
            'times': [r['avg_time'] for r in series],
            'f1_scores': [r['avg_f1_score'] for r in series]
        }
    
    def plot_speedup_comparison(self, output_dir: Path):
        """Compare speedup across processors"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        max_threads = 0
        
        for i, report in enumerate(self.reports):
            data = self.extract_thread_data(report)
            if data is None:
                continue
            
            cpu_name = self.cpu_names[i]
            color = self.colors[i % len(self.colors)]
            
            threads = data['threads']
            speedups = data['speedups']
            max_threads = max(max_threads, max(threads))
            
            ax.plot(threads, speedups, 'o-', linewidth=2.5, markersize=9,
                   label=cpu_name, color=color)
        
        # Ideal speedup line
        ideal_threads = range(1, max_threads + 1)
        ax.plot(ideal_threads, ideal_threads, '--', linewidth=2,
               label='Ideal (Linear)', color='gray', alpha=0.6)
        
        ax.set_xlabel('Number of Threads', fontweight='bold', fontsize=13)
        ax.set_ylabel('Speedup S(p) = T(1) / T(p)', fontweight='bold', fontsize=13)
        ax.set_title('Speedup Comparison Across Processors', 
                    fontweight='bold', fontsize=15, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        filename = output_dir / 'comparison_speedup.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'  ✓ {filename}')
        plt.close()
    
    def plot_efficiency_comparison(self, output_dir: Path):
        """Compare efficiency across processors"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, report in enumerate(self.reports):
            data = self.extract_thread_data(report)
            if data is None:
                continue
            
            cpu_name = self.cpu_names[i]
            color = self.colors[i % len(self.colors)]
            
            threads = data['threads']
            efficiencies = data['efficiencies']
            
            ax.plot(threads, efficiencies, 'o-', linewidth=2.5, markersize=9,
                   label=cpu_name, color=color)
        
        # Ideal efficiency line
        ax.axhline(y=1.0, linestyle='--', linewidth=2,
                  label='Ideal (100%)', color='gray', alpha=0.6)
        
        ax.set_xlabel('Number of Threads', fontweight='bold', fontsize=13)
        ax.set_ylabel('Efficiency E(p) = S(p) / p', fontweight='bold', fontsize=13)
        ax.set_title('Parallel Efficiency Comparison Across Processors', 
                    fontweight='bold', fontsize=15, pad=15)
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        filename = output_dir / 'comparison_efficiency.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'  ✓ {filename}')
        plt.close()
    
    def plot_time_comparison(self, output_dir: Path):
        """Compare absolute training times"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, report in enumerate(self.reports):
            data = self.extract_estimator_data(report)
            if data is None:
                continue
            
            cpu_name = self.cpu_names[i]
            color = self.colors[i % len(self.colors)]
            
            n_estimators = data['n_estimators']
            times = data['times']
            
            ax.plot(n_estimators, times, 'o-', linewidth=2.5, markersize=9,
                   label=cpu_name, color=color)
        
        ax.set_xlabel('Number of Estimators (Trees)', fontweight='bold', fontsize=13)
        ax.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=13)
        ax.set_title('Training Time Comparison Across Processors', 
                    fontweight='bold', fontsize=15, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        filename = output_dir / 'comparison_time_vs_estimators.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'  ✓ {filename}')
        plt.close()
    
    def plot_combined_comparison(self, output_dir: Path):
        """Create a comprehensive 2x2 comparison grid"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Speedup
        ax = axes[0, 0]
        max_threads = 0
        for i, report in enumerate(self.reports):
            data = self.extract_thread_data(report)
            if data:
                cpu_name = self.cpu_names[i][:30]  # Shorten for legend
                color = self.colors[i % len(self.colors)]
                ax.plot(data['threads'], data['speedups'], 'o-', linewidth=2,
                       markersize=7, label=cpu_name, color=color)
                max_threads = max(max_threads, max(data['threads']))
        
        ideal = range(1, max_threads + 1)
        ax.plot(ideal, ideal, '--', linewidth=1.5, color='gray', alpha=0.5, label='Ideal')
        ax.set_xlabel('Threads', fontweight='bold')
        ax.set_ylabel('Speedup', fontweight='bold')
        ax.set_title('Speedup vs Threads', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 2: Efficiency
        ax = axes[0, 1]
        for i, report in enumerate(self.reports):
            data = self.extract_thread_data(report)
            if data:
                cpu_name = self.cpu_names[i][:30]
                color = self.colors[i % len(self.colors)]
                ax.plot(data['threads'], data['efficiencies'], 'o-', linewidth=2,
                       markersize=7, label=cpu_name, color=color)
        
        ax.axhline(y=1.0, linestyle='--', linewidth=1.5, color='gray', alpha=0.5)
        ax.set_xlabel('Threads', fontweight='bold')
        ax.set_ylabel('Efficiency', fontweight='bold')
        ax.set_title('Efficiency vs Threads', fontweight='bold')
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 3: Time vs Estimators
        ax = axes[1, 0]
        for i, report in enumerate(self.reports):
            data = self.extract_estimator_data(report)
            if data:
                cpu_name = self.cpu_names[i][:30]
                color = self.colors[i % len(self.colors)]
                ax.plot(data['n_estimators'], data['times'], 'o-', linewidth=2,
                       markersize=7, label=cpu_name, color=color)
        
        ax.set_xlabel('Number of Estimators', fontweight='bold')
        ax.set_ylabel('Training Time (s)', fontweight='bold')
        ax.set_title('Time vs Estimators', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 4: Performance Summary Table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary table
        table_data = []
        headers = ['Processor', 'Max Speedup', 'Max Efficiency', 'Cores']
        
        for report in self.reports:
            data = self.extract_thread_data(report)
            if data:
                cpu = self.cpu_names[self.reports.index(report)][:40]
                max_speedup = f"{max(data['speedups']):.2f}x"
                max_eff = f"{max(data['efficiencies'])*100:.1f}%"
                cores = report['system_info']['logical_cores']
                table_data.append([cpu, max_speedup, max_eff, cores])
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='left', loc='center',
                           colWidths=[0.5, 0.15, 0.18, 0.1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax.set_title('Performance Summary', fontweight='bold', pad=20)
        
        fig.suptitle('Multi-Processor Scalability Comparison', 
                    fontweight='bold', fontsize=16, y=0.995)
        
        plt.tight_layout()
        filename = output_dir / 'comparison_combined.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'  ✓ {filename}')
        plt.close()
    
    def generate_all_plots(self, output_dir: Path):
        """Generate all comparison plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating multi-processor comparison plots...")
        self.plot_speedup_comparison(output_dir)
        self.plot_efficiency_comparison(output_dir)
        self.plot_time_comparison(output_dir)
        self.plot_combined_comparison(output_dir)
        print(f"\n✓ All comparison plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Compare scalability across multiple processors'
    )
    parser.add_argument('reports', nargs='+', help='JSON report files from benchmarks')
    parser.add_argument('--output-dir', '-o', default='comparison_scalability',
                       help='Output directory for comparison plots')
    
    args = parser.parse_args()
    
    if len(args.reports) < 2:
        print("Error: Need at least 2 report files to compare")
        sys.exit(1)
    
    # Load all reports
    reports = []
    for report_file in args.reports:
        print(f"Loading: {report_file}")
        with open(report_file, 'r') as f:
            reports.append(json.load(f))
    
    output_dir = Path(args.output_dir)
    
    # Generate comparison
    comparator = MultiProcessorComparison(reports)
    comparator.generate_all_plots(output_dir)
    
    print("\n✓ Multi-processor comparison complete!")


if __name__ == '__main__':
    main()
