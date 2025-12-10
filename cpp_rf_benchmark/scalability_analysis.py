#!/usr/bin/env python3
"""
Scalability Analysis for Random Forest Benchmark
Analyzes speedup, efficiency, and scaling behavior.
Usage: python scalability_analysis.py <report.json> [--output-dir DIR]
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 100

class ScalabilityAnalyzer:
    def __init__(self, report_data: Dict):
        self.report = report_data
        self.results = report_data['results']
        self.cpu_name = report_data['system_info']['cpu_model']
        self.logical_cores = report_data['system_info']['logical_cores']
        
    def analyze_thread_scaling(self) -> Dict:
        """Analyze speedup and efficiency for thread scaling experiments"""
        # Extract data for constant n_estimators
        thread_data = []
        for r in self.results:
            thread_data.append({
                'threads': r['n_threads'] if r['n_threads'] > 0 else self.logical_cores,
                'n_estimators': r['n_estimators'],
                'time': r['avg_time'],
                'std_time': r['std_time'],
                'f1': r['avg_f1_score'],
                'cpu': r['avg_cpu_usage']
            })
        
        # Group by n_estimators to find series with constant estimators
        by_estimators = {}
        for item in thread_data:
            key = item['n_estimators']
            if key not in by_estimators:
                by_estimators[key] = []
            by_estimators[key].append(item)
        
        # Find the series with most thread variations
        best_series = max(by_estimators.values(), key=len)
        best_series = sorted(best_series, key=lambda x: x['threads'])
        
        if len(best_series) < 2:
            return None
            
        # Calculate speedup and efficiency
        t1 = best_series[0]['time']  # Sequential time
        
        analysis = {
            'threads': [],
            'time': [],
            'std_time': [],
            'speedup': [],
            'efficiency': [],
            'f1_scores': [],
            'cpu_usage': [],
            't1': t1
        }
        
        for item in best_series:
            p = item['threads']
            t = item['time']
            speedup = t1 / t
            efficiency = speedup / p
            
            analysis['threads'].append(p)
            analysis['time'].append(t)
            analysis['std_time'].append(item['std_time'])
            analysis['speedup'].append(speedup)
            analysis['efficiency'].append(efficiency)
            analysis['f1_scores'].append(item['f1'])
            analysis['cpu_usage'].append(item['cpu'])
        
        # Estimate Amdahl's law parameters
        # S(p) = 1 / (s + (1-s)/p), where s is serial fraction
        # Using least squares to fit
        if len(analysis['threads']) >= 3:
            analysis['amdahl_s'] = self._estimate_serial_fraction(
                analysis['threads'], analysis['speedup']
            )
        else:
            analysis['amdahl_s'] = None
            
        return analysis
    
    def _estimate_serial_fraction(self, threads: List[int], speedups: List[float]) -> float:
        """Estimate serial fraction using Amdahl's law"""
        # Try different serial fractions and find best fit
        best_s = 0.0
        best_error = float('inf')
        
        for s in np.linspace(0, 0.5, 100):
            predicted = [1.0 / (s + (1-s)/p) for p in threads]
            error = np.mean([(pred - obs)**2 for pred, obs in zip(predicted, speedups)])
            if error < best_error:
                best_error = error
                best_s = s
        
        return best_s
    
    def analyze_estimator_scaling(self) -> Dict:
        """Analyze scaling with number of estimators"""
        # Find series with constant threads (preferably -1 or max)
        by_threads = {}
        for r in self.results:
            key = r['n_threads']
            if key not in by_threads:
                by_threads[key] = []
            by_threads[key].append(r)
        
        # Prefer -1 (all cores) or maximum threads
        if -1 in by_threads:
            series = by_threads[-1]
        else:
            max_threads = max(by_threads.keys())
            series = by_threads[max_threads]
        
        series = sorted(series, key=lambda x: x['n_estimators'])
        
        if len(series) < 2:
            return None
        
        analysis = {
            'n_estimators': [r['n_estimators'] for r in series],
            'time': [r['avg_time'] for r in series],
            'std_time': [r['std_time'] for r in series],
            'f1_scores': [r['avg_f1_score'] for r in series],
            'std_f1': [r['std_f1_score'] for r in series],
            'cpu_usage': [r['avg_cpu_usage'] for r in series]
        }
        
        # Calculate time per estimator
        analysis['time_per_estimator'] = [
            t / n for t, n in zip(analysis['time'], analysis['n_estimators'])
        ]
        
        return analysis
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        thread_analysis = self.analyze_thread_scaling()
        
        if thread_analysis is None:
            return {
                'cpu_name': self.cpu_name,
                'logical_cores': self.logical_cores,
                'max_speedup': None,
                'max_efficiency': None,
                'serial_fraction': None
            }
        
        return {
            'cpu_name': self.cpu_name,
            'logical_cores': self.logical_cores,
            'max_speedup': max(thread_analysis['speedup']),
            'max_efficiency': max(thread_analysis['efficiency']),
            'efficiency_at_max_cores': thread_analysis['efficiency'][-1],
            'serial_fraction': thread_analysis['amdahl_s']
        }


class ScalabilityVisualizer:
    def __init__(self, analyzer: ScalabilityAnalyzer, output_dir: Path):
        self.analyzer = analyzer
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_speedup(self, thread_analysis: Dict):
        """Plot speedup vs number of threads"""
        if thread_analysis is None:
            print("  ⚠ No thread scaling data available")
            return
        
        threads = thread_analysis['threads']
        speedup = thread_analysis['speedup']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Actual speedup
        ax.plot(threads, speedup, 'o-', linewidth=2, markersize=8, 
                label='Actual Speedup', color='#2E86AB')
        
        # Ideal speedup
        ax.plot(threads, threads, '--', linewidth=2, 
                label='Ideal (Linear) Speedup', color='#A23B72', alpha=0.7)
        
        # Amdahl's law prediction
        if thread_analysis['amdahl_s'] is not None:
            s = thread_analysis['amdahl_s']
            amdahl_speedup = [1.0 / (s + (1-s)/p) for p in threads]
            ax.plot(threads, amdahl_speedup, ':', linewidth=2,
                    label=f'Amdahl\'s Law (s={s:.3f})', color='#F18F01', alpha=0.8)
        
        ax.set_xlabel('Number of Threads', fontweight='bold', fontsize=12)
        ax.set_ylabel('Speedup S(p) = T(1) / T(p)', fontweight='bold', fontsize=12)
        ax.set_title(f'Speedup vs Number of Threads\n{self.analyzer.cpu_name}', 
                     fontweight='bold', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)
        
        # Add annotations for key points
        for i, (p, s) in enumerate(zip(threads, speedup)):
            if i == 0 or i == len(threads) - 1 or p in [4, 8, 16]:
                ax.annotate(f'{s:.2f}', 
                           xy=(p, s), xytext=(5, 5),
                           textcoords='offset points', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        filename = self.output_dir / 'speedup_vs_threads.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'  ✓ {filename}')
        plt.close()
    
    def plot_efficiency(self, thread_analysis: Dict):
        """Plot efficiency vs number of threads"""
        if thread_analysis is None:
            print("  ⚠ No thread scaling data available")
            return
        
        threads = thread_analysis['threads']
        efficiency = thread_analysis['efficiency']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Actual efficiency
        ax.plot(threads, efficiency, 'o-', linewidth=2, markersize=8,
                label='Actual Efficiency', color='#06A77D')
        
        # Ideal efficiency (100%)
        ax.axhline(y=1.0, linestyle='--', linewidth=2, 
                   label='Ideal Efficiency', color='#A23B72', alpha=0.7)
        
        ax.set_xlabel('Number of Threads', fontweight='bold', fontsize=12)
        ax.set_ylabel('Efficiency E(p) = S(p) / p', fontweight='bold', fontsize=12)
        ax.set_title(f'Parallel Efficiency vs Number of Threads\n{self.analyzer.cpu_name}', 
                     fontweight='bold', fontsize=14, pad=15)
        ax.set_ylim([0, min(1.2, max(efficiency) * 1.1)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)
        
        # Add percentage labels
        for i, (p, e) in enumerate(zip(threads, efficiency)):
            if i == 0 or i == len(threads) - 1 or p in [4, 8, 16]:
                ax.annotate(f'{e*100:.1f}%', 
                           xy=(p, e), xytext=(5, -15),
                           textcoords='offset points', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.4))
        
        plt.tight_layout()
        filename = self.output_dir / 'efficiency_vs_threads.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'  ✓ {filename}')
        plt.close()
    
    def plot_time_vs_estimators(self, estimator_analysis: Dict):
        """Plot training time vs number of estimators"""
        if estimator_analysis is None:
            print("  ⚠ No estimator scaling data available")
            return
        
        n_est = estimator_analysis['n_estimators']
        times = estimator_analysis['time']
        std_times = estimator_analysis['std_time']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.errorbar(n_est, times, yerr=std_times, fmt='o-', linewidth=2, 
                    markersize=8, capsize=5, capthick=2,
                    label='Training Time', color='#D62828')
        
        # Fit linear trend
        if len(n_est) >= 2:
            coeffs = np.polyfit(n_est, times, 1)
            poly = np.poly1d(coeffs)
            ax.plot(n_est, poly(n_est), '--', linewidth=2, 
                    label=f'Linear Fit (slope={coeffs[0]:.3f}s/tree)', 
                    color='#F77F00', alpha=0.7)
        
        ax.set_xlabel('Number of Estimators (Trees)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=12)
        ax.set_title(f'Training Time vs Number of Estimators\n{self.analyzer.cpu_name}', 
                     fontweight='bold', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=11)
        
        # Add time labels
        for x, y, std in zip(n_est, times, std_times):
            ax.annotate(f'{y:.2f}s\n±{std:.2f}', 
                       xy=(x, y), xytext=(0, 10),
                       textcoords='offset points', fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        filename = self.output_dir / 'time_vs_estimators.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'  ✓ {filename}')
        plt.close()
    
    def plot_f1_vs_estimators(self, estimator_analysis: Dict):
        """Plot F1-score vs number of estimators"""
        if estimator_analysis is None:
            print("  ⚠ No estimator scaling data available")
            return
        
        n_est = estimator_analysis['n_estimators']
        f1_scores = [f * 100 for f in estimator_analysis['f1_scores']]
        std_f1 = [s * 100 for s in estimator_analysis['std_f1']]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.errorbar(n_est, f1_scores, yerr=std_f1, fmt='o-', linewidth=2,
                    markersize=8, capsize=5, capthick=2,
                    label='F1-Score', color='#06A77D')
        
        ax.set_xlabel('Number of Estimators (Trees)', fontweight='bold', fontsize=12)
        ax.set_ylabel('F1-Score (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'Model Quality vs Number of Estimators\n{self.analyzer.cpu_name}', 
                     fontweight='bold', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=11)
        
        # Add value labels
        for x, y, std in zip(n_est, f1_scores, std_f1):
            ax.annotate(f'{y:.2f}%\n±{std:.2f}', 
                       xy=(x, y), xytext=(0, -20),
                       textcoords='offset points', fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.4))
        
        plt.tight_layout()
        filename = self.output_dir / 'f1_vs_estimators.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'  ✓ {filename}')
        plt.close()
    
    def plot_combined_speedup_efficiency(self, thread_analysis: Dict):
        """Combined plot of speedup and efficiency"""
        if thread_analysis is None:
            print("  ⚠ No thread scaling data available")
            return
        
        threads = thread_analysis['threads']
        speedup = thread_analysis['speedup']
        efficiency = thread_analysis['efficiency']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Speedup plot
        ax1.plot(threads, speedup, 'o-', linewidth=2, markersize=8, 
                label='Actual', color='#2E86AB')
        ax1.plot(threads, threads, '--', linewidth=2, 
                label='Ideal', color='#A23B72', alpha=0.7)
        ax1.set_xlabel('Number of Threads', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Speedup S(p)', fontweight='bold', fontsize=12)
        ax1.set_title('Speedup', fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Efficiency plot
        ax2.plot(threads, efficiency, 'o-', linewidth=2, markersize=8,
                label='Actual', color='#06A77D')
        ax2.axhline(y=1.0, linestyle='--', linewidth=2, 
                   label='Ideal', color='#A23B72', alpha=0.7)
        ax2.set_xlabel('Number of Threads', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Efficiency E(p)', fontweight='bold', fontsize=12)
        ax2.set_title('Efficiency', fontweight='bold', fontsize=13)
        ax2.set_ylim([0, min(1.2, max(efficiency) * 1.1)])
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        fig.suptitle(f'Scalability Analysis: {self.analyzer.cpu_name}', 
                     fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        filename = self.output_dir / 'combined_scalability.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'  ✓ {filename}')
        plt.close()
    
    def plot_all(self):
        """Generate all scalability plots"""
        print(f"\nGenerating scalability analysis plots...")
        
        thread_analysis = self.analyzer.analyze_thread_scaling()
        estimator_analysis = self.analyzer.analyze_estimator_scaling()
        
        # Thread scaling plots
        if thread_analysis:
            self.plot_speedup(thread_analysis)
            self.plot_efficiency(thread_analysis)
            self.plot_combined_speedup_efficiency(thread_analysis)
        
        # Estimator scaling plots
        if estimator_analysis:
            self.plot_time_vs_estimators(estimator_analysis)
            self.plot_f1_vs_estimators(estimator_analysis)
        
        print(f"\n✓ All plots saved to {self.output_dir}/")


def generate_text_report(analyzer: ScalabilityAnalyzer, output_dir: Path):
    """Generate text summary report"""
    stats = analyzer.get_summary_stats()
    thread_analysis = analyzer.analyze_thread_scaling()
    estimator_analysis = analyzer.analyze_estimator_scaling()
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SCALABILITY ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"CPU: {stats['cpu_name']}")
    report_lines.append(f"Logical Cores: {stats['logical_cores']}")
    report_lines.append("")
    
    if thread_analysis:
        report_lines.append("-" * 70)
        report_lines.append("THREAD SCALING ANALYSIS")
        report_lines.append("-" * 70)
        report_lines.append(f"Sequential time T(1): {thread_analysis['t1']:.3f} seconds")
        report_lines.append(f"Maximum speedup: {stats['max_speedup']:.2f}x")
        report_lines.append(f"Maximum efficiency: {stats['max_efficiency']*100:.1f}%")
        report_lines.append(f"Efficiency at max cores: {stats['efficiency_at_max_cores']*100:.1f}%")
        if stats['serial_fraction'] is not None:
            report_lines.append(f"Estimated serial fraction (Amdahl): {stats['serial_fraction']:.3f}")
        report_lines.append("")
        
        report_lines.append("Detailed Results:")
        report_lines.append(f"{'Threads':>10} {'Time(s)':>12} {'Speedup':>10} {'Efficiency':>12} {'F1-Score':>12}")
        report_lines.append("-" * 70)
        for i in range(len(thread_analysis['threads'])):
            p = thread_analysis['threads'][i]
            t = thread_analysis['time'][i]
            s = thread_analysis['speedup'][i]
            e = thread_analysis['efficiency'][i]
            f1 = thread_analysis['f1_scores'][i] * 100
            report_lines.append(f"{p:>10} {t:>12.3f} {s:>10.2f} {e*100:>11.1f}% {f1:>11.2f}%")
        report_lines.append("")
    
    if estimator_analysis:
        report_lines.append("-" * 70)
        report_lines.append("ESTIMATOR SCALING ANALYSIS")
        report_lines.append("-" * 70)
        avg_time_per_tree = np.mean(estimator_analysis['time_per_estimator'])
        report_lines.append(f"Average time per tree: {avg_time_per_tree:.4f} seconds")
        report_lines.append("")
        
        report_lines.append("Detailed Results:")
        report_lines.append(f"{'Estimators':>12} {'Time(s)':>12} {'Time/Tree(s)':>15} {'F1-Score':>12}")
        report_lines.append("-" * 70)
        for i in range(len(estimator_analysis['n_estimators'])):
            n = estimator_analysis['n_estimators'][i]
            t = estimator_analysis['time'][i]
            tpt = estimator_analysis['time_per_estimator'][i]
            f1 = estimator_analysis['f1_scores'][i] * 100
            report_lines.append(f"{n:>12} {t:>12.3f} {tpt:>15.4f} {f1:>11.2f}%")
        report_lines.append("")
    
    report_lines.append("=" * 70)
    
    # Print to console
    print("\n" + "\n".join(report_lines))
    
    # Save to file
    report_file = output_dir / 'scalability_report.txt'
    with open(report_file, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"\n✓ Text report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize Random Forest scalability'
    )
    parser.add_argument('report', help='JSON report file from benchmark')
    parser.add_argument('--output-dir', '-o', 
                       help='Output directory for plots (default: scalability_CPUNAME)')
    
    args = parser.parse_args()
    
    # Load report
    print(f"Loading report: {args.report}")
    with open(args.report, 'r') as f:
        report_data = json.load(f)
    
    # Create analyzer
    analyzer = ScalabilityAnalyzer(report_data)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        cpu_short = report_data['system_info']['cpu_model'].replace(' ', '_')[:50]
        output_dir = Path(f'scalability_{cpu_short}')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    visualizer = ScalabilityVisualizer(analyzer, output_dir)
    visualizer.plot_all()
    
    # Generate text report
    generate_text_report(analyzer, output_dir)
    
    print("\n✓ Scalability analysis complete!")


if __name__ == '__main__':
    main()
