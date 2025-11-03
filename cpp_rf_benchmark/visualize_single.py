#!/usr/bin/env python3
"""
Visualize single Random Forest benchmark results from JSON report.
Usage: python visualize_single.py <report.json>
"""

import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def load_report(filename):
    """Load JSON report"""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_benchmark_results(report, output_dir=None):
    """Create comprehensive visualization of benchmark results"""
    
    # Extract data
    system_info = report['system_info']
    results = report['results']
    dataset = report['dataset']
    config = report['config']
    
    estimators = [r['n_estimators'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    std_times = [r['std_time'] for r in results]
    avg_f1_scores = [r['avg_f1_score'] * 100 for r in results]
    std_f1_scores = [r['std_f1_score'] * 100 for r in results]
    avg_cpu = [r['avg_cpu_usage'] for r in results]
    max_cpu = [r['max_cpu_usage'] for r in results]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    
    # Title
    cpu_name = system_info['cpu_model']
    fig.suptitle(f'Random Forest Benchmark Results\n{cpu_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Create grid
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Training Time
    ax1 = fig.add_subplot(gs[0, :2])
    bars1 = ax1.bar(range(len(estimators)), avg_times, 
                    yerr=std_times, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Number of Estimators', fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax1.set_title('Training Time vs Number of Estimators', fontweight='bold', pad=10)
    ax1.set_xticks(range(len(estimators)))
    ax1.set_xticklabels(estimators)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, avg_times, std_times)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}s\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. CPU Usage
    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(estimators))
    width = 0.35
    bars2a = ax2.bar(x - width/2, avg_cpu, width, label='Average', alpha=0.8, color='orange')
    bars2b = ax2.bar(x + width/2, max_cpu, width, label='Maximum', alpha=0.8, color='red')
    ax2.set_xlabel('Estimators', fontweight='bold')
    ax2.set_ylabel('CPU Usage (%)', fontweight='bold')
    ax2.set_title('CPU Usage', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(estimators)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. F1-Score
    ax3 = fig.add_subplot(gs[1, :2])
    bars3 = ax3.bar(range(len(estimators)), avg_f1_scores,
                    yerr=std_f1_scores, capsize=5, alpha=0.7, color='green')
    ax3.set_xlabel('Number of Estimators', fontweight='bold')
    ax3.set_ylabel('F1-Score (%)', fontweight='bold')
    ax3.set_title('F1-Score vs Number of Estimators', fontweight='bold', pad=10)
    ax3.set_xticks(range(len(estimators)))
    ax3.set_xticklabels(estimators)
    ax3.set_ylim([max(0, min(avg_f1_scores) - 5), 100])
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val, std in zip(bars3, avg_f1_scores, std_f1_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # 4. Efficiency (F1-Score / Time)
    ax4 = fig.add_subplot(gs[1, 2])
    efficiency = [f1 / time if time > 0 else 0 
                  for f1, time in zip(avg_f1_scores, avg_times)]
    bars4 = ax4.bar(range(len(estimators)), efficiency, alpha=0.7, color='purple')
    ax4.set_xlabel('Estimators', fontweight='bold')
    ax4.set_ylabel('Efficiency (F1% / sec)', fontweight='bold')
    ax4.set_title('Training Efficiency', fontweight='bold', pad=10)
    ax4.set_xticks(range(len(estimators)))
    ax4.set_xticklabels(estimators)
    ax4.grid(True, alpha=0.3)
    
    # 5. Per-Core CPU Usage Heatmap
    ax5 = fig.add_subplot(gs[2, :])
    n_cores = system_info['logical_cores']
    per_core_data = []
    for r in results:
        per_core_data.append(r['cpu_metrics']['per_core_usage'])
    
    if per_core_data and len(per_core_data[0]) > 0:
        heatmap_data = np.array(per_core_data)
        im = ax5.imshow(heatmap_data.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        ax5.set_xlabel('Configuration (Estimators)', fontweight='bold')
        ax5.set_ylabel('CPU Core', fontweight='bold')
        ax5.set_title('CPU Usage Distribution Across Cores', fontweight='bold', pad=10)
        ax5.set_xticks(range(len(estimators)))
        ax5.set_xticklabels(estimators)
        ax5.set_yticks(range(min(n_cores, len(per_core_data[0]))))
        ax5.set_yticklabels(range(min(n_cores, len(per_core_data[0]))))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('CPU Usage (%)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(estimators)):
            for j in range(min(n_cores, len(per_core_data[0]))):
                text = ax5.text(i, j, f'{heatmap_data[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=7)
    
    # 6. Run Variations
    ax6 = fig.add_subplot(gs[3, 0])
    for i, r in enumerate(results):
        run_times = r['run_times']
        runs = list(range(1, len(run_times) + 1))
        ax6.plot(runs, run_times, marker='o', label=f'{r["n_estimators"]} trees', linewidth=2)
    ax6.set_xlabel('Run Number', fontweight='bold')
    ax6.set_ylabel('Time (seconds)', fontweight='bold')
    ax6.set_title('Training Time Variation Across Runs', fontweight='bold', pad=10)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. System Info Table
    ax7 = fig.add_subplot(gs[3, 1:])
    ax7.axis('tight')
    ax7.axis('off')
    
    info_data = [
        ['CPU Model', system_info['cpu_model']],
        ['Architecture', system_info['architecture']],
        ['Logical Cores', str(system_info['logical_cores'])],
        ['Physical Cores', str(system_info['physical_cores'])],
        ['Memory', f"{system_info['total_memory'] / (1024**3):.2f} GB"],
        ['OS', f"{system_info['os_name']} {system_info['os_version']}"],
        ['', ''],
        ['Train Samples', str(dataset['train_samples'])],
        ['Test Samples', str(dataset['test_samples'])],
        ['Features', str(dataset['n_features'])],
        ['Classes', str(dataset['n_classes'])],
        ['Max Depth', str(config['max_depth'])],
        ['Runs per Config', str(config['num_runs'])],
    ]
    
    table = ax7.table(cellText=info_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style table
    for i in range(len(info_data)):
        table[(i, 0)].set_facecolor('#E8E8E8')
        table[(i, 0)].set_text_props(weight='bold')
    
    ax7.set_title('System & Configuration Details', fontweight='bold', pad=10, loc='left')
    
    # Save figure
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path('.')
    
    output_file = output_path / f"benchmark_visualization_{system_info.get('cpu_model', 'unknown').replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")
    
    plt.show()

def print_summary(report):
    """Print text summary of results"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    system_info = report['system_info']
    results = report['results']
    summary = report.get('summary', {})
    
    print(f"\nCPU: {system_info['cpu_model']}")
    print(f"Cores: {system_info['logical_cores']} logical, {system_info['physical_cores']} physical")
    print(f"Memory: {system_info['total_memory'] / (1024**3):.2f} GB")
    
    print(f"\nResults:")
    print(f"{'Estimators':<12} {'Time (sec)':<15} {'F1-Score (%)':<15} {'CPU (%)':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['n_estimators']:<12} "
              f"{r['avg_time']:<15.3f} "
              f"{r['avg_f1_score']*100:<15.2f} "
              f"{r['avg_cpu_usage']:<12.1f}")
    
    if summary:
        print(f"\nOptimal Configurations:")
        print(f"  🎯 Best F1-Score: {summary['best_f1_score']*100:.2f}% "
              f"({summary['best_f1_score_estimators']} estimators)")
        print(f"  ⚡ Fastest: {summary['fastest_time']:.3f} sec "
              f"({summary['fastest_estimators']} estimators)")
        print(f"  ⭐ Most Efficient: {summary['most_efficient_estimators']} estimators")
    
    print("\n" + "="*80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_single.py <report.json> [output_dir]")
        sys.exit(1)
    
    report_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Loading report: {report_file}")
    report = load_report(report_file)
    
    print_summary(report)
    plot_benchmark_results(report, output_dir)

if __name__ == '__main__':
    main()
