#!/usr/bin/env python3
"""
Compare Random Forest benchmark results from two different processors.
Usage: python compare_processors.py <report1.json> <report2.json> [output_dir]
"""

import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

def load_report(filename):
    """Load JSON report"""
    with open(filename, 'r') as f:
        return json.load(f)

def get_processor_name(system_info):
    """Extract short processor name"""
    cpu = system_info['cpu_model']
    # Try to extract meaningful name
    if 'Intel' in cpu:
        parts = cpu.split()
        for i, part in enumerate(parts):
            if part.startswith('i') and any(c.isdigit() for c in part):
                return ' '.join(parts[i:i+2])
    elif 'AMD' in cpu:
        parts = cpu.split()
        for i, part in enumerate(parts):
            if 'Ryzen' in part:
                return ' '.join(parts[i:i+3])
    return cpu[:30] + '...' if len(cpu) > 30 else cpu

def plot_comparison(report1, report2, output_dir=None):
    """Create comprehensive comparison visualization"""
    
    # Extract data from both reports
    sys1 = report1['system_info']
    sys2 = report2['system_info']
    results1 = report1['results']
    results2 = report2['results']
    
    cpu1_name = get_processor_name(sys1)
    cpu2_name = get_processor_name(sys2)
    
    # Extract metrics
    estimators = [r['n_estimators'] for r in results1]
    
    times1 = [r['avg_time'] for r in results1]
    times2 = [r['avg_time'] for r in results2]
    
    f1_1 = [r['avg_f1_score'] * 100 for r in results1]
    f1_2 = [r['avg_f1_score'] * 100 for r in results2]
    
    cpu1 = [r['avg_cpu_usage'] for r in results1]
    cpu2 = [r['avg_cpu_usage'] for r in results2]
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'Random Forest Benchmark Comparison\n{cpu1_name} vs {cpu2_name}', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # 1. Training Time Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(estimators))
    width = 0.35
    bars1 = ax1.bar(x - width/2, times1, width, label=cpu1_name, alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, times2, width, label=cpu2_name, alpha=0.8, color='coral')
    
    ax1.set_xlabel('Number of Estimators', fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax1.set_title('Training Time Comparison', fontweight='bold', pad=10, fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(estimators)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s',
                    ha='center', va='bottom', fontsize=8)
    
    # 2. Speedup Factor
    ax2 = fig.add_subplot(gs[0, 2])
    speedup = [t2/t1 if t1 > 0 else 0 for t1, t2 in zip(times1, times2)]
    colors = ['green' if s > 1 else 'red' for s in speedup]
    bars_speedup = ax2.bar(range(len(estimators)), speedup, alpha=0.7, color=colors)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Equal Performance')
    ax2.set_xlabel('Estimators', fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontweight='bold')
    ax2.set_title(f'{cpu1_name} Speedup', fontweight='bold', pad=10, fontsize=12)
    ax2.set_xticks(range(len(estimators)))
    ax2.set_xticklabels(estimators)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars_speedup, speedup)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va='bottom' if val > 1 else 'top', fontsize=9, fontweight='bold')
    
    # 3. F1-Score Comparison
    ax3 = fig.add_subplot(gs[1, :2])
    bars3a = ax3.bar(x - width/2, f1_1, width, label=cpu1_name, alpha=0.8, color='green')
    bars3b = ax3.bar(x + width/2, f1_2, width, label=cpu2_name, alpha=0.8, color='lightgreen')
    
    ax3.set_xlabel('Number of Estimators', fontweight='bold')
    ax3.set_ylabel('F1-Score (%)', fontweight='bold')
    ax3.set_title('F1-Score Comparison', fontweight='bold', pad=10, fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(estimators)
    ax3.set_ylim([max(0, min(min(f1_1), min(f1_2)) - 2), 100])
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. CPU Usage Comparison
    ax4 = fig.add_subplot(gs[1, 2])
    bars4a = ax4.bar(x - width/2, cpu1, width, label=cpu1_name, alpha=0.8, color='orange')
    bars4b = ax4.bar(x + width/2, cpu2, width, label=cpu2_name, alpha=0.8, color='yellow')
    
    ax4.set_xlabel('Estimators', fontweight='bold')
    ax4.set_ylabel('CPU Usage (%)', fontweight='bold')
    ax4.set_title('Average CPU Usage', fontweight='bold', pad=10, fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(estimators)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Efficiency Comparison
    ax5 = fig.add_subplot(gs[2, 0])
    eff1 = [f/t if t > 0 else 0 for f, t in zip(f1_1, times1)]
    eff2 = [f/t if t > 0 else 0 for f, t in zip(f1_2, times2)]
    
    bars5a = ax5.bar(x - width/2, eff1, width, label=cpu1_name, alpha=0.8, color='purple')
    bars5b = ax5.bar(x + width/2, eff2, width, label=cpu2_name, alpha=0.8, color='pink')
    
    ax5.set_xlabel('Estimators', fontweight='bold')
    ax5.set_ylabel('Efficiency (F1% / sec)', fontweight='bold')
    ax5.set_title('Training Efficiency', fontweight='bold', pad=10, fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(estimators)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Time Reduction %
    ax6 = fig.add_subplot(gs[2, 1])
    time_reduction = [(t2-t1)/t2*100 if t2 > 0 else 0 for t1, t2 in zip(times1, times2)]
    colors = ['green' if tr > 0 else 'red' for tr in time_reduction]
    bars6 = ax6.bar(range(len(estimators)), time_reduction, alpha=0.7, color=colors)
    
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_xlabel('Estimators', fontweight='bold')
    ax6.set_ylabel('Time Reduction (%)', fontweight='bold')
    ax6.set_title(f'{cpu1_name} Time Savings', fontweight='bold', pad=10, fontsize=12)
    ax6.set_xticks(range(len(estimators)))
    ax6.set_xticklabels(estimators)
    ax6.grid(True, alpha=0.3)
    
    for bar, val in zip(bars6, time_reduction):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=9)
    
    # 7. Per-Core CPU Comparison
    ax7 = fig.add_subplot(gs[2, 2])
    core_avg1 = []
    core_avg2 = []
    
    for r in results1:
        cores = r['cpu_metrics']['per_core_usage']
        if cores:
            core_avg1.append(np.mean(cores))
    
    for r in results2:
        cores = r['cpu_metrics']['per_core_usage']
        if cores:
            core_avg2.append(np.mean(cores))
    
    if core_avg1 and core_avg2:
        ax7.plot(estimators, core_avg1, marker='o', label=cpu1_name, 
                linewidth=2, markersize=8, color='steelblue')
        ax7.plot(estimators, core_avg2, marker='s', label=cpu2_name, 
                linewidth=2, markersize=8, color='coral')
        ax7.set_xlabel('Estimators', fontweight='bold')
        ax7.set_ylabel('Avg Core Usage (%)', fontweight='bold')
        ax7.set_title('Average Per-Core Utilization', fontweight='bold', pad=10, fontsize=12)
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)
    
    # 8. System Comparison Table
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('tight')
    ax8.axis('off')
    
    comparison_data = [
        ['Metric', cpu1_name, cpu2_name, 'Difference'],
        ['', '', '', ''],
        ['CPU Model', sys1['cpu_model'][:40], sys2['cpu_model'][:40], ''],
        ['Logical Cores', str(sys1['logical_cores']), str(sys2['logical_cores']), 
         f"{sys1['logical_cores'] - sys2['logical_cores']:+d}"],
        ['Physical Cores', str(sys1['physical_cores']), str(sys2['physical_cores']), 
         f"{sys1['physical_cores'] - sys2['physical_cores']:+d}"],
        ['Memory (GB)', f"{sys1['total_memory']/(1024**3):.1f}", 
         f"{sys2['total_memory']/(1024**3):.1f}", ''],
        ['', '', '', ''],
        ['Avg Time (s)', f"{np.mean(times1):.3f}", f"{np.mean(times2):.3f}", 
         f"{(np.mean(times1) - np.mean(times2)):.3f}"],
        ['Avg F1-Score (%)', f"{np.mean(f1_1):.2f}", f"{np.mean(f1_2):.2f}", 
         f"{(np.mean(f1_1) - np.mean(f1_2)):+.2f}"],
        ['Avg CPU Usage (%)', f"{np.mean(cpu1):.1f}", f"{np.mean(cpu2):.1f}", 
         f"{(np.mean(cpu1) - np.mean(cpu2)):+.1f}"],
        ['Avg Speedup', f"1.00x", f"{np.mean(speedup):.2f}x", 
         f"{(np.mean(speedup) - 1)*100:+.1f}%"],
    ]
    
    table = ax8.table(cellText=comparison_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style table
    for i in range(len(comparison_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            elif i in [1, 6]:  # Separator rows
                cell.set_facecolor('#E8E8E8')
            elif j == 0:  # First column
                cell.set_facecolor('#F0F0F0')
                cell.set_text_props(weight='bold')
    
    ax8.set_title('Detailed Comparison', fontweight='bold', pad=15, loc='left', fontsize=14)
    
    # Save figure
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path('.')
    
    output_file = output_path / f"benchmark_comparison_{cpu1_name}_vs_{cpu2_name}.png".replace(' ', '_')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison saved to: {output_file}")
    
    plt.show()

def print_comparison_summary(report1, report2):
    """Print text summary of comparison"""
    sys1 = report1['system_info']
    sys2 = report2['system_info']
    results1 = report1['results']
    results2 = report2['results']
    
    cpu1 = get_processor_name(sys1)
    cpu2 = get_processor_name(sys2)
    
    print("\n" + "="*100)
    print(f"PROCESSOR COMPARISON: {cpu1} vs {cpu2}")
    print("="*100)
    
    print(f"\n{cpu1}:")
    print(f"  Model: {sys1['cpu_model']}")
    print(f"  Cores: {sys1['logical_cores']} logical, {sys1['physical_cores']} physical")
    
    print(f"\n{cpu2}:")
    print(f"  Model: {sys2['cpu_model']}")
    print(f"  Cores: {sys2['logical_cores']} logical, {sys2['physical_cores']} physical")
    
    print(f"\nPerformance Comparison:")
    print(f"{'Config':<12} {cpu1 + ' Time':<20} {cpu2 + ' Time':<20} {'Speedup':<12}")
    print("-" * 100)
    
    for r1, r2 in zip(results1, results2):
        speedup = r2['avg_time'] / r1['avg_time'] if r1['avg_time'] > 0 else 0
        print(f"{r1['n_estimators']} trees"
              f"{'':<4} {r1['avg_time']:<20.3f} {r2['avg_time']:<20.3f} {speedup:.2f}x")
    
    # Calculate average speedup
    avg_speedup = np.mean([r2['avg_time'] / r1['avg_time'] if r1['avg_time'] > 0 else 0 
                          for r1, r2 in zip(results1, results2)])
    
    print(f"\n{'='*100}")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    if avg_speedup > 1:
        print(f"✅ {cpu1} is {avg_speedup:.2f}x FASTER on average")
        print(f"   Time savings: {(1 - 1/avg_speedup)*100:.1f}%")
    elif avg_speedup < 1:
        print(f"❌ {cpu1} is {1/avg_speedup:.2f}x SLOWER on average")
    else:
        print(f"⚖️  Both processors perform equally")
    print("="*100)

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_processors.py <report1.json> <report2.json> [output_dir]")
        sys.exit(1)
    
    report1_file = sys.argv[1]
    report2_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"Loading reports:")
    print(f"  Report 1: {report1_file}")
    print(f"  Report 2: {report2_file}")
    
    report1 = load_report(report1_file)
    report2 = load_report(report2_file)
    
    print_comparison_summary(report1, report2)
    plot_comparison(report1, report2, output_dir)

if __name__ == '__main__':
    main()
