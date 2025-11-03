#!/usr/bin/env python3
"""
Compare Random Forest benchmark results from two different processors.
Generates separate high-resolution graphics files.
Usage: python compare_processors.py <report1.json> <report2.json>
"""

import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def load_report(filename):
    """Load JSON report"""
    with open(filename, 'r') as f:
        return json.load(f)

def get_processor_name(system_info):
    """Extract short processor name"""
    cpu = system_info['cpu_model']
    # Shorten for display
    cpu = cpu.replace(' with Radeon Graphics', '')
    cpu = cpu.replace('(R)', '').replace('(TM)', '')
    if len(cpu) > 40:
        return cpu[:37] + '...'
    return cpu

def get_short_name(system_info):
    """Get very short name for filenames"""
    cpu = system_info['cpu_model']
    cpu = cpu.replace('AMD ', '').replace('Intel ', '')
    cpu = cpu.replace('(R)', '').replace('(TM)', '')
    cpu = cpu.replace(' with Radeon Graphics', '')
    cpu = cpu.replace(' CPU', '').replace(' Processor', '')
    cpu = cpu.replace(' @ ', '_')
    cpu = cpu.replace(' ', '_')
    return cpu[:30]

def plot_training_time(results1, results2, cpu1_name, cpu2_name, output_dir):
    """Plot training time comparison"""
    estimators = [r['n_estimators'] for r in results1]
    times1 = [r['avg_time'] for r in results1]
    times2 = [r['avg_time'] for r in results2]
    std1 = [r['std_time'] for r in results1]
    std2 = [r['std_time'] for r in results2]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(estimators))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, times1, width, yerr=std1, capsize=5,
                   label=cpu1_name, alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, times2, width, yerr=std2, capsize=5,
                   label=cpu2_name, alpha=0.8, color='coral')
    
    ax.set_xlabel('Number of Estimators', fontweight='bold', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=12)
    ax.set_title(f'Training Time Comparison\n{cpu1_name} vs {cpu2_name}', 
                 fontweight='bold', pad=15, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = output_dir / '1_training_time_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename.name}')
    plt.close()

def plot_speedup(results1, results2, cpu1_name, cpu2_name, output_dir):
    """Plot speedup factor"""
    estimators = [r['n_estimators'] for r in results1]
    times1 = [r['avg_time'] for r in results1]
    times2 = [r['avg_time'] for r in results2]
    speedup = [t2/t1 if t1 > 0 else 0 for t1, t2 in zip(times1, times2)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['green' if s > 1 else 'red' if s < 1 else 'gray' for s in speedup]
    bars = ax.bar(range(len(estimators)), speedup, alpha=0.7, color=colors)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, 
               label='Equal Performance', zorder=0)
    ax.set_xlabel('Number of Estimators', fontweight='bold', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontweight='bold', fontsize=12)
    ax.set_title(f'{cpu1_name} Speedup Factor\n(Higher is better for CPU1)', 
                 fontweight='bold', pad=15, fontsize=14)
    ax.set_xticks(range(len(estimators)))
    ax.set_xticklabels(estimators)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, speedup)):
        height = bar.get_height()
        va = 'bottom' if val > 1 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va=va, fontsize=10, fontweight='bold')
    
    # Add average speedup
    avg_speedup = np.mean(speedup)
    ax.text(0.98, 0.98, f'Avg: {avg_speedup:.2f}x',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            ha='right', va='top')
    
    plt.tight_layout()
    filename = output_dir / '2_speedup_factor.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename.name}')
    plt.close()

def plot_f1_comparison(results1, results2, cpu1_name, cpu2_name, output_dir):
    """Plot F1-score comparison"""
    estimators = [r['n_estimators'] for r in results1]
    f1_1 = [r['avg_f1_score'] * 100 for r in results1]
    f1_2 = [r['avg_f1_score'] * 100 for r in results2]
    std1 = [r['std_f1_score'] * 100 for r in results1]
    std2 = [r['std_f1_score'] * 100 for r in results2]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(estimators))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, f1_1, width, yerr=std1, capsize=5,
                   label=cpu1_name, alpha=0.8, color='green')
    bars2 = ax.bar(x + width/2, f1_2, width, yerr=std2, capsize=5,
                   label=cpu2_name, alpha=0.8, color='lightgreen')
    
    ax.set_xlabel('Number of Estimators', fontweight='bold', fontsize=12)
    ax.set_ylabel('F1-Score (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'F1-Score Comparison\n{cpu1_name} vs {cpu2_name}', 
                 fontweight='bold', pad=15, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators)
    ax.set_ylim([max(0, min(min(f1_1), min(f1_2)) - 5), 100])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = output_dir / '3_f1_score_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename.name}')
    plt.close()

def plot_cpu_usage(results1, results2, cpu1_name, cpu2_name, output_dir):
    """Plot CPU usage comparison"""
    estimators = [r['n_estimators'] for r in results1]
    cpu1 = [r['avg_cpu_usage'] for r in results1]
    cpu2 = [r['avg_cpu_usage'] for r in results2]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(estimators))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cpu1, width, label=cpu1_name, 
                   alpha=0.8, color='orange')
    bars2 = ax.bar(x + width/2, cpu2, width, label=cpu2_name, 
                   alpha=0.8, color='gold')
    
    ax.set_xlabel('Number of Estimators', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average CPU Usage (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'CPU Usage Comparison\n{cpu1_name} vs {cpu2_name}', 
                 fontweight='bold', pad=15, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = output_dir / '4_cpu_usage_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename.name}')
    plt.close()

def plot_efficiency(results1, results2, cpu1_name, cpu2_name, output_dir):
    """Plot training efficiency comparison"""
    estimators = [r['n_estimators'] for r in results1]
    f1_1 = [r['avg_f1_score'] * 100 for r in results1]
    f1_2 = [r['avg_f1_score'] * 100 for r in results2]
    times1 = [r['avg_time'] for r in results1]
    times2 = [r['avg_time'] for r in results2]
    
    eff1 = [f/t if t > 0 else 0 for f, t in zip(f1_1, times1)]
    eff2 = [f/t if t > 0 else 0 for f, t in zip(f1_2, times2)]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(estimators))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, eff1, width, label=cpu1_name, 
                   alpha=0.8, color='purple')
    bars2 = ax.bar(x + width/2, eff2, width, label=cpu2_name, 
                   alpha=0.8, color='orchid')
    
    ax.set_xlabel('Number of Estimators', fontweight='bold', fontsize=12)
    ax.set_ylabel('Efficiency (F1% / second)', fontweight='bold', fontsize=12)
    ax.set_title(f'Training Efficiency Comparison\n{cpu1_name} vs {cpu2_name}', 
                 fontweight='bold', pad=15, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = output_dir / '5_efficiency_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename.name}')
    plt.close()

def plot_time_savings(results1, results2, cpu1_name, cpu2_name, output_dir):
    """Plot time reduction percentage"""
    estimators = [r['n_estimators'] for r in results1]
    times1 = [r['avg_time'] for r in results1]
    times2 = [r['avg_time'] for r in results2]
    time_reduction = [(t2-t1)/t2*100 if t2 > 0 else 0 for t1, t2 in zip(times1, times2)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['green' if tr > 0 else 'red' for tr in time_reduction]
    bars = ax.bar(range(len(estimators)), time_reduction, alpha=0.7, color=colors)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Number of Estimators', fontweight='bold', fontsize=12)
    ax.set_ylabel('Time Reduction (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'{cpu1_name} Time Savings vs {cpu2_name}\n(Positive = CPU1 is faster)', 
                 fontweight='bold', pad=15, fontsize=14)
    ax.set_xticks(range(len(estimators)))
    ax.set_xticklabels(estimators)
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, time_reduction):
        height = bar.get_height()
        va = 'bottom' if val > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va=va, fontsize=10, fontweight='bold')
    
    # Add average
    avg_reduction = np.mean(time_reduction)
    ax.text(0.98, 0.98, f'Avg: {avg_reduction:.1f}%',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            ha='right', va='top')
    
    plt.tight_layout()
    filename = output_dir / '6_time_savings.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename.name}')
    plt.close()

def plot_comparison_table(report1, report2, cpu1_name, cpu2_name, output_dir):
    """Create detailed comparison table"""
    sys1 = report1['system_info']
    sys2 = report2['system_info']
    results1 = report1['results']
    results2 = report2['results']
    
    times1 = [r['avg_time'] for r in results1]
    times2 = [r['avg_time'] for r in results2]
    f1_1 = [r['avg_f1_score'] * 100 for r in results1]
    f1_2 = [r['avg_f1_score'] * 100 for r in results2]
    cpu1_usage = [r['avg_cpu_usage'] for r in results1]
    cpu2_usage = [r['avg_cpu_usage'] for r in results2]
    speedup = [t2/t1 if t1 > 0 else 0 for t1, t2 in zip(times1, times2)]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    comparison_data = [
        ['METRIC', cpu1_name, cpu2_name, 'DIFFERENCE'],
        ['', '', '', ''],
        ['SYSTEM INFORMATION', '', '', ''],
        ['CPU Model', sys1['cpu_model'][:35], sys2['cpu_model'][:35], ''],
        ['Logical Cores', str(sys1['logical_cores']), str(sys2['logical_cores']), 
         f"{sys1['logical_cores'] - sys2['logical_cores']:+d}"],
        ['Physical Cores', str(sys1['physical_cores']), str(sys2['physical_cores']), 
         f"{sys1['physical_cores'] - sys2['physical_cores']:+d}"],
        ['Memory (GB)', f"{sys1['total_memory']/(1024**3):.1f}", 
         f"{sys2['total_memory']/(1024**3):.1f}", ''],
        ['', '', '', ''],
        ['AVERAGE PERFORMANCE', '', '', ''],
        ['Training Time (s)', f"{np.mean(times1):.2f}", f"{np.mean(times2):.2f}", 
         f"{(np.mean(times1) - np.mean(times2)):.2f}s"],
        ['F1-Score (%)', f"{np.mean(f1_1):.2f}", f"{np.mean(f1_2):.2f}", 
         f"{(np.mean(f1_1) - np.mean(f1_2)):+.2f}%"],
        ['CPU Usage (%)', f"{np.mean(cpu1_usage):.1f}", f"{np.mean(cpu2_usage):.1f}", 
         f"{(np.mean(cpu1_usage) - np.mean(cpu2_usage)):+.1f}%"],
        ['', '', '', ''],
        ['SPEEDUP ANALYSIS', '', '', ''],
        ['Average Speedup', '1.00x', f"{np.mean(speedup):.2f}x", 
         f"{(np.mean(speedup) - 1)*100:+.1f}%"],
        ['Time Reduction', '0%', 
         f"{np.mean([(t2-t1)/t2*100 if t2 > 0 else 0 for t1, t2 in zip(times1, times2)]):.1f}%", 
         f"{np.mean([(t2-t1)/t2*100 if t2 > 0 else 0 for t1, t2 in zip(times1, times2)]):.1f}%"],
        ['Winner', '✓' if np.mean(speedup) > 1 else '', 
         '✓' if np.mean(speedup) < 1 else '', 
         'TIE' if abs(np.mean(speedup) - 1) < 0.05 else ''],
    ]
    
    table = ax.table(cellText=comparison_data, cellLoc='left', loc='center',
                     colWidths=[0.30, 0.25, 0.25, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3.0)
    
    # Style table
    for i in range(len(comparison_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Main header
                cell.set_facecolor('#2E75B6')
                cell.set_text_props(weight='bold', color='white')
            elif i in [1, 7, 12, 13]:  # Separator rows
                cell.set_facecolor('#E8E8E8')
            elif i in [2, 8, 13]:  # Section headers
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            elif j == 0:  # First column
                cell.set_facecolor('#F0F0F0')
                cell.set_text_props(weight='bold')
    
    plt.title('Detailed Processor Comparison', 
              fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    filename = output_dir / '0_comparison_summary.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename.name}')
    plt.close()

def print_text_summary(report1, report2):
    """Print text summary to console"""
    sys1 = report1['system_info']
    sys2 = report2['system_info']
    results1 = report1['results']
    results2 = report2['results']
    
    cpu1 = get_processor_name(sys1)
    cpu2 = get_processor_name(sys2)
    
    print("\n" + "="*100)
    print(f"PROCESSOR COMPARISON")
    print("="*100)
    
    print(f"\n[CPU 1] {cpu1}")
    print(f"  Cores: {sys1['logical_cores']} logical / {sys1['physical_cores']} physical")
    print(f"  Memory: {sys1['total_memory']/(1024**3):.1f} GB")
    
    print(f"\n[CPU 2] {cpu2}")
    print(f"  Cores: {sys2['logical_cores']} logical / {sys2['physical_cores']} physical")
    print(f"  Memory: {sys2['total_memory']/(1024**3):.1f} GB")
    
    print(f"\n{'Config':<15} {'CPU1 Time':<15} {'CPU2 Time':<15} {'Speedup':<12} {'F1-Score Diff':<15}")
    print("-" * 100)
    
    for r1, r2 in zip(results1, results2):
        speedup = r2['avg_time'] / r1['avg_time'] if r1['avg_time'] > 0 else 0
        f1_diff = (r1['avg_f1_score'] - r2['avg_f1_score']) * 100
        print(f"{r1['n_estimators']} trees     "
              f"{r1['avg_time']:<15.2f} {r2['avg_time']:<15.2f} "
              f"{speedup:<12.2f}x {f1_diff:+.2f}%")
    
    # Calculate averages
    times1 = [r['avg_time'] for r in results1]
    times2 = [r['avg_time'] for r in results2]
    avg_speedup = np.mean([t2/t1 if t1 > 0 else 0 for t1, t2 in zip(times1, times2)])
    avg_time_reduction = np.mean([(t2-t1)/t2*100 if t2 > 0 else 0 for t1, t2 in zip(times1, times2)])
    
    print("\n" + "="*100)
    print(f"SUMMARY:")
    print(f"  Average Speedup: {avg_speedup:.2f}x")
    print(f"  Time Reduction:  {avg_time_reduction:.1f}%")
    
    if avg_speedup > 1.05:
        print(f"\n  ✅ {cpu1} is significantly FASTER ({avg_speedup:.2f}x)")
    elif avg_speedup < 0.95:
        print(f"\n  ❌ {cpu1} is SLOWER ({1/avg_speedup:.2f}x)")
    else:
        print(f"\n  ⚖️  Both processors perform similarly")
    print("="*100)

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_processors.py <report1.json> <report2.json>")
        sys.exit(1)
    
    report1_file = sys.argv[1]
    report2_file = sys.argv[2]
    
    if not Path(report1_file).exists():
        print(f"Error: File '{report1_file}' not found")
        sys.exit(1)
    if not Path(report2_file).exists():
        print(f"Error: File '{report2_file}' not found")
        sys.exit(1)
    
    print(f"\n📊 Loading benchmark reports...")
    print(f"  Report 1: {report1_file}")
    print(f"  Report 2: {report2_file}")
    
    report1 = load_report(report1_file)
    report2 = load_report(report2_file)
    
    cpu1_name = get_processor_name(report1['system_info'])
    cpu2_name = get_processor_name(report2['system_info'])
    cpu1_short = get_short_name(report1['system_info'])
    cpu2_short = get_short_name(report2['system_info'])
    
    # Create output directory
    output_dir = Path(f'comparison_{cpu1_short}_vs_{cpu2_short}')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}/")
    print("\n🎨 Generating comparison graphics...")
    
    # Generate all comparison plots
    plot_comparison_table(report1, report2, cpu1_name, cpu2_name, output_dir)
    plot_training_time(report1['results'], report2['results'], cpu1_name, cpu2_name, output_dir)
    plot_speedup(report1['results'], report2['results'], cpu1_name, cpu2_name, output_dir)
    plot_f1_comparison(report1['results'], report2['results'], cpu1_name, cpu2_name, output_dir)
    plot_cpu_usage(report1['results'], report2['results'], cpu1_name, cpu2_name, output_dir)
    plot_efficiency(report1['results'], report2['results'], cpu1_name, cpu2_name, output_dir)
    plot_time_savings(report1['results'], report2['results'], cpu1_name, cpu2_name, output_dir)
    
    print(f"\n✅ All comparison graphics generated successfully!")
    print(f"\nGenerated files in {output_dir}/:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  • {f.name}")
    
    # Print text summary
    print_text_summary(report1, report2)

if __name__ == '__main__':
    main()
