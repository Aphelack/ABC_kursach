#!/usr/bin/env python3
"""
Visualize single Random Forest benchmark results from JSON report.
Generates separate high-resolution graphics files.
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
plt.rcParams['font.size'] = 10

def load_report(filename):
    """Load JSON report"""
    with open(filename, 'r') as f:
        return json.load(f)

def get_cpu_name_short(cpu_model):
    """Get shortened CPU name for filenames"""
    # Remove common words and shorten
    name = cpu_model.replace(' with Radeon Graphics', '')
    name = name.replace('AMD ', '').replace('Intel ', '')
    name = name.replace('(R)', '').replace('(TM)', '')
    name = name.replace(' ', '_')
    return name[:50]  # Limit length

def plot_training_time(results, cpu_name, output_dir):
    """Plot training time vs estimators"""
    estimators = [r['n_estimators'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    std_times = [r['std_time'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(estimators)), avg_times, 
                  yerr=std_times, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xlabel('Number of Estimators', fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax.set_title(f'Training Time vs Number of Estimators\n{cpu_name}', 
                 fontweight='bold', pad=10)
    ax.set_xticks(range(len(estimators)))
    ax.set_xticklabels(estimators)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val, std in zip(bars, avg_times, std_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = output_dir / '1_training_time.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename}')
    plt.close()

def plot_f1_score(results, cpu_name, output_dir):
    """Plot F1-score vs estimators"""
    estimators = [r['n_estimators'] for r in results]
    avg_f1 = [r['avg_f1_score'] * 100 for r in results]
    std_f1 = [r['std_f1_score'] * 100 for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(estimators)), avg_f1,
                  yerr=std_f1, capsize=5, alpha=0.7, color='green')
    ax.set_xlabel('Number of Estimators', fontweight='bold')
    ax.set_ylabel('F1-Score (%)', fontweight='bold')
    ax.set_title(f'F1-Score vs Number of Estimators\n{cpu_name}', 
                 fontweight='bold', pad=10)
    ax.set_xticks(range(len(estimators)))
    ax.set_xticklabels(estimators)
    ax.set_ylim([max(0, min(avg_f1) - 5), min(100, max(avg_f1) + 5)])
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val, std in zip(bars, avg_f1, std_f1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = output_dir / '2_f1_score.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename}')
    plt.close()

def plot_cpu_usage(results, cpu_name, output_dir):
    """Plot CPU usage"""
    estimators = [r['n_estimators'] for r in results]
    avg_cpu = [r['avg_cpu_usage'] for r in results]
    max_cpu = [r['max_cpu_usage'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(estimators))
    width = 0.35
    bars1 = ax.bar(x - width/2, avg_cpu, width, label='Average', alpha=0.8, color='orange')
    bars2 = ax.bar(x + width/2, max_cpu, width, label='Maximum', alpha=0.8, color='red')
    ax.set_xlabel('Number of Estimators', fontweight='bold')
    ax.set_ylabel('CPU Usage (%)', fontweight='bold')
    ax.set_title(f'CPU Usage\n{cpu_name}', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = output_dir / '3_cpu_usage.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename}')
    plt.close()

def plot_cpu_timeline(results, cpu_name, output_dir):
    """Plot CPU usage over time"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for r in results:
        timeline = r['cpu_metrics']['timeline']
        if timeline:
            time_points = np.arange(len(timeline)) * 0.1  # 100ms sampling
            ax.plot(time_points, timeline, label=f'{r["n_estimators"]} trees', 
                   linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('CPU Usage (%)', fontweight='bold')
    ax.set_title(f'CPU Usage Over Time\n{cpu_name}', fontweight='bold', pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = output_dir / '4_cpu_timeline.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename}')
    plt.close()

def plot_per_core_heatmap(results, system_info, cpu_name, output_dir):
    """Plot per-core CPU usage heatmap with high resolution for many cores"""
    estimators = [r['n_estimators'] for r in results]
    n_cores = system_info['logical_cores']
    
    per_core_data = []
    for r in results:
        per_core_data.append(r['cpu_metrics']['per_core_usage'])
    
    if not per_core_data or len(per_core_data[0]) == 0:
        print('  ⚠ No per-core data available')
        return
    
    heatmap_data = np.array(per_core_data).T  # Transpose: cores x configs
    
    # Dynamic figure height based on number of cores
    # More cores = taller figure for better readability
    fig_height = max(8, min(30, n_cores * 0.25))
    
    fig, ax = plt.subplots(figsize=(12, fig_height))
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xlabel('Configuration (Number of Estimators)', fontweight='bold', fontsize=12)
    ax.set_ylabel('CPU Core', fontweight='bold', fontsize=12)
    ax.set_title(f'CPU Usage Distribution Across {n_cores} Cores\n{cpu_name}', 
                 fontweight='bold', pad=15, fontsize=14)
    
    ax.set_xticks(range(len(estimators)))
    ax.set_xticklabels(estimators)
    ax.set_yticks(range(len(per_core_data[0])))
    ax.set_yticklabels([f'Core {i}' for i in range(len(per_core_data[0]))])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('CPU Usage (%)', fontweight='bold', fontsize=11)
    
    # Add text annotations (only if not too many cores)
    if n_cores <= 32:
        for i in range(len(estimators)):
            for j in range(len(per_core_data[0])):
                text = ax.text(i, j, f'{heatmap_data[j, i]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    filename = output_dir / '5_per_core_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename}')
    plt.close()

def plot_efficiency(results, cpu_name, output_dir):
    """Plot training efficiency (F1/time)"""
    estimators = [r['n_estimators'] for r in results]
    avg_f1 = [r['avg_f1_score'] * 100 for r in results]
    avg_times = [r['avg_time'] for r in results]
    efficiency = [f1 / time if time > 0 else 0 for f1, time in zip(avg_f1, avg_times)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(estimators)), efficiency, alpha=0.7, color='purple')
    ax.set_xlabel('Number of Estimators', fontweight='bold')
    ax.set_ylabel('Efficiency (F1% / second)', fontweight='bold')
    ax.set_title(f'Training Efficiency\n{cpu_name}', fontweight='bold', pad=10)
    ax.set_xticks(range(len(estimators)))
    ax.set_xticklabels(estimators)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = output_dir / '6_efficiency.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename}')
    plt.close()

def plot_run_variations(results, cpu_name, output_dir):
    """Plot variation across runs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time variations
    for r in results:
        run_times = r['run_times']
        runs = list(range(1, len(run_times) + 1))
        ax1.plot(runs, run_times, marker='o', label=f'{r["n_estimators"]} trees', 
                linewidth=2, markersize=8)
    ax1.set_xlabel('Run Number', fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontweight='bold')
    ax1.set_title('Training Time Variation', fontweight='bold', pad=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 variations
    for r in results:
        run_f1 = [f * 100 for f in r['run_f1_scores']]
        runs = list(range(1, len(run_f1) + 1))
        ax2.plot(runs, run_f1, marker='s', label=f'{r["n_estimators"]} trees', 
                linewidth=2, markersize=8)
    ax2.set_xlabel('Run Number', fontweight='bold')
    ax2.set_ylabel('F1-Score (%)', fontweight='bold')
    ax2.set_title('F1-Score Variation', fontweight='bold', pad=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Variations Across Runs\n{cpu_name}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    filename = output_dir / '7_run_variations.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename}')
    plt.close()

def create_summary_table(report, output_dir):
    """Create summary table as image"""
    system_info = report['system_info']
    dataset = report['dataset']
    config = report['config']
    summary = report.get('summary', {})
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    info_data = [
        ['SYSTEM INFORMATION', ''],
        ['CPU Model', system_info['cpu_model']],
        ['Architecture', system_info['architecture']],
        ['Logical Cores', str(system_info['logical_cores'])],
        ['Physical Cores', str(system_info['physical_cores'])],
        ['Memory', f"{system_info['total_memory'] / (1024**3):.2f} GB"],
        ['OS', f"{system_info['os_name']} {system_info['os_version']}"],
        ['', ''],
        ['DATASET INFORMATION', ''],
        ['Train Samples', str(dataset['train_samples'])],
        ['Test Samples', str(dataset['test_samples'])],
        ['Features', str(dataset['n_features'])],
        ['Classes', str(dataset['n_classes'])],
        ['', ''],
        ['CONFIGURATION', ''],
        ['Max Depth', str(config['max_depth'])],
        ['Runs per Config', str(config['num_runs'])],
        ['Test Size', f"{config['test_size']*100}%"],
        ['', ''],
        ['BEST RESULTS', ''],
        ['Best F1-Score', f"{summary.get('best_f1_score', 0)*100:.2f}% ({summary.get('best_f1_score_estimators', 'N/A')} estimators)"],
        ['Fastest Time', f"{summary.get('fastest_time', 0):.2f}s ({summary.get('fastest_estimators', 'N/A')} estimators)"],
        ['Most Efficient', f"{summary.get('most_efficient_estimators', 'N/A')} estimators"],
    ]
    
    table = ax.table(cellText=info_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header rows
    for i in [0, 8, 14, 18]:
        table[(i, 0)].set_facecolor('#3498db')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_facecolor('#3498db')
    
    plt.title(f'Benchmark Summary\n{system_info["cpu_model"]}', 
              fontweight='bold', fontsize=14, pad=20)
    
    filename = output_dir / '0_summary.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'  ✓ {filename}')
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_single.py <report.json>")
        sys.exit(1)
    
    report_file = sys.argv[1]
    
    if not Path(report_file).exists():
        print(f"Error: File '{report_file}' not found")
        sys.exit(1)
    
    print(f"\n📊 Loading benchmark report: {report_file}")
    report = load_report(report_file)
    
    # Create output directory
    cpu_short = get_cpu_name_short(report['system_info']['cpu_model'])
    output_dir = Path(f'graphs_{cpu_short}')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}/")
    print("\n🎨 Generating graphics...")
    
    cpu_name = report['system_info']['cpu_model']
    results = report['results']
    system_info = report['system_info']
    
    # Generate all plots
    create_summary_table(report, output_dir)
    plot_training_time(results, cpu_name, output_dir)
    plot_f1_score(results, cpu_name, output_dir)
    plot_cpu_usage(results, cpu_name, output_dir)
    plot_cpu_timeline(results, cpu_name, output_dir)
    plot_per_core_heatmap(results, system_info, cpu_name, output_dir)
    plot_efficiency(results, cpu_name, output_dir)
    plot_run_variations(results, cpu_name, output_dir)
    
    print(f"\n✅ All graphics generated successfully in {output_dir}/")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  • {f.name}")

if __name__ == '__main__':
    main()
