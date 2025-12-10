# Random Forest Scalability Analysis - Quick Start

## 📊 Overview

Complete toolkit for analyzing Random Forest parallelization scalability. Automatically generates publication-quality graphs for your course work analyzing:
- **Speedup** S(p) = T(1) / T(p)
- **Efficiency** E(p) = S(p) / p  
- **Amdahl's Law** fitting
- Cross-processor comparison

## 🚀 Quick Start (3 commands)

```bash
# 1. Run all scalability experiments
./run_scalability.sh

# 2. Analyze results (generates all graphs)
python3 scalability_analysis.py scalability_results_*/exp1_thread_scaling.json

# 3. Compare two processors
python3 compare_scalability.py cpu1_report.json cpu2_report.json
```

## 📁 New Files Created

### Configuration Files
- `config_scalability_threads.json` - Thread scaling (1, 2, 4, 8, 16, 32 threads)
- `config_scalability_estimators.json` - Estimator scaling (25, 50, 100, 200, 400 trees)
- `config_scalability_dataset.json` - Dataset size scaling (10%, 25%, 50%, 75%, 100%)

### Analysis Scripts
- `scalability_analysis.py` - Main analysis tool, generates 5 graphs + text report
- `compare_scalability.py` - Multi-processor comparison, generates 4 comparison graphs
- `run_scalability.sh` - Automated experiment runner
- `example_analysis.py` - Interactive helper for common tasks

### Documentation
- `SCALABILITY_GUIDE_RU.md` - Complete guide in Russian (25+ pages)

## 📈 Generated Graphs

### Single Processor Analysis (`scalability_analysis.py`)

1. **speedup_vs_threads.png**
   - Actual speedup vs ideal (linear)
   - Amdahl's law prediction curve
   - Perfect for "Анализ масштабируемости" section

2. **efficiency_vs_threads.png**
   - Parallel efficiency E(p)
   - Shows degradation at high thread counts
   - Explains overhead and contention

3. **time_vs_estimators.png**
   - Training time vs number of trees
   - Linear fit with slope (time per tree)
   - Demonstrates work scalability

4. **f1_vs_estimators.png**
   - Model quality vs number of trees
   - Shows diminishing returns
   - Helps choose optimal n_estimators

5. **combined_scalability.png**
   - Side-by-side speedup and efficiency
   - Publication-ready overview

### Multi-Processor Comparison (`compare_scalability.py`)

6. **comparison_speedup.png**
   - Overlay multiple CPUs
   - Direct performance comparison

7. **comparison_efficiency.png**
   - Efficiency comparison across CPUs
   - Architecture insights

8. **comparison_time_vs_estimators.png**
   - Absolute time comparison
   - Shows raw performance differences

9. **comparison_combined.png**
   - 2x2 dashboard with summary table
   - Complete comparison overview

## 📊 Text Report

Each analysis generates `scalability_report.txt`:

```
======================================================================
SCALABILITY ANALYSIS REPORT
======================================================================

CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
Logical Cores: 80

----------------------------------------------------------------------
THREAD SCALING ANALYSIS
----------------------------------------------------------------------
Sequential time T(1): 45.234 seconds
Maximum speedup: 28.45x
Maximum efficiency: 89.2%
Efficiency at max cores: 35.6%
Estimated serial fraction (Amdahl): 0.035

Detailed Results:
   Threads       Time(s)    Speedup   Efficiency    F1-Score
----------------------------------------------------------------------
         1       45.234       1.00      100.0%      85.42%
         2       23.156       1.95       97.6%      85.38%
         4       11.892       3.80       95.1%      85.45%
         8        6.234      7.26       90.7%      85.41%
        16        3.421     13.22       82.6%      85.39%
        32        1.985     22.78       71.2%      85.43%
        80        1.590     28.45       35.6%      85.40%
```

## 🎯 Usage Examples

### Analyze existing report
```bash
python3 scalability_analysis.py rf_benchmark_Intel_Xeon_20251103.json
```

### Custom output directory
```bash
python3 scalability_analysis.py report.json --output-dir my_analysis
```

### Compare two processors
```bash
python3 compare_scalability.py \
    rf_benchmark_Xeon_E5-2697.json \
    rf_benchmark_Xeon_Gold_6230.json \
    --output-dir comparison_E5_vs_Gold
```

### Run specific experiment
```bash
./build/rf_benchmark ../dataset/creditcard.csv config_scalability_threads.json
```

### Interactive help
```bash
python3 example_analysis.py
```

## 📝 For Your Course Work

### Section: "Анализ масштабируемости"

Use these graphs and data:

1. **Speedup Analysis**
   - Include `speedup_vs_threads.png`
   - Discuss linear scaling region (typically up to 8-16 threads)
   - Explain saturation point

2. **Efficiency Analysis**  
   - Include `efficiency_vs_threads.png`
   - Discuss overhead and contention
   - Mention cache effects, NUMA

3. **Amdahl's Law**
   - Use estimated serial fraction from report
   - Compare theoretical vs actual speedup
   - Discuss parallel limitations

4. **Work Scalability**
   - Include `time_vs_estimators.png`
   - Show linear time growth
   - Mention constant quality (F1-score)

5. **Processor Comparison**
   - Include `comparison_*.png` graphs
   - Discuss architectural differences
   - Explain performance variations

### Sample Text Snippets

```
На рисунке X представлена зависимость ускорения от числа потоков. 
Как видно, до p=16 ускорение близко к линейному (S(16)=13.22, 
эффективность 82.6%). При дальнейшем увеличении числа потоков 
наблюдается отклонение от идеальной зависимости.

Используя закон Амдала, оценена последовательная доля s ≈ 0.035 (3.5%). 
Это ограничивает теоретическое максимальное ускорение величиной 
S_max ≈ 28.6x, что согласуется с экспериментом (S(80) = 28.45).
```

## 🔧 Requirements

Install Python dependencies:
```bash
pip3 install matplotlib seaborn numpy
# or
pip3 install -r requirements.txt
```

## 📖 Full Documentation

See `SCALABILITY_GUIDE_RU.md` for:
- Detailed experiment descriptions
- Troubleshooting guide
- Advanced experiments (NUMA, CPU frequency)
- Tips for course work writing

## 🎨 Graph Customization

All graphs use:
- High resolution (300 DPI)
- Professional color schemes
- Error bars (from multiple runs)
- Clear labels and legends
- Publication-ready formatting

Colors are chosen for:
- Screen readability
- Print quality
- Colorblind accessibility

## 💡 Tips

1. **Run with sufficient repetitions**: Set `num_runs: 5` in configs
2. **Minimize background load**: Close other applications
3. **Check CPU throttling**: Use `performance` governor
4. **Document environment**: Note CPU, OS, kernel version
5. **Save raw data**: Keep JSON reports for future analysis

## 🐛 Troubleshooting

**No graphs generated?**
```bash
pip3 install matplotlib seaborn numpy
```

**Few data points?**
- Check configs have 4+ different values
- Verify benchmark completed successfully

**High variance?**
- Increase `num_runs` to 7-10
- Check for background processes

**Memory error?**
- Reduce `n_estimators`
- Use smaller dataset

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Run all experiments | `./run_scalability.sh` |
| Analyze one CPU | `python3 scalability_analysis.py report.json` |
| Compare CPUs | `python3 compare_scalability.py r1.json r2.json` |
| Interactive help | `python3 example_analysis.py` |
| Custom experiment | `./build/rf_benchmark <dataset> <config>` |
| View example | See existing `rf_benchmark_*.json` files |

## 📚 What You Get

For your course work analysis section:
- ✅ 9 publication-quality graphs (300 DPI PNG)
- ✅ Detailed text reports with statistics
- ✅ Speedup and efficiency calculations
- ✅ Amdahl's law fitting
- ✅ Cross-processor comparisons
- ✅ Ready-to-use data tables
- ✅ Statistical analysis (mean, std dev)

All formatted and ready to insert into your document!

---

**Next Steps**: Run `./run_scalability.sh` to start experiments, then analyze results with the provided Python scripts. See `SCALABILITY_GUIDE_RU.md` for complete documentation.
