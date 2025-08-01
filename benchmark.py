import time
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from vllm import LLM, SamplingParams
import seaborn as sns

@dataclass
class SamplerConfig:
    name: str
    use_dynamic_temperature: bool = False
    initial_temperature: Optional[float] = None
    final_temperature: Optional[float] = None
    max_steps: Optional[int] = None
    use_xtc: bool = False
    xtc_exclude_top: Optional[int] = None
    xtc_exclusion_threshold: Optional[float] = None
    xtc_min_probability: Optional[float] = None
    use_dry: bool = False
    dry_multiplier: Optional[float] = None
    dry_base: Optional[float] = None
    dry_allowed_length: Optional[int] = None
    dry_sequence_breakers: Optional[List[int]] = None

class VLLMBenchmark:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            enable_chunked_prefill=True
        )
        
    def create_sampler_configs(self) -> List[SamplerConfig]:
        configs = [
            SamplerConfig(name="baseline"),
            
            SamplerConfig(
                name="dynamic_temp_light",
                use_dynamic_temperature=True,
                initial_temperature=1.0,
                final_temperature=0.4,
                max_steps=50,
            ),
            
            SamplerConfig(
                name="dynamic_temp_heavy",
                use_dynamic_temperature=True,
                initial_temperature=1.5,
                final_temperature=0.2
            ),
            
            SamplerConfig(
                name="xtc_light",
                use_xtc=True,
                xtc_exclude_top=5,
                xtc_exclusion_threshold=0.1,
                xtc_min_probability=0.01
            ),
            
            SamplerConfig(
                name="xtc_heavy",
                use_xtc=True,
                xtc_exclude_top=20,
                xtc_exclusion_threshold=0.25,
                xtc_min_probability=0.005
            ),
            
            SamplerConfig(
                name="dry_light",
                use_dry=True,
                dry_multiplier=0.8,
                dry_base=1.75,
                dry_allowed_length=2,
                dry_sequence_breakers=[198]
            ),
            
            SamplerConfig(
                name="dry_heavy",
                use_dry=True,
                dry_multiplier=0.5,
                dry_base=3.0,
                dry_allowed_length=5,
                dry_sequence_breakers=[11, 13, 46, 58, 198, 220]
            ),
            
            SamplerConfig(
                name="dynamic_xtc",
                use_dynamic_temperature=True,
                initial_temperature=1.2,
                final_temperature=0.3,
                max_steps=50,
                use_xtc=True,
                xtc_exclude_top=10,
                xtc_exclusion_threshold=0.15,
                xtc_min_probability=0.01
            ),
            
            SamplerConfig(
                name="xtc_dry",
                use_xtc=True,
                xtc_exclude_top=12,
                xtc_exclusion_threshold=0.18,
                xtc_min_probability=0.008,
                use_dry=True,
                dry_multiplier=0.7,
                dry_base=2.0,
                dry_allowed_length=3,
                dry_sequence_breakers=[11, 46, 198]
            ),
            
            SamplerConfig(
                name="all_samplers",
                use_dynamic_temperature=True,
                initial_temperature=1.1,
                final_temperature=0.35,
                max_steps=50,
                use_xtc=True,
                xtc_exclude_top=15,
                xtc_exclusion_threshold=0.2,
                xtc_min_probability=0.006,
                use_dry=True,
                dry_multiplier=0.6,
                dry_base=2.5,
                dry_allowed_length=4,
                dry_sequence_breakers=[11, 13, 46, 58, 198]
            )
        ]
        return configs
    
    def create_sampling_params(self, config: SamplerConfig) -> SamplingParams:
        params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100
        }
        
        if config.use_dynamic_temperature:
            params.update({
                "use_dynamic_temperature": True,
                "initial_temperature": config.initial_temperature,
                "final_temperature": config.final_temperature,
                "max_steps": config.max_steps,
            })
        
        if config.use_xtc:
            params.update({
                "use_xtc": True,
                "xtc_exclude_top": config.xtc_exclude_top,
                "xtc_exclusion_threshold": config.xtc_exclusion_threshold,
                "xtc_min_probability": config.xtc_min_probability
            })
        
        if config.use_dry:
            params.update({
                "use_dry": True,
                "dry_multiplier": config.dry_multiplier,
                "dry_base": config.dry_base,
                "dry_allowed_length": config.dry_allowed_length,
                "dry_sequence_breakers": config.dry_sequence_breakers
            })
        
        return SamplingParams(**params)
    
    def generate_conversations(self, num_conversations: int) -> List[List[Dict[str, str]]]:
        user_messages = [
            "Explain the concept of neural networks in machine learning and their real-world applications.",
            "Describe how photosynthesis works in plants and why it's crucial for life on Earth.",
            "What are the main causes of climate change and what can we do about it?",
            "How does quantum computing differ from classical computing? What are its advantages?",
            "Explain the basic principles of economics and how supply and demand work.",
            "What is the structure of DNA and why is it so important for living organisms?",
            "Describe the process of evolution by natural selection with examples.",
            "How do computer algorithms solve complex problems efficiently?",
            "What are the key features of democratic government systems?",
            "Explain how renewable energy sources work and their benefits.",
            "What is artificial intelligence and how is it transforming industries today?",
            "Describe the water cycle and explain its importance to life on our planet.",
            "How do vaccines work to prevent diseases? Are they safe?",
            "What are the fundamental forces in physics and how do they shape our universe?",
            "Explain how the internet works at a basic level and its impact on society."
        ]
        
        conversations = []
        for i in range(num_conversations):
            message = user_messages[i % len(user_messages)]
            conversation = [{"role": "user", "content": message}]
            conversations.append(conversation)
        return conversations
    
    def benchmark_batch_size(self, config: SamplerConfig, batch_size: int, 
                           num_batches: int = 5) -> Dict:
        sampling_params = self.create_sampling_params(config)
        
        batch_results = []
        total_requests = 0
        total_tokens = 0
        
        for batch_idx in range(num_batches):
            conversations = self.generate_conversations(batch_size)
            
            start_time = time.time()
            try:
                outputs = self.llm.chat(conversations, sampling_params=sampling_params)
                end_time = time.time()
                
                batch_duration = end_time - start_time
                batch_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
                
                batch_results.append({
                    "duration": batch_duration,
                    "requests": len(conversations),
                    "tokens": batch_tokens,
                    "success": True
                })
                
                total_requests += len(conversations)
                total_tokens += batch_tokens
                
            except Exception as e:
                print(f"Batch {batch_idx} failed: {str(e)}")
                batch_results.append({
                    "duration": 0,
                    "requests": 0,
                    "tokens": 0,
                    "success": False
                })
        
        successful_batches = [b for b in batch_results if b["success"]]
        
        if successful_batches:
            durations = [b["duration"] for b in successful_batches]
            avg_batch_duration = statistics.mean(durations)
            total_duration = sum(durations)
            
            throughput_rps = total_requests / total_duration
            tokens_per_second = total_tokens / total_duration
            avg_latency_per_request = avg_batch_duration / batch_size
            
            p95_duration = np.percentile(durations, 95)
            p99_duration = np.percentile(durations, 99)
            
            efficiency_score = throughput_rps / batch_size if batch_size > 0 else 0
            
        else:
            throughput_rps = 0
            tokens_per_second = 0
            avg_latency_per_request = 0
            total_duration = 0
            p95_duration = 0
            p99_duration = 0
            efficiency_score = 0
        
        return {
            "sampler": config.name,
            "batch_size": batch_size,
            "total_batches": num_batches,
            "successful_batches": len(successful_batches),
            "failed_batches": num_batches - len(successful_batches),
            "total_requests": total_requests,
            "total_time": total_duration,
            "throughput_rps": throughput_rps,
            "tokens_per_second": tokens_per_second,
            "avg_latency_per_request": avg_latency_per_request,
            "p95_batch_duration": p95_duration,
            "p99_batch_duration": p99_duration,
            "total_tokens": total_tokens,
            "efficiency_score": efficiency_score,
            "avg_batch_duration": avg_batch_duration if successful_batches else 0
        }
    
    def run_full_benchmark(self, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
                          num_batches: int = 5) -> pd.DataFrame:
        configs = self.create_sampler_configs()
        all_results = []
        
        total_tests = len(configs) * len(batch_sizes)
        current_test = 0
        
        for config in configs:
            print(f"\nTesting sampler: {config.name}")
            for batch_size in batch_sizes:
                current_test += 1
                print(f"  Batch size {batch_size} ({current_test}/{total_tests})")
                
                result = self.benchmark_batch_size(config, batch_size, num_batches)
                all_results.append(result)
                
                print(f"    Throughput: {result['throughput_rps']:.2f} req/s")
                print(f"    Tokens/s: {result['tokens_per_second']:.2f}")
                print(f"    Avg latency: {result['avg_latency_per_request']:.3f}s")
                print(f"    Batch duration: {result['avg_batch_duration']:.3f}s")
        
        return pd.DataFrame(all_results)
    
    def create_performance_plots(self, df: pd.DataFrame):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        baseline_data = df[df['sampler'] == 'baseline'].set_index('batch_size')
        samplers = df['sampler'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(samplers)))
        
        ax1 = axes[0, 0]
        for i, sampler in enumerate(samplers):
            sampler_data = df[df['sampler'] == sampler]
            ax1.plot(sampler_data['batch_size'], sampler_data['throughput_rps'], 
                    marker='o', label=sampler, color=colors[i], linewidth=2, markersize=6)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (requests/second)')
        ax1.set_title('Throughput vs Batch Size')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        ax2 = axes[0, 1]
        for i, sampler in enumerate(samplers):
            sampler_data = df[df['sampler'] == sampler]
            ax2.plot(sampler_data['batch_size'], sampler_data['tokens_per_second'], 
                    marker='s', label=sampler, color=colors[i], linewidth=2, markersize=6)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Tokens per Second')
        ax2.set_title('Token Generation Rate vs Batch Size')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        ax3 = axes[0, 2]
        for i, sampler in enumerate(samplers):
            sampler_data = df[df['sampler'] == sampler]
            ax3.plot(sampler_data['batch_size'], sampler_data['avg_latency_per_request'], 
                    marker='^', label=sampler, color=colors[i], linewidth=2, markersize=6)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Average Latency per Request (seconds)')
        ax3.set_title('Latency vs Batch Size')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        
        ax4 = axes[1, 0]
        for i, sampler in enumerate(samplers):
            if sampler != 'baseline':
                sampler_data = df[df['sampler'] == sampler].set_index('batch_size')
                try:
                    overhead = ((sampler_data['avg_latency_per_request'] - baseline_data['avg_latency_per_request']) / 
                               baseline_data['avg_latency_per_request'] * 100)
                    ax4.plot(overhead.index, overhead.values, 
                            marker='D', label=sampler, color=colors[i], linewidth=2, markersize=6)
                except:
                    continue
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Latency Overhead (%)')
        ax4.set_title('Latency Overhead vs Baseline')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_xscale('log', base=2)
        
        ax5 = axes[1, 1]
        for i, sampler in enumerate(samplers):
            if sampler != 'baseline':
                sampler_data = df[df['sampler'] == sampler].set_index('batch_size')
                try:
                    efficiency = ((sampler_data['throughput_rps'] / baseline_data['throughput_rps']) * 100)
                    ax5.plot(efficiency.index, efficiency.values, 
                            marker='v', label=sampler, color=colors[i], linewidth=2, markersize=6)
                except:
                    continue
        ax5.set_xlabel('Batch Size')
        ax5.set_ylabel('Throughput Efficiency (%)')
        ax5.set_title('Throughput Efficiency vs Baseline')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=100, color='red', linestyle='--', alpha=0.7)
        ax5.set_xscale('log', base=2)
        
        ax6 = axes[1, 2]
        for i, sampler in enumerate(samplers):
            sampler_data = df[df['sampler'] == sampler]
            ax6.plot(sampler_data['batch_size'], sampler_data['efficiency_score'], 
                    marker='*', label=sampler, color=colors[i], linewidth=2, markersize=8)
        ax6.set_xlabel('Batch Size')
        ax6.set_ylabel('Efficiency Score (req/s per batch size)')
        ax6.set_title('Batching Efficiency Score')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig('vllm_sampler_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_overhead_heatmap(self, df: pd.DataFrame):
        baseline_df = df[df['sampler'] == 'baseline'].set_index('batch_size')
        
        overhead_data = []
        samplers = [s for s in df['sampler'].unique() if s != 'baseline']
        batch_sizes = sorted(df['batch_size'].unique())
        
        for sampler in samplers:
            sampler_data = df[df['sampler'] == sampler].set_index('batch_size')
            overhead_row = []
            for bs in batch_sizes:
                try:
                    overhead = ((sampler_data.loc[bs, 'avg_latency_per_request'] - 
                               baseline_df.loc[bs, 'avg_latency_per_request']) / 
                               baseline_df.loc[bs, 'avg_latency_per_request'] * 100)
                    overhead_row.append(overhead)
                except:
                    overhead_row.append(0)
            overhead_data.append(overhead_row)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(overhead_data, 
                   xticklabels=[f'BS={bs}' for bs in batch_sizes],
                   yticklabels=samplers,
                   annot=True, fmt='.1f', cmap='RdYlBu_r',
                   center=0, cbar_kws={'label': 'Latency Overhead (%)'})
        plt.title('Latency Overhead Heatmap (% vs Baseline)')
        plt.xlabel('Batch Size')
        plt.ylabel('Sampler Configuration')
        plt.tight_layout()
        plt.savefig('vllm_sampler_overhead_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_results(self, df: pd.DataFrame):
        print("\n" + "="*80)
        print("VLLM SAMPLER PERFORMANCE ANALYSIS")
        print("="*80)
        
        baseline_df = df[df['sampler'] == 'baseline']
        
        print("\n1. OVERALL THROUGHPUT IMPACT:")
        print("-" * 50)
        for sampler in df['sampler'].unique():
            if sampler != 'baseline':
                sampler_df = df[df['sampler'] == sampler]
                throughput_ratios = []
                for bs in df['batch_size'].unique():
                    baseline_row = baseline_df[baseline_df['batch_size'] == bs]
                    sampler_row = sampler_df[sampler_df['batch_size'] == bs]
                    if not baseline_row.empty and not sampler_row.empty:
                        ratio = (sampler_row['throughput_rps'].iloc[0] / 
                               baseline_row['throughput_rps'].iloc[0])
                        throughput_ratios.append(ratio)
                
                if throughput_ratios:
                    avg_ratio = np.mean(throughput_ratios)
                    std_ratio = np.std(throughput_ratios)
                    print(f"{sampler:20s}: {avg_ratio:.3f}x ± {std_ratio:.3f}")
        
        print("\n2. BATCH SIZE SCALING ANALYSIS:")
        print("-" * 50)
        for sampler in df['sampler'].unique():
            sampler_data = df[df['sampler'] == sampler].sort_values('batch_size')
            if len(sampler_data) >= 2:
                min_throughput = sampler_data['throughput_rps'].iloc[0]
                max_throughput = sampler_data['throughput_rps'].iloc[-1]
                scaling_factor = max_throughput / min_throughput if min_throughput > 0 else 0
                print(f"{sampler:20s}: {scaling_factor:.2f}x scaling (BS1→BS32)")
        
        print("\n3. LATENCY OVERHEAD BY BATCH SIZE:")
        print("-" * 50)
        for bs in sorted(df['batch_size'].unique()):
            print(f"Batch Size {bs}:")
            baseline_latency = baseline_df[baseline_df['batch_size'] == bs]['avg_latency_per_request'].iloc[0]
            
            for sampler in df['sampler'].unique():
                if sampler != 'baseline':
                    sampler_row = df[(df['sampler'] == sampler) & (df['batch_size'] == bs)]
                    if not sampler_row.empty:
                        sampler_latency = sampler_row['avg_latency_per_request'].iloc[0]
                        overhead = ((sampler_latency / baseline_latency) - 1) * 100
                        print(f"  {sampler:18s}: {overhead:+.1f}%")
        
        print("\n4. TOKEN GENERATION EFFICIENCY:")
        print("-" * 50)
        for sampler in df['sampler'].unique():
            if sampler != 'baseline':
                sampler_df = df[df['sampler'] == sampler]
                token_ratios = []
                for bs in df['batch_size'].unique():
                    baseline_row = baseline_df[baseline_df['batch_size'] == bs]
                    sampler_row = sampler_df[sampler_df['batch_size'] == bs]
                    if not baseline_row.empty and not sampler_row.empty:
                        ratio = (sampler_row['tokens_per_second'].iloc[0] / 
                               baseline_row['tokens_per_second'].iloc[0])
                        token_ratios.append(ratio)
                
                if token_ratios:
                    avg_ratio = np.mean(token_ratios)
                    print(f"{sampler:20s}: {avg_ratio:.3f}x token generation rate")
        
        print("\n5. OPTIMIZATION RECOMMENDATIONS:")
        print("-" * 50)
        
        worst_overhead = {}
        for sampler in df['sampler'].unique():
            if sampler != 'baseline':
                sampler_df = df[df['sampler'] == sampler]
                total_overhead = 0
                count = 0
                for bs in df['batch_size'].unique():
                    baseline_row = baseline_df[baseline_df['batch_size'] == bs]
                    sampler_row = sampler_df[sampler_df['batch_size'] == bs]
                    if not baseline_row.empty and not sampler_row.empty:
                        overhead = ((sampler_row['avg_latency_per_request'].iloc[0] /
                                   baseline_row['avg_latency_per_request'].iloc[0]) - 1) * 100
                        total_overhead += overhead
                        count += 1
                if count > 0:
                    worst_overhead[sampler] = total_overhead / count
        
        print("Priority optimizations based on measured overhead:")
        sorted_overhead = sorted(worst_overhead.items(), key=lambda x: x[1], reverse=True)
        
        for i, (sampler, overhead) in enumerate(sorted_overhead[:3]):
            print(f"  {i+1}. {sampler} ({overhead:.1f}% avg overhead)")
            
            if 'dry' in sampler:
                print("     → Optimize sequence detection with rolling hash")
                print("     → Implement early exit for short sequences")
            if 'xtc' in sampler:
                print("     → Parallelize probability calculations")
                print("     → Cache exclusion sets for similar prompts")
            if 'dynamic' in sampler:
                print("     → Use lookup tables for temperature interpolation")
                print("     → Batch temperature updates across sequences")
        
        print("\n  General optimizations:")
        print("     → Implement CUDA kernels for custom sampling operations")
        print("     → Use memory pooling for temporary sampler data structures")
        print("     → Apply batched operations where samplers allow vectorization")
        print("     → Consider adaptive sampling based on batch characteristics")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--num-batches", type=int, default=3)
    args = parser.parse_args()
    
    print(f"Initializing VLLM with model: {args.model}")
    benchmark = VLLMBenchmark(args.model)
    
    print(f"Testing batch sizes: {args.batch_sizes}")
    print(f"Number of batches per test: {args.num_batches}")
    
    results_df = benchmark.run_full_benchmark(args.batch_sizes, args.num_batches)
    
    results_df.to_csv('vllm_sampler_benchmark_results.csv', index=False)
    print(f"\nResults saved to vllm_sampler_benchmark_results.csv")
    
    benchmark.create_performance_plots(results_df)
    benchmark.create_overhead_heatmap(results_df)
    benchmark.analyze_results(results_df)

if __name__ == "__main__":
    main()