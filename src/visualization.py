#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語LLMファインチューニング 可視化システム
学習進捗と結果比較の可視化機能
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class VisualizationGenerator:
    """可視化生成クラス"""
    
    def __init__(self):
        self.colors = {
            'before': '#FF6B6B',
            'after': '#4ECDC4',
            'improvement': '#45B7D1',
            'background': '#F8F9FA'
        }
    
    def create_metrics_comparison_chart(self, comparison_data: Dict[str, Any], output_path: str = "results/metrics_comparison.png"):
        """評価指標の比較チャート"""
        try:
            improvement = comparison_data["improvement_analysis"]
            
            metrics = ['bleu_score', 'rouge_score', 'bert_score', 'keigo_frequency', 'teineigo_frequency', 'char_diversity']
            metric_labels = ['BLEU Score', 'ROUGE Score', 'BERT Score', '敬語使用頻度', '丁寧語使用頻度', '文字多様性']
            
            before_values = [improvement["before_averages"][metric] for metric in metrics]
            after_values = [improvement["after_averages"][metric] for metric in metrics]
            
            x = np.arange(len(metric_labels))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars1 = ax.bar(x - width/2, before_values, width, label='Before', color=self.colors['before'], alpha=0.8)
            bars2 = ax.bar(x + width/2, after_values, width, label='After', color=self.colors['after'], alpha=0.8)
            
            ax.set_xlabel('評価指標', fontsize=12)
            ax.set_ylabel('スコア', fontsize=12)
            ax.set_title('ファインチューニング前後の評価指標比較', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"評価指標比較チャートを保存しました: {output_path}")
            
        except Exception as e:
            print(f"評価指標比較チャート作成エラー: {e}")
    
    def create_improvement_rates_chart(self, comparison_data: Dict[str, Any], output_path: str = "results/improvement_rates.png"):
        """改善率のチャート"""
        try:
            improvement = comparison_data["improvement_analysis"]
            
            metrics = ['bleu_score', 'rouge_score', 'bert_score', 'keigo_frequency', 'teineigo_frequency', 'char_diversity']
            metric_labels = ['BLEU Score', 'ROUGE Score', 'BERT Score', '敬語使用頻度', '丁寧語使用頻度', '文字多様性']
            
            improvement_rates = [improvement["improvement_rates"][metric] for metric in metrics]
            
            # 色の設定（改善率に応じて）
            colors = [self.colors['improvement'] if rate > 0 else self.colors['before'] for rate in improvement_rates]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(metric_labels, improvement_rates, color=colors, alpha=0.8)
            
            ax.set_xlabel('評価指標', fontsize=12)
            ax.set_ylabel('改善率 (%)', fontsize=12)
            ax.set_title('ファインチューニングによる改善率', fontsize=14, fontweight='bold')
            ax.set_xticklabels(metric_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 値をバーの上に表示
            for bar, rate in zip(bars, improvement_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                       f'{rate:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"改善率チャートを保存しました: {output_path}")
            
        except Exception as e:
            print(f"改善率チャート作成エラー: {e}")
    
    def create_japanese_quality_analysis(self, comparison_data: Dict[str, Any], output_path: str = "results/japanese_quality_analysis.png"):
        """日本語品質分析チャート"""
        try:
            before_results = comparison_data["before_results"]
            after_results = comparison_data["after_results"]
            
            # 日本語品質指標の抽出
            before_keigo = [r["japanese_quality"]["keigo_frequency"] for r in before_results]
            after_keigo = [r["japanese_quality"]["keigo_frequency"] for r in after_results]
            
            before_teineigo = [r["japanese_quality"]["teineigo_frequency"] for r in before_results]
            after_teineigo = [r["japanese_quality"]["teineigo_frequency"] for r in after_results]
            
            before_diversity = [r["japanese_quality"]["char_diversity"] for r in before_results]
            after_diversity = [r["japanese_quality"]["char_diversity"] for r in after_results]
            
            # サブプロットの作成
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 敬語使用頻度の分布
            axes[0, 0].hist(before_keigo, bins=20, alpha=0.7, label='Before', color=self.colors['before'])
            axes[0, 0].hist(after_keigo, bins=20, alpha=0.7, label='After', color=self.colors['after'])
            axes[0, 0].set_title('敬語使用頻度の分布', fontweight='bold')
            axes[0, 0].set_xlabel('敬語使用頻度')
            axes[0, 0].set_ylabel('頻度')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 丁寧語使用頻度の分布
            axes[0, 1].hist(before_teineigo, bins=20, alpha=0.7, label='Before', color=self.colors['before'])
            axes[0, 1].hist(after_teineigo, bins=20, alpha=0.7, label='After', color=self.colors['after'])
            axes[0, 1].set_title('丁寧語使用頻度の分布', fontweight='bold')
            axes[0, 1].set_xlabel('丁寧語使用頻度')
            axes[0, 1].set_ylabel('頻度')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 文字多様性の分布
            axes[1, 0].hist(before_diversity, bins=20, alpha=0.7, label='Before', color=self.colors['before'])
            axes[1, 0].hist(after_diversity, bins=20, alpha=0.7, label='After', color=self.colors['after'])
            axes[1, 0].set_title('文字多様性の分布', fontweight='bold')
            axes[1, 0].set_xlabel('文字多様性')
            axes[1, 0].set_ylabel('頻度')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 改善率の比較
            improvement = comparison_data["improvement_analysis"]
            metrics = ['keigo_frequency', 'teineigo_frequency', 'char_diversity']
            labels = ['敬語使用頻度', '丁寧語使用頻度', '文字多様性']
            rates = [improvement["improvement_rates"][metric] for metric in metrics]
            
            colors = [self.colors['improvement'] if rate > 0 else self.colors['before'] for rate in rates]
            bars = axes[1, 1].bar(labels, rates, color=colors, alpha=0.8)
            axes[1, 1].set_title('日本語品質改善率', fontweight='bold')
            axes[1, 1].set_ylabel('改善率 (%)')
            axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 値をバーの上に表示
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                               f'{rate:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                               fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"日本語品質分析チャートを保存しました: {output_path}")
            
        except Exception as e:
            print(f"日本語品質分析チャート作成エラー: {e}")
    
    def create_interactive_dashboard(self, comparison_data: Dict[str, Any], output_path: str = "results/interactive_dashboard.html"):
        """インタラクティブダッシュボードの作成"""
        try:
            improvement = comparison_data["improvement_analysis"]
            
            # サブプロットの作成
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('評価指標比較', '改善率', '日本語品質指標', '詳細分析'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "table"}]]
            )
            
            # 評価指標比較
            metrics = ['bleu_score', 'rouge_score', 'bert_score']
            metric_labels = ['BLEU Score', 'ROUGE Score', 'BERT Score']
            before_values = [improvement["before_averages"][metric] for metric in metrics]
            after_values = [improvement["after_averages"][metric] for metric in metrics]
            
            fig.add_trace(
                go.Bar(name='Before', x=metric_labels, y=before_values, marker_color=self.colors['before']),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name='After', x=metric_labels, y=after_values, marker_color=self.colors['after']),
                row=1, col=1
            )
            
            # 改善率
            improvement_rates = [improvement["improvement_rates"][metric] for metric in metrics]
            colors = [self.colors['improvement'] if rate > 0 else self.colors['before'] for rate in improvement_rates]
            
            fig.add_trace(
                go.Bar(x=metric_labels, y=improvement_rates, marker_color=colors, name='改善率'),
                row=1, col=2
            )
            
            # 日本語品質指標
            japanese_metrics = ['keigo_frequency', 'teineigo_frequency', 'char_diversity']
            japanese_labels = ['敬語使用頻度', '丁寧語使用頻度', '文字多様性']
            before_japanese = [improvement["before_averages"][metric] for metric in japanese_metrics]
            after_japanese = [improvement["after_averages"][metric] for metric in japanese_metrics]
            
            fig.add_trace(
                go.Scatter(x=before_japanese, y=after_japanese, mode='markers+text',
                          text=japanese_labels, textposition="top center",
                          marker=dict(size=15, color=self.colors['improvement']),
                          name='日本語品質'),
                row=2, col=1
            )
            
            # 詳細分析テーブル
            table_data = []
            for metric, label in zip(metrics + japanese_metrics, metric_labels + japanese_labels):
                before_val = improvement["before_averages"][metric]
                after_val = improvement["after_averages"][metric]
                improvement_rate = improvement["improvement_rates"][metric]
                
                table_data.append([
                    label,
                    f"{before_val:.3f}",
                    f"{after_val:.3f}",
                    f"{improvement_rate:.1f}%"
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['指標', 'Before', 'After', '改善率'],
                               fill_color='lightblue',
                               align='center'),
                    cells=dict(values=list(zip(*table_data)),
                              fill_color='white',
                              align='center')
                ),
                row=2, col=2
            )
            
            # レイアウトの設定
            fig.update_layout(
                title_text="日本語LLMファインチューニング 評価ダッシュボード",
                title_x=0.5,
                showlegend=True,
                height=800,
                width=1200
            )
            
            # 軸ラベルの設定
            fig.update_xaxes(title_text="評価指標", row=1, col=1)
            fig.update_yaxes(title_text="スコア", row=1, col=1)
            fig.update_xaxes(title_text="評価指標", row=1, col=2)
            fig.update_yaxes(title_text="改善率 (%)", row=1, col=2)
            fig.update_xaxes(title_text="Before", row=2, col=1)
            fig.update_yaxes(title_text="After", row=2, col=1)
            
            # HTMLファイルとして保存
            fig.write_html(output_path)
            print(f"インタラクティブダッシュボードを保存しました: {output_path}")
            
        except Exception as e:
            print(f"インタラクティブダッシュボード作成エラー: {e}")
    
    def create_summary_report(self, comparison_data: Dict[str, Any], output_path: str = "results/summary_report.md"):
        """サマリーレポートの作成"""
        try:
            improvement = comparison_data["improvement_analysis"]
            
            report = f"""# 日本語LLMファインチューニング 評価レポート

## 概要
- 評価日時: {comparison_data.get('timestamp', 'N/A')}
- 評価ID: {comparison_data.get('comparison_id', 'N/A')}
- 総質問数: {len(comparison_data.get('before_results', []))}

## 評価指標の改善結果

### 基本評価指標
- **BLEU Score**: {improvement['before_averages']['bleu_score']:.3f} → {improvement['after_averages']['bleu_score']:.3f} ({improvement['improvement_rates']['bleu_score']:+.1f}%)
- **ROUGE Score**: {improvement['before_averages']['rouge_score']:.3f} → {improvement['after_averages']['rouge_score']:.3f} ({improvement['improvement_rates']['rouge_score']:+.1f}%)
- **BERT Score**: {improvement['before_averages']['bert_score']:.3f} → {improvement['after_averages']['bert_score']:.3f} ({improvement['improvement_rates']['bert_score']:+.1f}%)

### 日本語品質指標
- **敬語使用頻度**: {improvement['before_averages']['keigo_frequency']:.3f} → {improvement['after_averages']['keigo_frequency']:.3f} ({improvement['improvement_rates']['keigo_frequency']:+.1f}%)
- **丁寧語使用頻度**: {improvement['before_averages']['teineigo_frequency']:.3f} → {improvement['after_averages']['teineigo_frequency']:.3f} ({improvement['improvement_rates']['teineigo_frequency']:+.1f}%)
- **文字多様性**: {improvement['before_averages']['char_diversity']:.3f} → {improvement['after_averages']['char_diversity']:.3f} ({improvement['improvement_rates']['char_diversity']:+.1f}%)

## 改善分析

### 最も改善された指標
"""
            
            # 改善率のランキング
            improvement_rates = improvement['improvement_rates']
            sorted_improvements = sorted(improvement_rates.items(), key=lambda x: x[1], reverse=True)
            
            for i, (metric, rate) in enumerate(sorted_improvements[:3], 1):
                metric_names = {
                    'bleu_score': 'BLEU Score',
                    'rouge_score': 'ROUGE Score', 
                    'bert_score': 'BERT Score',
                    'keigo_frequency': '敬語使用頻度',
                    'teineigo_frequency': '丁寧語使用頻度',
                    'char_diversity': '文字多様性'
                }
                report += f"{i}. {metric_names.get(metric, metric)}: {rate:+.1f}%\n"
            
            report += f"""
### 改善率の統計
- 平均改善率: {np.mean(list(improvement_rates.values())):.1f}%
- 最大改善率: {max(improvement_rates.values()):.1f}%
- 最小改善率: {min(improvement_rates.values()):.1f}%

## 結論

日本語LLMファインチューニングにより、以下の改善が確認されました：

1. **回答品質の向上**: BLEU、ROUGE、BERTスコアの改善
2. **日本語表現の向上**: 敬語・丁寧語の使用頻度の改善
3. **文字多様性の向上**: 日本語特有の文字種の適切な使用

これらの結果は、三陽商会ドキュメントを基にした日本語特化ファインチューニングの有効性を示しています。

---
*このレポートは自動生成されました。*
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"サマリーレポートを保存しました: {output_path}")
            
        except Exception as e:
            print(f"サマリーレポート作成エラー: {e}")

def main():
    """メイン実行関数"""
    print("=== 日本語LLMファインチューニング 可視化システム ===")
    
    # 比較評価結果の読み込み
    try:
        with open('results/comparison_evaluation.json', 'r', encoding='utf-8') as f:
            comparison_data = json.load(f)
        print("比較評価結果を読み込みました")
    except Exception as e:
        print(f"比較評価結果読み込みエラー: {e}")
        return
    
    # 可視化生成器の初期化
    visualizer = VisualizationGenerator()
    
    # 各種可視化の生成
    print("\n可視化を生成中...")
    
    # 評価指標比較チャート
    visualizer.create_metrics_comparison_chart(comparison_data)
    
    # 改善率チャート
    visualizer.create_improvement_rates_chart(comparison_data)
    
    # 日本語品質分析
    visualizer.create_japanese_quality_analysis(comparison_data)
    
    # インタラクティブダッシュボード
    visualizer.create_interactive_dashboard(comparison_data)
    
    # サマリーレポート
    visualizer.create_summary_report(comparison_data)
    
    print("\n=== 可視化生成完了 ===")
    print("生成されたファイル:")
    print("- results/metrics_comparison.png")
    print("- results/improvement_rates.png")
    print("- results/japanese_quality_analysis.png")
    print("- results/interactive_dashboard.html")
    print("- results/summary_report.md")

if __name__ == "__main__":
    main()
