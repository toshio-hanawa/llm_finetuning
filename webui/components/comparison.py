#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較表示コンポーネント
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any

from webui.components.styles import create_model_card, create_metric_card

class ComparisonDisplay:
    """比較結果表示クラス"""
    
    @staticmethod
    def display_comparison_result(result: Dict[str, Any]):
        """比較結果の表示"""
        if result["status"] != "success":
            st.error(f"エラー: {result['message']}")
            return
        
        # 結果データの取得
        base_result = result["results"]["base_model"]
        finetuned_result = result["results"]["finetuned_model"]
        comparison = result["comparison"]
        metrics = result.get("metrics", {})
        
        # マークダウンで比較結果エリアを作成
        st.markdown('<div class="comparison-result">', unsafe_allow_html=True)
        
        # 1. モデル出力の並列表示
        st.subheader("📝 モデル出力比較")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_model_card(
                "ベースモデル",
                base_result["text"],
                "base"
            )
        
        with col2:
            create_model_card(
                "ファインチューニング済みモデル",
                finetuned_result["text"],
                "finetuned"
            )
        
        # 2. 基本メトリクス表示
        st.subheader("⚡ パフォーマンス指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card(
                f"{base_result['inference_time']:.2f}s",
                "ベース推論時間"
            )
        
        with col2:
            create_metric_card(
                f"{finetuned_result['inference_time']:.2f}s",
                "FT推論時間",
                -comparison["speed_improvement"]  # 負の値で高速化を表現
            )
        
        with col3:
            create_metric_card(
                str(base_result['character_count']),
                "ベース文字数"
            )
        
        with col4:
            improvement = (finetuned_result['character_count'] - base_result['character_count']) / base_result['character_count'] * 100 if base_result['character_count'] > 0 else 0
            create_metric_card(
                str(finetuned_result['character_count']),
                "FT文字数",
                improvement
            )
        
        # 3. 詳細メトリクス分析
        if metrics and "base_metrics" in metrics:
            st.subheader("📊 詳細メトリクス分析")
            ComparisonDisplay._display_detailed_metrics(metrics)
        
        # 4. 可視化チャート
        st.subheader("📈 可視化分析")
        ComparisonDisplay._display_comparison_charts(result)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def _display_detailed_metrics(metrics: Dict[str, Any]):
        """詳細メトリクスの表示"""
        base_metrics = metrics["base_metrics"]
        finetuned_metrics = metrics["finetuned_metrics"]
        improvements = metrics["improvements"]
        
        # タブで整理
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 基本統計", "🈳 日本語特性", "✨ 言語品質", "📖 読みやすさ"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**ベースモデル**")
                st.json(base_metrics["basic_stats"])
            
            with col2:
                st.markdown("**ファインチューニング済み**")
                st.json(finetuned_metrics["basic_stats"])
        
        with tab2:
            # 日本語文字使用の可視化
            ComparisonDisplay._display_japanese_characteristics(
                base_metrics["japanese_characteristics"],
                finetuned_metrics["japanese_characteristics"]
            )
        
        with tab3:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                keigo_improvement = improvements.get("keigo_improvement", 0)
                create_metric_card(
                    f"{finetuned_metrics['linguistic_quality']['keigo_usage_rate']:.1f}%",
                    "敬語使用率",
                    keigo_improvement
                )
            
            with col2:
                vocab_improvement = improvements.get("vocabulary_improvement", 0)
                create_metric_card(
                    f"{finetuned_metrics['linguistic_quality']['vocabulary_richness']:.1f}%",
                    "語彙豊富さ",
                    vocab_improvement
                )
            
            with col3:
                similarity = metrics.get("similarity", 0)
                create_metric_card(
                    f"{similarity:.1f}%",
                    "類似度"
                )
        
        with tab4:
            ComparisonDisplay._display_readability_metrics(
                base_metrics["readability"],
                finetuned_metrics["readability"]
            )
    
    @staticmethod
    def _display_japanese_characteristics(base_chars: Dict, finetuned_chars: Dict):
        """日本語文字特性の可視化"""
        # データ準備
        categories = ['ひらがな', 'カタカナ', '漢字']
        base_values = [
            base_chars['hiragana_count'],
            base_chars['katakana_count'],
            base_chars['kanji_count']
        ]
        finetuned_values = [
            finetuned_chars['hiragana_count'],
            finetuned_chars['katakana_count'],
            finetuned_chars['kanji_count']
        ]
        
        # レーダーチャート
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=base_values,
            theta=categories,
            fill='toself',
            name='ベースモデル',
            line_color='#FF6B6B'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=finetuned_values,
            theta=categories,
            fill='toself',
            name='ファインチューニング済み',
            line_color='#4ECDC4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(base_values), max(finetuned_values)) * 1.1]
                )
            ),
            showlegend=True,
            title="日本語文字種使用比較",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _display_readability_metrics(base_readability: Dict, finetuned_readability: Dict):
        """読みやすさメトリクスの表示"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "平均文長（ベース）",
                f"{base_readability['avg_sentence_length']:.1f}文字"
            )
            st.metric(
                "文長分散（ベース）",
                f"{base_readability['sentence_length_variance']:.1f}"
            )
        
        with col2:
            improvement = (
                (finetuned_readability['avg_sentence_length'] - base_readability['avg_sentence_length'])
                / base_readability['avg_sentence_length'] * 100
                if base_readability['avg_sentence_length'] > 0 else 0
            )
            
            st.metric(
                "平均文長（FT）",
                f"{finetuned_readability['avg_sentence_length']:.1f}文字",
                f"{improvement:+.1f}%"
            )
            
            variance_improvement = (
                (finetuned_readability['sentence_length_variance'] - base_readability['sentence_length_variance'])
                / base_readability['sentence_length_variance'] * 100
                if base_readability['sentence_length_variance'] > 0 else 0
            )
            
            st.metric(
                "文長分散（FT）",
                f"{finetuned_readability['sentence_length_variance']:.1f}",
                f"{variance_improvement:+.1f}%"
            )
    
    @staticmethod
    def _display_comparison_charts(result: Dict[str, Any]):
        """比較チャートの表示"""
        base_result = result["results"]["base_model"]
        finetuned_result = result["results"]["finetuned_model"]
        
        # 推論時間比較
        col1, col2 = st.columns(2)
        
        with col1:
            # 推論時間バーチャート
            fig_time = go.Figure(data=[
                go.Bar(
                    name='推論時間',
                    x=['ベースモデル', 'ファインチューニング済み'],
                    y=[base_result['inference_time'], finetuned_result['inference_time']],
                    marker_color=['#FF6B6B', '#4ECDC4']
                )
            ])
            
            fig_time.update_layout(
                title="推論時間比較",
                yaxis_title="時間（秒）",
                height=300
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # 文字数比較
            fig_chars = go.Figure(data=[
                go.Bar(
                    name='文字数',
                    x=['ベースモデル', 'ファインチューニング済み'],
                    y=[base_result['character_count'], finetuned_result['character_count']],
                    marker_color=['#FF6B6B', '#4ECDC4']
                )
            ])
            
            fig_chars.update_layout(
                title="出力文字数比較",
                yaxis_title="文字数",
                height=300
            )
            
            st.plotly_chart(fig_chars, use_container_width=True)

# グローバルインスタンス
comparison_display = ComparisonDisplay()
