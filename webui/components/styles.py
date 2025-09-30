#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モダンなUI スタイル定義
"""

import streamlit as st

def apply_modern_styles():
    """モダンで洗練されたCSSスタイルを適用"""
    st.markdown("""
    <style>
    /* メインテーマの設定 */
    .main {
        padding-top: 1rem;
    }
    
    /* ヘッダースタイル */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem !important;
        margin-top: 0.5rem !important;
    }
    
    /* カードスタイル */
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .base-model-card {
        border-left: 4px solid #FF6B6B;
    }
    
    .finetuned-model-card {
        border-left: 4px solid #4ECDC4;
    }
    
    /* モデル出力エリア */
    .model-output {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        font-family: 'Hiragino Sans', 'Yu Gothic', sans-serif;
        line-height: 1.7;
        min-height: 150px;
    }
    
    .base-output {
        border-left: 4px solid #FF6B6B;
        background: linear-gradient(135deg, #fff5f5 0%, #fff 100%);
    }
    
    .finetuned-output {
        border-left: 4px solid #4ECDC4;
        background: linear-gradient(135deg, #f0fdfc 0%, #fff 100%);
    }
    
    /* メトリクスカード */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-top: 0.25rem;
    }
    
    /* 改善率インジケーター */
    .improvement-positive {
        color: #27ae60 !important;
        font-weight: 600;
    }
    
    .improvement-negative {
        color: #e74c3c !important;
        font-weight: 600;
    }
    
    .improvement-neutral {
        color: #95a5a6 !important;
        font-weight: 600;
    }
    
    /* ボタンスタイル */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* セレクトボックス */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e9ecef;
    }
    
    /* テキストエリア */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        font-family: 'Hiragino Sans', 'Yu Gothic', sans-serif;
    }
    
    /* 数値入力 */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e9ecef;
    }
    
    /* スライダー */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* ローディングスピナー */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* プログレスバー */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* サイドバー */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* 比較結果のアニメーション */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .comparison-result {
        animation: slideInUp 0.6s ease-out;
    }
    
    /* レスポンシブデザイン */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem !important;
        }
        
        .model-card {
            padding: 1rem;
        }
        
        .model-output {
            padding: 1rem;
        }
    }
    
    /* ダークモード対応 */
    @media (prefers-color-scheme: dark) {
        .model-card {
            background: #2d3748;
            border-color: #4a5568;
        }
        
        .model-output {
            background: #1a202c;
            border-color: #4a5568;
            color: #e2e8f0;
        }
        
        .metric-card {
            background: #2d3748;
            border-color: #4a5568;
        }
        
        .metric-value {
            color: #e2e8f0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """メインヘッダーの表示"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 LLM モデル比較システム</h1>
        <p>ベースモデル vs ファインチューニング済みモデル</p>
    </div>
    """, unsafe_allow_html=True)

def create_model_card(title: str, content: str, model_type: str = "base"):
    """モデル出力カード"""
    card_class = "base-model-card" if model_type == "base" else "finetuned-model-card"
    output_class = "base-output" if model_type == "base" else "finetuned-output"
    
    icon = "🔹" if model_type == "base" else "🔸"
    
    st.markdown(f"""
    <div class="model-card {card_class}">
        <h3>{icon} {title}</h3>
        <div class="model-output {output_class}">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(value: str, label: str, improvement: float = None):
    """メトリクスカード"""
    improvement_html = ""
    if improvement is not None:
        if improvement > 0:
            improvement_html = f'<div class="improvement-positive">↗ +{improvement:.1f}%</div>'
        elif improvement < 0:
            improvement_html = f'<div class="improvement-negative">↘ {improvement:.1f}%</div>'
        else:
            improvement_html = f'<div class="improvement-neutral">→ {improvement:.1f}%</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {improvement_html}
    </div>
    """, unsafe_allow_html=True)
