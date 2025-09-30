#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 LLM比較 WebUI - メインアプリケーション
Streamlit フロントエンド
"""

import streamlit as st
import time
import logging
import json
from typing import Dict, Any

# ローカルインポート
from webui.components.styles import apply_modern_styles, display_header
from webui.components.comparison import comparison_display
from webui.utils.api_client import api_client

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ページ設定
st.set_page_config(
    page_title="🤖 LLM比較システム",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# スタイル適用
apply_modern_styles()

class WebUIApp:
    """WebUIメインアプリケーションクラス"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """セッション状態の初期化"""
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'api_connected' not in st.session_state:
            st.session_state.api_connected = False
        if 'comparison_history' not in st.session_state:
            st.session_state.comparison_history = []
        if 'current_result' not in st.session_state:
            st.session_state.current_result = None
    
    def check_api_connection(self):
        """API接続確認"""
        try:
            result = api_client.sync_health_check()
            if result.get("status") == "healthy":
                st.session_state.api_connected = True
                st.session_state.models_loaded = result.get("models_loaded", False)
                return True
            else:
                st.session_state.api_connected = False
                return False
        except Exception as e:
            logger.error(f"API接続エラー: {e}")
            st.session_state.api_connected = False
            return False
    
    def display_sidebar(self):
        """サイドバーの表示"""
        with st.sidebar:
            st.markdown("## ⚙️ システム制御")
            
            # API接続状態
            if st.session_state.api_connected:
                st.success("✅ API接続済み")
            else:
                st.error("❌ API未接続")
                if st.button("🔄 接続確認"):
                    with st.spinner("接続確認中..."):
                        self.check_api_connection()
                    st.rerun()
            
            # モデル読み込み状態
            if st.session_state.api_connected:
                if st.session_state.models_loaded:
                    st.success("✅ モデル読み込み済み")
                else:
                    st.warning("⚠️ モデル未読み込み")
                    if st.button("📥 モデル読み込み"):
                        with st.spinner("モデル読み込み中..."):
                            result = api_client.sync_load_models()
                            if result.get("status") == "success":
                                st.session_state.models_loaded = True
                                st.success("モデル読み込み完了！")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(f"読み込みエラー: {result.get('message', '不明なエラー')}")
            
            st.markdown("---")
            
            # 推論パラメータ設定
            st.markdown("## 🎛️ 推論パラメータ")
            
            max_new_tokens = st.slider(
                "最大新規トークン数",
                min_value=50,
                max_value=512,
                value=256,
                step=10
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.7,
                step=0.1
            )
            
            top_p = st.slider(
                "Top-p",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.05
            )
            
            top_k = st.slider(
                "Top-k",
                min_value=1,
                max_value=100,
                value=50,
                step=5
            )
            
            # セッションに保存
            st.session_state.generation_params = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
            
            st.markdown("---")
            
            # 履歴管理
            st.markdown("## 📚 比較履歴")
            if st.session_state.comparison_history:
                st.markdown(f"**{len(st.session_state.comparison_history)}件の履歴**")
                if st.button("🗑️ 履歴クリア"):
                    st.session_state.comparison_history = []
                    st.rerun()
            else:
                st.markdown("履歴なし")
    
    def display_main_content(self):
        """メインコンテンツの表示"""
        # ヘッダー
        display_header()
        
        # API未接続の場合
        if not st.session_state.api_connected:
            st.error("""
            ### 🚨 API サーバーに接続できません
            
            以下の手順でAPIサーバーを起動してください：
            
            1. ターミナルで以下のコマンドを実行：
            ```bash
            cd /home/ubuntu/Documents/projects/llm_finetuning
            source webui_env/bin/activate
            python -m webui.api.main
            ```
            
            2. サーバー起動後、サイドバーの「🔄 接続確認」をクリック
            """)
            return
        
        # モデル未読み込みの場合
        if not st.session_state.models_loaded:
            st.warning("""
            ### ⚠️ モデルが読み込まれていません
            
            サイドバーの「📥 モデル読み込み」ボタンをクリックしてモデルを読み込んでください。
            """)
            return
        
        # メイン機能
        self.display_comparison_interface()
    
    def display_comparison_interface(self):
        """比較インターフェースの表示"""
        st.markdown("## 💬 プロンプト入力")
        
        # プリセット質問
        preset_questions = [
            "カスタム質問を入力",
            "三陽商会の主力事業について教えてください。",
            "三陽商会の海外展開戦略について説明してください。",
            "三陽商会の財務状況はどうですか？",
            "三陽商会の競合他社との違いは何ですか？",
            "三陽商会の今後の成長戦略について教えてください。"
        ]
        
        selected_preset = st.selectbox(
            "📋 プリセット質問を選択（またはカスタム質問を入力）",
            preset_questions
        )
        
        # プロンプト入力
        if selected_preset == "カスタム質問を入力":
            prompt = st.text_area(
                "🔤 質問を入力してください：",
                height=100,
                placeholder="例：三陽商会の主力事業について詳しく教えてください。"
            )
        else:
            prompt = st.text_area(
                "🔤 質問を入力してください：",
                value=selected_preset,
                height=100
            )
        
        # 実行ボタン
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button(
                "🚀 比較実行",
                type="primary",
                use_container_width=True,
                disabled=not prompt.strip()
            ):
                self.execute_comparison(prompt.strip())
        
        with col2:
            if st.button(
                "🔹 ベースのみ",
                use_container_width=True,
                disabled=not prompt.strip()
            ):
                self.execute_single_inference(prompt.strip(), "base")
        
        with col3:
            if st.button(
                "🔸 FTのみ",
                use_container_width=True,
                disabled=not prompt.strip()
            ):
                self.execute_single_inference(prompt.strip(), "finetuned")
        
        # 結果表示
        if st.session_state.current_result:
            st.markdown("---")
            self.display_results()
    
    def execute_comparison(self, prompt: str):
        """比較実行"""
        with st.spinner("🤖 モデル比較実行中..."):
            try:
                params = st.session_state.generation_params
                result = api_client.sync_compare_models(
                    prompt=prompt,
                    **params
                )
                
                if result.get("status") == "success":
                    st.session_state.current_result = result
                    # 履歴に追加
                    st.session_state.comparison_history.append({
                        "timestamp": time.time(),
                        "prompt": prompt,
                        "result": result,
                        "type": "comparison"
                    })
                    st.success("✅ 比較完了！")
                else:
                    st.error(f"❌ エラー: {result.get('message', '不明なエラー')}")
                    
            except Exception as e:
                logger.error(f"比較実行エラー: {e}")
                st.error(f"❌ 予期しないエラー: {str(e)}")
    
    def execute_single_inference(self, prompt: str, model_type: str):
        """単一モデル推論実行"""
        model_name = "ベースモデル" if model_type == "base" else "ファインチューニング済みモデル"
        
        with st.spinner(f"🤖 {model_name}で推論実行中..."):
            try:
                params = st.session_state.generation_params
                result = api_client.sync_single_inference(
                    prompt=prompt,
                    model_type=model_type,
                    **params
                )
                
                if result.get("status") == "success":
                    st.session_state.current_result = result
                    # 履歴に追加
                    st.session_state.comparison_history.append({
                        "timestamp": time.time(),
                        "prompt": prompt,
                        "result": result,
                        "type": "single",
                        "model_type": model_type
                    })
                    st.success(f"✅ {model_name}推論完了！")
                else:
                    st.error(f"❌ エラー: {result.get('message', '不明なエラー')}")
                    
            except Exception as e:
                logger.error(f"単一推論エラー: {e}")
                st.error(f"❌ 予期しないエラー: {str(e)}")
    
    def display_results(self):
        """結果表示"""
        result = st.session_state.current_result
        
        if result.get("type") == "single" or "results" not in result:
            # 単一モデル結果
            self.display_single_result(result)
        else:
            # 比較結果
            comparison_display.display_comparison_result(result)
    
    def display_single_result(self, result: Dict[str, Any]):
        """単一モデル結果の表示"""
        if result["status"] != "success":
            st.error(f"エラー: {result['message']}")
            return
        
        single_result = result["result"]
        model_type = single_result.get("model_type", "unknown")
        
        st.subheader(f"📝 {single_result['model_name']} 出力結果")
        
        # 結果カード
        card_type = "base" if model_type == "base" else "finetuned"
        model_name = "ベースモデル" if model_type == "base" else "ファインチューニング済みモデル"
        
        from webui.components.styles import create_model_card, create_metric_card
        
        create_model_card(
            model_name,
            single_result["text"],
            card_type
        )
        
        # メトリクス表示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card(
                f"{single_result['inference_time']:.2f}s",
                "推論時間"
            )
        
        with col2:
            create_metric_card(
                str(single_result['character_count']),
                "文字数"
            )
        
        with col3:
            create_metric_card(
                model_name,
                "使用モデル"
            )
        
        # 詳細メトリクス
        if "metrics" in result:
            st.subheader("📊 詳細分析")
            st.json(result["metrics"])
    
    def run(self):
        """アプリケーション実行"""
        # 初期API接続確認
        if not st.session_state.api_connected:
            self.check_api_connection()
        
        # サイドバー表示
        self.display_sidebar()
        
        # メインコンテンツ表示
        self.display_main_content()

def main():
    """メイン関数"""
    app = WebUIApp()
    app.run()

if __name__ == "__main__":
    main()
