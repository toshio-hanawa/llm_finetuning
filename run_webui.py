#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 LLM比較WebUI 統合起動スクリプト
FastAPI + Streamlit を同時起動
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path

class WebUILauncher:
    """WebUI統合起動クラス"""
    
    def __init__(self):
        self.api_process = None
        self.streamlit_process = None
        self.is_running = False
        
        # プロジェクトルートディレクトリを設定
        self.project_root = Path(__file__).parent
        os.chdir(self.project_root)
        
    def check_virtual_env(self):
        """仮想環境の確認"""
        venv_path = self.project_root / "webui_env"
        if not venv_path.exists():
            print("❌ 仮想環境が見つかりません")
            print("以下のコマンドで仮想環境を作成してください：")
            print("python3 -m venv webui_env")
            print("source webui_env/bin/activate")
            print("pip install -r requirements.txt")
            return False
        
        return True
    
    def get_python_executable(self):
        """仮想環境のPythonパスを取得"""
        venv_python = self.project_root / "webui_env" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        
        # フォールバック
        return sys.executable
    
    def start_api_server(self):
        """FastAPIサーバーの起動"""
        print("🔧 FastAPIサーバー起動中...")
        
        try:
            python_exe = self.get_python_executable()
            
            # FastAPI起動コマンド
            api_cmd = [
                python_exe, "-m", "uvicorn",
                "webui.api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ]
            
            # 環境変数の設定
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root)
            
            self.api_process = subprocess.Popen(
                api_cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print("✅ FastAPIサーバーが起動しました (http://localhost:8000)")
            print("📖 API ドキュメント: http://localhost:8000/api/docs")
            
            return True
            
        except Exception as e:
            print(f"❌ FastAPIサーバー起動エラー: {e}")
            return False
    
    def start_streamlit_app(self):
        """Streamlitアプリの起動"""
        print("🔧 Streamlitアプリ起動中...")
        
        # APIサーバーの起動を待機
        print("⏳ FastAPIサーバーの準備を待機中...")
        time.sleep(5)
        
        try:
            python_exe = self.get_python_executable()
            
            # Streamlit起動コマンド
            streamlit_cmd = [
                python_exe, "-m", "streamlit", "run",
                "streamlit_app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
            ]
            
            # 環境変数の設定
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root)
            
            self.streamlit_process = subprocess.Popen(
                streamlit_cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print("✅ Streamlitアプリが起動しました (http://localhost:8501)")
            
            return True
            
        except Exception as e:
            print(f"❌ Streamlitアプリ起動エラー: {e}")
            return False
    
    def monitor_processes(self):
        """プロセスの監視"""
        while self.is_running:
            try:
                # APIサーバーの監視
                if self.api_process and self.api_process.poll() is not None:
                    print("⚠️ FastAPIサーバーが停止しました")
                    self.is_running = False
                    break
                
                # Streamlitアプリの監視
                if self.streamlit_process and self.streamlit_process.poll() is not None:
                    print("⚠️ Streamlitアプリが停止しました")
                    self.is_running = False
                    break
                
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ プロセス監視エラー: {e}")
                break
    
    def signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        print("\n🛑 停止シグナルを受信しました。サーバーを停止中...")
        self.stop()
    
    def stop(self):
        """サーバーの停止"""
        self.is_running = False
        
        print("🔄 プロセス停止中...")
        
        # Streamlitプロセスの停止
        if self.streamlit_process:
            try:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=10)
                print("✅ Streamlitアプリを停止しました")
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
                print("⚡ Streamlitアプリを強制停止しました")
            except Exception as e:
                print(f"❌ Streamlit停止エラー: {e}")
        
        # FastAPIプロセスの停止
        if self.api_process:
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=10)
                print("✅ FastAPIサーバーを停止しました")
            except subprocess.TimeoutExpired:
                self.api_process.kill()
                print("⚡ FastAPIサーバーを強制停止しました")
            except Exception as e:
                print(f"❌ FastAPI停止エラー: {e}")
        
        print("🎯 すべてのサーバーが停止しました")
    
    def run(self):
        """WebUI統合起動"""
        print("🚀 LLM比較WebUI 統合起動スクリプト")
        print("=" * 50)
        
        # シグナルハンドラーの設定
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 仮想環境チェック
        if not self.check_virtual_env():
            return False
        
        try:
            # FastAPIサーバー起動
            if not self.start_api_server():
                return False
            
            # Streamlitアプリ起動
            if not self.start_streamlit_app():
                self.stop()
                return False
            
            self.is_running = True
            
            print("\n" + "=" * 50)
            print("🎉 WebUIが正常に起動しました！")
            print("=" * 50)
            print("📱 Streamlit UI: http://localhost:8501")
            print("🔗 FastAPI ドキュメント: http://localhost:8000/api/docs")
            print("❤️ システム監視: http://localhost:8000/api/health")
            print("=" * 50)
            print("終了するには Ctrl+C を押してください")
            print("=" * 50)
            
            # プロセス監視スレッド開始
            monitor_thread = threading.Thread(target=self.monitor_processes)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # メインループ
            try:
                while self.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ 起動エラー: {e}")
            self.stop()
            return False
        finally:
            if self.is_running:
                self.stop()

def main():
    """メイン関数"""
    launcher = WebUILauncher()
    success = launcher.run()
    
    if success:
        print("✅ WebUIが正常に停止しました")
        sys.exit(0)
    else:
        print("❌ WebUI起動に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()
