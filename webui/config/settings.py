#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI設定管理モジュール
"""

import os
from typing import Optional
from pydantic import BaseModel

class Settings(BaseModel):
    """アプリケーション設定"""
    
    # モデル設定
    BASE_MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    FINETUNED_MODEL_PATH: str = "./japanese_finetuned_model"
    
    # API設定
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # Streamlit設定
    STREAMLIT_HOST: str = "localhost"
    STREAMLIT_PORT: int = 8501
    
    # 推論設定
    MAX_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    TOP_K: int = 50
    
    # デバイス設定
    DEVICE: str = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    # ディレクトリ設定
    RESULTS_DIR: str = "./results"
    STATIC_DIR: str = "./webui/static"
    
    # ログ設定
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# グローバル設定インスタンス
settings = Settings()
