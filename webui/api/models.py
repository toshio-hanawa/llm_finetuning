#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデルサービス - ベースモデルとファインチューニング済みモデルの管理
"""

import torch
import time
import logging
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from webui.config.settings import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """モデル管理クラス"""
    
    def __init__(self):
        self.device = settings.DEVICE
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        self._models_loaded = False
        
        logger.info(f"使用デバイス: {self.device}")
    
    async def load_models(self) -> Dict[str, Any]:
        """モデルを非同期で読み込み"""
        if self._models_loaded:
            return {"status": "already_loaded", "message": "モデルは既に読み込み済みです"}
        
        try:
            start_time = time.time()
            
            # ベースモデルの読み込み
            logger.info(f"ベースモデル読み込み中: {settings.BASE_MODEL_NAME}")
            self.base_tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                settings.BASE_MODEL_NAME,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # パディングトークンの設定
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
            # ファインチューニング済みモデルの読み込み
            logger.info(f"ファインチューニング済みモデル読み込み中: {settings.FINETUNED_MODEL_PATH}")
            
            # ベースモデルを再度読み込み（LoRA適用のため）
            base_model_for_peft = AutoModelForCausalLM.from_pretrained(
                settings.BASE_MODEL_NAME,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # ファインチューニング用のトークナイザー
            self.finetuned_tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
            if self.finetuned_tokenizer.pad_token is None:
                self.finetuned_tokenizer.pad_token = self.finetuned_tokenizer.eos_token
            
            # 日本語特殊トークンの追加
            special_tokens = ["<|japanese|>", "<|question|>", "<|answer|>", "<|end|>"]
            self.finetuned_tokenizer.add_tokens(special_tokens)
            
            # 埋め込み層のリサイズ
            base_model_for_peft.resize_token_embeddings(len(self.finetuned_tokenizer))
            
            # PEFTモデルとして読み込み
            self.finetuned_model = PeftModel.from_pretrained(
                base_model_for_peft, 
                settings.FINETUNED_MODEL_PATH
            )
            
            load_time = time.time() - start_time
            self._models_loaded = True
            
            logger.info(f"モデル読み込み完了: {load_time:.2f}秒")
            
            return {
                "status": "success",
                "message": "モデルの読み込みが完了しました",
                "load_time": load_time,
                "base_vocab_size": len(self.base_tokenizer),
                "finetuned_vocab_size": len(self.finetuned_tokenizer),
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return {
                "status": "error",
                "message": f"モデルの読み込みに失敗しました: {str(e)}"
            }
    
    def is_loaded(self) -> bool:
        """モデルが読み込み済みかどうかを確認"""
        return self._models_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        if not self._models_loaded:
            return {"status": "not_loaded", "message": "モデルが読み込まれていません"}
        
        return {
            "status": "loaded",
            "base_model": {
                "name": settings.BASE_MODEL_NAME,
                "vocab_size": len(self.base_tokenizer),
                "device": str(self.base_model.device) if hasattr(self.base_model, 'device') else self.device
            },
            "finetuned_model": {
                "path": settings.FINETUNED_MODEL_PATH,
                "vocab_size": len(self.finetuned_tokenizer),
                "device": str(self.finetuned_model.base_model.device) if hasattr(self.finetuned_model.base_model, 'device') else self.device
            }
        }

# グローバルモデルマネージャーインスタンス
model_manager = ModelManager()
