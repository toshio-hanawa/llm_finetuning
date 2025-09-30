#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推論サービス - テキスト生成と比較機能
"""

import torch
import time
import asyncio
import logging
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from webui.api.models import model_manager
from webui.config.settings import settings

logger = logging.getLogger(__name__)

class InferenceService:
    """推論サービスクラス"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def _generate_text(self, model, tokenizer, prompt: str, **generation_kwargs) -> Tuple[str, float]:
        """テキスト生成の実行（同期処理）"""
        try:
            start_time = time.time()
            
            # プロンプトのフォーマット
            formatted_prompt = self._format_prompt(prompt)
            
            # トークン化
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.MAX_LENGTH
            )
            
            # デバイスに移動
            if model_manager.device == "cuda":
                inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}
            
            # 生成設定
            generation_config = {
                "max_new_tokens": generation_kwargs.get("max_new_tokens", 256),
                "temperature": generation_kwargs.get("temperature", settings.TEMPERATURE),
                "top_p": generation_kwargs.get("top_p", settings.TOP_P),
                "top_k": generation_kwargs.get("top_k", settings.TOP_K),
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }
            
            # テキスト生成
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **generation_config
                )
            
            # デコード
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            inference_time = time.time() - start_time
            
            # 空の応答の場合のフォールバック
            if not generated_text or len(generated_text.strip()) < 5:
                generated_text = "申し訳ございませんが、適切な応答を生成できませんでした。"
            
            return generated_text, inference_time
            
        except Exception as e:
            logger.error(f"テキスト生成エラー: {e}")
            return f"エラーが発生しました: {str(e)}", 0.0
    
    def _format_prompt(self, prompt: str) -> str:
        """プロンプトの標準化フォーマット"""
        return f"<|japanese|><|question|>{prompt}<|answer|>"
    
    async def generate_comparison(self, 
                                prompt: str, 
                                **generation_kwargs) -> Dict[str, Any]:
        """ベースモデルとファインチューニング済みモデルの並列比較"""
        if not model_manager.is_loaded():
            return {
                "status": "error",
                "message": "モデルが読み込まれていません。先にモデルを読み込んでください。"
            }
        
        try:
            # 並列実行でベースモデルとファインチューニング済みモデルの推論を実行
            loop = asyncio.get_event_loop()
            
            # partialを使用してキーワード引数を含む関数を作成
            base_generate_func = partial(
                self._generate_text,
                model_manager.base_model,
                model_manager.base_tokenizer,
                prompt,
                **generation_kwargs
            )
            
            finetuned_generate_func = partial(
                self._generate_text,
                model_manager.finetuned_model,
                model_manager.finetuned_tokenizer,
                prompt,
                **generation_kwargs
            )
            
            base_future = loop.run_in_executor(
                self.executor,
                base_generate_func
            )
            
            finetuned_future = loop.run_in_executor(
                self.executor,
                finetuned_generate_func
            )
            
            # 両方の結果を待機
            base_result, finetuned_result = await asyncio.gather(
                base_future, finetuned_future
            )
            
            base_text, base_time = base_result
            finetuned_text, finetuned_time = finetuned_result
            
            return {
                "status": "success",
                "prompt": prompt,
                "results": {
                    "base_model": {
                        "text": base_text,
                        "inference_time": base_time,
                        "character_count": len(base_text),
                        "model_name": settings.BASE_MODEL_NAME
                    },
                    "finetuned_model": {
                        "text": finetuned_text,
                        "inference_time": finetuned_time,
                        "character_count": len(finetuned_text),
                        "model_path": settings.FINETUNED_MODEL_PATH
                    }
                },
                "comparison": {
                    "speed_improvement": (base_time - finetuned_time) / base_time * 100 if base_time > 0 else 0,
                    "length_difference": len(finetuned_text) - len(base_text),
                    "total_time": max(base_time, finetuned_time)
                }
            }
            
        except Exception as e:
            logger.error(f"比較推論エラー: {e}")
            return {
                "status": "error",
                "message": f"比較推論でエラーが発生しました: {str(e)}"
            }
    
    async def generate_single(self, 
                            prompt: str, 
                            model_type: str = "finetuned",
                            **generation_kwargs) -> Dict[str, Any]:
        """単一モデルでの推論"""
        if not model_manager.is_loaded():
            return {
                "status": "error",
                "message": "モデルが読み込まれていません。"
            }
        
        try:
            if model_type == "base":
                model = model_manager.base_model
                tokenizer = model_manager.base_tokenizer
                model_name = settings.BASE_MODEL_NAME
            else:
                model = model_manager.finetuned_model
                tokenizer = model_manager.finetuned_tokenizer
                model_name = settings.FINETUNED_MODEL_PATH
            
            loop = asyncio.get_event_loop()
            
            # partialを使用してキーワード引数を含む関数を作成
            generate_func = partial(
                self._generate_text,
                model,
                tokenizer,
                prompt,
                **generation_kwargs
            )
            
            text, inference_time = await loop.run_in_executor(
                self.executor,
                generate_func
            )
            
            return {
                "status": "success",
                "prompt": prompt,
                "result": {
                    "text": text,
                    "inference_time": inference_time,
                    "character_count": len(text),
                    "model_name": model_name,
                    "model_type": model_type
                }
            }
            
        except Exception as e:
            logger.error(f"単一推論エラー: {e}")
            return {
                "status": "error",
                "message": f"推論でエラーが発生しました: {str(e)}"
            }

# グローバル推論サービスインスタンス
inference_service = InferenceService()
