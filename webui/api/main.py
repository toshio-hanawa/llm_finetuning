#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI メインアプリケーション
"""

import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

from webui.api.models import model_manager
from webui.api.inference import inference_service
from webui.utils.metrics import metrics_calculator
from webui.config.settings import settings

# ログ設定
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# FastAPIアプリケーション
app = FastAPI(
    title="🤖 LLM比較API",
    description="ベースモデルとファインチューニング済みモデルの比較API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストモデル
class CompareRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

class SingleInferenceRequest(BaseModel):
    prompt: str
    model_type: str = "finetuned"  # "base" or "finetuned"
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

# APIエンドポイント
@app.get("/api/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": model_manager.is_loaded()
    }

@app.post("/api/models/load")
async def load_models(background_tasks: BackgroundTasks):
    """モデルの読み込み"""
    if model_manager.is_loaded():
        return {
            "status": "already_loaded",
            "message": "モデルは既に読み込み済みです"
        }
    
    # バックグラウンドでモデルを読み込み
    result = await model_manager.load_models()
    return result

@app.get("/api/models/status")
async def get_model_status():
    """モデルの状態取得"""
    return model_manager.get_model_info()

@app.post("/api/inference/compare")
async def compare_models(request: CompareRequest):
    """モデル比較推論"""
    try:
        # 生成パラメータの準備
        generation_kwargs = {}
        if request.max_new_tokens:
            generation_kwargs["max_new_tokens"] = request.max_new_tokens
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            generation_kwargs["top_k"] = request.top_k
        
        # 推論実行
        result = await inference_service.generate_comparison(
            request.prompt, 
            **generation_kwargs
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["message"])
        
        # メトリクス計算
        base_text = result["results"]["base_model"]["text"]
        finetuned_text = result["results"]["finetuned_model"]["text"]
        
        metrics = metrics_calculator.compare_texts(base_text, finetuned_text)
        result["metrics"] = metrics
        
        return result
        
    except Exception as e:
        logger.error(f"比較推論エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/single")
async def single_inference(request: SingleInferenceRequest):
    """単一モデル推論"""
    try:
        # 生成パラメータの準備
        generation_kwargs = {}
        if request.max_new_tokens:
            generation_kwargs["max_new_tokens"] = request.max_new_tokens
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            generation_kwargs["top_k"] = request.top_k
        
        # 推論実行
        result = await inference_service.generate_single(
            request.prompt,
            request.model_type,
            **generation_kwargs
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["message"])
        
        # メトリクス計算
        text = result["result"]["text"]
        metrics = metrics_calculator.calculate_text_metrics(text)
        result["metrics"] = metrics
        
        return result
        
    except Exception as e:
        logger.error(f"単一推論エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings")
async def get_settings():
    """設定情報の取得"""
    return {
        "base_model_name": settings.BASE_MODEL_NAME,
        "finetuned_model_path": settings.FINETUNED_MODEL_PATH,
        "device": settings.DEVICE,
        "max_length": settings.MAX_LENGTH,
        "default_temperature": settings.TEMPERATURE,
        "default_top_p": settings.TOP_P,
        "default_top_k": settings.TOP_K
    }

# スタートアップイベント
@app.on_event("startup")
async def startup_event():
    """アプリケーション開始時の処理"""
    logger.info("🚀 LLM比較API が開始されました")
    logger.info(f"設定: {settings.BASE_MODEL_NAME} -> {settings.FINETUNED_MODEL_PATH}")
    logger.info(f"デバイス: {settings.DEVICE}")

@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    logger.info("🛑 LLM比較API が終了されました")

def main():
    """FastAPIサーバーの起動"""
    uvicorn.run(
        "webui.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()
