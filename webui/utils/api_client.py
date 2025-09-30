#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI クライアント - Streamlit用
"""

import httpx
import asyncio
import logging
from typing import Dict, Any, Optional
import streamlit as st

logger = logging.getLogger(__name__)

class APIClient:
    """FastAPI クライアントクラス"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout = httpx.Timeout(120.0)  # 2分のタイムアウト
    
    async def _make_request(self, 
                          method: str, 
                          endpoint: str, 
                          json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """HTTP リクエストの実行"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}{endpoint}"
                
                if method.upper() == "GET":
                    response = await client.get(url)
                elif method.upper() == "POST":
                    response = await client.post(url, json=json_data)
                else:
                    raise ValueError(f"サポートされていないHTTPメソッド: {method}")
                
                response.raise_for_status()
                return response.json()
                
        except httpx.TimeoutException:
            logger.error(f"タイムアウト: {endpoint}")
            return {
                "status": "error",
                "message": "リクエストがタイムアウトしました。APIサーバーが起動していることを確認してください。"
            }
        except httpx.ConnectError:
            logger.error(f"接続エラー: {endpoint}")
            return {
                "status": "error",
                "message": "APIサーバーに接続できません。サーバーが起動していることを確認してください。"
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTPエラー {e.response.status_code}: {endpoint}")
            try:
                error_detail = e.response.json()
                return {
                    "status": "error",
                    "message": f"APIエラー: {error_detail.get('detail', str(e))}"
                }
            except:
                return {
                    "status": "error",
                    "message": f"HTTPエラー {e.response.status_code}: {str(e)}"
                }
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return {
                "status": "error",
                "message": f"予期しないエラーが発生しました: {str(e)}"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        return await self._make_request("GET", "/api/health")
    
    async def load_models(self) -> Dict[str, Any]:
        """モデルの読み込み"""
        return await self._make_request("POST", "/api/models/load")
    
    async def get_model_status(self) -> Dict[str, Any]:
        """モデル状態の取得"""
        return await self._make_request("GET", "/api/models/status")
    
    async def compare_models(self, 
                           prompt: str,
                           max_new_tokens: int = 256,
                           temperature: float = 0.7,
                           top_p: float = 0.9,
                           top_k: int = 50) -> Dict[str, Any]:
        """モデル比較推論"""
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
        return await self._make_request("POST", "/api/inference/compare", data)
    
    async def single_inference(self,
                              prompt: str,
                              model_type: str = "finetuned",
                              max_new_tokens: int = 256,
                              temperature: float = 0.7,
                              top_p: float = 0.9,
                              top_k: int = 50) -> Dict[str, Any]:
        """単一モデル推論"""
        data = {
            "prompt": prompt,
            "model_type": model_type,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
        return await self._make_request("POST", "/api/inference/single", data)
    
    async def get_settings(self) -> Dict[str, Any]:
        """設定情報の取得"""
        return await self._make_request("GET", "/api/settings")

    # Streamlit用の同期ラッパー関数
    def sync_health_check(self) -> Dict[str, Any]:
        """同期版ヘルスチェック"""
        return asyncio.run(self.health_check())
    
    def sync_load_models(self) -> Dict[str, Any]:
        """同期版モデル読み込み"""
        return asyncio.run(self.load_models())
    
    def sync_get_model_status(self) -> Dict[str, Any]:
        """同期版モデル状態取得"""
        return asyncio.run(self.get_model_status())
    
    def sync_compare_models(self, **kwargs) -> Dict[str, Any]:
        """同期版モデル比較"""
        return asyncio.run(self.compare_models(**kwargs))
    
    def sync_single_inference(self, **kwargs) -> Dict[str, Any]:
        """同期版単一推論"""
        return asyncio.run(self.single_inference(**kwargs))
    
    def sync_get_settings(self) -> Dict[str, Any]:
        """同期版設定取得"""
        return asyncio.run(self.get_settings())

@st.cache_resource
def get_api_client(base_url: str = "http://localhost:8000") -> APIClient:
    """APIクライアントのキャッシュされたインスタンスを取得"""
    return APIClient(base_url)

# デフォルトクライアント
api_client = get_api_client()
