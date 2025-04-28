# -*- coding: utf-8 -*-
import requests
import json
import traceback
import time
from typing import Dict, Any, Optional, Union


class FeishuTableException(Exception):
    """飞书表格操作异常"""
    pass


class FeishuTable:
    def __init__(self, app_id: str = "cli_a7be125a83bc500b", 
                 app_secret: str = "MZmenmnuDlCdBnTYLV8nGfZR0k37NRf0",
                 base_id: str = "IwuXbGyVtaUl9pscR5McjEjtnje"):
        """
        初始化飞书表格操作类
        Args:
            app_id: 飞书应用ID
            app_secret: 飞书应用密钥
            base_id: 多维表格ID
        """
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_id = base_id
        self.fs_token = None
        self.proxies = {
            "http": "http://syt8296695317:300761@123.187.240.43:6141",
            "https": "http://syt8296695317:300761@123.187.240.43:6141"
        }
        # API接口地址
        self._api_base = "https://open.feishu.cn/open-apis/bitable/v1"
        self._token_api = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"

    def _check_response(self, response: requests.Response, operation: str) -> Dict:
        """检查API响应"""
        try:
            result = response.json()
            # 检查token是否失效
            if result.get("code") == 99991663:  # token失效的错误码
                self.fs_token = None
                return {"code": 99991663, "msg": "token失效"}
            if result.get("code", 0) != 0:
                error_msg = f"{operation}失败: {result.get('msg', '')}"
                raise FeishuTableException(error_msg)
            return result
        except json.JSONDecodeError:
            raise FeishuTableException(f"{operation}响应解析失败: {response.text}")

    def _get_headers(self) -> Dict:
        """获取请求头"""
        if not self.fs_token:
            self.get_access_token()
        return {
            'Authorization': f'Bearer {self.fs_token}',
            'Content-Type': 'application/json; charset=utf-8',
        }

    def get_access_token(self, check_count: int = 0) -> Optional[str]:
        """获取飞书访问令牌"""
        try:
            if check_count > 5:
                raise FeishuTableException("获取access token重试次数过多")
            
            post_data = {"app_id": self.app_id, "app_secret": self.app_secret}
            response = requests.post(self._token_api, json=post_data, proxies=self.proxies, timeout=30)
            result = self._check_response(response, "获取access token")
            
            self.fs_token = result["tenant_access_token"]
            return self.fs_token
        except Exception as e:
            print(traceback.format_exc())
            if check_count < 5:
                time.sleep(1)
                return self.get_access_token(check_count + 1)
            raise FeishuTableException(f"获取access token失败: {str(e)}")

    def get_table_records(self, table_id: str,
                         page_size: int = 100) -> Dict[str, Any]:
        """
        获取表格记录
        Args:
            table_id: 表格ID
            page_size: 每页记录数
        Returns:
            包含记录列表和分页信息的字典
        """
        try:
            params = {
                "page_size": page_size
            }

            url = f"{self._api_base}/apps/{self.base_id}/tables/{table_id}/records"
            response = requests.get(
                url, 
                params=params, 
                headers=self._get_headers(), 
                proxies=self.proxies, 
                timeout=30
            )
            
            result = self._check_response(response, "获取记录")
            return result.get("data", {})
        except Exception as e:
            raise FeishuTableException(f"获取表格记录失败: {str(e)}")

    def search_records(self, table_id: str, filter_conditions, 
                      page_size: int = 100) -> Dict[str, Any]:
        """
        搜索表格记录
        Args:
            table_id: 表格ID
            filter_conditions: 过滤条件
            page_size: 每页记录数
        Returns:
            搜索结果
        """
        try:
            json_data = {
                "page_size": page_size,
                "filter": filter_conditions
                # "view_id": "vew6NlVSqS"
            }
   
            url = f"{self._api_base}/apps/{self.base_id}/tables/{table_id}/records/search"
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=json_data,
                proxies=self.proxies,
                timeout=30
            )
            
            result = self._check_response(response, "搜索记录")
            return result.get("data", {})
        except Exception as e:
            raise FeishuTableException(f"搜索表格记录失败: {str(e)}")

    def update_record(self, table_id: str, record_id: str, 
                     fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新表格记录
        Args:
            table_id: 表格ID
            record_id: 记录ID
            fields: 要更新的字段
        Returns:
            更新后的记录
        """
        try:
            url = f"{self._api_base}/apps/{self.base_id}/tables/{table_id}/records/{record_id}"
            response = requests.put(
                url,
                headers=self._get_headers(),
                json={"fields": fields},
                proxies=self.proxies,
                timeout=30
            )
            
            result = self._check_response(response, "更新记录")
            return result.get("data", {})
        except Exception as e:
            raise FeishuTableException(f"更新表格记录失败: {str(e)}")

    def create_record(self, table_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建表格记录
        Args:
            table_id: 表格ID
            fields: 字段值
        Returns:
            创建的记录
        """
        try:
            url = f"{self._api_base}/apps/{self.base_id}/tables/{table_id}/records"
            response = requests.post(
                url,
                headers=self._get_headers(),
                json={"fields": fields},
                proxies=self.proxies,
                timeout=30
            )
            
            result = self._check_response(response, "创建记录")
            return result.get("data", {})
        except Exception as e:
            raise FeishuTableException(f"创建表格记录失败: {str(e)}")

    def batch_create_records(self, table_id: str, records: list) -> Dict[str, Any]:
        """
        批量创建记录
        Args:
            table_id: 表格ID
            records: 记录列表，每个记录是一个字段字典
        Returns:
            创建结果
        """
        try:
            url = f"{self._api_base}/apps/{self.base_id}/tables/{table_id}/records/batch_create"
            response = requests.post(
                url,
                headers=self._get_headers(),
                json={"records": [{"fields": r} for r in records]},
                proxies=self.proxies,
                timeout=30
            )
            
            result = self._check_response(response, "批量创建记录")
            return result.get("data", {})
        except Exception as e:
            raise FeishuTableException(f"批量创建记录失败: {str(e)}")

if __name__ == '__main__':
    # 初始化
    fs = FeishuTable()

    try:
        # 获取记录
        # records = fs.get_table_records("tbltQGa9A4BvMuQP")
        # print(records)
        
        # 分页搜索
        filter_condition = {
            "conjunction": "and",
            "conditions": [
                {
                    "field_name": "账号ID",
				    "operator": "is",
				    "value": [
					    "MzkwODcwOTE2MQe=="
				        ] 
                },
            ]
        }
        results = fs.search_records("tbltQGa9A4BvMuQP", filter_condition)
        print(results)
        # 获取第一条记录的最后采集文章信息时间
        last_collect_time = int(results.get("items", [])[0].get("fields", {}).get("最后采集文章信息时间", "").ljust(13, '0'))
        print(f"最后采集文章信息时间: {last_collect_time}")
        date_time = int(str("1739116800").ljust(13, '0'))
        if last_collect_time>date_time:
            print("需要更新")

        
        # # 批量创建记录
        # records_to_create = [
        #     {"字段1": "值1", "字段2": "值2"},
        #     {"字段1": "值3", "字段2": "值4"}
        # ]
        # fs.batch_create_records("table_id", records_to_create)
        
    except FeishuTableException as e:
        print(f"操作失败: {e}")