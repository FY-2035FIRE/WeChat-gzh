# -*- coding: utf-8 -*-
import pymysql
from dbutils.pooled_db import PooledDB
import logging

_host = 't.mysql.test.yao.com'
_port = 3306
_user = 'ccvs'
_password = 'd41d8cd98f00b204'
_database = 'ccvs'


class DataSource:
    _instance = None
    _is_init = False  # 添加初始化标志

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    # 初始化连接池
    def __init__(self, minconn=5, maxconn=20):
        # 避免重复初始化
        if self._is_init:
            return

        # 在DataSource类中添加日志功能
        self.logger = logging.getLogger('mysql_db')
        self.logger.setLevel(logging.INFO)
        # 避免重复添加处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # 优化连接池配置
        self.pool = PooledDB(
            creator=pymysql,
            mincached=minconn,     # 最小空闲连接数
            maxcached=maxconn,     # 最大空闲连接数
            maxshared=10,          # 添加共享连接数
            maxconnections=100,    # 最大连接数
            blocking=True,         # 连接池满时是否阻塞
            maxusage=10000,        # 单个连接最大使用次数
            setsession=[],         # 初始化连接的命令
            host=_host,
            port=_port,
            user=_user,
            password=_password,
            database=_database,
            charset='utf8mb4',     # 明确指定字符集
            autocommit=True
        )
        self._is_init = True
        self.logger.info("数据库连接池初始化完成")

    def get_connection(self):
        """获取数据库连接"""
        return self.pool.connection()

    def get_pool_status(self):
        """获取连接池状态"""
        status = {
            "size": self.pool._connections,
            "idle": len(self.pool._idle_cache),
            "shared": len(self.pool._shared_cache) if hasattr(self.pool, '_shared_cache') else 0
        }
        self.logger.info(f"连接池状态: {status}")
        return status

    def execute_query(self, sql, params=None):
        conn = self.pool.connection()
        cursor = conn.cursor()
        result_dict = []
        try:
            self.logger.debug(f"执行查询: {sql}, 参数: {params}")
            cursor.execute(sql, params)
            result = cursor.fetchall()
            column_names = [i[0] for i in cursor.description]
    
            for row in result:
                result_dict.append(dict(zip(column_names, row)))
        except Exception as e:
            self.logger.error(f"查询执行错误: {e}, SQL: {sql}")
            result_dict = None
        finally:
            cursor.close()
            conn.close()
    
        return result_dict

    def execute_update(self, sql, params=None):
        conn = self.pool.connection()
        cursor = conn.cursor()
        affected_rows = 0

        try:
            self.logger.debug(f"执行更新: {sql}, 参数: {params}")
            cursor.execute(sql, params)
            conn.commit()
            affected_rows = cursor.rowcount
            self.logger.info(f"更新执行成功: {affected_rows} 行受影响")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"更新执行错误: {e}, SQL: {sql}")
        finally:
            cursor.close()
            conn.close()

        return affected_rows

    def execute_update_batch(self, sql, params=None):
        conn = self.pool.connection()
        cursor = conn.cursor()
        affected_rows = 0

        try:
            self.logger.debug(f"执行批量更新: {sql}, 参数数量: {len(params) if params else 0}")
            cursor.executemany(sql, params)
            conn.commit()
            affected_rows = cursor.rowcount
            self.logger.info(f"批量更新执行成功: {affected_rows} 行受影响")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"批量更新执行错误: {e}, SQL: {sql}")
        finally:
            cursor.close()
            conn.close()
        
        return affected_rows

    def execute_insert(self, sql, params=None):
        conn = self.pool.connection()
        cursor = conn.cursor()
        last_insert_id = -1

        try:
            self.logger.debug(f"执行插入: {sql}, 参数: {params}")
            cursor.execute(sql, params)
            conn.commit()
            affected_rows = cursor.rowcount
            last_insert_id = cursor.lastrowid  # 获取自动生成的主键值
            self.logger.info(f"插入执行成功: {affected_rows} 行受影响, ID: {last_insert_id}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"插入执行错误: {e}, SQL: {sql}")
        finally:
            cursor.close()
            conn.close()

        return last_insert_id

    def execute_delete(self, sql, params=None):
        conn = self.pool.connection()
        cursor = conn.cursor()
        affected_rows = 0

        try:
            self.logger.debug(f"执行删除: {sql}, 参数: {params}")
            cursor.execute(sql, params)
            conn.commit()
            affected_rows = cursor.rowcount
            self.logger.info(f"删除执行成功: {affected_rows} 行受影响")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"删除执行错误: {e}, SQL: {sql}")
        finally:
            cursor.close()
            conn.close()
        
        return affected_rows

    def execute_transaction(self, operations):
        """
        执行事务操作
        :param operations: 包含SQL和参数的操作列表，格式为 [(sql1, params1), (sql2, params2), ...]
        :return: 是否成功
        """
        conn = self.pool.connection()
        cursor = conn.cursor()
        success = False
        
        try:
            conn.begin()  # 开始事务
            self.logger.debug(f"开始执行事务，操作数量: {len(operations)}")
            
            for sql, params in operations:
                cursor.execute(sql, params)
                
            conn.commit()
            self.logger.info("事务执行成功")
            success = True
        except Exception as e:
            conn.rollback()
            self.logger.error(f"事务执行错误: {e}")
        finally:
            cursor.close()
            conn.close()
        
        return success

dataSource = DataSource()
