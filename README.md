# 公众号流量主项目

这个项目用于爬取微信公众号文章数据，支持获取文章的标题、内容、阅读量、分享量、点赞量等数据。

## 技术栈

- Python 3.8+
- MySQL 数据库
- mitmproxy (用于网络请求拦截)
- 飞书多维表格 API (用于数据存储和展示)

## 数据库结构

主要数据表：

1. `articles` - 文章基本信息
   - id: 主键
   - title: 文章标题
   - content: 文章内容
   - publish_time: 发布时间
   - url: 文章链接

2. `article_stats` - 文章统计数据
   - article_id: 关联articles表的id
   - read_count: 阅读数
   - like_count: 点赞数
   - share_count: 分享数
   - update_time: 数据更新时间

## 核心功能

1. 文章数据采集
   - 通过mitmproxy拦截微信公众号请求
   - 自动提取文章内容和统计数据
   - 支持批量采集历史文章

2. 数据存储和同步
   - MySQL本地存储
   - 飞书多维表格云端备份
   - 定时同步数据

3. 数据分析
   - 文章数据趋势分析
   - 阅读量、点赞量等指标统计
   - 生成数据报表

## 使用说明

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 配置数据库
   - 导入`fy_gzh_2025-03-28.sql`初始化数据库
   - 修改`utils/mysql_db_utils.py`中的数据库配置

3. 启动采集
   ```bash
   # mitmdump -p 8080 -s wechat_gzh_mitm.py -q
   # mitmdump -s wechat_gzh_mitm.py -q
   ```