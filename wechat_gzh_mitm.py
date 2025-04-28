# -*- coding: utf-8 -*-
import mitmproxy.http
import re
import json
from utils.mysql_db_utils import dataSource
from datetime import datetime, timedelta


def save_db_data(dict_data, headers, cookies, url_home):
    tab_keys = "biz,dict_data_list,headers,cookies,url_home,usage_record"
    tab_len = "%s,%s,%s,%s,%s,%s"
    sql = """INSERT INTO llz_wechat_gzh_mitm ({})VALUES({})""".format(tab_keys, tab_len)
    # 执行插入操作
    __biz = re.findall(r"__biz=(.*?)&", url_home)[0]
    tab_values = (__biz,json.dumps(dict_data), json.dumps(headers), json.dumps(cookies), url_home, 0)
    dataSource.execute_insert(sql, tab_values)


def parse_data_to_json(html, cookies, headers, url_home):
    try:
        
        # 使用更安全的方式提取msgList
        msg_list_match = re.search(r"var msgList = '(.*?)';", html)
        if not msg_list_match:
            print("未找到msgList数据")
            return
        
        msg_list = msg_list_match.group(1)
        # 简化字符串替换
        # data_msg_list = msg_list.replace("\n", '').replace("&quot;", '"')
        data_msg_list = msg_list.replace("\n", '').replace("&quot;", '"').replace("&quot;:", '').replace("&quot;,", '')
        
        # 直接解析JSON，避免多余的序列化步骤
        try:
            data_res = json.loads(data_msg_list)
        except json.JSONDecodeError:
            print("JSON解析失败")
            return
            
        list_ = data_res.get("list", [])
        if not list_:
            print("文章列表为空")
            return
            
        dict_data = []
        for article in list_:
            try:
                # print(f"正在处理article = {article}")
                # 提取文章信息
                if extract_article_info(article, dict_data) is False:
                    break
            except Exception as e:
                print(f"解析文章错误：{e}")
                
        if dict_data:
            save_db_data(dict_data, headers, cookies, url_home)
        else:
            print("未提取到有效文章数据")
            
    except Exception as e:
        print(f"解析数据失败：{type(e).__name__}, {e}")

def extract_article_info(article, dict_data):
    """提取单篇文章信息"""
    comm_msg_info = article.get('comm_msg_info', {})
    date_time = comm_msg_info.get('datetime', "")

    # 获取3天前的时间戳，只处理3天内的文章
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    last_collect_time = int((today - timedelta(days=3)).timestamp())
    date_time_ = int(date_time)
    if date_time_ < last_collect_time:
        print("无需追踪发表时间超过 3 天的文章...")
        return False
    
    app_msg_ext_info = article.get('app_msg_ext_info')
    if not app_msg_ext_info:
        print("app_msg_ext_info 为空")
        return
        
    # 处理主文章
    process_single_article(app_msg_ext_info, date_time, dict_data)
    
    # 处理多篇文章列表
    multi_articles = app_msg_ext_info.get('multi_app_msg_item_list', [])
    for sub_article in multi_articles:
        process_single_article(sub_article, date_time, dict_data)


def process_single_article(article_info, date_time, dict_data):
    """处理单个文章数据"""
    title = article_info.get('title', '')
    content_url = article_info.get('content_url', '')
    cover = article_info.get('cover', '')
    item_show_type = article_info.get('item_show_type', '')
    print(f"title = {title}, content_url = {content_url}, cover = {cover}, item_show_type = {item_show_type}")
    if item_show_type != 0 or not content_url or not title or not cover:
        print(f"文章信息不完整，跳过")
        return
    
    try:
        sn_match = re.search(r"sn=(.*?)&", content_url)
        sn = sn_match.group(1) if sn_match else ""
        
        dict_data.append({
            "date_time": date_time, 
            "title": title, 
            "content_url": content_url, 
            "cover": cover, 
            "sn": sn
        })
    except Exception as e:
        print(f"处理文章URL错误：{e}, URL: {content_url}")


class GetWeChatData:
    try:
        def response(self, flow: mitmproxy.http.HTTPFlow):
            url_home = flow.request.url
            # print(url_home)
            # 过滤不需要的数据包
            if 'https://mp.weixin.qq.com/mp/profile_ext?action=home&__biz=' not in url_home:
                return
            print("================================================")
            # print("1", flow.request.data)
            # print("2", flow.request.cookies)
            # print("3", flow.request.headers)
            # print("4", flow.response.text)
            # # print("5", flow.request.query)
            # # print("5", flow.request.content)
            # # print("5", flow.request.text)
            # # print("5", flow.request.json())

            print(f"url_home = {url_home}") 

            cookies = {}
            ck = flow.request.cookies
            ck_res = [cookies.update({i[0]: i[1]}) for i in ck.items()]
            print(f"cookies = {cookies}")

            headers = {}
            hd = flow.request.headers
            hd_res = [headers.update({i[0]: i[1]}) for i in hd.items()]
            print(f"headers = {headers}")

            if flow.response is None:
                print("响应为空")
                return
            html = flow.response.text
            
            parse_data_to_json(html, cookies, headers, url_home)

            print("================================================")
    except ConnectionRefusedError:
        print("目标拒绝连接")
    except ConnectionResetError:
        print("目标连接重置")
    except Exception as e:
        print(f"{type(e).__name__}", e)


addons = [
    GetWeChatData()
]


