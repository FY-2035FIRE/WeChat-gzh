# -*- coding: utf-8 -*-
from math import log
import traceback
import time
import random
import re
import json
import requests
from utils.mysql_db_utils import dataSource
from utils.feishu_bitable_utils import FeishuTable, FeishuTableException
from datetime import datetime
# 使用tqdm替代rich.progress作为进度条显示
from tqdm import tqdm

proxies = {
    "http": "http://syt8283719718:371170@101.91.243.187:1110",
    "https": "http://syt8283719718:371170@101.91.243.187:1110"
}
# 初始化
# fs = FeishuTable()

def get_all_html(url, headers, params, max_retries=3):
    check_nums = 0
    while check_nums < max_retries:
        try:
            time.sleep(random.randint(38, 69))
            response = requests.get(url, headers=headers, params=params, timeout=20, proxies=proxies)
            return response.text
        except Exception as e:
            print(traceback.print_exc())
            print(f"get_all_html：check_nums={check_nums} 请求页信息出错 原因：{e} url={url} json_data={params}")
            check_nums += 1
    
    print("get_all_html 多次请求失败，", url, headers, params)
    return None


def get_wechat_details_info(id_, d_info, cookies, headers, url_home, check_nums=0):
    try:
        if check_nums >= 3:
            print("get_wechat_details_info 多次请求失败，", d_info)
            return
        url = "https://mp.weixin.qq.com/s"
        title = d_info.get('title')
        content_url = d_info.get('content_url')
        date_time = d_info.get('date_time')
        cover = d_info.get('cover')
        date_time_ = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(date_time))

        __biz = re.findall(r"__biz=(.*?)&", content_url)[0]
        mid = re.findall(r"mid=(.*?)&", content_url)[0]
        idx = re.findall(r"idx=(.*?)&", content_url)[0]
        sn = re.findall(r"sn=(.*?)&", content_url)[0]
        chksm = re.findall(r"chksm=(.*?)&", content_url)[0]
        scene = re.findall(r"scene=(.*?)#", content_url)[0]

        key = re.findall(r"key=(.*?)&", url_home)[0]
        uin = re.findall(r"uin=(.*?)&", url_home)[0]
        pass_ticket = re.findall(r"pass_ticket=(.*?)&", url_home)[0]
        version = re.findall(r"version=(.*?)&", url_home)[0]
        devicetype = re.findall(r"devicetype=(.*?)&", url_home)[0]

        params = {
            "__biz": __biz,
            "mid": mid,
            "idx": idx,
            "sn": sn,
            "chksm": chksm,
            "scene": scene,
            "key": key,
            "ascene": "1",
            "uin": uin,
            "devicetype": devicetype,
            "version": version,
            "lang": "zh_CN",
            "acctmode": "0",
            "pass_ticket": pass_ticket,
            "wx_header": "1"
        }

        html = get_all_html(url, headers, params)

        if html is None:
            print("get_wechat_details_info 多次请求失败，", d_info)
            return

        try:
            title = re.findall(r"var title = '(.*?)';", html)[0]
        except:
            title = title
        try:
            nickname = re.findall(r'var nickname = htmlDecode\("(.*?)"\);', html)[0]
        except:
            nickname = ""
        try:
            user_name = re.findall(r'var user_name = "(.*?)";', html)[0]
        except:
            user_name = ""
        try:
            createTime = re.findall(r"var createTime = '(.*?)';", html)[0]
        except:
            createTime = ""
        try:
            provinceName = re.findall(r"provinceName: '(.*?)'", html)[0]
        except:
            provinceName = ""
        try:
            hit_nickname = re.findall(r"hit_nickname: '(.*?)'", html)[0]
            if hit_nickname == "" or hit_nickname is None:
                hit_nickname = "原创"
        except:
            hit_nickname = "原创"
        try:
            read_num_new_ = re.findall(r"var read_num_new = '(.*?)'", html)
            if len(read_num_new_) == 0:
                read_num = re.findall(r"var read_num = '(.*?)'", html)[0]
            else:
                read_num = read_num_new_[0]
        except:
            read_num = 0
        try:
            msg_link = re.findall(r'msg_link = "(.*?)";', html)[0]
        except:
            msg_link = ""
        try:
            like_count = re.findall(r"old_like_count: '(.*?)'", html)[0]
        except:
            like_count = 0
        try:
            share_count = re.findall(r"share_count: '(.*?)'", html)[0]
        except:
            share_count = 0
        try:
            comment_count = re.findall(r"comment_count: '(.*?)'", html)[0]
        except:
            comment_count = 0
        try:
            is_mp_video = re.findall(r"is_mp_video: '(.*?)'", html)[0]
        except:
            is_mp_video = 0

        if "video_id: '" in html:
            article_type = "视频"
        else:
            article_type = "图文"

        # res_dict = {
        #     "title": title,
        #     "content_url": msg_link,
        #     "cover_url": cover,
        #     "uuid": sn,

        #     "nickname": nickname,
        #     "user_name": user_name,
        #     "publish_time": createTime,
        #     "province_name": provinceName,
        #     "hit_nickname": hit_nickname,
        #     "article_type": article_type,
        #     "read_num": read_num,
        #     "share_count": share_count,
        #     "like_count": like_count,
        #     "comment_count": comment_count,
        # }
        # print("res_dict = ", res_dict)
        if int(read_num) >= 2000:
            if article_type == "视频" and is_mp_video == "0":
                print(f"[1] 《{title}》阅读量大于 2000，且不可下载的视频文章不入库...")
                return
            print(f"[1] 《{title}》阅读量大于 2000，开始入库...")
            send_gzh_article_data(title, msg_link,cover,sn,nickname,user_name,provinceName,createTime,hit_nickname,article_type,read_num,like_count,share_count,comment_count)
        else:
            print(f"[0] 《{title}》阅读量小于 2000，属于低质量文章不入库...")

    except Exception as e:
        print(traceback.print_exc(), e)
        check_nums += 1
        get_wechat_details_info(id_, d_info, cookies, headers, url_home, check_nums)


def d_info_dict_data_list(id_, dict_data_list, headers, cookies, url_home):

    for d_info in dict_data_list:
        get_wechat_details_info(id_, d_info, cookies, headers, url_home)
       
    sql = """update llz_wechat_gzh_mitm set usage_record='1' where id = {};""".format(id_)
    dataSource.execute_update(sql)


def get_db_data():
    empty_count = 0  # 记录空结果次数
    while True:
        check_sql = """SELECT * FROM llz_wechat_gzh_mitm WHERE usage_record=0 limit 100;"""
        res = dataSource.execute_query(check_sql)
        
        if res is None or len(res) == 0:
            empty_count += 1
            if empty_count%20 == 0 or empty_count == 1:
                # 使用tqdm创建时间进度条，显示程序运行时长
                for _ in tqdm(range(100), desc="程序自检中"):
                    time.sleep(0.1)
            else:
                time.sleep(10)
            continue
        # 重置计数器
        empty_count = 0

        for r in res:
            id_ = r['id']
            biz_ = r['biz']
            dict_data_list = json.loads(r['dict_data_list'])
            headers = json.loads(r['headers'])
            cookies = json.loads(r['cookies'])
            url_home = r['url_home']
            create_time = r['create_time']

            target_time = datetime.strptime(str(create_time), "%Y-%m-%d %H:%M:%S")
            time_difference = datetime.now() - target_time
            seconds_difference = abs(int(time_difference.total_seconds()))
            if seconds_difference > 30*60:
                sql = """update llz_wechat_gzh_mitm set usage_record='2' where id = {};""".format(id_)
                dataSource.execute_update(sql)
                print("本批数据存在 时间大于 30 分钟，cookies可能已过期，跳过")
                continue
            print('当前在追踪%s 创建的第 %s 批次的 %s 篇文章信息.....' %(create_time,id_,len(dict_data_list)))
            d_info_dict_data_list(id_, dict_data_list, headers, cookies, url_home)


def send_gzh_article_data(title, content_url,cover_url,uuid,nickname,user_name,province,publish_time,hit_nickname,article_type,read_num,like_count,share_count,comment_count):
    tab_keys = "title, content_url,cover_url,uuid,nickname,user_name,province,publish_time,hit_nickname,article_type,read_num,like_count,share_count,comment_count"    
    tab_len = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"
    sql = """INSERT INTO llz_wechat_gzh_article ({})VALUES({})ON DUPLICATE KEY UPDATE read_num = VALUES(read_num), like_count = VALUES(like_count), share_count = VALUES(share_count), comment_count = VALUES(comment_count)""".format(tab_keys, tab_len)
    tab_values = (title, content_url,cover_url,uuid,nickname,user_name,province,publish_time,hit_nickname,article_type,read_num,like_count,share_count,comment_count)
    dataSource.execute_insert(sql, tab_values)

if __name__ == '__main__':
    
    get_db_data()
