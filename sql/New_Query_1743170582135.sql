-- --------------------------------------------------------
-- 主机:                           127.0.0.1
-- 服务器版本:                        5.7.22-log - MySQL Community Server (GPL)
-- 服务器操作系统:                      Win64
-- HeidiSQL 版本:                  12.5.0.6677
-- --------------------------------------------------------

-- 视图 fy_fire2035.v_fy_gzh_publish_task 结构
-- 要求：
    -- 1、将素材库按n_type进行内部变量编号。
    -- 2、根据发布规则中素材数量m_number和对应的n_type值，取出内部变量编号相应m_number个素材记录（不同流量主取编号时是累加状态）。
    -- 3、素材只取已经洗稿并待发布的,素材类型n_type = m_type。
-- 发布规则如：
    -- id:1,llz_name:lc,p_platform:今日头条,m_type:视频,m_number:2
    -- id:2,llz_name:sxf,p_platform:今日头条,m_type:视频,m_number:2
-- 查询结果：
    -- llz_name:lc,p_platform:今日头条,m_type:视频,n_title:视频标题1,n_content:视频链接1,n_cover:视频封面1,n_tags:视频标签1
    -- llz_name:lc,p_platform:今日头条,m_type:视频,n_title:视频标题2,n_content:视频链接2,n_cover:视频封面2,n_tags:视频标签2
CREATE VIEW v_fy_gzh_publish_task AS
SELECT 
    r.llz_name,
    r.p_platform,
    m.id AS m_id,
    m.n_type, 
    m.n_title,
    m.n_content,
    m.n_cover,
    m.n_tags,
    m.n_info_ext, 
    m.update_time
FROM 
    llz_publish_rule r
JOIN 
    (
        SELECT 
            ml.*,
            (
                SELECT COUNT(*) + 1 
                FROM llz_material_libs ml2 
                WHERE ml2.n_type = ml.n_type 
                    AND ml2.update_time > ml.update_time 
                    AND ml2.is_laundty = 1 
                    AND ml2.is_publish = 0
                    
            ) AS row_num
        FROM 
            llz_material_libs ml
        WHERE 
            ml.is_laundty = 1 
            AND ml.is_publish = 0
    ) m ON m.n_type = r.m_type
JOIN 
    (
        SELECT 
            r1.id,
            r1.m_type,
            (
                SELECT IFNULL(SUM(r2.m_number), 0) 
                FROM llz_publish_rule r2 
                WHERE r2.m_type = r1.m_type AND r2.id <= r1.id
            ) AS end_num,
            (
                SELECT IFNULL(SUM(r2.m_number), 0) 
                FROM llz_publish_rule r2 
                WHERE r2.m_type = r1.m_type AND r2.id < r1.id
            ) + 1 AS start_num
        FROM 
            llz_publish_rule r1
    ) ranges ON r.id = ranges.id
WHERE 
    m.row_num BETWEEN ranges.start_num AND ranges.end_num
ORDER BY 
    r.llz_name,r.p_platform,r.m_type;
 

CREATE VIEW v_fy_gzh_article_median_stats AS
SELECT 
    title,
    cnt AS article_count,
    AVG(median_value) AS median_read_num,
    MAX(content_url) AS content_url,
    MAX(cover_url) AS cover_url
FROM (
    SELECT 
        a.title,
        a.read_num AS median_value,
        a.content_url,
        a.cover_url,
        (SELECT COUNT(*) FROM llz_wechat_gzh_article WHERE title = a.title AND article_type = '视频') AS cnt,
        (SELECT COUNT(*) FROM llz_wechat_gzh_article WHERE title = a.title AND article_type = '视频' AND read_num <= a.read_num) AS position
    FROM 
        llz_wechat_gzh_article a        
        LEFT JOIN llz_material_libs t2 ON 
		    t2.o_type='视频' AND 
		    (
		        -- 计算两个标题的相似度
		        (LENGTH(a.title) - ABS(LENGTH(a.title) - LENGTH(t2.title))) / LENGTH(a.title) > 0.9 AND
		        (
		            -- 检查是否有共同的前缀（至少80%相同）
		            SUBSTRING(a.title, 1, FLOOR(LENGTH(a.title) * 0.8)) = 
		            SUBSTRING(t2.title, 1, FLOOR(LENGTH(a.title) * 0.8))
		            OR
		            -- 或者编辑距离近似（使用简化的方法）
		            (LENGTH(a.title) - LENGTH(REPLACE(a.title, SUBSTRING(t2.title, 1, 10), ''))) / 10 > 0.8
		        )
		    ) 
    WHERE 
        a.article_type = '视频'
        AND a.is_del = 0
        AND t2.id IS NULL
) AS ranked
WHERE 
    (cnt % 2 = 1 AND position = (cnt + 1) / 2) OR
    (cnt % 2 = 0 AND (position = cnt / 2 OR position = cnt / 2 + 1))
GROUP BY 
    title
ORDER BY median_read_num DESC

