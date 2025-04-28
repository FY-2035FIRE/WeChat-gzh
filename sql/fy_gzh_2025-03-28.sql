-- --------------------------------------------------------
-- 主机:                           127.0.0.1
-- 服务器版本:                        5.7.22-log - MySQL Community Server (GPL)
-- 服务器操作系统:                      Win64
-- HeidiSQL 版本:                  12.5.0.6677
-- --------------------------------------------------------

-- 导出 fy_fire2035 的数据库结构
CREATE DATABASE IF NOT EXISTS `fy_fire2035` /*!40100 DEFAULT CHARACTER SET utf8mb4 */;
USE `fy_fire2035`;

-- 导出  表 fy_fire2035.llz_wechat_gzh_article 结构
CREATE TABLE `llz_wechat_gzh_article` (
	`id` INT(11) NOT NULL AUTO_INCREMENT,
	`title` VARCHAR(100) NOT NULL COMMENT '标题' COLLATE 'utf8mb4_general_ci',
	`content_url` LONGTEXT NOT NULL COMMENT '文章URL' COLLATE 'utf8mb4_general_ci',
	`cover_url` LONGTEXT NOT NULL COMMENT '封面URL' COLLATE 'utf8mb4_general_ci',
	`uuid` VARCHAR(50) NOT NULL COMMENT '公众号取sn值' COLLATE 'utf8mb4_general_ci',
	`nickname` VARCHAR(50) NOT NULL COMMENT '名称' COLLATE 'utf8mb4_general_ci',
	`user_name` VARCHAR(50) NULL DEFAULT NULL COMMENT '账号' COLLATE 'utf8mb4_general_ci',
	`province` VARCHAR(50) NOT NULL COMMENT 'IP城市' COLLATE 'utf8mb4_general_ci',
	`publish_time` TIMESTAMP NULL DEFAULT NULL COMMENT '文章发布时间',
	`hit_nickname` VARCHAR(50) NOT NULL COMMENT '引用来源' COLLATE 'utf8mb4_general_ci',
	`article_type` VARCHAR(50) NOT NULL COMMENT '文章类型' COLLATE 'utf8mb4_general_ci',
	`read_num` INT(11) NOT NULL COMMENT '阅读量',
	`like_count` INT(11) NOT NULL COMMENT '点赞量',
	`share_count` INT(11) NOT NULL COMMENT '分享量',
	`comment_count` INT(11) NOT NULL COMMENT '评论量',
	`is_del` TINYINT(4) NOT NULL DEFAULT '0',
	`create_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	`update_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	PRIMARY KEY (`id`) USING BTREE,
	UNIQUE INDEX `uuid` (`uuid`) USING BTREE,
	INDEX `idx_title_type` (`title`, `article_type`) USING BTREE,
	INDEX `idx_publish_time` (`publish_time`) USING BTREE,
	INDEX `idx_is_del` (`is_del`) USING BTREE
)
COMMENT='公众号文章'
COLLATE='utf8mb4_general_ci'
ENGINE=InnoDB
;


-- 导出  表 fy_fire2035.llz_wechat_gzh_mitm 结构
CREATE TABLE `llz_wechat_gzh_mitm` (
	`id` BIGINT(20) UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增id',
	`biz` VARCHAR(50) NOT NULL COMMENT '公众号业务标识' COLLATE 'utf8_general_ci',
	`dict_data_list` LONGTEXT NULL DEFAULT NULL COMMENT '公众号主页列表信息' COLLATE 'utf8_general_ci',
	`headers` LONGTEXT NULL DEFAULT NULL COMMENT '公众号headers' COLLATE 'utf8_general_ci',
	`cookies` LONGTEXT NOT NULL COMMENT '公众号cookies' COLLATE 'utf8_general_ci',
	`url_home` LONGTEXT NOT NULL COMMENT '公众号主页链接' COLLATE 'utf8_general_ci',
	`create_time` DATETIME NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
	`update_time` DATETIME NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
	`usage_record` INT(11) NULL DEFAULT NULL COMMENT '未使用0，使用1，过期2',
	PRIMARY KEY (`id`) USING BTREE,
	INDEX `idx_usage_record` (`usage_record`) USING BTREE
)
COMMENT='微信公众号'
COLLATE='utf8_general_ci'
ENGINE=InnoDB
;


CREATE TABLE `llz_publish_rule` (
	`id` INT(11) NOT NULL,
	`llz_name` VARCHAR(50) NOT NULL COMMENT '流量主名称' COLLATE 'utf8_general_ci',
	`p_platform` VARCHAR(50) NOT NULL COMMENT '发布平台' COLLATE 'utf8_general_ci',
	`m_type` VARCHAR(50) NOT NULL COMMENT '素材类型：视频、图文、话题' COLLATE 'utf8_general_ci',
	`m_number` INT(11) NOT NULL DEFAULT '0' COMMENT '每次发布数量',
	`create_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	`update_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	PRIMARY KEY (`id`) USING BTREE,
	INDEX `idx_m_type` (`m_type`) USING BTREE,
	INDEX `idx_llz_platform` (`llz_name`, `p_platform`) USING BTREE
)
COMMENT='发布规则'
COLLATE='utf8_general_ci'
ENGINE=InnoDB
;


CREATE TABLE `llz_material_libs` (
	`id` INT(11) NOT NULL AUTO_INCREMENT,
	`title` VARCHAR(100) NOT NULL COMMENT '标题' COLLATE 'utf8_general_ci',
	`o_type` VARCHAR(50) NOT NULL COMMENT '素材类型：视频、图文、话题' COLLATE 'utf8_general_ci',
	`o_content` LONGTEXT NOT NULL COMMENT '原素材链接' COLLATE 'utf8_general_ci',
	`o_cover` LONGTEXT NULL DEFAULT NULL COMMENT '原封面链接' COLLATE 'utf8_general_ci',
	`n_title` VARCHAR(100) NULL DEFAULT NULL COMMENT '新标题（30字内）' COLLATE 'utf8_general_ci',
	`n_type` VARCHAR(50) NULL DEFAULT NULL COMMENT '新素材类型：视频、图文、话题' COLLATE 'utf8_general_ci',
	`n_content` LONGTEXT NULL DEFAULT NULL COMMENT '新素材链接' COLLATE 'utf8_general_ci',
	`n_cover` LONGTEXT NULL DEFAULT NULL COMMENT '新封面链接' COLLATE 'utf8_general_ci',
	`n_tags` VARCHAR(150) NULL DEFAULT NULL COMMENT '标签、话题' COLLATE 'utf8_general_ci',
	`n_info_ext` VARCHAR(500) NULL DEFAULT NULL COMMENT '扩展信息：分辨率、码率、备注等' COLLATE 'utf8_general_ci',
	`is_laundty` TINYINT(4) NOT NULL DEFAULT '0' COMMENT '是否完成洗稿',
	`is_publish` TINYINT(4) NOT NULL DEFAULT '0' COMMENT '是否发布',
	`create_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	`update_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	PRIMARY KEY (`id`) USING BTREE,
	INDEX `idx_title_otype` (`title`, `o_type`) USING BTREE,
	INDEX `idx_ntype_status` (`n_type`, `is_laundty`, `is_publish`) USING BTREE,
	INDEX `idx_update_time` (`update_time`) USING BTREE
)
COMMENT='素材库'
COLLATE='utf8_general_ci'
ENGINE=InnoDB
;


