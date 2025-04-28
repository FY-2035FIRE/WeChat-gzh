# _*_ coding: utf-8 _*_
import cv2
import numpy as np
import os
import datetime
import logging
import tempfile
from typing import Dict, Tuple, Any, Optional, List, Union
from moviepy.editor import VideoFileClip, AudioFileClip, vfx
from pydub import AudioSegment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('video_processor')

# 输出参数配置
OUTPUT_PARAMS = {
    'remove_temp': True,
    'codec': 'libx264',
    'audio_codec': 'aac',
    'audio_bufsize': 2000,
    'threads': 4,  # 优化多线程
    'preset': 'medium',
    'ffmpeg_params': ['-movflags', '+faststart']
}

# 处理参数
PROCESS_CONFIG = {
    'base_scale_range': (0.95, 0.99),
    'speed_range': (0.95, 1.05),
    'noise_db_adjust': -20,
    'max_retries': 3,
    'temp_dir_name': 'video_temp',
    'new_file_prefix': 'new_',
    'file_concat': '_',
}


class VideoProcessor:
    """视频处理类，提供视频处理的各种功能"""

    def __init__(self, config: Dict = None, output_params: Dict = None):
        """
        初始化视频处理器
        
        Args:
            config: 处理配置参数
            output_params: 输出参数配置
        """
        self.config = config or PROCESS_CONFIG
        self.output_params = output_params or OUTPUT_PARAMS
        
    def get_current_datetime(self, format: str = "%Y%m%d%H%M%S") -> str:
        """
        获取当前日期和时间，并格式化为指定的字符串格式。
        
        Args:
            format: 格式化字符串，默认为 "%Y%m%d%H%M%S"
            
        Returns:
            格式化后的日期和时间字符串
        """
        current_time = datetime.datetime.now()
        formatted_datetime = current_time.strftime(format)
        return formatted_datetime
    
    def copy_file(self, input_path: str, output_path: str) -> None:
        """
        复制文件
        
        Args:
            input_path: 输入文件的路径
            output_path: 输出文件的路径
        """
        # 检查输入文件是否存在
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"输入文件 '{input_path}' 不存在")

        try:
            # 打开源文件（以二进制模式读取）
            with open(input_path, 'rb') as src_file:
                # 打开目标文件（以二进制模式写入）
                with open(output_path, 'wb') as dst_file:
                    # 分块读取并写入（例如，每次读取 1024 字节）
                    while True:
                        chunk = src_file.read(1024)
                        if not chunk:
                            break
                        dst_file.write(chunk)
            logger.info(f"文件已成功复制到 {output_path}")
        except Exception as e:
            logger.error(f"复制文件时出错: {e}")
            raise

    def process_path(self, file_path: str, new_file_prefix: str) -> Dict[str, str]:
        """
        处理文件路径并创建临时目录
        
        Args:
            file_path: 输入文件路径
            new_file_prefix: 新文件前缀
            
        Returns:
            包含处理路径信息的字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        base_name = os.path.basename(file_path)
        dir_path = os.path.dirname(file_path)
        file_name, file_extension = os.path.splitext(base_name)
        new_file_name = new_file_prefix + file_name

        # 创建临时目录
        temp_dir = os.path.join(dir_path, self.config.get("temp_dir_name"))
        os.makedirs(temp_dir, exist_ok=True)
        
        return {
            'src_input_path': file_path,  # 原始视频路径
            'src_input_dir': dir_path,  # 原视频目录
            'src_input_file_name': file_name,   # 原视频名
            'src_input_file_extension': file_extension,     # 原视频后缀
            'cur_input_path': file_path,   # 当前待处理的视频路径
            'cur_input_audio_path': file_path,   # 当前待处理的音频路径 默认原视频
            'cur_step': '',  # 已经处理的步骤
            'cur_audio_step': '',  # 已经处理的步骤
            'output_temp_dir': os.path.join(dir_path, self.config.get('temp_dir_name')),  # 临时输出目录
            'output_new_file_name': new_file_name,  # 处理后最终要输出的文件名
            'output_new_file_path': os.path.join(dir_path, new_file_name) + file_extension,  # 最终输出路径
            'output_file_concat': self.config.get('file_concat')  # 过程文件拼接步骤的符号
        }

    def metadata_process(self, file_info: Dict[str, str]) -> Tuple[str, str]:
        """
        清除视频元数据
        
        Args:
            file_info: 文件信息字典
            
        Returns:
            (输出路径, 当前步骤)
        """
        op_name = '清除元数据'
        try:
            step = file_info['cur_step'] + file_info['output_file_concat'] + op_name
            input_path = file_info['cur_input_path']
            temp_file_name = file_info['src_input_file_name'] + file_info['output_file_concat'] + step
            tmp_dir = file_info['output_temp_dir']
            output_path = tmp_dir + os.sep + temp_file_name + file_info['src_input_file_extension']

            logger.info(f"{op_name} 处理开始")
            os.system(f"ffmpeg -y -i \"{input_path}\" -map_metadata -1 -c:v copy -c:a copy \"{output_path}\"")
            logger.info(f"{op_name} 处理完成")
            return output_path, step
        except Exception as e:
            logger.error(f"{op_name} 处理失败 错误信息 {e}")
            raise

    def _process_frame(self, frame: np.ndarray, base_scale: float, 
                      bg_color: Tuple[int, int, int], bd_color: Tuple[int, int, int],
                      alpha: float, time_str: str, width: int, height: int,
                      position: Tuple[int, int], font_scale: float, thickness: int) -> np.ndarray:
        """
        处理单个视频帧
        
        Args:
            frame: 原始帧
            base_scale: 缩放比例
            bg_color: 背景颜色
            bd_color: 边框颜色
            alpha: 透明度
            time_str: 时间戳字符串
            width: 帧宽度
            height: 帧高度
            position: 水印位置
            font_scale: 字体大小
            thickness: 字体粗细
            
        Returns:
            处理后的帧
        """
        # 基础处理 - 缩放
        h, w = frame.shape[:2]
        scale = base_scale - 0.02
        new_w = max(int(w * scale) // 2 * 2, 2)
        new_h = max(int(h * scale) // 2 * 2, 2)

        resized = cv2.resize(frame, (new_w, new_h))
        pad_vert = h - new_h
        pad_horz = w - new_w

        # 添加边框保持原始分辨率
        processed_frame = cv2.copyMakeBorder(
            resized,
            pad_vert // 2, pad_vert - pad_vert // 2,
            pad_horz // 2, pad_horz - pad_horz // 2,
            cv2.BORDER_CONSTANT, value=bd_color
        )

        # 深度处理 - 添加水印和背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 文字位置计算（右下角）
        (text_width, text_height), _ = cv2.getTextSize(time_str, font, font_scale, thickness)
        padding = 5
        x = width - text_width - position[0] - padding * 2
        y = height - position[1] - text_height - padding

        # 创建叠加层
        overlay = processed_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), bg_color, -1)

        # 混合背景
        cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)

        # 添加文字
        cv2.putText(processed_frame, time_str,
                    (x + padding, y + text_height + padding),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    
        return processed_frame

    def advanced_process_video(self, file_info: Dict[str, str] = None, 
                              bg_color: Tuple[int, int, int] = (255, 218, 185),
                              alpha: float = 0.05, skip_interval: int = 30,
                              position: Tuple[int, int] = (50, 50), 
                              font_scale: float = 1,
                              thickness: int = 1, 
                              enable_speed: bool = True, 
                              bd_color: Tuple[int, int, int] = (153, 51, 255)) -> Tuple[str, str, int,int]:
        """
        高级视频处理：缩放、边框填充和深度处理
        
        Args:
            file_info: 文件信息字典
            bg_color: 背景颜色RGB元组
            alpha: 透明度值
            skip_interval: 跳帧间隔
            position: 水印位置
            font_scale: 字体大小
            thickness: 字体粗细
            enable_speed: 是否启用变速
            bd_color: 边框颜色
            
        Returns:
            (输出路径, 当前步骤)
        """
        op_name = '缩放+边框填充+深度处理'
        try:
            if not file_info:
                raise ValueError("file_info 参数缺失")
                
            # 初始化路径和配置
            step = file_info['cur_step'] + file_info['output_file_concat'] + op_name
            input_path = file_info['cur_input_path']
            tmp_dir = file_info['output_temp_dir']
            temp_file_name = file_info['src_input_file_name'] + file_info['output_file_concat'] + step
            output_path = tmp_dir + os.sep + temp_file_name + file_info['src_input_file_extension']
            temp_video_path = tmp_dir + os.sep + temp_file_name + "_temp.mp4"

            # 读取视频
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")

            # 获取原始视频属性
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 初始化处理参数
            low_scale, high_scale = self.config.get('base_scale_range')
            low_speed, high_speed = self.config.get('speed_range')
            base_scale = np.random.uniform(low_scale, high_scale)
            speed_factor = np.random.uniform(low_speed, high_speed) if enable_speed else 1.0

            # 速度控制逻辑
            target_fps = original_fps * speed_factor
            output_frame_duration = 1.0 / target_fps

            # 创建输出视频对象
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, int(target_fps), (width, height))

            # 帧处理控制变量
            frame_counter = 0
            frame_count = 0
            
            logger.info(f"初始化信息结束 变速: {speed_factor:.2f} 缩放比例:{base_scale:.2f} "
                       f"背景颜色:{bg_color} 背景不透明度{alpha} 边框颜色:{bd_color}")
            logger.info(f"{op_name} opencv 帧处理开始")

            # 主处理循环
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 帧删除逻辑
                frame_counter += 1
                if skip_interval > 0 and frame_counter % skip_interval == 0:
                    continue

                # 时间戳计算
                current_time_sec = frame_count * output_frame_duration
                minutes = int(current_time_sec // 60)
                seconds = int(current_time_sec % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"

                # 处理帧
                processed_frame = self._process_frame(
                    frame, base_scale, bg_color, bd_color, alpha, 
                    time_str, width, height, position, font_scale, thickness
                )

                # 写入处理后的帧
                out.write(processed_frame)
                frame_count += 1

            cap.release()
            out.release()
            logger.info(f"{op_name} opencv 帧处理并写入临时文件完成")

            # 音频处理
            audio_output_path, audio_cur_step = self.audio_processing(file_info)
            audio_clip = AudioFileClip(audio_output_path)
            
            # 音频速度调整
            if skip_interval > 0:
                actual_speed = speed_factor * (skip_interval / (skip_interval - 1))
                audio_clip = audio_clip.fx(vfx.speedx, actual_speed)

            # 合并视频和音频
            video_clip = VideoFileClip(temp_video_path)
            logger.info(f"{op_name} movepy 生成视频文件开始")
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(output_path, **self.output_params)
            logger.info(f"{op_name} movepy 生成视频文件结束")
            logger.info(f"{op_name} 处理完成！输出文件已保存至: {output_path}")
            
            # 清理临时文件
            try:
                if os.path.exists(temp_video_path) and self.output_params.get('remove_temp', True):
                    os.remove(temp_video_path)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")
                
            return output_path, step, width, height

        except Exception as e:
            logger.error(f"{op_name} 处理失败 错误信息: {str(e)}")
            raise

    def repeat_until_duration(self, seg: AudioSegment, duration: int) -> AudioSegment:
        """
        重复音频片段直到达到指定时长
        
        Args:
            seg: 音频片段
            duration: 目标时长
            
        Returns:
            处理后的音频片段
        """
        if len(seg) == 0:
            return seg
        repeat_times = (duration // len(seg)) + 1
        return (seg * repeat_times)[:duration]

    def audio_processing(self, file_info: Dict[str, str]) -> Tuple[str, str]:
        """
        音频处理：偏移和添加噪音
        
        Args:
            file_info: 文件信息字典
            
        Returns:
            (输出路径, 当前步骤)
        """
        op_name = '音频处理偏移噪音'
        try:
            logger.info(f"{op_name} 开始")
            step = file_info['cur_audio_step'] + file_info['output_file_concat'] + '音频处理偏移+噪音'
            input_audio = file_info['cur_input_audio_path']
            input_noise_path = file_info['input_noise_path']
            tmp_dir = file_info['output_temp_dir']

            temp_file_name = file_info['src_input_file_name'] + file_info['output_file_concat'] + step
            output_path = tmp_dir + os.sep + temp_file_name + file_info['output_file_concat'] + "processed_audio.mp3"

            # 检查音频文件是否存在
            if not os.path.exists(input_audio):
                raise FileNotFoundError(f"音频文件 '{input_audio}' 不存在。")

            # 处理音频
            logger.info(f"{op_name} 开始处理音频")
            try:
                logger.info(f"{op_name} 加载音频信息")
                audio = AudioSegment.from_file(input_audio)
            except Exception as e:
                raise ValueError(f"加载音频文件 '{input_audio}' 时出错: {e}")

            # 音调随机偏移
            logger.info(f"{op_name} 音调随机偏移")
            pitch_shift_factor = np.random.uniform(0.95, 1.05)
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * pitch_shift_factor)
            })

            logger.info(f"{op_name} 添加环境底噪")
            # 添加环境底噪 - 生产模式
            noise_db_adjust = self.config.get('noise_db_adjust', -20)
            noise = AudioSegment.from_file(input_noise_path) + noise_db_adjust

            # 设置噪声的帧率和声道与音频一致
            logger.info(f"{op_name} 设置噪声的帧率和声道与音频一致")
            noise = noise.set_frame_rate(audio.frame_rate).set_channels(audio.channels)
            noise = self.repeat_until_duration(noise, len(audio))

            processed_audio = audio.overlay(noise)
            logger.info(f"{op_name} 导出处理完的音频信息到 【{output_path}】")
            processed_audio.export(output_path, format="mp3")
            logger.info(f"{op_name} 导出处理完的音频信息完成")
            
            return output_path, step

        except Exception as e:
            logger.error(f"{op_name} 处理失败 错误信息 {e}")
            raise

    def auto_process(self, input_video_path: str, input_noise_path: str) -> Tuple[str,int,int]:
        """
        视频处理入口
        
        Args:
            input_video_path: 输入视频路径
            input_noise_path: 输入噪音文件路径
            
        Returns:
            处理后的视频路径
        """
        try:
            process_start_time = datetime.datetime.now()
            logger.info(f"开始处理视频: {input_video_path}")

            # 初始化文件信息
            _file_info = self.process_path(input_video_path, self.config.get('new_file_prefix', ''))
            _file_info['input_noise_path'] = input_noise_path
            
            # 定义处理步骤
            video_steps = ['清除元数据', '关键帧+缩放+边框填充+深度处理']

            # 执行处理步骤
            for step in video_steps:
                logger.info(f"【{step}】 【开始】处理【视频】【{_file_info.get('src_input_file_name')}】")
                start_time = datetime.datetime.now()

                if step == '清除元数据':
                    output_path, cur_step = self.metadata_process(_file_info)
                    _file_info['cur_input_path'] = output_path
                    _file_info['cur_step'] = cur_step
                elif step == '关键帧+缩放+边框填充+深度处理':
                    bd_color = (0, 0, 0)
                    bg_color = (255, 222, 173)
                    output_path, cur_step,width, height = self.advanced_process_video(
                        file_info=_file_info, bg_color=bg_color, bd_color=bd_color, alpha=0.05
                    )
                    _file_info['cur_input_path'] = output_path
                    _file_info['cur_step'] = cur_step
                else:
                    logger.warning(f"暂不支持该操作 {step}")
                    
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"【{step}】 【结束】处理【视频】【{_file_info.get('src_input_file_name')}】 耗时: {duration:.2f} 秒")

            # 合并音视频
            start_time = datetime.datetime.now()
            final_out_path = _file_info.get('output_new_file_path')
            self.copy_file(_file_info['cur_input_path'], final_out_path)

            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"【合并音视频】 【结束】 耗时: {duration:.2f} 秒")

            total_duration = (end_time - process_start_time).total_seconds()
            logger.info(f"【任务总耗时】: {total_duration:.2f} 秒")

            return final_out_path,width,height
            
        except Exception as e:
            logger.error(f"视频处理失败: {e}", exc_info=True)
            logger.error("本次处理异常 返回空字符串")
            return '',0,0


def main():
    """主函数"""
    # 要转换的视频源文件
    input_video_path = r"D:\tmp\t2.mp4"
    # 使用的噪音文件
    input_noise_path = r"D:\tmp\white_noise.mp3"
    
    # 创建处理器并执行处理
    processor = VideoProcessor()
    final_path,width,height = processor.auto_process(input_video_path, input_noise_path)
    print(f'视频宽高: {width}x{height}')
    print(f'返回结果: {final_path}')


if __name__ == '__main__':
    main()