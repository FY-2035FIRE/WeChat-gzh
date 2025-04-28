# _*_ coding: utf-8 _*_
import cv2
import numpy as np
import os
import datetime
import logging
import tempfile
import json
import time
import concurrent.futures
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List, Union, Callable
from moviepy.editor import VideoFileClip, AudioFileClip, vfx
from pydub import AudioSegment
import traceback

# 检查是否支持CUDA
CUDA_SUPPORT = cv2.cuda.getCudaEnabledDeviceCount() > 0

# 配置日志 - 使用JSON格式
class JsonFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# 配置日志
logger = logging.getLogger('video_processor')
logger.setLevel(logging.INFO)

# 添加控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(JsonFormatter())
logger.addHandler(console_handler)

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
    'chunk_size': 1024 * 1024,  # 1MB 分块大小
    'max_workers': 4,  # 并行处理的最大工作线程数
    'preview_duration': 5,  # 预览视频的时长（秒）
    'supported_video_formats': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
    'supported_audio_formats': ['.mp3', '.wav', '.aac', '.ogg', '.flac']
}

# 自定义异常类
class VideoProcessError(Exception):
    """视频处理错误基类"""
    pass

class FileFormatError(VideoProcessError):
    """文件格式错误"""
    pass

class ProcessingError(VideoProcessError):
    """处理过程错误"""
    pass

class ResourceError(VideoProcessError):
    """资源错误"""
    pass

# 处理器基类 - 责任链模式
class ProcessorHandler(ABC):
    """处理器基类，实现责任链模式"""
    
    def __init__(self, name: str):
        self.name = name
        self.next_handler = None
        
    def set_next(self, handler):
        """设置下一个处理器"""
        self.next_handler = handler
        return handler
        
    def process(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理并传递给下一个处理器"""
        try:
            logger.info(f"【{self.name}】处理开始")
            start_time = time.time()
            
            # 执行具体处理逻辑
            result = self.handle(file_info)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"【{self.name}】处理完成，耗时: {duration:.2f}秒")
            
            # 传递给下一个处理器
            if self.next_handler:
                return self.next_handler.process(result)
            return result
            
        except Exception as e:
            logger.error(f"【{self.name}】处理失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise ProcessingError(f"{self.name}处理失败: {str(e)}")
    
    @abstractmethod
    def handle(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """具体处理逻辑，由子类实现"""
        pass

# 工具类
class VideoUtils:
    """视频处理工具类"""
    
    @staticmethod
    def get_current_datetime(format: str = "%Y%m%d%H%M%S") -> str:
        """获取当前日期和时间，并格式化为指定的字符串格式"""
        current_time = datetime.datetime.now()
        formatted_datetime = current_time.strftime(format)
        return formatted_datetime
    
    @staticmethod
    def copy_file(input_path: str, output_path: str, chunk_size: int = 1024 * 1024) -> None:
        """
        复制文件，使用指定的块大小
        
        Args:
            input_path: 输入文件的路径
            output_path: 输出文件的路径
            chunk_size: 读取块大小，默认1MB
        """
        # 检查输入文件是否存在
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"输入文件 '{input_path}' 不存在")

        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 打开源文件（以二进制模式读取）
            with open(input_path, 'rb') as src_file:
                # 打开目标文件（以二进制模式写入）
                with open(output_path, 'wb') as dst_file:
                    # 分块读取并写入
                    while True:
                        chunk = src_file.read(chunk_size)
                        if not chunk:
                            break
                        dst_file.write(chunk)
            logger.info(f"文件已成功复制到 {output_path}")
        except Exception as e:
            logger.error(f"复制文件时出错: {e}")
            raise
    
    @staticmethod
    def is_supported_format(file_path: str, supported_formats: List[str]) -> bool:
        """
        检查文件格式是否受支持
        
        Args:
            file_path: 文件路径
            supported_formats: 支持的格式列表
            
        Returns:
            是否支持该格式
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext in supported_formats
    
    @staticmethod
    def create_preview(input_path: str, output_path: str, duration: int = 5) -> str:
        """
        创建视频预览
        
        Args:
            input_path: 输入视频路径
            output_path: 输出预览视频路径
            duration: 预览时长（秒）
            
        Returns:
            预览视频路径
        """
        try:
            # 使用ffmpeg创建预览视频
            cmd = [
                'ffmpeg', '-y', '-i', input_path, 
                '-t', str(duration), 
                '-c:v', 'libx264', '-c:a', 'aac',
                '-preset', 'ultrafast', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"创建预览视频失败: {e.stderr.decode()}")
            raise ProcessingError(f"创建预览视频失败: {e}")
    
    @staticmethod
    def retry_operation(operation: Callable, max_retries: int = 3, retry_delay: float = 1.0, *args, **kwargs):
        """
        重试操作
        
        Args:
            operation: 要重试的操作函数
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            *args, **kwargs: 传递给操作函数的参数
            
        Returns:
            操作结果
        """
        retries = 0
        last_exception = None
        
        while retries < max_retries:
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                retries += 1
                logger.warning(f"操作失败，正在重试 ({retries}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
        
        # 所有重试都失败
        logger.error(f"操作在 {max_retries} 次重试后仍然失败: {str(last_exception)}")
        raise last_exception

# 具体处理器实现
class PathProcessor(ProcessorHandler):
    """路径处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("路径处理")
        self.config = config
        
    def handle(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理文件路径并创建临时目录"""
        file_path = file_info.get('input_video_path')
        new_file_prefix = self.config.get('new_file_prefix', '')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        # 检查文件格式是否支持
        if not VideoUtils.is_supported_format(file_path, self.config.get('supported_video_formats')):
            raise FileFormatError(f"不支持的视频格式: {os.path.splitext(file_path)[1]}")
            
        base_name = os.path.basename(file_path)
        dir_path = os.path.dirname(file_path)
        file_name, file_extension = os.path.splitext(base_name)
        new_file_name = new_file_prefix + file_name

        # 使用tempfile创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="video_proc_", dir=dir_path)
        
        result = {
            'src_input_path': file_path,  # 原始视频路径
            'src_input_dir': dir_path,  # 原视频目录
            'src_input_file_name': file_name,   # 原视频名
            'src_input_file_extension': file_extension,     # 原视频后缀
            'cur_input_path': file_path,   # 当前待处理的视频路径
            'cur_input_audio_path': file_path,   # 当前待处理的音频路径 默认原视频
            'cur_step': '',  # 已经处理的步骤
            'cur_audio_step': '',  # 已经处理的步骤
            'output_temp_dir': temp_dir,  # 临时输出目录
            'output_new_file_name': new_file_name,  # 处理后最终要输出的文件名
            'output_new_file_path': os.path.join(dir_path, new_file_name) + file_extension,  # 最终输出路径
            'output_file_concat': self.config.get('file_concat'),  # 过程文件拼接步骤的符号
            'input_noise_path': file_info.get('input_noise_path'),  # 噪音文件路径
            'preview_path': os.path.join(temp_dir, f"preview_{file_name}{file_extension}"),  # 预览视频路径
            'width': 0,  # 视频宽度
            'height': 0,  # 视频高度
        }
        
        # 创建预览视频
        if self.config.get('create_preview', True):
            preview_duration = self.config.get('preview_duration', 5)
            VideoUtils.create_preview(file_path, result['preview_path'], preview_duration)
            logger.info(f"已创建预览视频: {result['preview_path']}")
            
        return result

class MetadataProcessor(ProcessorHandler):
    """元数据处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("清除元数据")
        self.config = config
        
    def handle(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """清除视频元数据"""
        step = file_info['cur_step'] + file_info['output_file_concat'] + self.name
        input_path = file_info['cur_input_path']
        temp_file_name = file_info['src_input_file_name'] + file_info['output_file_concat'] + step
        tmp_dir = file_info['output_temp_dir']
        output_path = os.path.join(tmp_dir, temp_file_name + file_info['src_input_file_extension'])

        # 使用subprocess代替os.system，更安全且可以捕获输出
        try:
            cmd = [
                'ffmpeg', '-y', '-i', input_path, 
                '-map_metadata', '-1', 
                '-c:v', 'copy', 
                '-c:a', 'copy', 
                output_path
            ]
            result = subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"元数据清除成功: {output_path}")
            
            # 更新文件信息
            file_info['cur_input_path'] = output_path
            file_info['cur_step'] = step
            
            return file_info
            
        except subprocess.CalledProcessError as e:
            logger.error(f"清除元数据失败: {e.stderr.decode()}")
            raise ProcessingError(f"清除元数据失败: {e}")

class AudioProcessor(ProcessorHandler):
    """音频处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("音频处理")
        self.config = config
        
    def handle(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理音频：偏移和添加噪音"""
        step = file_info['cur_audio_step'] + file_info['output_file_concat'] + '音频处理偏移+噪音'
        input_audio = file_info['cur_input_audio_path']
        input_noise_path = file_info['input_noise_path']
        tmp_dir = file_info['output_temp_dir']

        temp_file_name = file_info['src_input_file_name'] + file_info['output_file_concat'] + step
        output_path = os.path.join(tmp_dir, temp_file_name + file_info['output_file_concat'] + "processed_audio.mp3")

        # 检查音频文件是否存在
        if not os.path.exists(input_audio):
            raise FileNotFoundError(f"音频文件 '{input_audio}' 不存在。")
            
        # 检查噪音文件是否存在
        if not os.path.exists(input_noise_path):
            raise FileNotFoundError(f"噪音文件 '{input_noise_path}' 不存在。")

        # 使用重试机制处理音频
        def process_audio():
            # 加载音频
            audio = AudioSegment.from_file(input_audio)
            
            # 音调随机偏移
            pitch_shift_factor = np.random.uniform(0.95, 1.05)
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * pitch_shift_factor)
            })

            # 添加环境底噪
            noise_db_adjust = self.config.get('noise_db_adjust', -20)
            noise = AudioSegment.from_file(input_noise_path) + noise_db_adjust

            # 设置噪声的帧率和声道与音频一致
            noise = noise.set_frame_rate(audio.frame_rate).set_channels(audio.channels)
            
            # 重复噪音直到达到音频长度
            def repeat_until_duration(seg, duration):
                if len(seg) == 0:
                    return seg
                repeat_times = (duration // len(seg)) + 1
                return (seg * repeat_times)[:duration]
                
            noise = repeat_until_duration(noise, len(audio))

            # 叠加噪音
            processed_audio = audio.overlay(noise)
            
            # 导出处理后的音频
            processed_audio.export(output_path, format="mp3")
            return output_path
            
        # 使用重试机制
        max_retries = self.config.get('max_retries', 3)
        output_path = VideoUtils.retry_operation(process_audio, max_retries)
        
        # 更新文件信息
        file_info['cur_audio_output_path'] = output_path
        file_info['cur_audio_step'] = step
        
        return file_info

class FrameProcessor(ProcessorHandler):
    """帧处理器"""
    
    def __init__(self, config: Dict[str, Any], 
                 bg_color: Tuple[int, int, int] = (255, 222, 173),
                 bd_color: Tuple[int, int, int] = (0, 0, 0),
                 alpha: float = 0.05,
                 skip_interval: int = 30,
                 position: Tuple[int, int] = (50, 50),
                 font_scale: float = 1,
                 thickness: int = 1,
                 enable_speed: bool = True):
        super().__init__("视频帧处理")
        self.config = config
        self.bg_color = bg_color
        self.bd_color = bd_color
        self.alpha = alpha
        self.skip_interval = skip_interval
        self.position = position
        self.font_scale = font_scale
        self.thickness = thickness
        self.enable_speed = enable_speed
        
    def _process_frame(self, frame: np.ndarray, base_scale: float, 
                      time_str: str, width: int, height: int) -> np.ndarray:
        """
        处理单个视频帧
        """
        # 基础处理 - 缩放
        h, w = frame.shape[:2]
        scale = base_scale - 0.02
        new_w = max(int(w * scale) // 2 * 2, 2)
        new_h = max(int(h * scale) // 2 * 2, 2)

        # 使用CUDA加速（如果可用）
        if CUDA_SUPPORT:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_resized = cv2.cuda.resize(gpu_frame, (new_w, new_h))
            resized = gpu_resized.download()
        else:
            resized = cv2.resize(frame, (new_w, new_h))
            
        pad_vert = h - new_h
        pad_horz = w - new_w

        # 添加边框保持原始分辨率
        processed_frame = cv2.copyMakeBorder(
            resized,
            pad_vert // 2, pad_vert - pad_vert // 2,
            pad_horz // 2, pad_horz - pad_horz // 2,
            cv2.BORDER_CONSTANT, value=self.bd_color
        )

        # 深度处理 - 添加水印和背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 文字位置计算（右下角）
        (text_width, text_height), _ = cv2.getTextSize(time_str, font, self.font_scale, self.thickness)
        padding = 5
        x = width - text_width - self.position[0] - padding * 2
        y = height - self.position[1] - text_height - padding

        # 创建叠加层
        overlay = processed_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), self.bg_color, -1)

        # 混合背景
        cv2.addWeighted(overlay, self.alpha, processed_frame, 1 - self.alpha, 0, processed_frame)

        # 添加文字
        cv2.putText(processed_frame, time_str,
                    (x + padding, y + text_height + padding),
                    font, self.font_scale, (255, 255, 255), self.thickness, cv2.LINE_AA)
                    
        return processed_frame
        
    def _process_video_chunk(self, chunk_info):
        """处理视频的一个分块"""
        start_frame, end_frame, cap, output_path, base_scale, speed_factor, width, height = chunk_info
        
        # 创建输出视频对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS) * speed_factor), (width, height))
        
        # 跳到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_counter = 0
        frame_count = 0
        output_frame_duration = 1.0 / (cap.get(cv2.CAP_PROP_FPS) * speed_factor)
        
        # 处理分配的帧
        for i in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 帧删除逻辑
            frame_counter += 1
            if self.skip_interval > 0 and frame_counter % self.skip_interval == 0:
                continue
                
            # 时间戳计算
            current_time_sec = frame_count * output_frame_duration
            minutes = int(current_time_sec // 60)
            seconds = int(current_time_sec % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            # 处理帧
            processed_frame = self._process_frame(
                frame, base_scale, time_str, width, height
            )
            
            # 写入处理后的帧
            out.write(processed_frame)
            frame_count += 1
            
        out.release()
        return output_path, frame_count
        
    def handle(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """高级视频处理：缩放、边框填充和深度处理"""
        step = file_info['cur_step'] + file_info['output_file_concat'] + "缩放+边框填充+深度处理"
        input_path = file_info['cur_input_path']
        tmp_dir = file_info['output_temp_dir']
        temp_file_name = file_info['src_input_file_name'] + file_info['output_file_concat'] + step
        output_path = os.path.join(tmp_dir, temp_file_name + file_info['src_input_file_extension'])
        
        # 读取视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
            
        # 获取原始视频属性
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 更新文件信息中的视频尺寸
        file_info['width'] = width
        file_info['height'] = height
        
        # 初始化处理参数
        low_scale, high_scale = self.config.get('base_scale_range')
        low_speed, high_speed = self.config.get('speed_range')
        base_scale = np.random.uniform(low_scale, high_scale)
        speed_factor = np.random.uniform(low_speed, high_speed) if self.enable_speed else 1.0
        
        logger.info(f"初始化信息: 变速: {speed_factor:.2f} 缩放比例:{base_scale:.2f} "
                   f"背景颜色:{self.bg_color} 背景不透明度{self.alpha} 边框颜色:{self.bd_color}")
        
        # 并行处理视频帧
        max_workers = self.config.get('max_workers', 4)
        
        # 创建临时目录用于存储分块处理结果
        chunks_dir = os.path.join(tmp_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # 计算每个工作线程处理的帧数
        frames_per_worker = total_frames // max_workers
        if frames_per_worker < 100:  # 如果每个工作线程处理的帧数太少，则不使用并行处理
            max_workers = 1
            frames_per_worker = total_frames
            
        # 准备分块处理任务
        tasks = []
        for i in range(max_workers):
            start_frame = i * frames_per_worker
            end_frame = min((i + 1) * frames_per_worker, total_frames)
            chunk_output = os.path.join(chunks_dir, f"chunk_{i}.mp4")
            
            tasks.append((
                start_frame, end_frame, cv2.VideoCapture(input_path), 
                chunk_output, base_scale, speed_factor, width, height
            ))
            
        # 并行处理视频分块
        chunk_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_video_chunk, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_path, frame_count = future.result()
                    chunk_results.append((chunk_path, frame_count))
                    logger.info(f"视频分块处理完成: {chunk_path}, 处理了 {frame_count} 帧")
                except Exception as e:
                    logger.error(f"视频分块处理失败: {str(e)}")
                    raise
        
        # 合并视频分块
        if max_workers > 1:
            # 创建合并文件列表
            chunks_list_file = os.path.join(tmp_dir, "chunks_list.txt")
            with open(chunks_list_file, 'w') as f:
                for chunk_path, _ in chunk_results:
                    f.write(f"file '{chunk_path}'\n")
                    
            # 使用ffmpeg合并视频分块
            temp_video_path = os.path.join(tmp_dir, f"{temp_file_name}_temp.mp4")
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', chunks_list_file, '-c', 'copy', temp_video_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # 只有一个分块，直接使用
            temp_video_path = chunk_results[0][0]
        
        # 处理音频
        audio_output_path = file_info.get('cur_audio_output_path')
        if not audio_output_path:
            # 如果没有处理过的音频，则处理
            audio_processor = AudioProcessor(self.config)
            file_info = audio_processor.handle(file_info)
            audio_output_path = file_info.get('cur_audio_output_path')
            
        # 加载音频
        audio_clip = AudioFileClip(audio_output_path)
        
        # 音频速度调整
        if self.skip_interval > 0:
            actual_speed = speed_factor * (self.skip_interval / (self.skip_interval - 1))
            audio_clip = audio_clip.fx(vfx.speedx, actual_speed)
            
        # 合并视频和音频
        video_clip = VideoFileClip(temp_video_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, **OUTPUT_PARAMS)
        
        # 清理临时文件
        if self.config.get('remove_temp', True):
            try:
                # 删除分块文件
                for chunk_path, _ in chunk_results:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                        
                # 删除分块列表文件
                if os.path.exists(chunks_list_file):
                    os.remove(chunks_list_file)
                    
                # 删除临时合并视频
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                    
                # 删除分块目录
                if os.path.exists(chunks_dir):
                    shutil.rmtree(chunks_dir)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")
        
        # 更新文件信息
        file_info['cur_input_path'] = output_path
        file_info['cur_step'] = step
        
        return file_info

class FinalProcessor(ProcessorHandler):
    """最终处理器 - 负责最终输出和清理"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("最终处理")
        self.config = config
        
    def handle(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理最终输出并清理临时文件"""
        try:
            # 复制最终处理结果到目标路径
            final_output_path = file_info.get('output_new_file_path')
            current_path = file_info.get('cur_input_path')
            
            logger.info(f"正在复制最终处理结果到: {final_output_path}")
            VideoUtils.copy_file(current_path, final_output_path, 
                                chunk_size=self.config.get('chunk_size', 1024 * 1024))
            
            # 清理临时目录
            if self.config.get('remove_temp', True):
                temp_dir = file_info.get('output_temp_dir')
                try:
                    if os.path.exists(temp_dir):
                        logger.info(f"清理临时目录: {temp_dir}")
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"清理临时目录失败: {str(e)}")
            
            # 更新文件信息
            file_info['final_output_path'] = final_output_path
            
            return file_info
            
        except Exception as e:
            logger.error(f"最终处理失败: {str(e)}")
            raise ProcessingError(f"最终处理失败: {str(e)}")


class VideoProcessor:
    """视频处理类，使用责任链模式组织处理流程"""
    
    def __init__(self, config: Dict[str, Any] = None, output_params: Dict[str, Any] = None):
        """
        初始化视频处理器
        
        Args:
            config: 处理配置参数
            output_params: 输出参数配置
        """
        self.config = config or PROCESS_CONFIG
        self.output_params = output_params or OUTPUT_PARAMS
        
        # 初始化处理链
        self.chain = self._build_processing_chain()
        
    def _build_processing_chain(self) -> ProcessorHandler:
        """构建处理链"""
        # 路径处理器
        path_processor = PathProcessor(self.config)
        
        # 元数据处理器
        metadata_processor = MetadataProcessor(self.config)
        
        # 帧处理器
        frame_processor = FrameProcessor(
            self.config,
            bg_color=(255, 222, 173),
            bd_color=(0, 0, 0),
            alpha=0.05,
            skip_interval=30,
            position=(50, 50),
            font_scale=1,
            thickness=1,
            enable_speed=True
        )
        
        # 最终处理器
        final_processor = FinalProcessor(self.config)
        
        # 构建处理链
        path_processor.set_next(metadata_processor)
        metadata_processor.set_next(frame_processor)
        frame_processor.set_next(final_processor)
        
        return path_processor
    
    def process(self, input_video_path: str, input_noise_path: str) -> Tuple[str, int, int]:
        """
        处理视频
        
        Args:
            input_video_path: 输入视频路径
            input_noise_path: 输入噪音文件路径
            
        Returns:
            Tuple[str, int, int]: (处理后的视频路径, 视频宽度, 视频高度)
        """
        try:
            process_start_time = time.time()
            logger.info(f"开始处理视频: {input_video_path}")
            
            # 初始化文件信息
            file_info = {
                'input_video_path': input_video_path,
                'input_noise_path': input_noise_path
            }
            
            # 执行处理链
            result = self.chain.process(file_info)
            
            # 获取处理结果
            final_path = result.get('final_output_path', '')
            width = result.get('width', 0)
            height = result.get('height', 0)
            
            process_end_time = time.time()
            total_duration = process_end_time - process_start_time
            logger.info(f"视频处理完成，总耗时: {total_duration:.2f}秒")
            
            return final_path, width, height
            
        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}")
            logger.error(traceback.format_exc())
            return '', 0, 0
    
    def create_preview(self, input_path: str, duration: int = 5) -> str:
        """
        创建视频预览
        
        Args:
            input_path: 输入视频路径
            duration: 预览时长（秒）
            
        Returns:
            预览视频路径
        """
        try:
            # 获取输入视频的目录和文件名
            dir_path = os.path.dirname(input_path)
            file_name = os.path.basename(input_path)
            name, ext = os.path.splitext(file_name)
            
            # 创建预览视频路径
            preview_path = os.path.join(dir_path, f"preview_{name}{ext}")
            
            # 使用VideoUtils创建预览
            return VideoUtils.create_preview(input_path, preview_path, duration)
            
        except Exception as e:
            logger.error(f"创建预览视频失败: {str(e)}")
            raise


def main():
    """主函数"""
    # 要转换的视频源文件
    input_video_path = r"D:\tmp\t2.mp4"
    # 使用的噪音文件
    input_noise_path = r"D:\tmp\white_noise.mp3"
    
    # 创建处理器并执行处理
    processor = VideoProcessor()
    final_path, width, height = processor.process(input_video_path, input_noise_path)
    print(f'视频宽高: {width}x{height}')
    print(f'返回结果: {final_path}')
    
    # 创建预览视频
    try:
        preview_path = processor.create_preview(final_path, duration=5)
        print(f'预览视频: {preview_path}')
    except Exception as e:
        print(f'创建预览视频失败: {str(e)}')


if __name__ == '__main__':
    main()