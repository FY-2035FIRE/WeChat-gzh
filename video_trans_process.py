# _*_ coding: utf-8 _*_
import cv2
import numpy as np
from moviepy.editor import *
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import datetime

### 基础方法 start ###
### 基础方法 start ###
### 基础方法 start ###

def get_current_datetime(format="%Y%m%d%H%M%S"):
    """
    获取当前日期和时间，并格式化为指定的字符串格式。
    参数:
        format (str): 格式化字符串，默认为 "%Y-%m-%d %H:%M:%S"。
    返回:
        str: 格式化后的日期和时间字符串。
    """
    current_time = datetime.datetime.now()
    formatted_datetime = current_time.strftime(format)
    return formatted_datetime
def copy_file(input_path, output_path):
    """
    参数:
        input_path (str): 输入文件的路径。
        output_path (str): 输出文件的路径。
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
        print(f"文件已成功复制到 {output_path}")
    except Exception as e:
        print(f"复制文件时出错: {e}")

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


### 基础方法 end ###
### 基础方法 end ###
### 基础方法 end ###

def process_path(file_path: str, new_file_prefix: str):
    """处理文件路径并创建临时目录"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    base_name = os.path.basename(file_path)
    dir_path = os.path.dirname(file_path)
    file_name, file_extension = os.path.splitext(base_name)
    new_file_name = new_file_prefix + file_name

    # 创建临时目录
    temp_dir = os.path.join(dir_path, PROCESS_CONFIG.get("temp_dir_name"))
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
        'output_temp_dir': os.path.join(dir_path, PROCESS_CONFIG.get('temp_dir_name')),  # 临时输出目录 用于存放过程中产生的文件
        'output_new_file_name': new_file_name,  # 处理后最终要输出的文件名
        'output_new_file_path': os.path.join(dir_path, new_file_name) + file_extension,  # 处理后最终要输出的文件名
        'output_file_concat': PROCESS_CONFIG.get('file_concat')  # 过程文件拼接步骤的 符号
    }



def metadata_process(file_info):
    """ 清除元数据 """
    op_name = '清除元数据'
    try:
        step = file_info['cur_step'] + file_info['output_file_concat'] + '清除元数据'
        input_path = file_info['cur_input_path']
        temp_file_name = file_info['src_input_file_name'] + file_info['output_file_concat'] + step
        tmp_dir = file_info['output_temp_dir']
        output_path = tmp_dir + os.sep + temp_file_name + file_info['src_input_file_extension']

        os.system(f"ffmpeg -y -i {input_path} -map_metadata -1 -c:v copy -c:a copy {output_path}")
        return output_path, step
    except Exception as e:
        print(f"{op_name} 处理失败 错误信息 {e}")
        raise e

def advanced_process_video(file_info=None, bg_color=(255, 218, 185),
                           alpha=0.05, skip_interval=30,
                           position=(50, 50), font_scale=1,
                           thickness=1, enable_speed=True, bd_color=(153,51,255)):
    op_name = '缩放+边框填充+深度处理'
    try:
        if not file_info:
            raise Exception("file_info 参数缺失")
        # 初始化路径和配置（保持原有逻辑）
        step = file_info['cur_step'] + file_info['output_file_concat'] + op_name
        input_path = file_info['cur_input_path']
        tmp_dir = file_info['output_temp_dir']
        temp_file_name = file_info['src_input_file_name'] + file_info['output_file_concat'] + step
        output_path = tmp_dir + os.sep + temp_file_name + file_info['src_input_file_extension']

        # 读取视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        # 获取原始视频属性
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化处理参数
        low_scale, high_scale = PROCESS_CONFIG.get('base_scale_range')
        low_speed, high_speed = PROCESS_CONFIG.get('speed_range')
        base_scale = np.random.uniform(low_scale, high_scale)
        speed_factor = np.random.uniform(low_speed, high_speed) if enable_speed else 1.0

        # ===== 改进的速度控制核心逻辑 =====
        target_fps = original_fps * speed_factor  # 计算目标帧率
        frame_interval = 1.0 / original_fps  # 原始帧间隔时间
        output_frame_duration = 1.0 / target_fps  # 目标帧间隔时间

        # 创建输出视频对象（使用调整后的帧率）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = tmp_dir + os.sep + temp_file_name + "_temp.mp4"
        out = cv2.VideoWriter(temp_video_path, fourcc, int(target_fps), (width, height))


        # ===== 新增帧删除控制变量 =====
        frame_counter = 0         # 原始帧计数器
        output_frame_count = 0    # 实际输出帧计数器
        skip_next = False        # 删除标记

        # 时间轴控制变量
        time_accumulator = 0.0  # 时间累加器（原始时间轴）
        last_valid_frame = None  # 用于减速时的帧重复
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_count = 0
        print(f"初始化信息结束 变速: {speed_factor:.2f} 缩放比例:{base_scale:.2f} 背景颜色:{bg_color} 背景不透明度{alpha} 边框颜色:{bd_color}" )
        print(f"{op_name} opencv 帧处理开始")

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 直接退出循环，避免过度填充

            # ===== 新增帧删除逻辑 =====
            frame_counter += 1
            if skip_interval > 0 and frame_counter % skip_interval == 0:
                skip_next = True
                continue  # 直接跳过本帧处理
            else:
                skip_next = False

            process_frame = frame
            # ===== 基础处理 start=====
            '''缩放'''
            h, w = process_frame.shape[:2]
            scale = base_scale - 0.02
            new_w = max(int(w * scale) // 2 * 2, 2)
            new_h = max(int(h * scale) // 2 * 2, 2)

            resized = cv2.resize(process_frame, (new_w, new_h))
            pad_vert = h - new_h
            pad_horz = w - new_w

            '''边框'''
            # 添加边框保持原始分辨率
            processed_frame = cv2.copyMakeBorder(
                resized,
                pad_vert // 2, pad_vert - pad_vert // 2,
                pad_horz // 2, pad_horz - pad_horz // 2,
                cv2.BORDER_CONSTANT, value=bd_color
            )
            # ===== 基础处理 end=====


            # ===== 深度处理 start=====

            # 时间戳计算（使用处理后的时间轴）
            current_time_sec = frame_count * output_frame_duration
            minutes = int(current_time_sec // 60)
            seconds = int(current_time_sec % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            # 文字位置计算（右下角）
            (text_width, text_height), _ = cv2.getTextSize(time_str, font, font_scale, thickness)
            padding = 5
            x = width - text_width - position[0] - padding * 2
            y = height - position[1] - text_height - padding

            # 创建叠加层
            overlay = processed_frame.copy()

            cv2.rectangle(overlay, (0, 0), (width, height), bg_color, -1)

            '''透明背景图层'''
            # 混合背景
            cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)

            '''时间戳水印'''
            # 添加文字
            cv2.putText(processed_frame, time_str,
                        (x + padding, y + text_height + padding),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # ===== 深度处理 end=====

            # 写入处理后的帧
            out.write(processed_frame)
            frame_count += 1

        cap.release()
        out.release()
        print(f"{op_name} opencv 帧处理并写入临时文件完成")

        # ===== 音频同步调整 =====

        audio_output_path, audio_cur_step = audio_processing(file_info)
        audio_clip = AudioFileClip(audio_output_path)
        if skip_interval > 0:
            # 计算实际帧率调整系数
            actual_speed = speed_factor * (skip_interval / (skip_interval - 1))
            audio_clip = audio_clip.fx(vfx.speedx, actual_speed)

        # 合并并输出最终视频
        video_clip = VideoFileClip(temp_video_path)
        print(f"{op_name} movepy 生成视频文件开始")
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, **OUTPUT_PARAMS)
        print(f"{op_name} movepy 生成视频文件结束")
        print(f"{op_name} 处理完成！输出文件已保存至: {output_path}")
        return output_path, step

    except Exception as e:
        print(f"{op_name} 处理失败 错误信息: {str(e)}")
        raise e


def audio_processing(file_info):
    """
    音频处理偏移噪音
    1、原音频导出
    2、增加噪音文件
    3、随机音轨偏移
    """
    op_name = '音频处理偏移噪音'
    try:
        print(f"{op_name} 开始")
        step = file_info['cur_audio_step'] +file_info['output_file_concat']+ '音频处理偏移+噪音'
        input_audio = file_info['cur_input_audio_path']
        input_noise_path = file_info['input_noise_path']
        file_name = file_info['src_input_file_name']
        file_dir = file_info['src_input_dir']
        tmp_dir = file_info['output_temp_dir']

        temp_file_name = file_info['src_input_file_name'] + file_info['output_file_concat'] + step
        output_path = tmp_dir + os.sep + temp_file_name + file_info['output_file_concat']+"processed_audio.mp3"

        # 检查音频文件是否存在
        if not os.path.exists(input_audio):
            raise FileNotFoundError(f"音频文件 '{input_audio}' 不存在。")

        # 处理音频
        print(f"{op_name} 开始处理音频.......")
        try:
            print(f"{op_name} 加载音频信息.......")
            audio = AudioSegment.from_file(input_audio)

        except Exception as e:
            raise ValueError(f"加载音频文件 '{input_audio}' 时出错: {e}")

        # 音调随机偏移
        print(f"{op_name} 音调随机偏移.......")
        pitch_shift_factor = np.random.uniform(0.95, 1.05)

        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * pitch_shift_factor)
        })

        print(f"{op_name} 添加环境底噪.......")
        # 添加环境底噪
        ############# 调试噪音放开 ###########
        ############# 调试噪音放开 ###########
        ############# 调试噪音放开 ###########
        # # 原声音调降低到原音量的1%（降低约40 dB）
        # original_volume = audio.dBFS  # 当前音量（分贝）
        # target_volume = original_volume - 40  # 降低到1%（近似于-40dB）
        # audio = audio.apply_gain(target_volume - original_volume)
        #
        # # 添加环境底噪并将其放入最大（放大20 dB）
        # noise = AudioSegment.from_file(input_noise_path) + 20  # 放大20 dB
        # noise = noise.apply_gain(original_volume + 20 - noise.dBFS)  # 确保噪声音量足够大
        ############ 调试噪音放开 ###########
        ############# 调试噪音放开 ###########
        ############# 调试噪音放开 ###########

        # 调试需要注释本行
        noise = AudioSegment.from_file(input_noise_path) - 20  # 减小20 dB

        # 设置噪声的帧率和声道与音频一致
        print(f"{op_name} 设置噪声的帧率和声道与音频一致.......")
        noise = noise.set_frame_rate(audio.frame_rate).set_channels(audio.channels)

        def repeat_until_duration(seg, duration):
            if len(seg) == 0:
                return seg
            repeat_times = (duration // len(seg)) + 1
            return (seg * repeat_times)[:duration]

        noise = repeat_until_duration(noise, len(audio))

        processed_audio = audio.overlay(noise)
        print(f"{op_name} 导出处理完的音频信息到 【{output_path}】.......")
        processed_audio.export(output_path, format="mp3")
        print(f"{op_name} 导出处理完的音频信息.......完成")
        return output_path, step

    except Exception as e:
        print(f"{op_name} 处理失败 错误信息 {e}")
        raise e

def auto_process(input_video_path, input_noise_path):
    '''视频处理入口'''
    try:
        process_start_time = datetime.datetime.now()  # 记录开始时间

        _file_info = process_path(input_video_path, PROCESS_CONFIG.get('new_file_prefix',''))
        _file_info['input_noise_path'] = input_noise_path
        video_steps = ['清除元数据', '关键帧+缩放+边框填充+深度处理']

        for step in video_steps:
            print(f"【{step}】 【开始】处理【视频】【{_file_info.get('src_input_file_name')}】<<<<<<<<<<<<<<<<<<<<<<")
            start_time = datetime.datetime.now()  # 记录开始时间

            if step == '清除元数据':
                output_path, cur_step = metadata_process(_file_info)
                _file_info['cur_input_path'] = output_path
                _file_info['cur_step'] = cur_step
            elif step == '关键帧+缩放+边框填充+深度处理':
                bd_color = (0, 0, 0)
                bg_color = (255, 222, 173)
                output_path, cur_step = advanced_process_video(file_info=_file_info,bg_color=bg_color, bd_color=bd_color,alpha=0.05)
                _file_info['cur_input_path'] = output_path
                _file_info['cur_step'] = cur_step
            else:
                print(f"暂不支持改操作 {step}")
            end_time = datetime.datetime.now()  # 记录结束时间
            duration = (end_time - start_time).total_seconds()  # 计算耗时
            print(f"【{step}】 【结束】处理【视频】【{_file_info.get('src_input_file_name')}】>>>>>>>>>>>>>>>>>>>> 耗时: {duration:.2f} 秒")

        #合并音视频
        start_time = datetime.datetime.now()  # 记录开始时间
        final_out_path = _file_info.get('output_new_file_path')
        copy_file(_file_info['cur_input_path'],final_out_path)

        end_time = datetime.datetime.now()  # 记录结束时间
        duration = (end_time - start_time).total_seconds()  # 计算耗时
        print(f"【合并音视频】 【结束】>>>>>>>>>>>>>>>>>>>> 耗时: {duration:.2f} 秒")

        duration = (end_time - process_start_time).total_seconds()  # 计算耗时
        print(f"【任务总耗时】:【 {duration:.2f} 秒】")

        return final_out_path
    except Exception as e:
        print(e)
        print("本次处理异常 返回空字符串", e)
        return ''


if __name__ == '__main__':
    # 要转换的视频源文件
    input_video_path = r"D:\tmp\t2.mp4"
    # input_video_path = r"D:\tmp\test1.mp4"
    # input_video_path = r"D:\tmp\video(3).mp4"
    # input_video_path = r"D:\tmp\video1.mp4"
    # 使用的噪音文件
    input_noise_path = r"D:\tmp\white_noise.mp3"
    # 文件名前缀
    final_path = auto_process(input_video_path, input_noise_path)
    print(f'返回结果: {final_path}')
