import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import sys
import os
import subprocess  # для запуска ffmpeg
from tqdm import tqdm  # импортируем tqdm для прогресс-бара
import colorama  # для цветного форматирования текста в консоли
from colorama import Fore, Back, Style
import msvcrt  # для обработки ввода с клавиатуры (Windows)
import signal  # для обработки прерывания

# Опциональная поддержка PyAV (если установлен)
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    print(f"{Fore.YELLOW}PyAV не установлен. Для некоторых функций будет использоваться ffmpeg.{Style.RESET_ALL}")

# Попытка импорта curses для Linux или windows-curses для Windows
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    # Если модуль недоступен, используем альтернативную реализацию
    CURSES_AVAILABLE = False
    print(f"{Fore.YELLOW}.{Style.RESET_ALL}")

# Инициализация colorama
colorama.init(autoreset=True)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, increment_path, set_logging, scale_coords, non_max_suppression
from utils.torch_utils import select_device, time_synchronized

from sort import Sort


def draw_boxes(img, bbox, identities=None, categories=None, names=None, colors=None):
    """
    Рисует bounding box и ID объекта на изображении.
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = 2  # Толщина линии

        # Вычисляем ширину и высоту бокса
        width = x2 - x1
        height = y2 - y1

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        
        # Используем белый цвет для всех рамок, если включен флаг white-boxes
        if hasattr(opt, 'white_boxes') and opt.white_boxes:
            color = [255, 255, 255]  # Белый цвет в формате BGR
        else:
            color = colors[cat]

        # Отрисовка bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)
        
        # Рисуем диагональные линии внутри bbox, если включен флаг cross
        if hasattr(opt, 'cross') and opt.cross:
            # Определяем толщину линий креста
            cross_thickness = 1  # По умолчанию
            if hasattr(opt, 'cross_thickness'):
                cross_thickness = opt.cross_thickness
                
            # Диагональ из левого верхнего в правый нижний
            cv2.line(img, (x1, y1), (x2, y2), color, cross_thickness)
            # Диагональ из правого верхнего в левый нижний
            cv2.line(img, (x2, y1), (x1, y2), color, cross_thickness)
        
        # Вычисляем примерную высоту текста для метки
        font_scale = 0.6
        font_thickness = max(tl - 1, 1)
        label_height = int(cv2.getTextSize("ID", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][1] * 1.5)
        
        # Добавляем красный квадрат в левом углу bounding box
        # Размер квадрата равен высоте текста
        square_size = label_height
        red_color = [0, 0, 255]  # Красный цвет в формате BGR
        cv2.rectangle(img, (x1, y1), (x1 + square_size, y1 + square_size), red_color, -1)  # Закрашенный квадрат

        label = f"ID {id}: {names[cat]}"
        tf = max(tl - 1, 1)  # Толщина шрифта
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, tf, cv2.LINE_AA)
        
        # Добавляем текст с размерами прямо у правой стороны рамки
        # Используем очень простой подход с прямым текстом (не вертикальным)
        
        # Надпись для ширины - сверху справа
        cv2.putText(img, f"W={width}", (x2 + 2, y1 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
        
        # Надпись для высоты - снизу справа
        cv2.putText(img, f"H={height}", (x2 + 2, y2 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
        
    return img


def apply_datamosh(input_path, output_path):
    """Применяет эффект datamosh к видео"""
    if not AV_AVAILABLE:
        print(f"{Fore.YELLOW}Для эффекта datamosh требуется PyAV. Установите его с помощью 'pip install av'{Style.RESET_ALL}")
        # Просто копируем файл без эффекта
        try:
            import shutil
            shutil.copy2(input_path, output_path)
        except Exception as e:
            print(f"{Fore.RED}Ошибка при копировании файла: {e}{Style.RESET_ALL}")
        return

    container = av.open(input_path)
    
    # Получаем параметры входного видео
    in_stream = container.streams.video[0]
    codec_name = in_stream.codec_context.name
    
    # Создаем выходной контейнер с нужным кодеком
    output = av.open(output_path, mode='w')
    output_stream = output.add_stream(codec_name)
    
    # Копируем параметры и конвертируем в целые числа где нужно
    output_stream.width = int(in_stream.width)
    output_stream.height = int(in_stream.height)
    output_stream.pix_fmt = in_stream.pix_fmt
    
    # Устанавливаем битрейт и другие параметры
    output_stream.bit_rate = 2000000  # 2 Mbps
    output_stream.options = {'crf': '23'}  # Качество сжатия
    
    try:
        # Сохраняем первый I-frame
        first_frame = True
        skip_frames = 2  # Пропускаем каждый второй I-frame для усиления эффекта
        frame_count = 0
        
        for frame in container.decode(video=0):
            frame_count += 1
            
            # Сохраняем первый кадр как есть
            if first_frame:
                packet = output_stream.encode(frame)
                if packet:
                    output.mux(packet)
                first_frame = False
                continue
                
            # Пропускаем некоторые I-frames для создания эффекта
            if frame_count % skip_frames != 0:
                # Изменяем тип кадра на P-frame
                frame.pict_type = 'P'
                
            packet = output_stream.encode(frame)
            if packet:
                output.mux(packet)
        
        # Записываем оставшиеся кадры
        packet = output_stream.encode(None)
        if packet:
            output.mux(packet)
            
    finally:
        # Закрываем файлы
        container.close()
        output.close()


def add_audio_to_video(input_video_with_audio, output_video_without_audio, final_output_path):
    """
    Добавляет аудио из исходного видео в выходное видео с использованием ffmpeg.
    Делает это в два этапа: сначала извлекает аудио, затем соединяет его с видео.
    
    Args:
        input_video_with_audio: Путь к исходному видео с аудио
        output_video_without_audio: Путь к выходному видео без аудио
        final_output_path: Путь для сохранения итогового видео с аудио
    
    Returns:
        bool: True если аудио успешно добавлено, False в случае ошибки
    """
    print(f"{Fore.MAGENTA}===== ДОБАВЛЕНИЕ АУДИО К ОТТРЕЧЕННОМУ ВИДЕО ====={Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Исходное видео с аудио: {Fore.WHITE}{input_video_with_audio}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Оттреченное видео без аудио: {Fore.WHITE}{output_video_without_audio}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Результат будет сохранен в: {Fore.WHITE}{final_output_path}{Style.RESET_ALL}")
    
    # Проверяем наличие исходных файлов
    if not os.path.exists(input_video_with_audio):
        print(f"{Fore.RED}Ошибка: Исходное видео не найдено: {input_video_with_audio}{Style.RESET_ALL}")
        return False
        
    if not os.path.exists(output_video_without_audio):
        print(f"{Fore.RED}Ошибка: Оттреченное видео не найдено: {output_video_without_audio}{Style.RESET_ALL}")
        return False
    
    # Проверка наличия аудио в исходном видео
    cmd_check = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_video_with_audio
    ]
    
    try:
        result = subprocess.run(cmd_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        has_audio = "audio" in result.stdout.lower()
        
        if has_audio:
            print(f"{Fore.GREEN}✓ Аудиодорожка найдена в исходном видео.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ Аудиодорожка НЕ найдена в исходном видео.{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}✗ Ошибка при проверке аудио: {e}{Style.RESET_ALL}")
        return False
    
    # Создаем папку для итогового файла, если она не существует
    final_dir = os.path.dirname(final_output_path)
    if not os.path.exists(final_dir):
        os.makedirs(final_dir, exist_ok=True)
    
    # Объединение видео и аудио
    cmd = [
        "ffmpeg", "-y",
        "-i", output_video_without_audio,  # Видео без аудио
        "-i", input_video_with_audio,      # Исходное видео с аудио
        "-c:v", "copy",                    # Копируем видео без перекодирования
        "-c:a", "aac",                     # Аудиокодек AAC
        "-b:a", "192k",                    # Битрейт аудио
        "-map", "0:v:0",                   # Берем видео из первого источника
        "-map", "1:a:0",                   # Берем аудио из второго источника
        "-shortest",                       # Длительность по кратчайшему потоку
        final_output_path
    ]
    
    try:
        print(f"{Fore.YELLOW}Выполнение команды: {' '.join(cmd)}{Style.RESET_ALL}")
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if process.returncode == 0:
            print(f"{Fore.GREEN}✓ Аудио успешно добавлено в оттреченное видео.{Style.RESET_ALL}")
        else:
            stderr = process.stderr.decode('utf-8', errors='replace')
            print(f"{Fore.RED}✗ Ошибка ffmpeg: {stderr}{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}✗ Ошибка при добавлении аудио: {e}{Style.RESET_ALL}")
        return False
    
    # Проверка наличия аудио в результирующем видео
    cmd_check = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        final_output_path
    ]
    
    try:
        result = subprocess.run(cmd_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        has_audio = "audio" in result.stdout.lower()
        
        if has_audio:
            print(f"{Fore.GREEN}✓ Аудиодорожка найдена в результирующем видео.{Style.RESET_ALL}")
            print(f"{Fore.GREEN}==== ОПЕРАЦИЯ УСПЕШНО ЗАВЕРШЕНА ===={Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}Итоговое видео с аудио: {Fore.WHITE}{os.path.abspath(final_output_path)}{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}✗ Аудиодорожка НЕ найдена в результирующем видео.{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}✗ Ошибка при проверке аудио: {e}{Style.RESET_ALL}")
        return False


# Переопределяем sys.stdout для фильтрации нежелательных выводов
original_stdout = sys.stdout

class FilteredStdout:
    def __init__(self, original):
        self.original = original
        self.current_line = ""
    
    def write(self, text):
        # Если строка содержит "video" и номер кадра (например, "video 1/1 (123/3085)"),
        # не выводим её
        if "video 1/1" in text and "/3085)" in text:
            return
        
        # Заменяем все цветовые коды на MAGENTA
        text = text.replace(Fore.CYAN, Fore.MAGENTA)
        text = text.replace(Fore.GREEN, Fore.MAGENTA)
        text = text.replace(Fore.YELLOW, Fore.MAGENTA)
        text = text.replace(Fore.RED, Fore.MAGENTA)
        text = text.replace(Fore.WHITE, Fore.MAGENTA)
        
        # Заменяем все фоновые цвета на MAGENTA (кроме случаев, где нужен чёрный текст на розовом фоне)
        if Back.GREEN in text:
            text = text.replace(Back.GREEN, Back.MAGENTA)
        if Back.YELLOW in text:
            text = text.replace(Back.YELLOW, Back.MAGENTA)
        if Back.RED in text:
            text = text.replace(Back.RED, Back.MAGENTA)
        if Back.CYAN in text:
            text = text.replace(Back.CYAN, Back.MAGENTA)
            
        self.original.write(text)
    
    def flush(self):
        self.original.flush()

# Используем эту функцию для переопределения вывода в начале обработки
def setup_filtered_output():
    sys.stdout = FilteredStdout(original_stdout)

# Возвращаем оригинальный вывод в конце обработки
def restore_original_output():
    sys.stdout = original_stdout


def detect():
    global early_stop_flag
    early_stop_flag = False
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Загрузка модели YOLO
    print(f"{Fore.MAGENTA}Загрузка модели {Fore.WHITE}{weights}{Fore.MAGENTA}...{Style.RESET_ALL}")
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if half:
        model.half()
        print(f"{Fore.MAGENTA}Используется режим {Fore.WHITE}half precision{Style.RESET_ALL}")

    # Загрузка датасета
    print(f"{Fore.MAGENTA}Загрузка источника {Fore.WHITE}{source}{Fore.MAGENTA}...{Style.RESET_ALL}")
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    # Фильтруем нежелательный вывод
    setup_filtered_output()
    
    # Получаем общее количество кадров для прогресс-бара
    total_frames = 0
    if hasattr(dataset, 'cap') and dataset.cap is not None:
        total_frames = int(dataset.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{Fore.MAGENTA}Всего кадров: {Fore.WHITE}{total_frames}{Style.RESET_ALL}")
    
    # Получение имен классов и цветов
    names = model.module.names if hasattr(model, 'module') else model.names
    
   
    for i, name in enumerate(names):
        if name.lower() == "person":
            names[i] = "Player"
            
    # Setup colors
    if opt.white_boxes:
        colors = [[255, 255, 255] for _ in range(len(names))]
    else:
        colors = [[np.random.randint(0, 255), 0, 255] for _ in range(len(names))]  # Изменение: используем розовый (magenta) [r, 0, 255]
    
    print(f"{Fore.MAGENTA}Доступные классы: {Fore.WHITE}{', '.join(names)}{Style.RESET_ALL}")

    # Инициализация трекера
    sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
    if opt.track:
        print(f"{Fore.MAGENTA}Трекинг объектов {Fore.WHITE}активирован{Style.RESET_ALL}")

    # Добавляем функцию проверки нажатия Escape
    def check_exit_key():
        global early_stop_flag
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # Escape имеет код 27
            if ord(key) == 27:
                print(f"\n{Fore.YELLOW}Завершение обработки по нажатию Escape...{Style.RESET_ALL}")
                early_stop_flag = True
                return True
        return False
    
    # Инициализация видеописателей
    vid_writer, track_writer = None, None
    vid_path, track_path = None, None
    
    # Вывод всех активных опций
    active_options = []
    if opt.track:
        active_options.append(f"{Fore.WHITE}--track{Style.RESET_ALL}")
    if opt.show_track:
        active_options.append(f"{Fore.WHITE}--show-track{Style.RESET_ALL}")
    if hasattr(opt, 'white_boxes') and opt.white_boxes:
        active_options.append(f"{Fore.WHITE}--white-boxes{Style.RESET_ALL}")
    if hasattr(opt, 'cross') and opt.cross:
        cross_opt = f"{Fore.WHITE}--cross{Style.RESET_ALL}"
        if hasattr(opt, 'cross_thickness'):
            cross_opt += f" ({Fore.WHITE}толщина: {opt.cross_thickness}{Style.RESET_ALL})"
        active_options.append(cross_opt)
    if opt.datamosh:
        active_options.append(f"{Fore.WHITE}--datamosh{Style.RESET_ALL}")
    if opt.add_audio:
        active_options.append(f"{Fore.WHITE}--add-audio{Style.RESET_ALL}")
    
    if active_options:
        print(f"{Fore.MAGENTA}Активные опции: {', '.join(active_options)}{Style.RESET_ALL}")
    
    # Выводим информацию об устройстве
    device_info = "CPU" if opt.device == "cpu" or opt.device == "" else f"GPU:{opt.device}"
    print(f"{Fore.MAGENTA}Устройство: {Fore.WHITE}{device_info}{Style.RESET_ALL}")
    
    # Выводим информацию о клавишах управления
    print(f"{Fore.MAGENTA}Управление: {Fore.WHITE}Escape - завершить обработку{Style.RESET_ALL}")
    
    # Отступ перед прогресс-баром
    print("\n")
    
    # Инициализация переменных для отслеживания прогресса
    start_time = time.time()
    frame_count = 0
    
    # Создаем прогресс-бар с общим количеством кадров, если доступно
    pbar_args = {
        "desc": f"{Fore.MAGENTA}Обработка видео{Style.RESET_ALL}",
        "unit": "кадр",
        "ncols": 100,
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    }
    if total_frames > 0:
        pbar_args["total"] = total_frames
    
    with tqdm(**pbar_args) as pbar:
        for path, img, im0s, vid_cap in dataset:
            # Проверяем, не нажата ли клавиша Escape
            if check_exit_key():
                print(f"\n{Fore.YELLOW}Обработка остановлена пользователем на кадре {frame_count}/{total_frames}.{Style.RESET_ALL}")
                break
                
            frame_count += 1
            
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Предсказание YOLO
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes)

            for i, det in enumerate(pred):
                p, s, im0 = path, '', im0s.copy()

                # Создаем черное изображение для второго видео
                black_img = np.zeros_like(im0)

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    dets_to_sort = np.empty((0, 6))
                    for x1, y1, x2, y2, conf, detclass in det.cpu().numpy():
                        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                    if opt.track:
                        tracked_dets = sort_tracker.update(dets_to_sort)

                        if len(tracked_dets) > 0:
                            bbox_xyxy = tracked_dets[:, :4]
                            identities = tracked_dets[:, 8]
                            categories = tracked_dets[:, 4]

                            # Рисуем bbox и ID на обоих изображениях
                            im0 = draw_boxes(im0, bbox_xyxy, identities, categories, names, colors)
                            black_img = draw_boxes(black_img, bbox_xyxy, identities, categories, names, colors)

                            if opt.show_track:
                                # Копируем объекты на черный фон
                                for i, box in enumerate(bbox_xyxy):
                                    x1, y1, x2, y2 = map(int, box)
                                    # Добавляем небольшой отступ вокруг объекта
                                    padding = 10
                                    y1 = max(0, y1 - padding)
                                    y2 = min(im0.shape[0], y2 + padding)
                                    x1 = max(0, x1 - padding)
                                    x2 = min(im0.shape[1], x2 + padding)
                                    
                                    # Копируем область с объектом из основного видео на черный фон
                                    black_img[y1:y2, x1:x2] = im0[y1:y2, x1:x2]

                # Запись в видеофайл
                if vid_path != p:
                    vid_path = p
                    track_path = str(save_dir / ('tracks_' + Path(p).name))

                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if isinstance(track_writer, cv2.VideoWriter):
                        track_writer.release()

                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(str(save_dir / Path(p).name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    track_writer = cv2.VideoWriter(track_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                if vid_writer is not None:
                    vid_writer.write(im0)
                if track_writer is not None:
                    track_writer.write(black_img)
                
                # Обновляем прогресс-бар с информацией о скорости и оставшемся времени
                elapsed_time = time.time() - start_time
                fps_avg = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                if total_frames > 0:
                    remaining_frames = total_frames - frame_count
                    estimated_time = remaining_frames / fps_avg if fps_avg > 0 else 0
                    
                    # Форматируем оставшееся время в часы:минуты:секунды
                    hours, remainder = divmod(estimated_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    
                    pbar.set_postfix({
                        "FPS": f"{Fore.WHITE}{fps_avg:.1f}{Style.RESET_ALL}",
                        "ETA": f"{Fore.MAGENTA}{time_str}{Style.RESET_ALL}"
                    })
                else:
                    pbar.set_postfix({
                        "FPS": f"{Fore.WHITE}{fps_avg:.1f}{Style.RESET_ALL}"
                    })
                    
                pbar.update(1)
    
    # Восстанавливаем оригинальный вывод
    restore_original_output()
    
    # Информация о завершении
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Проверяем, было ли завершение по Escape
    if early_stop_flag:
        print(f"\n{Back.MAGENTA}{Fore.BLACK}{Style.BRIGHT} ОБРАБОТКА ПРЕРВАНА ПОЛЬЗОВАТЕЛЕМ {Style.RESET_ALL}")
    else:
        print(f"\n{Back.MAGENTA}{Fore.BLACK}{Style.BRIGHT} ОБРАБОТКА ЗАВЕРШЕНА {Style.RESET_ALL}")
    
    print(f"{Fore.MAGENTA}Общее время: {Fore.WHITE}{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Обработано кадров: {Fore.WHITE}{frame_count}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Средняя скорость: {Fore.WHITE}{frame_count / total_time:.1f} FPS{Style.RESET_ALL}")
    
    # Пути к результатам
    print(f"\n{Fore.MAGENTA}Результаты сохранены в:{Style.RESET_ALL}")
    original_output_path = str(save_dir / Path(source).name)
    track_output_path = None
    
    if vid_writer is not None:
        print(f"{Fore.WHITE}{original_output_path}{Style.RESET_ALL}")
    
    if track_writer is not None and opt.show_track:
        track_output_path = str(save_dir / ('tracks_' + Path(source).name))
        print(f"{Fore.WHITE}{track_output_path}{Style.RESET_ALL}")
        
    print(f"\n{Back.MAGENTA}{Fore.MAGENTA}{Style.BRIGHT}{'=' * 80}{Style.RESET_ALL}")

    # После записи обычного видео, применяем datamosh если включена опция
    if isinstance(track_writer, cv2.VideoWriter):
        track_writer.release()
        track_writer = None  # Освобождаем ресурс
        
        if opt.datamosh:
            try:
                print(f"{Fore.MAGENTA}Применение эффекта datamosh...{Style.RESET_ALL}")
                # Путь к временному файлу
                temp_path = str(save_dir / 'temp_tracks.mp4')
                final_path = str(save_dir / ('tracks_' + Path(p).name))
                
                # Даем время на освобождение файла
                time.sleep(1)
                
                # Переименовываем оригинальный файл
                os.rename(track_path, temp_path)
                
                # Применяем datamosh
                apply_datamosh(temp_path, final_path)
                
                # Даем время на освобождение файла
                time.sleep(1)
                
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                print(f"{Fore.MAGENTA}Эффект datamosh успешно применен!{Style.RESET_ALL}")
                track_output_path = final_path
                    
            except Exception as e:
                print(f"{Fore.MAGENTA}Ошибка при создании datamosh эффекта: {e}{Style.RESET_ALL}")
                # В случае ошибки возвращаем оригинальный файл
                if os.path.exists(temp_path):
                    if not os.path.exists(final_path):
                        os.rename(temp_path, final_path)
        
        # Добавляем аудио из исходного видео, если опция включена
        if opt.add_audio and opt.show_track and Path(source).is_file() and track_output_path:
            try:
                print(f"{Fore.MAGENTA}Добавление аудио в выходное видео...{Style.RESET_ALL}")
                audio_result_path = str(save_dir / ('audio_tracks_' + Path(source).name))
                
                # Проверяем существование файлов
                if not os.path.exists(source):
                    print(f"{Fore.RED}Исходный файл не найден: {source}{Style.RESET_ALL}")
                elif not os.path.exists(track_output_path):
                    print(f"{Fore.RED}Выходной файл трекинга не найден: {track_output_path}{Style.RESET_ALL}")
                else:
                    # Добавляем аудио с использованием ffmpeg
                    success = add_audio_to_video(
                        input_video_with_audio=source,
                        output_video_without_audio=track_output_path,
                        final_output_path=audio_result_path
                    )
                    
                    if success:
                        print(f"{Fore.MAGENTA}Видео с аудио сохранено в: {Fore.WHITE}{audio_result_path}{Style.RESET_ALL}")
                    
            except Exception as e:
                print(f"{Fore.RED}Ошибка при добавлении аудио: {e}{Style.RESET_ALL}")

    # Освобождение ресурсов
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
        vid_writer = None


def print_help_and_prompt():
    """
    Интерактивный выбор опций с использованием навигации стрелками и возврат аргументов для argparse
    """
    # Очистка экрана консоли
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # ASCII-арт логотип
    owl_art = """                                                                             ==                     ==
                 <^\()/^>               <^\()/^>
                  \/  \/                 \/  \/
                   /__\      .  '  .      /__\ 
      ==            /\    .     |     .    /\            ==
   <^\()/^>       !_\/       '  |  '       \/_!       <^\()/^>
    \/  \/     !_/I_||  .  '   \'/   '  .  ||_I\_!     \/  \/
     /__\     /I_/| ||      -== + ==-      || |\_I\     /__\
     /_ \   !//|  | ||  '  .   /.\   .  '  || |  |\\!   /_ \
    (-   ) /I/ |  | ||       .  |  .       || |  | \I\ (=   )
     \__/!//|  |  | ||    '     |     '    || |  |  |\\!\__/
     /  \I/ |  |  | ||       '  .  '    *  || |  |  | \I/  \
    {_ __}  |  |  | ||                     || |  |  |  {____}
 _!__|= ||  |  |  | ||   *      +          || |  |  |  ||  |__!_
 _I__|  ||__|__|__|_||          A          ||_|__|__|__||- |__I_
 -|--|- ||--|--|--|-||       __/_\__  *    ||-|--|--|--||= |--|-
  |  |  ||  |  |  | ||      /\-'o'-/\      || |  |  |  ||  |  |
  |  |= ||  |  |  | ||     _||:<_>:||_     || |  |  |  ||= |  |
  |  |- ||  |  |  | || *  /\_/=====\_/\  * || |  |  |  ||= |  |
  |  |- ||  |  |  | ||  __|:_:_[I]_:_:|__  || |  |  |  ||- |  | 
 _|__|  ||__|__|__|_||:::::::::::::::::::::||_|__|__|__||  |__|_
 -|--|= ||--|--|--|-||:::::::::::::::::::::||-|--|--|--||- |--|-
  jgs|- ||  |  |  | ||:::::::::::::::::::::|| |  |  |  ||= |  | 
~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^~~~~~~~~~
    """
    
    # Speesh logo
    speesh_logo = """
     
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠶⠚⠉⢉⣩⠽⠟⠛⠛⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⠉⠀⢀⣠⠞⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡞⠁⠀⠀⣰⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣠⡤⠤⠄⢤⣄⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢰⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠴⠒⠋⠉⠀⠀⠀⣀⣤⠴⠒⠋⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⡄⠀⠀⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⢳⡄⢀⡴⠚⠉⠀⠀⠀⠀⠀⣠⠴⠚⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠘⣧⠀⠀⠀⠀⠀⠀⠀⠀⣰⠃⠀⠀⠹⡏⠀⠀⠀⠀⠀⣀⣴⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠳⢬⣳⣄⣠⠤⠤⠶⠶⠒⠋⠀⠀⠀⠀⠹⡀⠀⠀⠀⠀⠈⠉⠛⠲⢦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠤⠖⠋⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢳⠦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣠⠖⠋⠀⠀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠀⢃⠈⠙⠲⣄⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢠⠞⠁⠀⠀⠀⢀⢾⠃⠀⠀⠀⠀⠀⠀⠀⠀⢢⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⣹⠮⣄⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣰⠋⠀⠀⢀⡤⡴⠃⠈⠦⣀⠀⠀⠀⠀⠀⠀⢀⣷⢸⠀⠀⠀⠀⢀⣀⠘⡄⠤⠤⢤⠔⠒⠂⠉⠁⠀⠀⠀⠑⢄⡀⠀⠀⠙⢦⡀⠀⠀⠀
⠀⠀⠀⠀⣼⠃⠀⠀⢠⣞⠟⠀⠀⠀⡄⠀⠉⠒⠢⣤⣤⠄⣼⢻⠸⠀⠀⠀⠀⠉⢤⠀⢿⡖⠒⠊⢦⠤⠤⣀⣀⡀⠀⠀⠀⠈⠻⡝⠲⢤⣀⠙⢦⠀⠀
⠀⠀⠀⢰⠃⠀⠀⣴⣿⠎⠀⠀⢀⣜⠤⠄⢲⠎⠉⠀⠀⡼⠸⠘⡄⡇⠀⠀⠀⠀⢸⠀⢸⠘⢆⠀⠘⡄⠀⠀⠀⢢⠉⠉⠀⠒⠒⠽⡄⠀⠈⠙⠮⣷⡀
⠀⠀⠀⡟⠀⠀⣼⢻⠧⠐⠂⠉⡜⠀⠀⡰⡟⠀⠀⠀⡰⠁⡇⠀⡇⡇⠀⠀⠀⠀⢺⠇⠀⣆⡨⢆⠀⢽⠀⠀⠀⠈⡷⡄⠀⠀⠀⠀⠹⡄⠀⠀⠀⠈⠁
⠀⠀⢸⠃⠀⠀⢃⠎⠀⠀⠀⣴⠃⠀⡜⠹⠁⠀⠀⡰⠁⢠⠁⠀⢸⢸⠀⠀⠀⢠⡸⢣⠔⡏⠀⠈⢆⠀⣇⠀⠀⠀⢸⠘⢆⠀⠀⠀⠀⢳⠀⠀⠀⠀⠀
⠀⠀⢸⠀⠀⠀⡜⠀⠀⢀⡜⡞⠀⡜⠈⠏⠀⠈⡹⠑⠒⠼⡀⠀⠀⢿⠀⠀⠀⢀⡇⠀⢇⢁⠀⠀⠈⢆⢰⠀⠀⠀⠈⡄⠈⢢⠀⠀⠀⠈⣇⠀⠀⠀⠀
⠀⠀⢸⡀⠀⢰⠁⠀⢀⢮⠀⠇⡜⠀⠘⠀⠀⢰⠃⠀⠀⡇⠈⠁⠀⢘⡄⠀⠀⢸⠀⠀⣘⣼⠤⠤⠤⣈⡞⡀⠀⠀⠀⡇⠰⡄⢣⡀⠀⠀⢻⠀⠀⠀⠀
⠀⠀⠈⡇⠀⡜⠀⢀⠎⢸⢸⢰⠁⠀⠄⠀⢠⠃⠀⠀⢸⠀⠀⠀⠀⠀⡇⠀⠀⡆⠀⠀⣶⣿⡿⠿⡛⢻⡟⡇⠀⠀⠀⡇⠀⣿⣆⢡⠀⠀⢸⡇⠀⠀⠀
⠀⠀⢠⡏⠀⠉⢢⡎⠀⡇⣿⠊⠀⠀⠀⢠⡏⠀⠀⠀⠎⠀⠀⠀⠀⠀⡇⠀⡸⠀⠀⠀⡇⠀⢰⡆⡇⢸⢠⢹⠀⠀⠀⡇⠀⢹⠈⢧⣣⠀⠘⡇⠀⠀⠀
⠀⠀⢸⡇⠀⠀⠀⡇⠀⡇⢹⠀⠀⠀⢀⡾⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢠⠃⠀⠀⠠⠟⡯⣻⣇⢃⠇⢠⠏⡇⠀⢸⡆⠀⢸⠀⠈⢳⡀⠀⡇⠀⠀⠀
⠀⠀⠀⣇⠀⡔⠋⡇⠀⢱⢼⠀⠀⡂⣼⡇⢹⣶⣶⣶⣤⣤⣀⠀⠀⠀⣇⠇⠀⠀⠀⠀⣶⡭⢃⣏⡘⠀⡎⠀⠇⠀⡾⣷⠀⣼⠀⠀⠀⢻⡄⡇⠀⠀⠀
⠀⠀⠀⣹⠜⠋⠉⠓⢄⡏⢸⠀⠀⢳⡏⢸⠹⢀⣉⢭⣻⡽⠿⠛⠓⠀⠋⠀⠀⠀⠀⠀⠘⠛⠛⠓⠀⡄⡇⠀⢸⢰⡇⢸⡄⡟⠀⠀⠀⠀⢳⡇⠀⠀⠀
⠀⣠⠞⠁⠀⠀⠀⠀⠀⢙⠌⡇⠀⣿⠁⠀⡇⡗⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⠀⠀⠀⠀⠀⠀⠁⠁⠀⢸⣼⠀⠈⣇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢸⠁⠀⠀⢀⡠⠔⠚⠉⠉⢱⣇⢸⢧⠀⠀⠸⣱⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⡤⠦⡔⠀⠀⠀⠀⠀⢀⡼⠀⠀⣼⡏⠀⠀⢹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢸⠀⠀⠀⠋⠀⠀⠀⢀⡠⠤⣿⣾⣇⣧⠀⠀⢫⡆⠀⠀⠀⠀⠀⠀⠀⢨⠀⠀⣠⠇⠀⠀⢀⡠⣶⠋⠀⠀⡸⣾⠁⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢸⡄⠀⠀⠀⠀⠠⠊⠁⠀⠀⢸⢃⠘⡜⡵⡀⠈⢿⡱⢲⡤⠤⢀⣀⣀⡀⠉⠉⣀⡠⡴⠚⠉⣸⢸⠀⠀⢠⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⢧⠀⠀⠀⠀⠀⠀⠀⣀⠤⠚⠚⣤⣵⡰⡑⡄⠀⢣⡈⠳⡀⠀⠀⠀⢨⡋⠙⣆⢸⠀⠀⣰⢻⡎⠀⠀⡎⡇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠈⢷⡀⠀⠀⠀⠀⠀⠁⠀⠀⠀⡸⢌⣳⣵⡈⢦⡀⠳⡀⠈⢦⡀⠀⠘⠏⠲⣌⠙⢒⠴⡧⣸⡇⠀⡸⢸⠇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢠⣿⠢⡀⠀⠀⠀⠠⠄⡖⠋⠀⠀⠙⢿⣳⡀⠑⢄⠹⣄⡀⠙⢄⡠⠤⠒⠚⡖⡇⠀⠘⣽⡇⢠⠃⢸⢀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⣾⠃⠀⠀⠀⠀⠀⢀⡼⣄⠀⠀⠀⠀⠀⠑⣽⣆⠀⠑⢝⡍⠒⠬⢧⣀⡠⠊⠀⠸⡀⠀⢹⡇⡎⠀⡿⢸⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⡼⠁⠀⠀⠀⠀⠀⠀⢀⠻⣺⣧⠀⠀⠀⠰⢢⠈⢪⡷⡀⠀⠙⡄⠀⠀⠱⡄⠀⠀⠀⢧⠀⢸⡻⠀⢠⡇⣾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢰⠇⠀⠀⠀⠀⠀⠀⠀⢸⠀⡏⣿⠀⠀⠀⠀⢣⢇⠀⠑⣄⠀⠀⠸⡄⠀⠀⠘⡄⠀⠀⠸⡀⢸⠁⠀⡾⢰⡏⢳⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
           `"""
 
    
    # Выводим комбинированный ASCII-арт с розовым цветом
    print(f"{Fore.MAGENTA}{owl_art}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{speesh_logo}{Style.RESET_ALL}")
    

    branding = f"""
{Fore.WHITE}╔═════════════════════════════════════════════════════════════════════════╗
║ {Fore.MAGENTA}speesh.ru{Fore.WHITE} | {Fore.MAGENTA}rbdclan.moscow{Fore.WHITE} | {Fore.MAGENTA}Sergey Golovchan © 2025{Fore.WHITE}               ║
╚═════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(branding)
    
    # Выводим заголовок
    title_box = f"""
{Fore.WHITE}╔{'═' * 65}╗
║{Back.MAGENTA}{Fore.BLACK}{Style.BRIGHT}{'YOLO v7 DETECTION & TRACKING SYSTEM':^65}{Style.RESET_ALL}{Fore.WHITE}║
╚{'═' * 65}╝{Style.RESET_ALL}
    """
    print(title_box)
    
    # Собираем команду на основе выбора пользователя
    cmd = []
    
    # 1. Выбор модели
    model_options = [
        "yolov7.pt (стандартная модель)",
        "yolov7-tiny.pt (быстрая, менее точная)",
        "yolov7-w6.pt (большая, более точная)",
        "Свой вариант (ввести путь вручную)"
    ]
    
    print(f"\n{Fore.MAGENTA}Выберите модель:{Style.RESET_ALL}")
    model_choice = create_interactive_menu("ВЫБОР МОДЕЛИ", model_options)
    
    if model_choice == 0:
        weights = "yolov7.pt"
    elif model_choice == 1:
        weights = "yolov7-tiny.pt"
    elif model_choice == 2:
        weights = "yolov7-w6.pt"
    elif model_choice == 3:
        print(f"\n{Fore.MAGENTA}Введите путь к файлу весов:{Style.RESET_ALL}")
        weights = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
    else:
        print(f"{Fore.MAGENTA}Выход из программы.{Style.RESET_ALL}")
        sys.exit(0)
    
    cmd.extend(["--weights", weights])
    
    # 2. Выбор источника
    print(f"\n{Fore.MAGENTA}Выберите источник:{Style.RESET_ALL}")
    source_options = [
        "Видеофайл (выбрать файл)",
        "Веб-камера (0)",
        "Другая камера (ввести номер)",
        "Папка с изображениями"
    ]
    
    source_choice = create_interactive_menu("ВЫБОР ИСТОЧНИКА", source_options)
    
    if source_choice == 0:
        print(f"\n{Fore.MAGENTA}Введите путь к видеофайлу:{Style.RESET_ALL}")
        source = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
    elif source_choice == 1:
        source = "0"
    elif source_choice == 2:
        print(f"\n{Fore.MAGENTA}Введите номер камеры:{Style.RESET_ALL}")
        source = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
    elif source_choice == 3:
        print(f"\n{Fore.MAGENTA}Введите путь к папке с изображениями:{Style.RESET_ALL}")
        source = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
    else:
        print(f"{Fore.MAGENTA}Выход из программы.{Style.RESET_ALL}")
        sys.exit(0)
    
    cmd.extend(["--source", source])
    
    # 3. Выбор размера изображения
    img_size_options = [
        "640 (стандартный размер, баланс скорости и точности)",
        "320 (быстрее, менее точно)",
        "416 (компромисс)",
        "512 (компромисс)",
        "768 (точнее, медленнее)",
        "1024 (самый точный, самый медленный)",
        "Свой вариант (ввести вручную)"
    ]
    
    print(f"\n{Fore.MAGENTA}Выберите размер входного изображения:{Style.RESET_ALL}")
    img_size_choice = create_interactive_menu("РАЗМЕР ИЗОБРАЖЕНИЯ", img_size_options)
    
    img_sizes = [640, 320, 416, 512, 768, 1024]
    if img_size_choice < 6:
        img_size = img_sizes[img_size_choice]
    elif img_size_choice == 6:
        print(f"\n{Fore.MAGENTA}Введите размер изображения (целое число):{Style.RESET_ALL}")
        img_size = int(input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip())
        print(f"{Style.RESET_ALL}")
    else:
        print(f"{Fore.MAGENTA}Выход из программы.{Style.RESET_ALL}")
        sys.exit(0)
    
    cmd.extend(["--img-size", str(img_size)])
    
    # 4. Выбор порогов уверенности и IOU
    section_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'НАСТРОЙКИ ДЕТЕКЦИИ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(section_header)
    
    conf_options = [
        "0.25 (стандартный, умеренные требования)",
        "0.1 (больше объектов, больше ложных срабатываний)",
        "0.4 (меньше объектов, меньше ложных срабатываний)",
        "0.5 (строгие требования к уверенности)",
        "Свой вариант (ввести вручную)"
    ]
    
    conf_choice = create_interactive_menu("ПОРОГ УВЕРЕННОСТИ", conf_options)
    conf_values = [0.25, 0.1, 0.4, 0.5]
    
    if conf_choice < 4:
        conf_thres = conf_values[conf_choice]
    elif conf_choice == 4:
        print(f"\n{Fore.MAGENTA}Введите порог уверенности (от 0.0 до 1.0):{Style.RESET_ALL}")
        conf_thres = float(input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip())
        print(f"{Style.RESET_ALL}")
    else:
        conf_thres = 0.25
    
    cmd.extend(["--conf-thres", str(conf_thres)])
    
    iou_options = [
        "0.45 (стандартный)",
        "0.3 (менее строгое подавление пересечений)",
        "0.6 (более строгое подавление пересечений)",
        "Свой вариант (ввести вручную)"
    ]
    
    iou_choice = create_interactive_menu("ПОРОГ IOU", iou_options)
    iou_values = [0.45, 0.3, 0.6]
    
    if iou_choice < 3:
        iou_thres = iou_values[iou_choice]
    elif iou_choice == 3:
        print(f"\n{Fore.MAGENTA}Введите порог IOU (от 0.0 до 1.0):{Style.RESET_ALL}")
        iou_thres = float(input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip())
        print(f"{Style.RESET_ALL}")
    else:
        iou_thres = 0.45
    
    cmd.extend(["--iou-thres", str(iou_thres)])
    
    # 5. Устройство для вычислений
    hw_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'АППАРАТНАЯ ЧАСТЬ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(hw_header)
    
    device_options = [
        "CPU",
        "GPU:0 (первая видеокарта)",
        "GPU:1 (вторая видеокарта, если есть)",
        "GPU:2 (третья видеокарта, если есть)",
        "Свой вариант (ввести вручную)"
    ]
    
    print(f"\n{Fore.MAGENTA}Выберите устройство для вычислений:{Style.RESET_ALL}")
    device_choice = create_interactive_menu("ВЫБОР УСТРОЙСТВА", device_options)
    
    device_values = ["cpu", "0", "1", "2"]
    if device_choice < 4:
        device = device_values[device_choice]
    elif device_choice == 4:
        print(f"\n{Fore.MAGENTA}Введите устройство (cpu, 0, 1, 2, ...):{Style.RESET_ALL}")
        device = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
    else:
        device = "cpu"
    
    cmd.extend(["--device", device])
    
    # 6. Фильтрация классов
    classes_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'НАСТРОЙКА КЛАССОВ ОБЪЕКТОВ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(classes_header)
    
    print(f"\n{Fore.MAGENTA}Хотите фильтровать классы объектов?{Style.RESET_ALL}")
    filter_options = ["Нет (обнаруживать все классы)", "Да (выбрать классы)"]
    filter_choice = create_interactive_menu("ФИЛЬТРАЦИЯ КЛАССОВ", filter_options)
    
    if filter_choice == 1:
        class_options = [
            "0: person (человек)",
            "1: bicycle (велосипед)",
            "2: car (машина)",
            "3: motorcycle (мотоцикл)",
            "4: airplane (самолет)",
            "5: bus (автобус)",
            "6: train (поезд)",
            "7: truck (грузовик)",
            "8: boat (лодка)",
            "9: traffic light (светофор)",
            "10: fire hydrant (пожарный гидрант)",
            "11: stop sign (знак остановки)",
            "12: parking meter (парковочный счетчик)",
            "13: bench (скамейка)",
            "14: bird (птица)",
            "15: cat (кошка)",
            "16: dog (собака)",
            "17: horse (лошадь)",
            "18: sheep (овца)",
            "19: cow (корова)",
            "20: elephant (слон)",
            "21: bear (медведь)",
            "22: zebra (зебра)",
            "23: giraffe (жираф)",
            "24: backpack (рюкзак)",
            "25: umbrella (зонт)",
            "26: handbag (сумка)",
            "27: tie (галстук)",
            "28: suitcase (чемодан)",
            "29: frisbee (фрисби)",
            "30: skis (лыжи)",
            "31: snowboard (сноуборд)",
            "32: sports ball (спортивный мяч)",
            "33: kite (воздушный змей)",
            "34: baseball bat (бейсбольная бита)",
            "35: baseball glove (бейсбольная перчатка)",
            "36: skateboard (скейтборд)",
            "37: surfboard (доска для серфинга)",
            "38: tennis racket (теннисная ракетка)",
            "39: bottle (бутылка)",
            "40: wine glass (бокал для вина)",
            "41: cup (чашка)",
            "42: fork (вилка)",
            "43: knife (нож)",
            "44: spoon (ложка)",
            "45: bowl (миска)",
            "46: banana (банан)",
            "47: apple (яблоко)",
            "48: sandwich (сэндвич)",
            "49: orange (апельсин)",
            "50: broccoli (брокколи)",
            "51: carrot (морковь)",
            "52: hot dog (хот-дог)",
            "53: pizza (пицца)",
            "54: donut (пончик)",
            "55: cake (торт)",
            "56: chair (стул)",
            "57: couch (диван)",
            "58: potted plant (комнатное растение)",
            "59: bed (кровать)",
            "60: dining table (обеденный стол)",
            "61: toilet (туалет)",
            "62: tv (телевизор)",
            "63: laptop (ноутбук)",
            "64: mouse (мышь)",
            "65: remote (пульт)",
            "66: keyboard (клавиатура)",
            "67: cell phone (мобильный телефон)",
            "68: microwave (микроволновка)",
            "69: oven (духовка)",
            "70: toaster (тостер)",
            "71: sink (раковина)",
            "72: refrigerator (холодильник)",
            "73: book (книга)",
            "74: clock (часы)",
            "75: vase (ваза)",
            "76: scissors (ножницы)",
            "77: teddy bear (плюшевый мишка)",
            "78: hair drier (фен)",
            "79: toothbrush (зубная щетка)"
        ]
        
        print(f"\n{Fore.MAGENTA}Выберите классы для обнаружения (пробел для выбора, Enter для подтверждения):{Style.RESET_ALL}")
        class_choices = create_interactive_menu("ВЫБОР КЛАССОВ", class_options, multi_select=True)
        
        if class_choices:
            # Извлекаем номера классов из выбранных опций
            classes = [choice for choice in class_choices]
            cmd.append("--classes")
            cmd.extend([str(c) for c in classes])
    
    # 7. Опции трекинга и отображения
    feature_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(feature_header)
    
    feature_options = [
        "Трекинг объектов (--track)",
        "Показать трекинг на отдельном видео (--show-track)",
        "Белые рамки (--white-boxes)",
        "Диагональный крест (--cross)",
        "Эффект datamosh (--datamosh)",
        "Добавить аудио из исходника (--add-audio)",
        "Расширенное сглаживание (--augment)",
        "Разрешить перезапись (--exist-ok)",
        "Класс-агностик NMS (--agnostic-nms)",
        "Уникальный цвет для каждого трека (--unique-track-color)"
    ]
    
    print(f"\n{Fore.MAGENTA}Выберите дополнительные опции (пробел для выбора, Enter для подтверждения):{Style.RESET_ALL}")
    feature_choices = create_interactive_menu("ДОПОЛНИТЕЛЬНЫЕ ОПЦИИ", feature_options, multi_select=True)
    
    # Словарь соответствия индекса опции и аргумента
    feature_args = {
        0: "--track",
        1: "--show-track",
        2: "--white-boxes",
        3: "--cross",
        4: "--datamosh",
        5: "--add-audio",
        6: "--augment",
        7: "--exist-ok",
        8: "--agnostic-nms",
        9: "--unique-track-color"
    }
    
    # Добавляем выбранные опции в командную строку
    for choice in feature_choices:
        cmd.append(feature_args[choice])
    
    # Если выбран крест, запрашиваем толщину
    if 3 in feature_choices:
        cross_options = [
            "1 (тонкий)",
            "2 (средний)",
            "3 (толстый)"
        ]
        print(f"\n{Fore.MAGENTA}Выберите толщину креста:{Style.RESET_ALL}")
        cross_choice = create_interactive_menu("ТОЛЩИНА КРЕСТА", cross_options)
        if cross_choice is not None:
            thickness = cross_choice + 1  # индексы с 0, но толщина с 1
            cmd.extend(["--cross-thickness", str(thickness)])
    
    # 8. Папка для сохранения результатов
    save_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'НАСТРОЙКИ СОХРАНЕНИЯ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(save_header)
    
    save_options = [
        "По умолчанию (runs/detect)",
        "Указать папку вручную"
    ]
    
    print(f"\n{Fore.MAGENTA}Куда сохранить результаты?{Style.RESET_ALL}")
    save_choice = create_interactive_menu("ПАПКА СОХРАНЕНИЯ", save_options)
    
    if save_choice == 1:
        print(f"\n{Fore.MAGENTA}Введите путь для сохранения результатов:{Style.RESET_ALL}")
        save_path = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
        cmd.extend(["--project", save_path])
    
    # 9. Примеры готовых команд
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Fore.MAGENTA}{speesh_logo}{Style.RESET_ALL}")
    
    # Отображаем бренды
    branding = f"""
{Fore.WHITE}╔═════════════════════════════════════════════════════════════════════════╗
║ {Fore.MAGENTA}speesh.ru{Fore.WHITE} | {Fore.MAGENTA}rbdclan.moscow{Fore.WHITE} | {Fore.MAGENTA}Sergey Golovchan © 2025{Fore.WHITE}               ║
╚═════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(branding)
    
    # Заголовок "Настройка завершена"
    complete_header = f"""
{Fore.WHITE}╔{'═' * 65}╗
║{Back.MAGENTA}{Fore.BLACK}{Style.BRIGHT}{'НАСТРОЙКА ЗАВЕРШЕНА':^65}{Style.RESET_ALL}{Fore.WHITE}║
╚{'═' * 65}╝{Style.RESET_ALL}
    """
    print(complete_header)
    
    # Выводим итоговую команду в красивой рамке
    cmd_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'ЗАПУСК С ПАРАМЕТРАМИ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(cmd_header)
    
    cmd_str = " ".join(cmd)
    cmd_box = f"""
{Fore.WHITE}╔{'═' * 65}╗
║{Fore.WHITE} python detect_or_track.py {cmd_str:<34}{Fore.WHITE}║
╚{'═' * 65}╝{Style.RESET_ALL}
    """
    print(cmd_box)
    
    # Примеры готовых команд в красивой рамке
    examples_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'ПРИМЕРЫ ГОТОВЫХ КОМАНД ДЛЯ БУДУЩЕГО ИСПОЛЬЗОВАНИЯ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(examples_header)
    
    # Создаем рамку для примеров
    print(f"{Fore.WHITE}╔{'═' * 65}╗{Style.RESET_ALL}")
    examples = [
        "python detect_or_track.py --weights yolov7.pt --source video.mp4 --img-size 640 --track",
        "python detect_or_track.py --weights yolov7.pt --source 0 --img-size 320 --conf-thres 0.4 --track",
        "python detect_or_track.py --weights yolov7.pt --source video.mp4 --img-size 1024 --track --show-track --white-boxes",
        "python detect_or_track.py --weights yolov7.pt --source video.mp4 --track --cross --cross-thickness 2 --datamosh",
        "python detect_or_track.py --weights yolov7-tiny.pt --source video.mp4 --device 0 --img-size 416 --track"
    ]
    
    for example in examples:
        print(f"{Fore.WHITE}║ {Fore.MAGENTA}{example:<63}{Fore.WHITE}║{Style.RESET_ALL}")
    
    print(f"{Fore.WHITE}╚{'═' * 65}╝{Style.RESET_ALL}")
    
    # Запрос подтверждения запуска
    confirm_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'ЗАПУСК':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(confirm_header)
    
    confirm_options = ["Запустить сейчас", "Выйти без запуска"]
    confirm_choice = create_interactive_menu("ЗАПУСК", confirm_options)
    
    if confirm_choice == 0:
        return cmd
    else:
        print(f"{Fore.MAGENTA}Выход из программы.{Style.RESET_ALL}")
        sys.exit(0)


def create_interactive_menu(title, options, multi_select=False):
    """
    Создает интерактивное меню с выбором через стрелки.
    
    Args:
        title: Заголовок меню
        options: Список опций для выбора
        multi_select: Разрешить выбор нескольких опций (с помощью пробела)
    
    Returns:
        Индекс выбранной опции или список индексов, если multi_select=True
    """
    # Используем упрощенное меню без curses
    # Константы для специальных клавиш в Windows
    KEY_UP = 72
    KEY_DOWN = 80
    KEY_ENTER = 13
    KEY_SPACE = 32
    KEY_ESC = 27
    KEY_Q = 113

    # Текущая выбранная опция
    current_row = 0
    # Для мультивыбора - список выбранных опций
    selected = [False] * len(options)
    
    # Отображение меню
    while True:
        # Очистка экрана
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Отображаем логотип Speesh
        speesh_logo = """
     
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠶⠚⠉⢉⣩⠽⠟⠛⠛⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⠉⠀⢀⣠⠞⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡞⠁⠀⠀⣰⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣠⡤⠤⠄⢤⣄⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢰⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠴⠒⠋⠉⠀⠀⠀⣀⣤⠴⠒⠋⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⡄⠀⠀⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⢳⡄⢀⡴⠚⠉⠀⠀⠀⠀⠀⣠⠴⠚⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠘⣧⠀⠀⠀⠀⠀⠀⠀⠀⣰⠃⠀⠀⠹⡏⠀⠀⠀⠀⠀⣀⣴⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠳⢬⣳⣄⣠⠤⠤⠶⠶⠒⠋⠀⠀⠀⠀⠹⡀⠀⠀⠀⠀⠈⠉⠛⠲⢦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠤⠖⠋⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢳⠦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣠⠖⠋⠀⠀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠀⢃⠈⠙⠲⣄⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢠⠞⠁⠀⠀⠀⢀⢾⠃⠀⠀⠀⠀⠀⠀⠀⠀⢢⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⣹⠮⣄⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣰⠋⠀⠀⢀⡤⡴⠃⠈⠦⣀⠀⠀⠀⠀⠀⠀⢀⣷⢸⠀⠀⠀⠀⢀⣀⠘⡄⠤⠤⢤⠔⠒⠂⠉⠁⠀⠀⠀⠑⢄⡀⠀⠀⠙⢦⡀⠀⠀⠀
⠀⠀⠀⠀⣼⠃⠀⠀⢠⣞⠟⠀⠀⠀⡄⠀⠉⠒⠢⣤⣤⠄⣼⢻⠸⠀⠀⠀⠀⠉⢤⠀⢿⡖⠒⠊⢦⠤⠤⣀⣀⡀⠀⠀⠀⠈⠻⡝⠲⢤⣀⠙⢦⠀⠀
⠀⠀⠀⢰⠃⠀⠀⣴⣿⠎⠀⠀⢀⣜⠤⠄⢲⠎⠉⠀⠀⡼⠸⠘⡄⡇⠀⠀⠀⠀⢸⠀⢸⠘⢆⠀⠘⡄⠀⠀⠀⢢⠉⠉⠀⠒⠒⠽⡄⠀⠈⠙⠮⣷⡀
⠀⠀⠀⡟⠀⠀⣼⢻⠧⠐⠂⠉⡜⠀⠀⡰⡟⠀⠀⠀⡰⠁⡇⠀⡇⡇⠀⠀⠀⠀⢺⠇⠀⣆⡨⢆⠀⢽⠀⠀⠀⠈⡷⡄⠀⠀⠀⠀⠹⡄⠀⠀⠀⠈⠁
⠀⠀⢸⠃⠀⠀⢃⠎⠀⠀⠀⣴⠃⠀⡜⠹⠁⠀⠀⡰⠁⢠⠁⠀⢸⢸⠀⠀⠀⢠⡸⢣⠔⡏⠀⠈⢆⠀⣇⠀⠀⠀⢸⠘⢆⠀⠀⠀⠀⢳⠀⠀⠀⠀⠀
⠀⠀⢸⠀⠀⠀⡜⠀⠀⢀⡜⡞⠀⡜⠈⠏⠀⠈⡹⠑⠒⠼⡀⠀⠀⢿⠀⠀⠀⢀⡇⠀⢇⢁⠀⠀⠈⢆⢰⠀⠀⠀⠈⡄⠈⢢⠀⠀⠀⠈⣇⠀⠀⠀⠀
⠀⠀⢸⡀⠀⢰⠁⠀⢀⢮⠀⠇⡜⠀⠘⠀⠀⢰⠃⠀⠀⡇⠈⠁⠀⢘⡄⠀⠀⢸⠀⠀⣘⣼⠤⠤⠤⣈⡞⡀⠀⠀⠀⡇⠰⡄⢣⡀⠀⠀⢻⠀⠀⠀⠀
⠀⠀⠈⡇⠀⡜⠀⢀⠎⢸⢸⢰⠁⠀⠄⠀⢠⠃⠀⠀⢸⠀⠀⠀⠀⠀⡇⠀⠀⡆⠀⠀⣶⣿⡿⠿⡛⢻⡟⡇⠀⠀⠀⡇⠀⣿⣆⢡⠀⠀⢸⡇⠀⠀⠀
⠀⠀⢠⡏⠀⠉⢢⡎⠀⡇⣿⠊⠀⠀⠀⢠⡏⠀⠀⠀⠎⠀⠀⠀⠀⠀⡇⠀⡸⠀⠀⠀⡇⠀⢰⡆⡇⢸⢠⢹⠀⠀⠀⡇⠀⢹⠈⢧⣣⠀⠘⡇⠀⠀⠀
⠀⠀⢸⡇⠀⠀⠀⡇⠀⡇⢹⠀⠀⠀⢀⡾⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢠⠃⠀⠀⠠⠟⡯⣻⣇⢃⠇⢠⠏⡇⠀⢸⡆⠀⢸⠀⠈⢳⡀⠀⡇⠀⠀⠀
⠀⠀⠀⣇⠀⡔⠋⡇⠀⢱⢼⠀⠀⡂⣼⡇⢹⣶⣶⣶⣤⣤⣀⠀⠀⠀⣇⠇⠀⠀⠀⠀⣶⡭⢃⣏⡘⠀⡎⠀⠇⠀⡾⣷⠀⣼⠀⠀⠀⢻⡄⡇⠀⠀⠀
⠀⠀⠀⣹⠜⠋⠉⠓⢄⡏⢸⠀⠀⢳⡏⢸⠹⢀⣉⢭⣻⡽⠿⠛⠓⠀⠋⠀⠀⠀⠀⠀⠘⠛⠛⠓⠀⡄⡇⠀⢸⢰⡇⢸⡄⡟⠀⠀⠀⠀⢳⡇⠀⠀⠀
⠀⣠⠞⠁⠀⠀⠀⠀⠀⢙⠌⡇⠀⣿⠁⠀⡇⡗⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⠀⠀⠀⠀⠀⠀⠁⠁⠀⢸⣼⠀⠈⣇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢸⠁⠀⠀⢀⡠⠔⠚⠉⠉⢱⣇⢸⢧⠀⠀⠸⣱⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⡤⠦⡔⠀⠀⠀⠀⠀⢀⡼⠀⠀⣼⡏⠀⠀⢹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢸⠀⠀⠀⠋⠀⠀⠀⢀⡠⠤⣿⣾⣇⣧⠀⠀⢫⡆⠀⠀⠀⠀⠀⠀⠀⢨⠀⠀⣠⠇⠀⠀⢀⡠⣶⠋⠀⠀⡸⣾⠁⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢸⡄⠀⠀⠀⠀⠠⠊⠁⠀⠀⢸⢃⠘⡜⡵⡀⠈⢿⡱⢲⡤⠤⢀⣀⣀⡀⠉⠉⣀⡠⡴⠚⠉⣸⢸⠀⠀⢠⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⢧⠀⠀⠀⠀⠀⠀⠀⣀⠤⠚⠚⣤⣵⡰⡑⡄⠀⢣⡈⠳⡀⠀⠀⠀⢨⡋⠙⣆⢸⠀⠀⣰⢻⡎⠀⠀⡎⡇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠈⢷⡀⠀⠀⠀⠀⠀⠁⠀⠀⠀⡸⢌⣳⣵⡈⢦⡀⠳⡀⠈⢦⡀⠀⠘⠏⠲⣌⠙⢒⠴⡧⣸⡇⠀⡸⢸⠇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢠⣿⠢⡀⠀⠀⠀⠠⠄⡖⠋⠀⠀⠙⢿⣳⡀⠑⢄⠹⣄⡀⠙⢄⡠⠤⠒⠚⡖⡇⠀⠘⣽⡇⢠⠃⢸⢀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⣾⠃⠀⠀⠀⠀⠀⢀⡼⣄⠀⠀⠀⠀⠀⠑⣽⣆⠀⠑⢝⡍⠒⠬⢧⣀⡠⠊⠀⠸⡀⠀⢹⡇⡎⠀⡿⢸⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⡼⠁⠀⠀⠀⠀⠀⠀⢀⠻⣺⣧⠀⠀⠀⠰⢢⠈⢪⡷⡀⠀⠙⡄⠀⠀⠱⡄⠀⠀⠀⢧⠀⢸⡻⠀⢠⡇⣾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢰⠇⠀⠀⠀⠀⠀⠀⠀⢸⠀⡏⣿⠀⠀⠀⠀⢣⢇⠀⠑⣄⠀⠀⠸⡄⠀⠀⠘⡄⠀⠀⠸⡀⢸⠁⠀⡾⢰⡏⢳⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
           `"""
       
        print(f"{Fore.MAGENTA}{speesh_logo}{Style.RESET_ALL}")
        
        # Отображаем бренды
        branding = f"""
{Fore.WHITE}╔═════════════════════════════════════════════════════════════════╗
║ {Fore.MAGENTA}speesh.ru{Fore.WHITE} | {Fore.MAGENTA}rbdclan.moscow{Fore.WHITE} | {Fore.MAGENTA}Sergey Golovchan © 2025{Fore.WHITE}      ║
╚═════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """
        print(branding)
        
        # Отображаем заголовок с красивым форматированием
        header = f"""
{Fore.WHITE}╔{'═' * (len(title) + 8)}╗
║  {Back.MAGENTA}{Fore.BLACK}{Style.BRIGHT} {title} {Style.RESET_ALL}{Fore.WHITE}  ║
╚{'═' * (len(title) + 8)}╝{Style.RESET_ALL}
        """
        print(header)
        
        # Отображаем опции с отступами и форматированием
        print(f"{Fore.WHITE}┌{'─' * 60}┐{Style.RESET_ALL}")
        for idx, option in enumerate(options):
            # Создаем строку опции
            if multi_select:
                prefix = f"{Fore.MAGENTA}[{Fore.WHITE}{'X' if selected[idx] else ' '}{Fore.MAGENTA}]{Style.RESET_ALL} "
                option_str = f"{prefix}{option}"
            else:
                option_str = option
            
            # Выводим опцию с соответствующим цветом
            if idx == current_row:
                print(f"{Fore.WHITE}│ {Back.MAGENTA}{Fore.BLACK}{Style.BRIGHT} {option_str:<58} {Style.RESET_ALL}{Fore.WHITE}│{Style.RESET_ALL}")
            else:
                print(f"{Fore.WHITE}│ {Fore.WHITE}{option_str:<58}{Fore.WHITE} │{Style.RESET_ALL}")
        print(f"{Fore.WHITE}└{'─' * 60}┘{Style.RESET_ALL}")
        
        # Выводим подсказки снизу с лучшим форматированием
        footer = f"""
{Fore.WHITE}╔═════════════════════════════════════════════════════════════════╗
║ {Fore.MAGENTA}↑/↓{Fore.WHITE}: Навигация {' ':^8} """
        
        if multi_select:
            footer += f"{Fore.MAGENTA}Пробел{Fore.WHITE}: Выбор {' ':^8} {Fore.MAGENTA}Enter{Fore.WHITE}: Подтвердить {' ':^3} {Fore.MAGENTA}q{Fore.WHITE}: Выход "
        else:
            footer += f"{Fore.MAGENTA}Enter{Fore.WHITE}: Выбор {' ':^13} {Fore.MAGENTA}q{Fore.WHITE}: Выход {' ':^21}"
            
        footer += f"""║
╚═════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """
        print(footer)
        
        try:
            # Обработка ввода (для Windows)
            key = ord(msvcrt.getch())
            
            # Если была нажата специальная клавиша (стрелки)
            if key == 224:  # Префикс для специальных клавиш
                key = ord(msvcrt.getch())  # Получаем код клавиши
                
                if key == KEY_UP and current_row > 0:
                    current_row -= 1
                elif key == KEY_DOWN and current_row < len(options) - 1:
                    current_row += 1
            
            # Обычные клавиши
            elif key == KEY_ENTER:  # Enter
                if multi_select:
                    # Возвращаем индексы выбранных опций
                    result = [i for i, s in enumerate(selected) if s]
                    if not result and current_row is not None:
                        # Если ничего не выбрано, возвращаем текущую позицию
                        result = [current_row]
                    return result
                else:
                    # Возвращаем индекс выбранной опции
                    return current_row
            
            elif key == KEY_SPACE and multi_select:  # Пробел для мультивыбора
                # Переключение выбора для мультивыбора
                selected[current_row] = not selected[current_row]
            
            elif key == KEY_ESC or key == KEY_Q:  # Выход
                return None if not multi_select else []
        
        except Exception as e:
            print(f"Ошибка при обработке ввода: {e}")
            # Упрощенное меню с вводом цифр в случае ошибок
            print("\nИспользуйте цифры для выбора опции:")
            for i, option in enumerate(options):
                print(f"{i+1}. {option}")
            try:
                choice = input("\nВыберите номер: ")
                if choice.isdigit() and 0 < int(choice) <= len(options):
                    return int(choice) - 1
                else:
                    print("Некорректный ввод. Попробуйте еще раз.")
                    time.sleep(1.5)
            except:
                # В случае любой ошибки, возвращаем первый вариант
                return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help="don't increment name if exists")
    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font thickness')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--unique-track-color', action='store_true', help='use unique color for each track')
    parser.add_argument('--datamosh', action='store_true', help='apply datamosh effect to tracking video')
    parser.add_argument('--white-boxes', action='store_true', help='use white color for all bounding boxes')
    parser.add_argument('--cross', action='store_true', help='draw diagonal cross lines inside bounding boxes')
    parser.add_argument('--cross-thickness', type=int, default=1, choices=[1, 2, 3], help='thickness of cross lines (1-3)')
    parser.add_argument('--add-audio', action='store_true', help='add audio from source video to tracking output')
    parser.add_argument('--help-prompt', action='store_true', help='show help and prompt for parameters')

    # Проверяем, были ли переданы аргументы командной строки
    if len(sys.argv) == 1:
        # Если аргументов нет, запускаем интерактивный режим
        try:
            args = print_help_and_prompt()
            if args:
                opt = parser.parse_args(args)
            else:
                sys.exit(0)
        except Exception as e:
            print(f"Ошибка при использовании интерактивного меню: {e}")
            print("Запуск стандартного режима помощи...")
            parser.print_help()
            sys.exit(1)
    else:
        opt = parser.parse_args()
        # Если передан флаг --help-prompt, запускаем интерактивный режим
        if opt.help_prompt:
            try:
                args = print_help_and_prompt()
                if args:
                    opt = parser.parse_args(args)
                else:
                    sys.exit(0)
            except Exception as e:
                print(f"Ошибка при использовании интерактивного меню: {e}")
                print("Запуск стандартного режима помощи...")
                parser.print_help()
                sys.exit(1)

    random.seed(1)
    
    # Сохраняем источник видео в глобальной переменной
    global source_video_path
    source_video_path = None
    
    # Обработка прерывания Ctrl+C
    def handle_interrupt(sig, frame):
        print(f"\n{Fore.YELLOW}Обработка прервана пользователем.{Style.RESET_ALL}")
        restore_original_output()
        
        # Запрашиваем добавление аудио, если видео было обработано
        if 'opt' in globals() and hasattr(opt, 'add_audio') and opt.add_audio:
            process_video_with_audio()
        sys.exit(0)
    
    # Регистрируем обработчик прерывания
    signal.signal(signal.SIGINT, handle_interrupt)
    
    with torch.no_grad():
        detect()
    
    # Запрашиваем добавление аудио после завершения обработки
    if 'opt' in globals() and hasattr(opt, 'add_audio') and opt.add_audio:
        process_video_with_audio()

# 2. Добавим функцию для обработки видео с аудио после завершения работы
def process_video_with_audio():
    """Добавляет аудио к последнему обработанному видео."""
    global source_video_path
    
    try:
        # Восстанавливаем оригинальный вывод, если он был изменен
        if 'restore_original_output' in globals():
            restore_original_output()
        
        print(f"\n{Fore.MAGENTA}Обработка видео с аудио...{Style.RESET_ALL}")
        
        # Если путь к исходному видео не был сохранен, используем параметр source
        if source_video_path is None and 'opt' in globals():
            source_video_path = opt.source
        
        # Проверяем, существует ли исходный файл
        if not os.path.isfile(source_video_path):
            print(f"{Fore.RED}Исходный файл не найден: {source_video_path}{Style.RESET_ALL}")
            return
        
        # Находим последнюю папку с экспериментами
        base_dir = opt.project if 'opt' in globals() else "runs/detect"
        exp_dirs = [d for d in os.listdir(base_dir) if d.startswith("exp")]
        if not exp_dirs:
            print(f"{Fore.RED}Папки с результатами не найдены в {base_dir}{Style.RESET_ALL}")
            return
        
        latest_exp = max(exp_dirs, key=lambda d: int(d.replace("exp", "") or "0"))
        latest_path = os.path.join(base_dir, latest_exp)
        
        # Находим трековое видео
        track_videos = [f for f in os.listdir(latest_path) if f.startswith('tracks_') and f.endswith('.mp4')]
        if not track_videos:
            # Если трекового видео нет, ищем обычное обработанное видео
            videos = [f for f in os.listdir(latest_path) if f.endswith('.mp4') and not f.startswith('audio_')]
            if not videos:
                print(f"{Fore.RED}Обработанные видео не найдены в {latest_path}{Style.RESET_ALL}")
                return
            track_video_path = os.path.join(latest_path, videos[0])
        else:
            track_video_path = os.path.join(latest_path, track_videos[0])
        
        # Определяем имя выходного файла
        output_name = f"audio_{os.path.basename(track_video_path)}"
        output_path = os.path.join(latest_path, output_name)
        
        # Добавляем аудио
        print(f"{Fore.MAGENTA}Добавление аудио из {Fore.WHITE}{source_video_path}{Fore.MAGENTA} в {Fore.WHITE}{track_video_path}{Style.RESET_ALL}")
        
        success = add_audio_to_video(
            input_video_with_audio=source_video_path,
            output_video_without_audio=track_video_path,
            final_output_path=output_path
        )
        
        if success:
            print(f"{Fore.GREEN}Аудио успешно добавлено в видео: {Fore.WHITE}{output_path}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Не удалось добавить аудио.{Style.RESET_ALL}")
    
    except Exception as e:
        print(f"{Fore.RED}Ошибка при обработке видео с аудио: {e}{Style.RESET_ALL}")

# 3. Модифицируем функцию source_choice для сохранения пути
def print_help_and_prompt():
    """
    Интерактивный выбор опций с использованием навигации стрелками и возврат аргументов для argparse
    """
    # Очистка экрана консоли
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # ASCII-арт логотип
    owl_art = """                                                                             ==                     ==
                 <^\()/^>               <^\()/^>
                  \/  \/                 \/  \/
                   /__\      .  '  .      /__\ 
      ==            /\    .     |     .    /\            ==
   <^\()/^>       !_\/       '  |  '       \/_!       <^\()/^>
    \/  \/     !_/I_||  .  '   \'/   '  .  ||_I\_!     \/  \/
     /__\     /I_/| ||      -== + ==-      || |\_I\     /__\
     /_ \   !//|  | ||  '  .   /.\   .  '  || |  |\\!   /_ \
    (-   ) /I/ |  | ||       .  |  .       || |  | \I\ (=   )
     \__/!//|  |  | ||    '     |     '    || |  |  |\\!\__/
     /  \I/ |  |  | ||       '  .  '    *  || |  |  | \I/  \
    {_ __}  |  |  | ||                     || |  |  |  {____}
 _!__|= ||  |  |  | ||   *      +          || |  |  |  ||  |__!_
 _I__|  ||__|__|__|_||          A          ||_|__|__|__||- |__I_
 -|--|- ||--|--|--|-||       __/_\__  *    ||-|--|--|--||= |--|-
  |  |  ||  |  |  | ||      /\-'o'-/\      || |  |  |  ||  |  |
  |  |= ||  |  |  | ||     _||:<_>:||_     || |  |  |  ||= |  |
  |  |- ||  |  |  | || *  /\_/=====\_/\  * || |  |  |  ||= |  |
  |  |- ||  |  |  | ||  __|:_:_[I]_:_:|__  || |  |  |  ||- |  | 
 _|__|  ||__|__|__|_||:::::::::::::::::::::||_|__|__|__||  |__|_
 -|--|= ||--|--|--|-||:::::::::::::::::::::||-|--|--|--||- |--|-
  jgs|- ||  |  |  | ||:::::::::::::::::::::|| |  |  |  ||= |  | 
~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^~~~~~~~~~
    """
    
    # Speesh logo
    speesh_logo = """
       .-.,     ,.-.
 '-.  /:::\\   //:::\\  .-'
 '-.\|':':' `"` ':':'|/.-'
 `-./`. .-=-. .-=-. .`\.-`
   /=- /     |     \\ -=\\
  ;   |      |      |   ;
  |=-.|______|______|.-=|
  |==  \\  0 /_\\ 0  /  ==|
  |=   /'---( )---'\\   =|
   \\   \\:   .'.   :/   /
    `\\= '--`   `--' =/'
       `-=._     _.=-'
           `"""
 
    
    # Выводим комбинированный ASCII-арт с розовым цветом
    print(f"{Fore.MAGENTA}{owl_art}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{speesh_logo}{Style.RESET_ALL}")
    
    # Отображаем бренды
    branding = f"""
{Fore.WHITE}╔═════════════════════════════════════════════════════════════════════════╗
║ {Fore.MAGENTA}speesh.ru{Fore.WHITE} | {Fore.MAGENTA}rbdclan.moscow{Fore.WHITE} | {Fore.MAGENTA}Sergey Golovchan © 2025{Fore.WHITE}               ║
╚═════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(branding)
    
    # Выводим заголовок
    title_box = f"""
{Fore.WHITE}╔{'═' * 65}╗
║{Back.MAGENTA}{Fore.BLACK}{Style.BRIGHT}{'YOLO v7 DETECTION & TRACKING SYSTEM':^65}{Style.RESET_ALL}{Fore.WHITE}║
╚{'═' * 65}╝{Style.RESET_ALL}
    """
    print(title_box)
    
    # Собираем команду на основе выбора пользователя
    cmd = []
    
    # 1. Выбор модели
    model_options = [
        "yolov7.pt (стандартная модель)",
        "yolov7-tiny.pt (быстрая, менее точная)",
        "yolov7-w6.pt (большая, более точная)",
        "Свой вариант (ввести путь вручную)"
    ]
    
    print(f"\n{Fore.MAGENTA}Выберите модель:{Style.RESET_ALL}")
    model_choice = create_interactive_menu("ВЫБОР МОДЕЛИ", model_options)
    
    if model_choice == 0:
        weights = "yolov7.pt"
    elif model_choice == 1:
        weights = "yolov7-tiny.pt"
    elif model_choice == 2:
        weights = "yolov7-w6.pt"
    elif model_choice == 3:
        print(f"\n{Fore.MAGENTA}Введите путь к файлу весов:{Style.RESET_ALL}")
        weights = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
    else:
        print(f"{Fore.MAGENTA}Выход из программы.{Style.RESET_ALL}")
        sys.exit(0)
    
    cmd.extend(["--weights", weights])
    
    # 2. Выбор источника
    print(f"\n{Fore.MAGENTA}Выберите источник:{Style.RESET_ALL}")
    source_options = [
        "Видеофайл (выбрать файл)",
        "Веб-камера (0)",
        "Другая камера (ввести номер)",
        "Папка с изображениями"
    ]

    source_choice = create_interactive_menu("ВЫБОР ИСТОЧНИКА", source_options)
    
    global source_video_path
    
    if source_choice == 0:
        print(f"\n{Fore.MAGENTA}Введите путь к видеофайлу:{Style.RESET_ALL}")
        source = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        source_video_path = source  # Сохраняем путь к исходному видео
        print(f"{Style.RESET_ALL}")
    elif source_choice == 1:
        source = "0"
    elif source_choice == 2:
        print(f"\n{Fore.MAGENTA}Введите номер камеры:{Style.RESET_ALL}")
        source = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
    elif source_choice == 3:
        print(f"\n{Fore.MAGENTA}Введите путь к папке с изображениями:{Style.RESET_ALL}")
        source = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
    else:
        print(f"{Fore.MAGENTA}Выход из программы.{Style.RESET_ALL}")
        sys.exit(0)
    
    cmd.extend(["--source", source])
    
    # 3. Выбор размера изображения
    img_size_options = [
        "640 (стандартный размер, баланс скорости и точности)",
        "320 (быстрее, менее точно)",
        "416 (компромисс)",
        "512 (компромисс)",
        "768 (точнее, медленнее)",
        "1024 (самый точный, самый медленный)",
        "Свой вариант (ввести вручную)"
    ]
    # RBDCLAN
    print(f"\n{Fore.MAGENTA}Выберите размер входного изображения:{Style.RESET_ALL}")
    img_size_choice = create_interactive_menu("РАЗМЕР ИЗОБРАЖЕНИЯ", img_size_options)
    
    img_sizes = [640, 320, 416, 512, 768, 1024]
    if img_size_choice < 6:
        img_size = img_sizes[img_size_choice]
    elif img_size_choice == 6:
        print(f"\n{Fore.MAGENTA}Введите размер изображения (целое число):{Style.RESET_ALL}")
        img_size = int(input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip())
        print(f"{Style.RESET_ALL}")
    else:
        print(f"{Fore.MAGENTA}Выход из программы.{Style.RESET_ALL}")
        sys.exit(0)
    
    cmd.extend(["--img-size", str(img_size)])
    
    # 4. Выбор порогов уверенности и IOU
    section_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'НАСТРОЙКИ ДЕТЕКЦИИ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(section_header)
    
    conf_options = [
        "0.25 (стандартный, умеренные требования)",
        "0.1 (больше объектов, больше ложных срабатываний)",
        "0.4 (меньше объектов, меньше ложных срабатываний)",
        "0.5 (строгие требования к уверенности)",
        "Свой вариант (ввести вручную)"
    ]
    # LOL
    conf_choice = create_interactive_menu("ПОРОГ УВЕРЕННОСТИ", conf_options)
    conf_values = [0.25, 0.1, 0.4, 0.5]
    
    if conf_choice < 4:
        conf_thres = conf_values[conf_choice]
    elif conf_choice == 4:
        print(f"\n{Fore.MAGENTA}Введите порог уверенности (от 0.0 до 1.0):{Style.RESET_ALL}")
        conf_thres = float(input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip())
        print(f"{Style.RESET_ALL}")
    else:
        conf_thres = 0.25
    
    cmd.extend(["--conf-thres", str(conf_thres)])
    
    iou_options = [
        "0.45 (стандартный)",
        "0.3 (менее строгое подавление пересечений)",
        "0.6 (более строгое подавление пересечений)",
        "Свой вариант (ввести вручную)"
    ]
    
    iou_choice = create_interactive_menu("ПОРОГ IOU", iou_options)
    iou_values = [0.45, 0.3, 0.6]
    
    if iou_choice < 3:
        iou_thres = iou_values[iou_choice]
    elif iou_choice == 3:
        print(f"\n{Fore.MAGENTA}Введите порог IOU (от 0.0 до 1.0):{Style.RESET_ALL}")
        iou_thres = float(input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip())
        print(f"{Style.RESET_ALL}")
    else:
        iou_thres = 0.45
    
    cmd.extend(["--iou-thres", str(iou_thres)])
    
    # 5. Устройство для вычислений
    hw_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'АППАРАТНАЯ ЧАСТЬ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(hw_header)
    
    device_options = [
        "CPU",
        "GPU:0 (первая видеокарта)",
        "GPU:1 (вторая видеокарта, если есть)",
        "GPU:2 (третья видеокарта, если есть)",
        "Свой вариант (ввести вручную)"
    ]
    
    print(f"\n{Fore.MAGENTA}Выберите устройство для вычислений:{Style.RESET_ALL}")
    device_choice = create_interactive_menu("ВЫБОР УСТРОЙСТВА", device_options)
    
    device_values = ["cpu", "0", "1", "2"]
    if device_choice < 4:
        device = device_values[device_choice]
    elif device_choice == 4:
        print(f"\n{Fore.MAGENTA}Введите устройство (cpu, 0, 1, 2, ...):{Style.RESET_ALL}")
        device = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
    else:
        device = "cpu"
    
    cmd.extend(["--device", device])
    # Я просто не выхожу на улицу
    # 6. Фильтрация классов
    classes_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'НАСТРОЙКА КЛАССОВ ОБЪЕКТОВ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(classes_header)
    
    print(f"\n{Fore.MAGENTA}Хотите фильтровать классы объектов?{Style.RESET_ALL}")
    filter_options = ["Нет (обнаруживать все классы)", "Да (выбрать классы)"]
    filter_choice = create_interactive_menu("ФИЛЬТРАЦИЯ КЛАССОВ", filter_options)
    
    if filter_choice == 1:
        class_options = [
            "0: person (человек)",
            "1: bicycle (велосипед)",
            "2: car (машина)",
            "3: motorcycle (мотоцикл)",
            "4: airplane (самолет)",
            "5: bus (автобус)",
            "6: train (поезд)",
            "7: truck (грузовик)",
            "8: boat (лодка)",
            "9: traffic light (светофор)",
            "10: fire hydrant (пожарный гидрант)",
            "11: stop sign (знак остановки)",
            "12: parking meter (парковочный счетчик)",
            "13: bench (скамейка)",
            "14: bird (птица)",
            "15: cat (кошка)",
            "16: dog (собака)",
            "17: horse (лошадь)",
            "18: sheep (овца)",
            "19: cow (корова)",
            "20: elephant (слон)",
            "21: bear (медведь)",
            "22: zebra (зебра)",
            "23: giraffe (жираф)",
            "24: backpack (рюкзак)",
            "25: umbrella (зонт)",
            "26: handbag (сумка)",
            "27: tie (галстук)",
            "28: suitcase (чемодан)",
            "29: frisbee (фрисби)",
            "30: skis (лыжи)",
            "31: snowboard (сноуборд)",
            "32: sports ball (спортивный мяч)",
            "33: kite (воздушный змей)",
            "34: baseball bat (бейсбольная бита)",
            "35: baseball glove (бейсбольная перчатка)",
            "36: skateboard (скейтборд)",
            "37: surfboard (доска для серфинга)",
            "38: tennis racket (теннисная ракетка)",
            "39: bottle (бутылка)",
            "40: wine glass (бокал для вина)",
            "41: cup (чашка)",
            "42: fork (вилка)",
            "43: knife (нож)",
            "44: spoon (ложка)",
            "45: bowl (миска)",
            "46: banana (банан)",
            "47: apple (яблоко)",
            "48: sandwich (сэндвич)",
            "49: orange (апельсин)",
            "50: broccoli (брокколи)",
            "51: carrot (морковь)",
            "52: hot dog (хот-дог)",
            "53: pizza (пицца)",
            "54: donut (пончик)",
            "55: cake (торт)",
            "56: chair (стул)",
            "57: couch (диван)",
            "58: potted plant (комнатное растение)",
            "59: bed (кровать)",
            "60: dining table (обеденный стол)",
            "61: toilet (туалет)",
            "62: tv (телевизор)",
            "63: laptop (ноутбук)",
            "64: mouse (мышь)",
            "65: remote (пульт)",
            "66: keyboard (клавиатура)",
            "67: cell phone (мобильный телефон)",
            "68: microwave (микроволновка)",
            "69: oven (духовка)",
            "70: toaster (тостер)",
            "71: sink (раковина)",
            "72: refrigerator (холодильник)",
            "73: book (книга)",
            "74: clock (часы)",
            "75: vase (ваза)",
            "76: scissors (ножницы)",
            "77: teddy bear (плюшевый мишка)",
            "78: hair drier (фен)",
            "79: toothbrush (зубная щетка)"
        ]
        
        print(f"\n{Fore.MAGENTA}Выберите классы для обнаружения (пробел для выбора, Enter для подтверждения):{Style.RESET_ALL}")
        class_choices = create_interactive_menu("ВЫБОР КЛАССОВ", class_options, multi_select=True)
        
        if class_choices:
            # Извлекаем номера классов из выбранных опций
            classes = [choice for choice in class_choices]
            cmd.append("--classes")
            cmd.extend([str(c) for c in classes])
    
    # 7. Опции трекинга и отображения
    feature_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(feature_header)
    
    feature_options = [
        "Трекинг объектов (--track)",
        "Показать трекинг на отдельном видео (--show-track)",
        "Белые рамки (--white-boxes)",
        "Диагональный крест (--cross)",
        "Эффект datamosh (--datamosh)",
        "Добавить аудио из исходника (--add-audio)",
        "Расширенное сглаживание (--augment)",
        "Разрешить перезапись (--exist-ok)",
        "Класс-агностик NMS (--agnostic-nms)",
        "Уникальный цвет для каждого трека (--unique-track-color)"
    ]
    
    print(f"\n{Fore.MAGENTA}Выберите дополнительные опции (пробел для выбора, Enter для подтверждения):{Style.RESET_ALL}")
    feature_choices = create_interactive_menu("ДОПОЛНИТЕЛЬНЫЕ ОПЦИИ", feature_options, multi_select=True)
    
    # Словарь соответствия индекса опции и аргумента
    feature_args = {
        0: "--track",
        1: "--show-track",
        2: "--white-boxes",
        3: "--cross",
        4: "--datamosh",
        5: "--add-audio",
        6: "--augment",
        7: "--exist-ok",
        8: "--agnostic-nms",
        9: "--unique-track-color"
    }
    
    # Добавляем выбранные опции в командную строку
    for choice in feature_choices:
        cmd.append(feature_args[choice])
    
    # Если выбран крест, запрашиваем толщину
    if 3 in feature_choices:
        cross_options = [
            "1 (тонкий)",
            "2 (средний)",
            "3 (толстый)"
        ]
        print(f"\n{Fore.MAGENTA}Выберите толщину креста:{Style.RESET_ALL}")
        cross_choice = create_interactive_menu("ТОЛЩИНА КРЕСТА", cross_options)
        if cross_choice is not None:
            thickness = cross_choice + 1  # индексы с 0, но толщина с 1
            cmd.extend(["--cross-thickness", str(thickness)])
    
    # 8. Папка для сохранения результатов
    save_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'НАСТРОЙКИ СОХРАНЕНИЯ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(save_header)
    
    save_options = [
        "По умолчанию (runs/detect)",
        "Указать папку вручную"
    ]
    
    print(f"\n{Fore.MAGENTA}Куда сохранить результаты?{Style.RESET_ALL}")
    save_choice = create_interactive_menu("ПАПКА СОХРАНЕНИЯ", save_options)
    
    if save_choice == 1:
        print(f"\n{Fore.MAGENTA}Введите путь для сохранения результатов:{Style.RESET_ALL}")
        save_path = input(f"{Fore.WHITE}> {Fore.MAGENTA}").strip()
        print(f"{Style.RESET_ALL}")
        cmd.extend(["--project", save_path])
    
    # 9. Примеры готовых команд
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Fore.MAGENTA}{speesh_logo}{Style.RESET_ALL}")
    
    # Отображаем бренды
    branding = f"""
{Fore.WHITE}╔═════════════════════════════════════════════════════════════════════════╗
║ {Fore.MAGENTA}speesh.ru{Fore.WHITE} | {Fore.MAGENTA}rbdclan.moscow{Fore.WHITE} | {Fore.MAGENTA}Sergey Golovchan © 2025{Fore.WHITE}               ║
╚═════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(branding)
    
    # Заголовок "Настройка завершена"
    complete_header = f"""
{Fore.WHITE}╔{'═' * 65}╗
║{Back.MAGENTA}{Fore.BLACK}{Style.BRIGHT}{'НАСТРОЙКА ЗАВЕРШЕНА':^65}{Style.RESET_ALL}{Fore.WHITE}║
╚{'═' * 65}╝{Style.RESET_ALL}
    """
    print(complete_header)
    
    # Выводим итоговую команду в красивой рамке
    cmd_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'ЗАПУСК С ПАРАМЕТРАМИ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(cmd_header)
    
    cmd_str = " ".join(cmd)
    cmd_box = f"""
{Fore.WHITE}╔{'═' * 65}╗
║{Fore.WHITE} python detect_or_track.py {cmd_str:<34}{Fore.WHITE}║
╚{'═' * 65}╝{Style.RESET_ALL}
    """
    print(cmd_box)
    
    # Примеры готовых команд в красивой рамке
    examples_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'ПРИМЕРЫ ГОТОВЫХ КОМАНД ДЛЯ БУДУЩЕГО ИСПОЛЬЗОВАНИЯ':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(examples_header)
    
    # Создаем рамку для примеров
    print(f"{Fore.WHITE}╔{'═' * 65}╗{Style.RESET_ALL}")
    examples = [
        "python detect_or_track.py --weights yolov7.pt --source video.mp4 --img-size 640 --track",
        "python detect_or_track.py --weights yolov7.pt --source 0 --img-size 320 --conf-thres 0.4 --track",
        "python detect_or_track.py --weights yolov7.pt --source video.mp4 --img-size 1024 --track --show-track --white-boxes",
        "python detect_or_track.py --weights yolov7.pt --source video.mp4 --track --cross --cross-thickness 2 --datamosh",
        "python detect_or_track.py --weights yolov7-tiny.pt --source video.mp4 --device 0 --img-size 416 --track"
    ]
    
    for example in examples:
        print(f"{Fore.WHITE}║ {Fore.MAGENTA}{example:<63}{Fore.WHITE}║{Style.RESET_ALL}")
    
    print(f"{Fore.WHITE}╚{'═' * 65}╝{Style.RESET_ALL}")
    
    # Запрос подтверждения запуска
    confirm_header = f"""
{Fore.WHITE}┌{'─' * 65}┐
│{Fore.MAGENTA}{'ЗАПУСК':^65}{Fore.WHITE}│
└{'─' * 65}┘{Style.RESET_ALL}
    """
    print(confirm_header)
    
    confirm_options = ["Запустить сейчас", "Выйти без запуска"]
    confirm_choice = create_interactive_menu("ЗАПУСК", confirm_options)
    
    if confirm_choice == 0:
        return cmd
    else:
        print(f"{Fore.MAGENTA}Выход из программы.{Style.RESET_ALL}")
        sys.exit(0)
    # Ты реал это все прочитал? Красава, горжусь что у меня есть такие подписчики. 
