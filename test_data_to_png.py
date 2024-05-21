# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from src.utils.video_reader import YUVReader
from src.utils.video_writer import PNGWriter

import os

from tqdm.auto import tqdm


def convert_one_seq_to_png(src_path, width, height, dst_path):
    src_reader = YUVReader(src_path, width, height, src_format='420')
    png_writer = PNGWriter(dst_path, width, height)
    rgb = src_reader.read_one_frame(dst_format='rgb')
    processed_frame = 0
    with tqdm() as progress_bar:
        while not src_reader.eof:
            png_writer.write_one_frame(rgb=rgb, src_format='rgb')
            processed_frame += 1
            progress_bar.update(1)
            rgb = src_reader.read_one_frame(dst_format='rgb')
    print(src_path, processed_frame)


def main():
    src_paths = [
        "input/MCV/videoSRC01_1280x720_30.yuv",
        "input/MCV/videoSRC03_1280x720_30.yuv",
        "input/MCV/videoSRC04_1280x720_30.yuv",
        "input/MCV/videoSRC05_1280x720_25.yuv",
        "input/MCV/videoSRC06_1280x720_25.yuv",
        "input/MCV/videoSRC07_1280x720_25.yuv",
        "input/MCV/videoSRC13_1280x720_30.yuv",
        "input/MCV/videoSRC15_1280x720_30.yuv",
        "input/MCV/videoSRC18_1280x720_25.yuv",
        "input/MCV/videoSRC21_1280x720_24.yuv",
        "input/MCV/videoSRC23_1280x720_24.yuv"
    ]
    width = 1280
    height = 720
    dst_paths = [
        "media/MCV/videoSRC01_1280x720_30_YUV",
        "media/MCV/videoSRC03_1280x720_30_YUV",
        "media/MCV/videoSRC04_1280x720_30_YUV",
        "media/MCV/videoSRC05_1280x720_25_YUV",
        "media/MCV/videoSRC06_1280x720_25_YUV",
        "media/MCV/videoSRC07_1280x720_25_YUV",
        "media/MCV/videoSRC13_1280x720_30_YUV",
        "media/MCV/videoSRC15_1280x720_30_YUV",
        "media/MCV/videoSRC18_1280x720_25_YUV",
        "media/MCV/videoSRC21_1280x720_24_YUV",
        "media/MCV/videoSRC23_1280x720_24_YUV"
    ]

    for i in range(len(src_paths)):
        src_path = src_paths[i]
        dst_path = dst_paths[i]
        convert_one_seq_to_png(src_path, width, height, dst_path)
        os.remove(src_path)
        print(f"Finished: {src_path}")

if __name__ == "__main__":
    main()
