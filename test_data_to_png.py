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
        # "input/Beauty_1920x1080_120fps_420_8bit_YUV.yuv",
        "input/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv",
        "input/ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv",
        "input/YachtRide_1920x1080_120fps_420_8bit_YUV.yuv",
        "input/Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv",
        "input/Jockey_1920x1080_120fps_420_8bit_YUV.yuv",
        "input/ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv"
    ]
    width = 1920
    height = 1080
    dst_paths = [
        # "media/UVG/Beauty_1920x1080_120fps_420_8bit_YUV",
        "media/UVG/HoneyBee_1920x1080_120fps_420_8bit_YUV",
        "media/UVG/ReadySteadyGo_1920x1080_120fps_420_8bit_YUV",
        "media/UVG/YachtRide_1920x1080_120fps_420_8bit_YUV",
        "media/UVG/Bosphorus_1920x1080_120fps_420_8bit_YUV",
        "media/UVG/Jockey_1920x1080_120fps_420_8bit_YUV",
        "media/UVG/ShakeNDry_1920x1080_120fps_420_8bit_YUV"
    ]

    for i in range(len(src_paths)):
        src_path = src_paths[i]
        dst_path = dst_paths[i]
        convert_one_seq_to_png(src_path, width, height, dst_path)
        os.remove(src_path)
        print(f"Finished: {src_path}")

if __name__ == "__main__":
    main()
