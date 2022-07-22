#!/usr/bin/env python3
"""
Adapted from https://github.com/facebookresearch/ParlAI/blob/dff9aabb5024c30c81e146cebffbc88bc6431b61/parlai/tasks/eli5/data_creation/download_reddit_qalist.py
"""

import io
import gc
import json

import msgspec
import os
import gzip
import zstandard as zstd
from multiprocessing import Process, cpu_count, Queue

from typing import Iterator, Dict, List
from argparse import ArgumentParser
from os.path import join as pjoin
from time import time
from utils import filter_tokenize, Comment
from tqdm import tqdm

def setup_args():
    """
    Set up args.
    """
    parser = ArgumentParser()
    reddit = parser.add_argument_group("Download Reddit Docs")
    reddit.add_argument("-sy", "--start_year", default=2022, type=int, metavar="N", help="starting year")
    reddit.add_argument("-ey", "--end_year", default=2022, type=int, metavar="N", help="end year")
    reddit.add_argument("-sm", "--start_month", default=5, type=int, metavar="N", help="starting month")
    reddit.add_argument("-em", "--end_month", default=5, type=int, metavar="N", help="end month")
    reddit.add_argument("-o", "--output_dir", default="res/", type=str, help="where to save the output")

    reddit.add_argument("--valid_split_percentage", default=0.0002, type=float, help="The percentage of the train set used as validation set in case there's no validation split")
    reddit.add_argument("--tokenizer", default="facebook/bart-large", type=str, help="tokenizer to use")
    reddit.add_argument("--max_context_length", default=6, type=int, help="max context length")
    reddit.add_argument("--dump_interval", default=2**10, type=int)
    return parser.parse_args().__dict__


def process():
    opt = setup_args()

    input_dir = pjoin(opt["output_dir"], "reddit_tmp")
    output_dir = pjoin(opt["output_dir"], "processed_data")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for year in range(opt["start_year"], opt["end_year"] + 1):
        st_month = opt["start_month"] if year == opt["start_year"] else 1
        end_month = opt["end_month"] if year == opt["end_year"] else 12
        months = range(st_month, end_month + 1)
        for month in months:

            if month < 10:
                comments_file = pjoin(input_dir, f"RC_{year}-0{month}.zst")
                output_train_file = pjoin(output_dir, f"DLGS_{year}_0{month}.train.zst")
                output_valid_file = pjoin(output_dir, f"DLGS_{year}_0{month}.valid.zst")
            else:
                comments_file = pjoin(input_dir, f"RC_{year}-{month}.zst")
                output_train_file = pjoin(output_dir, f"DLGS_{year}_{month}.train.zst")
                output_valid_file = pjoin(output_dir, f"DLGS_{year}_{month}.valid.zst")

            num_process = max(1, cpu_count() - 1)
            maxsize = 1000 * num_process
            dict_queue = Queue(maxsize=maxsize)
            filtered_queue = Queue(maxsize=maxsize)

            # read comments
            st_time = time()

            # init data filter process
            workers = []
            for _ in range(num_process):
                filter_data_p = Process(target=filter_data, args=(dict_queue, filtered_queue), daemon=True)
                filter_data_p.start()
                workers.append(filter_data_p)

            # init data reader process
            read_file_p = Process(target=read_file, args=(comments_file, dict_queue, num_process), daemon=True)
            read_file_p.start()

            # collect leaf from filtered queue
            collected_leaf = collect_leaf(filtered_queue, num_process)

            # wait for all processes to finish
            read_file_p.join()
            for worker in workers:
                worker.join()

            del dict_queue
            del filtered_queue
            gc.collect()
            print(f"Finish reading, consuming time: {time() - st_time}")

            # construct trees
            submissions, submission_num = construct_trees(collected_leaf)

            # delete variable and clear memory
            del collected_leaf
            gc.collect()

            # construct dialogue samples and write to file
            construct_dlgs(
                output_train_file,
                output_valid_file,
                submissions,
                submission_num,
                opt["max_context_length"],
                opt["dump_interval"],
                opt["valid_split_percentage"]
            )

            # delete variable and clear memory
            del submissions
            gc.collect()



def read_file(comments_file: str, dict_queue: Queue, num_workers: int):
    print(f"Start reading comments file: {comments_file}")

    fh = open(comments_file, "rb")
    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
    stream_reader = dctx.stream_reader(fh)
    f = io.TextIOWrapper(stream_reader, encoding="utf-8")
    decoder = msgspec.json.Decoder(Comment)

    st_time = time()
    for i, line in enumerate(f):
        if (i + 1) % 100000 == 0:
            print(f"read {i+1} lines comments, consuming time: {time() - st_time}")
        parsed = decoder.decode(line.encode("utf-8"))
        dict_queue.put(parsed)

    fh.close()

    for _ in range(num_workers):
        dict_queue.put(None)
    print(f"End of reading. found {i} lines in total")


def filter_data(dict_queue: Queue, filtered_queue: Queue) -> None:
    print("Start filtering data")
    count = 0

    while True:
        parsed = dict_queue.get()
        if parsed is None:
            break
        if filter_tokenize(parsed):
            continue
        content, id, link_id, parent_id = (
            parsed.body,
            parsed.id,
            parsed.link_id[3:],
            parsed.parent_id[3:],
        )
        count += 1
        if count % 10000 == 0:
            print(f"filtered {count} lines")
        filtered_queue.put([content, id, link_id, parent_id])
    filtered_queue.put(None)
    print("End of filtering data")


def collect_leaf(filtered_queue: Queue, num_process: int) -> None:
    leaf_list = []
    finished_num = 0
    while True:
        data = filtered_queue.get()
        if data is None:
            finished_num += 1
            if finished_num == num_process:
                break
        else:
            leaf_list.append(data)

    print(f"End of collecting leaf")
    return leaf_list


def construct_trees(collected_leaf):
    print("Start constructing trees")
    st_time = time()
    submissions = dict()
    count = 0
    submission_num = 0
    for leaf in collected_leaf:
        count += 1
        (content, id, link_id, parent_id) = leaf
        if link_id in submissions:
            submissions[link_id][id] = (content, parent_id, id, False, False)
        else:
            submissions[link_id] = dict()
            submission_num += 1
            submissions[link_id][id] = (content, parent_id, id, False, False)
    for link_id, submission in submissions.items():
        for id, (content, parent_id, _, has_child, responsed) in submission.items():
            if has_child:
                continue
            while True:
                if parent_id in submission:
                    (_content, _parent_id, _id, _has_child, responsed) = submission[parent_id]
                    if _has_child:
                        break
                    submission[parent_id] = (_content, _parent_id, _id, True, responsed)
                    parent_id = _parent_id
                else:
                    break
    print(f"End of constructing trees. found {count} leaf after filtering. consumed {time() - st_time:.2f} s")
    return submissions, submission_num


def construct_dlgs(output_train_file: str, output_valid_file: str, submissions: Dict, submission_num: int, max_context_length: int = 6, dump_interval: int = 2**10, valid_split_percentage: float = 0.0002) -> None:
    print("Start constructing dialogue samples")

    # Record statistics data
    stats_data = {}

    # init tqdm
    pbar = tqdm(total=submission_num)
    st_time = time()

    # init zstd compressor
    f_train = open(output_train_file, "wb")
    f_valid = open(output_valid_file, "wb")
    cctx_train = zstd.ZstdCompressor()
    cctx_valid = zstd.ZstdCompressor()
    compressor_train = cctx_train.stream_writer(f_train)
    compressor_valid = cctx_valid.stream_writer(f_valid)

    count = 0
    submission_count = 0
    valid_steps = int(1/valid_split_percentage)
    for link_id, submission in submissions.items():
        pbar.update(1)
        submission_count += 1
        if submission_count % valid_steps == 0:
            to_valid = True
        else:
            to_valid = False

        # Start building the dialogue from the leaf. Also ignore empty turns (placeholder)
        for id, (content, parent_id, _, has_child, responsed) in submission.items():

            if has_child:
                continue

            dlg = []
            id_list = []
            responsed_list = []
            while True:
                dlg.append(content)
                id_list.append(id)
                responsed_list.append(responsed)
                try:
                    id = parent_id
                    (content, parent_id, id, has_child, responsed) = submission[id]
                except KeyError:
                    dlg = []
                finally:
                    if not dlg or not has_child:
                        break
                    if link_id == parent_id:
                        dlg.append(content)
                        id_list.append(id)
                        responsed_list.append(responsed)
                        break
            # Some validation, assert not empty
            if len(dlg) < 2:
                continue

            dlg = dlg[::-1]
            id_list = id_list[::-1]
            responsed_list = responsed_list[::-1]

            # generate dialogue samples - (context, response) pair
            for i, flag in enumerate(responsed_list[1:max_context_length+1], 1):
                if flag:
                    continue
                dlg_obj = {
                    "context": dlg[:i],
                    "response": dlg[i],
                }

                # write the sample to file
                count += 1
                line = json.dumps(dlg_obj) + "\n"
                if to_valid:
                    compressor_valid.write(line.encode("utf-8"))
                else:
                    compressor_train.write(line.encode("utf-8"))
                if count % dump_interval == 0:
                    compressor_train.flush()
                    compressor_valid.flush()

                # set the comment (that already been response) as True
                (content, parent_id, id, has_child, responsed) = submission[id_list[i]]
                submission[id_list[i]] = (content, parent_id, id, has_child, True)

                stats_data[i] = stats_data.get(i, 0) + 1

    pbar.close()
    compressor_train.flush(zstd.FLUSH_FRAME)
    compressor_valid.flush(zstd.FLUSH_FRAME)
    f_train.close()
    f_valid.close()
    print(f"End of constructing dialogue samples, consumed {time() - st_time:.2f} s")
    print(f"Total {count} samples")
    print(f"Data stats info: {stats_data}")


def read_processed_file(filepath):

    fh = open(filepath, "rb")
    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
    stream_reader = dctx.stream_reader(fh)
    f = io.TextIOWrapper(stream_reader, encoding="utf-8")
    decoder = msgspec.json.Decoder()

    for i, line in enumerate(f):
        sample = decoder.decode(line.encode("utf-8"))

    fh.close()



if __name__ == "__main__":
    process()
