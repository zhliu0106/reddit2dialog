#!/usr/bin/env python3
"""
Adapted from https://github.com/facebookresearch/ParlAI/blob/dff9aabb5024c30c81e146cebffbc88bc6431b61/parlai/tasks/eli5/data_creation/download_reddit_qalist.py
"""

import io
import json
import os
import gzip
import zstandard as zstd
from multiprocessing import Queue, Process, cpu_count
from typing import Iterator, Dict, List
from argparse import ArgumentParser
from os.path import join as pjoin
from time import time
from utils import tokenize, do_filter


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

    reddit.add_argument("--min_dialogue_length", default=3, type=int, help="min dialogue length")
    reddit.add_argument("--dump_interval", default=2 ** 10, type=int)
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
                output_file = pjoin(output_dir, f"DLGS_{year}_0{month}.txt.gz")
            else:
                comments_file = pjoin(input_dir, f"RC_{year}-{month}.zst")
                output_file = pjoin(output_dir, f"DLGS_{year}_{month}.txt.gz")

            num_process = max(1, cpu_count() - 1)
            maxsize = 10 * num_process
            line_queue = Queue(maxsize=maxsize)
            dict_queue = Queue(maxsize=maxsize)
            filtered_queue = Queue(maxsize=maxsize)
            res_queue = Queue(1)

            # read comments
            st_time = time()
            read_file_p = Process(target=read_file, args=(comments_file, line_queue, num_process), daemon=True)
            collect_leaf_p = Process(target=collect_leaf, args=(filtered_queue, res_queue), daemon=True)
            line2dict_workers = []
            filter_workers = []
            for _ in range(num_process):
                line2dict_p = Process(target=line2dict, args=(line_queue, dict_queue), daemon=True)
                line2dict_p.start()
                line2dict_workers.append(line2dict_p)

                filter_data_p = Process(target=filter_data, args=(dict_queue, filtered_queue), daemon=True)
                filter_data_p.start()
                filter_workers.append(filter_data_p)
            read_file_p.start()
            collect_leaf_p.start()
            read_file_p.join()
            for worker in line2dict_workers:
                worker.join()
            for worker in filter_workers:
                worker.join()
            filtered_queue.put(None)
            collected_leaf = res_queue.get()
            collect_leaf_p.join()

            print(f"Finish reading, consuming time: {time() - st_time}")

            # construct trees
            submissions, submission2subreddit = construct_trees(collected_leaf)

            # construct dialogue samples and write to file
            dlgs2write_queue = Queue(maxsize=maxsize)
            store_sample_p = Process(target=store_sample, args=(output_file, dlgs2write_queue), daemon=True)
            store_sample_p.start()
            construct_dlgs(
                submissions,
                submission2subreddit,
                dlgs2write_queue,
                opt["min_dialogue_length"],
            )
            dlgs2write_queue.put(None)
            store_sample_p.join()


def read_file(comments_file: str, line_queue: Queue, num_workers: int) -> Iterator[str]:
    print(f"Start reading comments file: {comments_file}")
    fh = open(comments_file, "rb")
    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
    stream_reader = dctx.stream_reader(fh)
    f = io.TextIOWrapper(stream_reader, encoding="utf-8")

    for i, line in enumerate(f):
        if (i + 1) % 100000 == 0:
            print(f"read {i+1} lines comments")
        line_queue.put(line)

    fh.close()

    for _ in range(num_workers):
        line_queue.put(None)
    print(f"End of reading. found {i} lines in total")


def line2dict(line_queue: Queue, dict_queue: Queue) -> None:
    print("Start processing lines")
    count = 0
    while True:
        line = line_queue.get()
        if line is None:
            break
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[build] Error parsing line ({str(e)})\n - {line}")
            continue
        count += 1
        if count % 10000 == 0:
            print(f"load dict {count} lines")
        dict_queue.put(parsed)
    dict_queue.put(None)
    print("End of turning line to dict")


def filter_data(dict_queue: Queue, filtered_queue: Queue) -> None:
    print("Start filtering data")
    count = 0
    while True:
        parsed = dict_queue.get()
        if parsed is None:
            break
        if do_filter(parsed):
            continue
        content, id, link_id, parent_id, subreddit = (
            tokenize(parsed["body"]),
            parsed["id"],
            parsed["link_id"][3:],
            parsed["parent_id"][3:],
            parsed["subreddit"],
        )
        count += 1
        if count % 10000 == 0:
            print(f"filtered {count} lines")
        filtered_queue.put([content, id, link_id, parent_id, subreddit])
    # filtered_queue.put(None)
    print("End of filtering data")


def collect_leaf(filtered_queue: Queue, res_queue: Queue) -> None:
    leaf_list = []
    while True:
        data = filtered_queue.get()
        if data is None:
            break
        leaf_list.append(data)
    res_queue.put(leaf_list)
    print(f"End of collecting leaf")


def construct_trees(collected_leaf):
    st_time = time()
    submissions = dict()
    submission2subreddit = {}
    count = 0
    for leaf in collected_leaf:
        count += 1
        (content, id, link_id, parent_id, subreddit) = leaf
        if link_id in submissions:
            submissions[link_id][id] = (content, parent_id, id, False)
        else:
            submissions[link_id] = dict()
            submissions[link_id][id] = (content, parent_id, id, False)
            submission2subreddit[link_id] = subreddit
    for link_id, submission in submissions.items():
        for id, (content, parent_id, _, has_child) in submission.items():
            if has_child:
                continue
            while True:
                if parent_id in submission:
                    (_content, _parent_id, _id, _has_child) = submission[parent_id]
                    if _has_child:
                        break
                    submission[parent_id] = (_content, _parent_id, _id, True)
                    parent_id = _parent_id
                else:
                    break
    print(f"End of constructing trees. found {count} leaf after filtering. consumed {time() - st_time:.2f} s")
    return submissions, submission2subreddit


def construct_dlgs(
    submissions: Dict, submission2subreddit: Dict, dlgs2write_queue: Queue, min_dialogue_length: int = 3
) -> None:
    stats_data = {}
    st_time = time()
    for link_id, submission in submissions.items():
        # Start building the dialogue from the leaf. Also ignore empty turns (placeholder)
        for id, (content, parent_id, _, has_child) in submission.items():

            if has_child:
                continue

            dlg, ids = [], []
            while True:
                dlg.append(content)
                ids.append(id)
                try:
                    id = parent_id
                    (content, parent_id, id, has_child) = submission[id]
                except KeyError:
                    dlg = []
                finally:
                    if not dlg or not has_child:
                        break
                    if link_id == parent_id:
                        dlg.append(content)
                        ids.append(id)
                        break
            # Some validation, set min dialogue length and assert not empty
            if not dlg or len(dlg) < min_dialogue_length:
                continue

            if not ids or len(ids) != len(dlg) or not all(i.strip() for i in ids):
                continue

            try:
                # Lowercase the subreddit
                subreddit = submission2subreddit[link_id].strip().lower()
            except KeyError:
                continue

            if not subreddit:
                continue

            stats_data[len(dlg)] = stats_data.get(len(dlg), 0) + 1

            dlg_obj = {
                "domain": subreddit,
                "turns_with_ids": list(zip(ids, dlg))[::-1],
            }

            dlgs2write_queue.put(json.dumps(dlg_obj) + "\n")

    print(f"End of constructing dialogue samples, consumed {time() - st_time:.2f} s")
    print(f"Final data stats: {stats_data}")


def store_sample(output_file: str, dlgs2write_queue: Queue):
    print(f"Start storing samples to {output_file}")
    # f = gzip.open(output_file, "wt", encoding="utf-8")
    f = open(output_file, "w", encoding="utf-8")
    while True:
        dlg = dlgs2write_queue.get()
        if dlg is None:
            break
        f.write(dlg)
    f.close()
    print(f"End of storing samples to {output_file}")


if __name__ == "__main__":
    process()
