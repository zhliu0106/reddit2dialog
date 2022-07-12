#!/usr/bin/env python3
"""
Adapted from https://github.com/facebookresearch/ParlAI/blob/dff9aabb5024c30c81e146cebffbc88bc6431b61/parlai/tasks/eli5/data_creation/download_reddit_qalist.py
"""

import re
import os
import requests
import subprocess

from argparse import ArgumentParser
from bs4 import BeautifulSoup
from os.path import join as pjoin
from time import sleep, time
from iopath.common.file_io import PathManager

REDDIT_URL = "https://files.pushshift.io/reddit/"


# collects URLs for monthly dumps, has to be robust to file type changes
def gather_dump_urls(base_url, mode):
    page = requests.get(base_url + mode)
    soup = BeautifulSoup(page.content, "lxml")
    files = [it for it in soup.find_all(attrs={"class": "file"})]
    f_urls = [
        tg.find_all(lambda x: x.has_attr("href"))[0]["href"]
        for tg in files
        if len(tg.find_all(lambda x: x.has_attr("href"))) > 0
    ]
    date_to_url = {}
    for url_st in f_urls:
        ls = re.findall(r"20[0-9]{2}-[0-9]{2}", url_st)
        if len(ls) > 0:
            yr, mt = ls[0].split("-")
            date_to_url[(int(yr), int(mt))] = base_url + mode + url_st[1:]
    return date_to_url


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
    return parser.parse_args().__dict__


def download(mode="submissions"):
    opt = setup_args()
    output_dir = opt["output_dir"]
    reddit_tmp_dir = pjoin(output_dir, "reddit_tmp")
    if not os.path.exists(reddit_tmp_dir):
        os.mkdir(reddit_tmp_dir)

    assert mode in ["submissions", "comments"], "mode must be either submissions or comments"
    date_to_urls = gather_dump_urls(REDDIT_URL, mode)

    st_time = time()

    # get monthly reddit dumps
    for year in range(opt["start_year"], opt["end_year"] + 1):
        st_month = opt["start_month"] if year == opt["start_year"] else 1
        end_month = opt["end_month"] if year == opt["end_year"] else 12
        months = range(st_month, end_month + 1)
        for month in months:
            comments_url = date_to_urls[(year, month)]
            try:
                f_name = pjoin(reddit_tmp_dir, comments_url.split("/")[-1])
                tries_left = 5
                while tries_left:
                    try:
                        print("downloading %s %2f" % (f_name, time() - st_time))
                        subprocess.run(["wget", "-P", reddit_tmp_dir, comments_url], stdout=subprocess.PIPE)
                        print("downloading end %s %2f" % (f_name, time() - st_time))
                        tries_left = 0
                    except EOFError:
                        sleep(10)
                        print("failed reading file %s file, another %d tries" % (f_name, tries_left))
                        PathManager.rm(f_name)
                        tries_left -= 1
            except FileNotFoundError:
                sleep(60)
                print("failed downloading %s" % (comments_url))


if __name__ == "__main__":
    download("submissions")
    download("comments")
