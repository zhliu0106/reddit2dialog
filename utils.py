import re
import signal
from contextlib import contextmanager
from nltk.tokenize import word_tokenize

FILTERED_REDDITS = []
with open("filtered_reddits.txt") as f:
    for line in f:
        FILTERED_REDDITS.extend(line.strip().split())

URL_REGEX = re.compile(f"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+")
HTML_PAIRS = [
    ("&amp;", " & "),
    ("&quot", ' " '),
    ("&apos", " ' "),
    ("&gt;", " > "),
    ("&lt;", " < "),
]


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("timed_out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def tokenize(string):
    for a, b in HTML_PAIRS:
        string = string.replace(a, b)
    for a in ["\n", "\r", "<", ">"]:
        string = string.replace(a, " ")
    tokens = word_tokenize(string.strip())
    res = " ".join(tokens)
    return res


def do_filter(comment):
    content = comment["body"]
    if not content.strip():
        return True
    if comment["subreddit"].lower() in FILTERED_REDDITS:
        return True
    if comment["author"] == "AutoModerator":
        return True
    if content in ["[removed]", "[deleted]"] or "your submission has been removed" in content.lower():
        return True
    if " " not in content and len(content) > 2048:
        return True
    if len(content) < 5:
        return True
    if len(word_tokenize(content)) > 128:
        return True
    if not content[0].isascii():
        return True
    try:
        with time_limit(1):
            if URL_REGEX.search(content) is not None:
                return True
    except TimeoutException:
        return True
    return False
