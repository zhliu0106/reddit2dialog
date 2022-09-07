import re
import signal
import msgspec
from contextlib import contextmanager

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data/zhliu/plm/bart-large")

FILTERED_REDDITS = []
with open("filtered_reddits.txt") as f:
    for line in f:
        FILTERED_REDDITS.extend(line.strip().split())

URL_REGEX = re.compile(f"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+")
EMOJIS_REGEX = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002500-\U00002BEF"  # chinese char
    "\U00002702-\U000027B0"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"  # dingbats
    "\u3030"
    "]+",
    re.UNICODE,
)
HTML_PAIRS = [
    ("&amp;", " & "),
    ("&quot", ' " '),
    ("&apos", " ' "),
    ("&gt;", " > "),
    ("&lt;", " < "),
]


class Comment(msgspec.Struct):
    body: str
    id: str
    link_id: str
    parent_id: str
    subreddit: str
    author: str


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


def preprocess(string):
    for a, b in HTML_PAIRS:
        string = string.replace(a, b)
    for a in ["\n", "\r", "<", ">", "``", "''", "*"]:
        string = string.replace(a, " ")
    string = re.sub(EMOJIS_REGEX, '', string)
    res = " ".join(string.strip().split())
    return res


def filter_tokenize(comment):
    content = comment.body

    if not content.strip():
        return True
    if comment.subreddit.lower() in FILTERED_REDDITS:
        return True
    if comment.author == "AutoModerator":
        return True
    if content in ["[removed]", "[deleted]"] or "your submission has been removed" in content.lower():
        return True
    if " " not in content and len(content) > 2048:
        return True
    if len(content) < 5:
        return True
    if not content[0].isascii():
        return True
    try:
        with time_limit(1):
            if URL_REGEX.search(content) is not None:
                return True
    except TimeoutException:
        return True
    comment.body = preprocess(comment.body)
    if len(comment.body.split()) > 128:
        return True
    if len(tokenizer.tokenize(comment.body)) > 128:
        return True
    return False
