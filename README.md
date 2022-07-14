# reddit2dialog

reddit2dialog was created for NLPer who are interested in dialogue model but lack pretraining data.

With reddit2dialog, you can easily download reddit comments and turn it to dialogue pair format (context, response).

## Requirement

Make sure you have the following dependencies installed.

- python>=3.8
- transformers
- requests
- bs4
- lxml
- msgspec
- zstandard
- iopath

```shell
pip install -r requirements.txt
```

## Run



### Download

First, download the reddit comments data you need:

```shell
python download.py \
    -sy 2021 \ # start year
    -ey 2022 \ #   end year
    -sm 5 \    # start month
    -em 5 \    #   end month
    -o data_dir/
```

### Process

Process the data just downloaded:

```shell
python process.py \
    -sy 2021 \ # start year
    -ey 2022 \ #   end year
    -sm 5 \    # start month
    -em 5 \    #   end month
    -o data_dir/
```

## Reference

- https://github.com/facebookresearch/ParlAI/issues/2838
- https://files.pushshift.io/reddit/
- https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/eli5
