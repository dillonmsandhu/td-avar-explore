## How to use:

Run Everything:	`python run_all.py --script cov_lstd.py`
Run one config group:	`python -m algos.cov_lstd --base-config min`
Run one specific env:	`python -m algos.cov_lstd --base-config visual --env-ids Pong-misc`
Custom Batch Name:	`python run_all.py --suffix my_big_test`

Config groups can be found in `configs.CONFIG_REGISTRY` in `configs.py`

The python environemnt can be activated with `pyenv activate purejaxrl`, with requirements in `requirements.txt`