# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wilywonka1/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wilywonka1/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A refresh-aware counts forecaster for GTA VI's Exophase trophy earn rates. The case history is compressed to its distinct page states (one per Exophase refresh epoch); players tracked and per-trophy earners are forecast as counts, and rates directly, by an ensemble of damped trend models (logit rate against log time and log players, log earners against log players and log time, plus a recent-window fit) whose slopes are shrunk toward the cross-trophy median. Internal backtests on earlier refresh epochs set how far the median may move from the last read and how wide the quantile spread is. A history with a single distinct state is treated as a stale page awaiting its first refresh: the median shifts down in logit and the spread widens. Python 3.13, standard library only, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=sqbwfzsd&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
