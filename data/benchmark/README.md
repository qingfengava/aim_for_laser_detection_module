# Benchmark Dataset Layout

将每个场景的一次固定回放日志放在如下路径（`cmd_log.jsonl`）：

- `data/benchmark/static/cmd_log.jsonl`
- `data/benchmark/high_speed/cmd_log.jsonl`
- `data/benchmark/occlusion/cmd_log.jsonl`
- `data/benchmark/weak_light/cmd_log.jsonl`
- `data/benchmark/red_blue_switch/cmd_log.jsonl`

日志来源：
1. 启动程序：`./run.sh run 1`
2. 稳定运行该场景后，拷贝 `/dev/shm/cmd_log.jsonl` 到对应场景目录。
3. 保证每个场景至少 `300` 帧，且采集流程固定（相机参数/曝光/机位一致）。

回归命令：

```bash
./tools/benchmark/run_regression.sh
```

首次建立基线：

```bash
./tools/benchmark/run_regression.sh \
  config/benchmark/classic_acceptance_manifest.json \
  config/benchmark/classic_baseline.json \
  /tmp/classic_regression_report.json \
  update-baseline
```

