# Tic-tac-toe AI Experiment

We use tic-tac-toe as a toy problem here to learn how to train an AI to play it well.

## Goals

The main goal is to see how far VLMs can be pushed to understand Tic Tac Toe.

The two sub-goals are to have the AI -
1. To understand tic-tac-toe grid from visuals e.g. phone camera.
2. To offer a formidable gameplay as an opponent.

# Evaluating

## Strategic Performance

### Generate Examples

```sh
python -m src.gameplay.llm_generate_evaluations --player "o3-mini"
```

### Evaluate
Use [this notebook](./src/notebooks/gameplay_results.ipynb) to evaluate and visualize.

### Demo

A demo / playground for the evaluator can be served via -

```sh
./run_ui_demo.sh
```

## Vision Performance

```sh
python -m src.vision.evaluate
```

This will create a png grid with results.

For adhoc experiments, use the notebook with similar name.
