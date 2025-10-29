# Tic Tac Toe AI Experiment

>[!NOTE]
> Being an exploration, this project is and will perhaps forever be WIP.

We use tic-tac-toe as a toy problem here to learn how to train an AI to play it well.

## Goals

The goal is to learn, and see how far foundational models can be equipped with ability to understand and strategize, with Tic Tac Toe as a motivating example.

Tic Tac Toe is trivial to solve classically, which makes it an interesting toy example to explore.

The two sub-goals are to have the AI -
1. To understand tic-tac-toe grid from visuals e.g. phone camera.
2. To offer a formidable gameplay as an opponent.

# Strategy

Goal for strategy is for a model to infer the best line of play given any position as an array. Subgoals include determine various game states, e.g. whose play is it, whether the game has ended, whether the configuration is invalid.

## Custom Models

To train a custom model, use a command like below -

```sh
python -m src.vision.train --model "cnnv3"
```

## Evaluation
Step 1. Generate Examples

```sh
python -m src.gameplay.llm_generate_evaluations --player "o3-mini"
```

Step 2. Evaluate

Use [this notebook](./src/notebooks/gameplay_results.ipynb) to evaluate and visualize.

### Results

![Gameplay Results](https://res.cloudinary.com/dzcghojgi/image/upload/v1761737830/157475fc-0694-48b7-8b03-18049964d5f5.png)

## Demo

A demo / playground for the evaluator can be served via -

```sh
./run_ui_demo.sh
```

# Vision

## Evaluations

### Running

To run a new evaluation, execute the following -
```sh
python -m src.vision.evaluate --model "o3-mini"
```

For adhoc experiments, use the notebook with similar name.

### Some Results

| Model               |    Accuracy     | Inference Time | Result Grids                                                                                                                                                                                                                   |
| ------------------- | :-------------: | -------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| OpenAI GPT 4.1      | 52/100 correct  | _Not recorded_ | [<img src="https://res.cloudinary.com/dzcghojgi/image/upload/w_160/v1761736706/result_grid_20251017_033038_x36wdo.png">](https://res.cloudinary.com/dzcghojgi/image/upload/v1761736706/result_grid_20251017_033038_x36wdo.png) |
| InternVL3.5 (Local) | 10/100 correct  | _Not recorded_ | [<img src="https://res.cloudinary.com/dzcghojgi/image/upload/w_160/v1761736707/result_grid_20251017_034451_rrag5m.png">](https://res.cloudinary.com/dzcghojgi/image/upload/v1761736707/result_grid_20251017_034451_rrag5m.png) |
| OpenAI o3           | 75/100 correct  |         19.71s | [<img src="https://res.cloudinary.com/dzcghojgi/image/upload/w_160/v1761736706/result_grid_20251018_000545_ktcrdv.png">](https://res.cloudinary.com/dzcghojgi/image/upload/v1761736706/result_grid_20251018_000545_ktcrdv.png) |
| Custom CNN Model v3 | 100/100 correct |            6ms | [<img src="https://res.cloudinary.com/dzcghojgi/image/upload/w_160/v1761736707/result_grid_20251022_230449_ifqkg7.png">](https://res.cloudinary.com/dzcghojgi/image/upload/v1761736707/result_grid_20251022_230449_ifqkg7.png) |

