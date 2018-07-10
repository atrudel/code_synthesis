# Roadmap

In this folder you will find some resources to generate, mix, create program for **GOL 2 players**.

We already generated some of them and they are available in `generated` folder.

```
generated
    `-- tiling
    `-- scripts
        `-- Players.ipynb
        `-- Program_creator.ipynb
        `-- utils.py
```

## `tiling`

> ⚠️ For now, those patterns haven't been tested inside the game.

Those program are created using a tiling technics: it takes a pattern of size less than $(64, 64)$ and create a tile using it.

## `scripts`

The scripts used to generate the different program.

- `Players.ipynb`: notebook started to analyze the different patterns available with `Golly`.
- `Program_creator.ipynb`: for now, used to generate tilings.
- `utils.py`: contain some useful functions.
- `pattern_less64.csv`: used for the work-in-progress. Contains the `.rle` file that are $\leq (64, 64)$.

## Todo

There are two more technics we need to explore:

- [] handcraft software: a visual interface (like `Golly`) with a $(64, 64)$ grid in order to draw your program. We could have tools like: a pen (to draw freely, cell by cell), import a pattern and translate or rotate it on the canvas.
- [] random: take a list of different patterns and randomly places them on the canvas. One improvement could be to choose a ratio between different kind of patterns (stables, tingling, moving, generators,...).
