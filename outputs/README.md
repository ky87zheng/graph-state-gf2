# Output directory

For direct reproduction of the paper figures, place the released Mode-2 summary CSV here as:

```text
figure_points_results.csv
```

Then set in `simulate_modes.py`:

```python
EXPERIMENT_MODE = 2
RUN_SIMULATION = False
```

and run:

```bash
python simulate_modes.py
```

The script will generate:

```text
Fig1_Time_Performance.png
Fig2_Global_Overheads.png
```

If the Monte Carlo simulation is run from scratch (`RUN_SIMULATION = True`), checkpoint files are created under `checkpoints_seed1234/` and the summary CSV is written automatically.
