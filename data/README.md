# Data

The dataset is not committed to this repository (14.5 MB, and Kaggle's terms cover redistribution).

## Download

Kaggle competition: [Don't Get Kicked!](https://www.kaggle.com/c/DontGetKicked)

```bash
kaggle competitions download -c DontGetKicked -f training.csv -p .
unzip training.csv.zip
```

Expected file: `data/training.csv` — 72,983 rows, 34 columns, target `IsBadBuy`.

The competition's `Carvana_Data_Dictionary.txt` describes each column.
