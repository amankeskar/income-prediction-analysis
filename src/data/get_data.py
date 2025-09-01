from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTFILE = RAW_DIR / "adult.csv"

def main():
    print("Downloading 'adult' dataset from OpenML...")
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X, y = adult.data, adult.target

    # Clean column names
    X.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in X.columns]
    target = "income"
    df = pd.concat([X, y.rename(target)], axis=1)

    print("Shape:", df.shape)
    print("Target distribution:\n", df[target].value_counts(dropna=False))

    df.to_csv(OUTFILE, index=False)
    print(f"Saved raw dataset to {OUTFILE.resolve()}")

if __name__ == "__main__":
    main()
