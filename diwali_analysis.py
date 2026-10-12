"""Mini analysis of a clearly labeled sample Diwali shopping dataset."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data_path = Path(__file__).with_name("diwali_sample.csv")
    df = pd.read_csv(data_path)
    print("Sample data preview:")
    print(df.head())
    print("\nSummary by area:")
    print(df.groupby("Area")[["Orders", "SpendINR"]].sum().sort_values("SpendINR", ascending=False))
    print("\nSummary by gender:")
    print(df.groupby("Gender")[["Orders", "SpendINR"]].sum())

    area_spend = df.groupby("Area")["SpendINR"].sum().sort_values(ascending=False)
    area_spend.plot(kind="bar", title="Sample Diwali Spending by Area", ylabel="Spend (INR)")
    plt.tight_layout()
    output_path = Path(__file__).with_name("spending_by_area.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nChart saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
