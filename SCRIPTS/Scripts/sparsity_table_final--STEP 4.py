import pandas as pd

sparsity_table = pd.read_csv("sparsity_table_20May.csv", index_col=0)

def color_sparsity(val):
    if val > 80:
        return 'background-color: red'
    elif val > 50 and val < 80:
        return 'background-color: orange'
    elif val > 20 and val < 50:
        return 'background-color: yellow'
    else:
        return 'background-color: lightblue'

styled_df = sparsity_table.style.applymap(color_sparsity)
styled_df_path = "styled_sparsity_table_2.html"
styled_df.to_html(styled_df_path, escape=False)
print(f"\n Styled HTML saved to: {styled_df_path}")

def classify_sparsity(val):
    if val > 80:
        return 'red'
    elif val > 50 and val < 80:
        return 'orange'
    elif val > 20 and val < 50:
        return 'yellow'
    else:
        return 'blue'

summary = {}

for index, row in sparsity_table.iterrows():
    color_counts = {'red': 0, 'orange': 0, 'yellow': 0, 'blue': 0}
    for val in row:
        color = classify_sparsity(val)
        color_counts[color] += 1
    summary[index] = color_counts

print("\n=== Sparsity Table (% of missing values) ===\n")
print(sparsity_table.round(2))

print("\n=== Color Summary by Driver ===")
for driver, counts in summary.items():
    print(f"\n{driver}:")
    for color, count in counts.items():
        print(f"  {color.capitalize():<9}: {count} columns")
