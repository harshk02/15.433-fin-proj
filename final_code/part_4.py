import ratios
import Task_2
import pandas as pd

pt2_composite_scores = Task_2.metrics_df
pt3_metrics = ratios.monthly_summary

print(pt2_composite_scores)
print(pt3_metrics)


# Create a DataFrame with Fund ID as index and Composite Risk Index as values
part2_index = pd.DataFrame({
    'Composite Risk Index': pt2_composite_scores['Composite Risk Index'].values,
    'Volatility_Normalized': pt2_composite_scores['Volatility_Normalized'].values
}, index=pt2_composite_scores['Fund ID'])
# print(part2_index)

# Create a DataFrame with Part 3 metrics
part3_index = pd.DataFrame({
    'LPM_standardized': pt3_metrics['LPM_standardized'].values,
    'Downside Deviation_standardized': pt3_metrics['Downside Deviation_standardized'].values,
    'Fund Name': pt3_metrics['Fund Name'].values,
}, index=pt3_metrics.index)
# print("\nPart 3 Metrics:")
# print(part3_index)

# Convert indices to strings for consistent comparison
part2_index.index = part2_index.index.astype(str)
part3_index.index = part3_index.index.astype(str)

# Find common indices
common_indices = part2_index.index.intersection(part3_index.index)
# print("\nCommon Indices:")
# print(common_indices)

# Filter both DataFrames to only include common indices
part2_index = part2_index.loc[common_indices]
part3_index = part3_index.loc[common_indices]

# print("\nPart 2 Index Values:")
# print(part2_index.index)
# print("\nPart 3 Index Values:")
# print(part3_index.index)


# Create final composite index by combining part 2 and part 3 metrics
final_composite = pd.DataFrame(index=part2_index.index)
final_composite['Final_Composite_Score'] = 0.0

# Iterate through part 2 indices and find matches in part 3
for idx in part2_index.index:
    final_composite.loc[idx, 'Final_Composite_Score'] = (
        0.8 * part2_index.loc[idx, 'Composite Risk Index'] +
        0.1 * part3_index.loc[idx, 'LPM_standardized'] +
        0.1 * part3_index.loc[idx, 'Downside Deviation_standardized']
        )

# Sort values and calculate tertiles for risk categorization
sorted_scores = final_composite['Final_Composite_Score'].sort_values()
tertile_1 = sorted_scores.quantile(1/3)
tertile_2 = sorted_scores.quantile(2/3)

# Add risk category column based on tertiles
final_composite['Risk_Category'] = 'Medium Risk'
final_composite.loc[final_composite['Final_Composite_Score'] <= tertile_1, 'Risk_Category'] = 'Low Risk'
final_composite.loc[final_composite['Final_Composite_Score'] > tertile_2, 'Risk_Category'] = 'High Risk'

# Add individual component columns
final_composite['Volatility_Normalized'] = part2_index['Volatility_Normalized']
final_composite['Downside_Deviation_Standardized'] = part3_index['Downside Deviation_standardized']
final_composite.insert(0, 'Fund_Name', part3_index['Fund Name'])
# Reorder columns to make Risk_Category second
cols = ['Fund_Name', 'Risk_Category', 'Final_Composite_Score', 'Volatility_Normalized', 'Downside_Deviation_Standardized']
final_composite = final_composite[cols]



print("\nFinal Composite Scores:")
print(final_composite.sort_values('Final_Composite_Score', ascending=False))