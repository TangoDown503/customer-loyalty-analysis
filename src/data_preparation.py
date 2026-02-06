# # Data Preparation
# * This notebook will analyze, transform, and prepare data for model training


import pandas as pd


flight = pd.read_csv("../data/raw/Customer Flight Activity.csv")
loyalty = pd.read_csv("../data/raw/Customer Loyalty History.csv")


print("\n--- Unique Values in Categorical Variables ---")
print(f"Gender: {loyalty['Gender'].nunique()}")
print(f"Education: {loyalty['Education'].nunique()}")
print(f"Marital Status: {loyalty['Marital Status'].nunique()}")
print(f"Loyalty Card: {loyalty['Loyalty Card'].nunique()}")
print(f"Enrollment Type: {loyalty['Enrollment Type'].nunique()}")
print(f"Loyalty Number: {loyalty['Loyalty Number'].nunique()} unique IDs")
print(f"Salaries: {loyalty['Salary'].nunique()} unique salary numbers")
print(f"Unique Countries: {loyalty['Country'].nunique()}")
print(f"Unique Provinces: {loyalty['Province'].nunique()}")
print(f"Unique Cities: {loyalty['City'].nunique()}")


def print_value_percentages(df, column_name):
    """Print percentage distribution for a column in table format."""
    percentages = df[column_name].value_counts(normalize=True) * 100
    counts = df[column_name].value_counts()
    max_value_len = max(len(str(v)) for v in percentages.index)
    max_count_len = max(len(str(c)) for c in counts.values)

    print(f"\n{column_name}:")
    print("-----------------------------------")
    print(f"{'Value':<{max_value_len}} | {'Count':>{max_count_len}} | Percentage")
    print("-----------------------------------")

    for value in percentages.index:
        count = counts[value]
        pct = percentages[value]
        print(f"{value:<{max_value_len}} | {count:>{max_count_len}} | {pct:>6.2f}%")

    print("-----------------------------------")


print_value_percentages(loyalty, "Gender")
print_value_percentages(loyalty, "Education")
print_value_percentages(loyalty, "Marital Status")
print_value_percentages(loyalty, "Loyalty Card")
print_value_percentages(loyalty, "Enrollment Type")


numeric_cols = loyalty.select_dtypes(include=["int64", "float64"]).columns


flight["Has Activity"] = (
    (flight["Total Flights"] > 0)
    | (flight["Points Accumulated"] > 0)
    | (flight["Points Redeemed"] > 0)
)
active_flight = flight[flight["Has Activity"]]
active_flight["Date"] = pd.to_datetime(active_flight[["Year", "Month"]].assign(Day=1))
most_recent = active_flight.loc[
    active_flight.groupby("Loyalty Number")["Date"].idxmax()
]

customer_flight_summary = (
    flight.groupby("Loyalty Number")
    .agg(
        {
            "Total Flights": "sum",  # Total flights taken
            "Distance": "sum",  # Total distance flown
            "Points Accumulated": "sum",  # Total points accumulated
            "Points Redeemed": "sum",  # Total points redeemed
            "Dollar Cost Points Redeemed": "sum",  # Total dollar value redeemed
        }
    )
    .reset_index()
)

customer_flight_summary = customer_flight_summary.merge(
    most_recent[["Loyalty Number", "Year", "Month"]], on="Loyalty Number", how="left"
)

customer_flight_summary = customer_flight_summary.rename(
    columns={"Year": "Most Recent Year", "Month": "Most Recent Month"}
)
print("\n--- Customer Flight Summary ---\n\n")
customer_flight_summary.head(10)


flight_numeric_cols = customer_flight_summary.select_dtypes(
    include=["int64", "float64"]
).columns


# Create a datetime for December 2018
reference_date = pd.to_datetime("2018-12-01")

# Create a datetime column for most recent activity
customer_flight_summary["Most Recent Date"] = pd.to_datetime(
    customer_flight_summary.rename(
        columns={"Most Recent Year": "year", "Most Recent Month": "month"}
    ).assign(day=1)[["year", "month", "day"]]
)

# Calculate months difference
customer_flight_summary["Months Since Activity"] = (
    reference_date.year - customer_flight_summary["Most Recent Date"].dt.year
) * 12 + (reference_date.month - customer_flight_summary["Most Recent Date"].dt.month)

# Create churn column: 1 if >3 months, 0 if <=3 months
customer_flight_summary["Churn"] = (
    (customer_flight_summary["Months Since Activity"] >= 3)
    | (customer_flight_summary["Months Since Activity"].isna())
).astype(int)

print("\n--- Churn Analysis ---")
print(f"\nChurn distribution:\n{customer_flight_summary['Churn'].value_counts()}")

never_used = customer_flight_summary["Most Recent Date"].isna().sum()
total_churned = customer_flight_summary["Churn"].sum()
churned_after_using = total_churned - never_used
ratio = never_used / total_churned


# Calculate counts
never_used = customer_flight_summary["Most Recent Date"].isna().sum()
churned_after_using = total_churned - never_used
active_customers = (customer_flight_summary["Churn"] == 0).sum()

# Create data for pie chart
categories = ["Never Used Service", "Churned After Using", "Active Customers"]
counts = [never_used, churned_after_using, active_customers]

# For customers with activity
customers_with_activity = customer_flight_summary[
    customer_flight_summary["Most Recent Date"].notna()
].copy()

# Initialize columnsobjective
customer_flight_summary["Points Most Recent"] = None
customer_flight_summary["Activity 1 Month Before"] = None
customer_flight_summary["Activity 2 Months Before"] = None
customer_flight_summary["Activity 3 Months Before"] = None
customer_flight_summary["Overall Trend"] = None
customer_flight_summary["Avg Monthly Points"] = None
customer_flight_summary["Activity Volatility"] = None


for idx, row in customers_with_activity.iterrows():
    loyalty_num = row["Loyalty Number"]
    reference_date = row["Most Recent Date"]

    # Get customer history
    customer_data = flight[flight["Loyalty Number"] == loyalty_num].copy()
    customer_data["Date"] = pd.to_datetime(
        customer_data[["Year", "Month"]].assign(Day=1)
    )
    customer_data = customer_data.sort_values("Date")

    # Calculate dates
    date_1_month_before = reference_date - pd.DateOffset(months=1)
    date_2_months_before = reference_date - pd.DateOffset(months=2)
    date_3_months_before = reference_date - pd.DateOffset(months=3)

    # Get points for each period
    points_recent = customer_data[customer_data["Date"] == reference_date][
        "Points Accumulated"
    ].sum()
    points_1_month = customer_data[customer_data["Date"] == date_1_month_before][
        "Points Accumulated"
    ].sum()
    points_2_months = customer_data[customer_data["Date"] == date_2_months_before][
        "Points Accumulated"
    ].sum()
    points_3_months = customer_data[customer_data["Date"] == date_3_months_before][
        "Points Accumulated"
    ].sum()

    # Calculate absolute changes
    change_1_month = abs(points_recent - points_1_month)
    change_2_months = abs(points_1_month - points_2_months)
    change_3_months = abs(points_2_months - points_3_months)
    customer_flight_summary.loc[idx, "Points Most Recent"] = points_recent
    customer_flight_summary.loc[idx, "Activity 1 Month Before"] = change_1_month
    customer_flight_summary.loc[idx, "Activity 2 Months Before"] = change_2_months
    customer_flight_summary.loc[idx, "Activity 3 Months Before"] = change_3_months

    # Store actual point values
    customer_flight_summary.loc[idx, "Points Most Recent"] = points_recent
    customer_flight_summary.loc[idx, "Overall Trend"] = points_recent - points_3_months
    customer_flight_summary.loc[idx, "Avg Monthly Points"] = (
        points_recent + points_1_month + points_2_months + points_3_months
    ) / 4
    customer_flight_summary.loc[idx, "Activity Volatility"] = (
        change_1_month + change_2_months + change_3_months
    ) / 3



customer_complete = customer_flight_summary.merge(
    loyalty, on="Loyalty Number", how="left"
)

master_data = customer_complete[customer_complete["Most Recent Date"].notna()].copy()
master_data["Customer Age (Years)"] = 2018 - master_data["Enrollment Year"]
master_data.columns


print("\nChurn by Province:")
print(
    master_data.groupby("Province")["Churn"]
    .agg(["mean"])
    .sort_values("mean", ascending=False)
)

print("\nChurn by City:")
print(
    master_data.groupby("City")["Churn"]
    .agg(["mean"])
    .sort_values("mean", ascending=False)
)

columns_to_drop = [
    "Loyalty Number",
    "Churn",
    "Most Recent Date",
    "Most Recent Year",
    "Most Recent Month",
    "Months Since Activity",
    "Country",
    "Postal Code",
    "City",
    "Cancellation Year",
    "Cancellation Month",
    "Enrollment Month",
    "Enrollment Year",
]

loyalty_numbers = master_data["Loyalty Number"].copy()
y = master_data["Churn"].copy()

X_master = master_data.drop(columns=columns_to_drop, errors="ignore")

print("=== FINAL CLEAN DATASET ===")
print(f"Shape: {X_master.shape}")
print(f"Columns: {X_master.columns.tolist()}")
print(f"\nNull values:\n{X_master.isnull().sum()}")


print("\nSalary statistics (non-null values):")
print(X_master["Salary"].describe())

skewness = X_master["Salary"].skew()
print(f"\nSkewness: {skewness:.2f}")
if abs(skewness) < 0.5:
    print("→ Distribution is fairly symmetric (use MEAN)")
elif abs(skewness) < 1:
    print("→ Distribution is moderately skewed (use MEDIAN)")
else:
    print("→ Distribution is highly skewed (use MEDIAN)")


print(f"\n Netative Values Count: {(X_master['Salary'] < 0).sum()}")
print(X_master[X_master["Salary"] < 0]["Salary"].head(20))

X_master["Salary"] = X_master["Salary"].abs()
salary_median = X_master["Salary"].median()
X_master["Salary"] = X_master["Salary"].fillna(salary_median)
print("\n After cleaning:")
print(f"Negative salaries: {(X_master['Salary'] < 0).sum()}")
print(f"Null salaries: {X_master['Salary'].isnull().sum()}")


X_master.to_csv("../data/processed/master_features.csv", index=False)
tracking = pd.DataFrame({"Loyalty Number": loyalty_numbers, "Churn": y})
tracking.to_csv("../data/processed/tracking_labels.csv", index=False)
print("===== Files Saved Succesfully =====")
