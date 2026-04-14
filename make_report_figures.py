from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# CONFIG
DATA_FILE = "FINAL_NORMALIZED_FULL.csv"
OUTPUT_DIR = Path("report_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
DISTANCE_ORDER = ["Mid", "Far", "Near"]   

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

PALETTE = {
    "Mid": "#4C78A8",
    "Far": "#9ECAE9",
    "Near": "#E15759",
}

UNIT_MAP = {
    "Aluminum Dissolved": "µg/L",
    "Aluminum Total Recoverable": "µg/L",
    "Calcium Dissolved": "mg/L",
    "Calcium Total": "mg/L",
    "Copper Dissolved": "µg/L",
    "Copper Total Recoverable": "µg/L",
    "Lead Dissolved": "µg/L",
    "Lead Total Recoverable": "µg/L",
    "Mercury Dissolved": "µg/L",
    "Mercury Total": "µg/L",
    "Methyl Mercury": "µg/L",
    "Nickel Dissolved": "µg/L",
    "Nickel Total Recoverable": "µg/L",
    "Nitrogen Kjeldahl Dissolved": "mg/L",
    "Nitrogen Kjeldahl Total": "mg/L",
    "pH": "Unitless",
    "pH (Field)": "Unitless",
    "Phosphorus Total": "mg/L",
    "Phosphorus Total Dissolved": "mg/L",
    "Turbidity": "NTU",
    "Vanadium Total Recoverable": "µg/L",
    "Vanadium Dissolved": "µg/L",
    "OXYGEN BIOCHEMICAL DEMAND": "mg/L",
    "Oxygen Dissolved (Field Meter)": "mg/L",
    "Oxygen dissolved % saturation": "%",
}

CHEMISTRY_VARS = list(UNIT_MAP.keys())

LAKE_COORDS = {
    "Isadore": (57.23041, -111.60697),
    "Mildred": (57.05556, -111.58889),
    "Kearl": (57.29170, -111.23330),
    "McClelland": (57.49125, -111.27844),
    "Namur": (56.48447, -110.83511),
    "Gregoire": (57.44440, -112.62110),
    "AR6": (57.02000, -111.50000),
}

# HELPERS
def derive_lake(site_name: str) -> str:
    if pd.isna(site_name):
        return "Unknown"
    s = str(site_name).upper()
    if "ISADORE" in s:
        return "Isadore"
    if "MILDRED" in s:
        return "Mildred"
    if "KEARL" in s:
        return "Kearl"
    if "MCCLELLAND" in s:
        return "McClelland"
    if "NAMUR" in s:
        return "Namur"
    if "GREGOIRE" in s:
        return "Gregoire"
    return "Unknown"


def with_unit(var_name: str) -> str:
    unit = UNIT_MAP.get(var_name, "")
    return f"{var_name} ({unit})" if unit and unit != "Unitless" else var_name


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def save_fig(filename: str):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def make_boxplot(df, var, p_text, filename):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    sns.boxplot(
        data=df,
        x="Distance Group",
        y=var,
        order=DISTANCE_ORDER,
        palette=PALETTE,
        ax=ax
    )
    ax.set_title(f"{with_unit(var)} by Distance Group (p = {p_text})")
    ax.set_xlabel("Distance Group")
    ax.set_ylabel(with_unit(var))
    save_fig(filename)


def make_dual_boxplot(df, var1, var2, p1, p2, filename):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)

    sns.boxplot(
        data=df,
        x="Distance Group",
        y=var1,
        order=DISTANCE_ORDER,
        palette=PALETTE,
        ax=axes[0]
    )
    axes[0].set_title(f"{with_unit(var1)} (p = {p1})")
    axes[0].set_xlabel("")
    axes[0].set_ylabel(with_unit(var1))

    sns.boxplot(
        data=df,
        x="Distance Group",
        y=var2,
        order=DISTANCE_ORDER,
        palette=PALETTE,
        ax=axes[1]
    )
    axes[1].set_title(f"{with_unit(var2)} (p = {p2})")
    axes[1].set_xlabel("Distance Group")
    axes[1].set_ylabel(with_unit(var2))

    save_fig(filename)


# LOAD DATA
df = pd.read_csv(DATA_FILE)

if "Sampling Timestamp" in df.columns:
    df["Sampling Timestamp"] = pd.to_datetime(df["Sampling Timestamp"], errors="coerce")

if "Site Name" in df.columns:
    df["Lake"] = df["Site Name"].apply(derive_lake)
else:
    df["Lake"] = "Unknown"

df["Latitude"] = df["Lake"].map(lambda x: LAKE_COORDS.get(x, (np.nan, np.nan))[0])
df["Longitude"] = df["Lake"].map(lambda x: LAKE_COORDS.get(x, (np.nan, np.nan))[1])

ar6_lat, ar6_lon = LAKE_COORDS["AR6"]
df["Distance (km)"] = haversine(df["Latitude"], df["Longitude"], ar6_lat, ar6_lon)

CHEMISTRY_VARS = [c for c in CHEMISTRY_VARS if c in df.columns]
df = df[df["Distance Group"].isin(DISTANCE_ORDER)].copy()

# SECTION 2 TABLES

# Table 2.1.1 - Geographic coordinates and AR6 distance
lake_distance_table = (
    df.groupby(["Lake", "Latitude", "Longitude", "Distance Group"], dropna=False)["Distance (km)"]
    .mean()
    .reset_index()
)

lake_order = ["Mildred", "Isadore", "Kearl", "McClelland", "Gregoire", "Namur"]
lake_distance_table["Lake"] = pd.Categorical(
    lake_distance_table["Lake"], categories=lake_order, ordered=True
)
lake_distance_table = lake_distance_table.sort_values("Lake").copy()

lake_distance_table["Latitude (°N)"] = lake_distance_table["Latitude"].round(5)
lake_distance_table["Longitude (°W)"] = lake_distance_table["Longitude"].round(5)
lake_distance_table["AR6 Mean Distance (km)"] = lake_distance_table["Distance (km)"].round(2)

lake_distance_table = lake_distance_table[
    ["Lake", "Latitude (°N)", "Longitude (°W)", "AR6 Mean Distance (km)", "Distance Group"]
]
lake_distance_table.to_csv(OUTPUT_DIR / "table_2_1_1_lake_coordinates_distance.csv", index=False)

# Table 2.2.1 - Dataset summary
start_year = int(df["Sampling Timestamp"].dt.year.min()) if "Sampling Timestamp" in df.columns else np.nan
end_year = int(df["Sampling Timestamp"].dt.year.max()) if "Sampling Timestamp" in df.columns else np.nan

dataset_summary = pd.DataFrame({
    "Attribute": [
        "Number of Lakes",
        "Distance Groups",
        "Total Observations",
        "Time Span",
        "Number of Variables",
        "Data Source",
        "Data Structure",
        "Sampling Type",
        "Missing Data",
        "Preprocessing Applied",
    ],
    "Value": [
        "6",
        "Near, Mid, Far",
        f"{len(df)}",
        f"{start_year}-{end_year}",
        f"{len(CHEMISTRY_VARS)}",
        "OSMP (Oil Sands Monitoring Program), WQP (Water Quality Portal)",
        "Wide format (one event per row)",
        "Surface water measurements",
        "Yes",
        "Standardization, aggregation, median imputation",
    ]
})
dataset_summary.to_csv(OUTPUT_DIR / "table_2_2_1_dataset_summary.csv", index=False)

# Table 2.2.2 - Variable categories
variable_categories = pd.DataFrame({
    "Category": [
        "Metals",
        "Nutrients",
        "Major Ions",
        "Physical/Chemical Indicators",
    ],
    "Variables": [
        "Aluminum Dissolved, Aluminum Total Recoverable, Copper Dissolved, Copper Total Recoverable, Lead Dissolved, Lead Total Recoverable, Mercury Dissolved, Mercury Total, Methyl Mercury, Nickel Dissolved, Nickel Total Recoverable, Vanadium Dissolved, Vanadium Total Recoverable",
        "Nitrogen Kjeldahl Dissolved, Nitrogen Kjeldahl Total, Phosphorus Total, Phosphorus Total Dissolved",
        "Calcium Dissolved, Calcium Total",
        "pH, pH (Field), Turbidity, Oxygen Dissolved (Field Meter), Oxygen Dissolved % Saturation, Oxygen Biochemical Demand",
    ]
})
variable_categories.to_csv(OUTPUT_DIR / "table_2_2_2_variable_categories.csv", index=False)

# Table 2.3.1 - Grouped units
grouped_units = pd.DataFrame({
    "Units": ["µg/L", "mg/L", "Unitless", "NTU", "Percentage (%)"],
    "Variables": [
        "Aluminum Dissolved, Aluminum Total Recoverable, Copper Dissolved, Copper Total Recoverable, Lead Dissolved, Lead Total Recoverable, Mercury Dissolved, Mercury Total, Methyl Mercury, Nickel Dissolved, Nickel Total Recoverable, Vanadium Dissolved, Vanadium Total Recoverable",
        "Calcium Dissolved, Calcium Total, Nitrogen Kjeldahl Dissolved, Nitrogen Kjeldahl Total, Phosphorus Total, Phosphorus Total Dissolved, Oxygen Dissolved (Field Meter), Oxygen Biochemical Demand",
        "pH, pH (Field)",
        "Turbidity",
        "Oxygen Dissolved % Saturation",
    ]
})
grouped_units.to_csv(OUTPUT_DIR / "table_2_3_1_grouped_units.csv", index=False)

# SECTION 2 FIGURES

# Figure 2.4.1 - Correlation heatmap
corr_df = df[CHEMISTRY_VARS].apply(pd.to_numeric, errors="coerce")
corr_df = corr_df.dropna(axis=1, how="all")
corr_df = corr_df.loc[:, corr_df.nunique(dropna=True) > 1]
corr_matrix = corr_df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.25,
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("Correlation Between Chemical Variables")
save_fig("figure_2_4_1_correlation_heatmap.png")


# OPTIONAL EDA FIGURES

# Figure 2.4.2 - Representative distribution + boxplot
eda_var = "Aluminum Dissolved" if "Aluminum Dissolved" in df.columns else CHEMISTRY_VARS[0]

fig, axes = plt.subplots(2, 1, figsize=(7, 7))

sns.histplot(
    data=df,
    x=eda_var,
    hue="Distance Group",
    hue_order=DISTANCE_ORDER,
    palette=PALETTE,
    kde=True,
    bins=20,
    ax=axes[0]
)
axes[0].set_title(f"Distribution of {eda_var}")
axes[0].set_xlabel(with_unit(eda_var))
axes[0].set_ylabel("Count")

sns.boxplot(
    data=df,
    x="Distance Group",
    y=eda_var,
    order=DISTANCE_ORDER,
    palette=PALETTE,
    ax=axes[1]
)
axes[1].set_title(f"{eda_var} by Distance Group")
axes[1].set_xlabel("Distance Group")
axes[1].set_ylabel(with_unit(eda_var))

save_fig("figure_2_4_2_distribution_boxplot.png")


# PCA
pca_source = df.copy()
pca_source["Sampling Date"] = pca_source["Sampling Timestamp"].dt.date
pca_source = pca_source.dropna(subset=["Sampling Date"])

df_pca = pca_source.groupby(
    ["Site Name", "Sampling Date", "Distance Group"], as_index=False
).agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan)

if "Lake" not in df_pca.columns:
    df_pca["Lake"] = df_pca["Site Name"].apply(derive_lake)

pca_vars = [c for c in CHEMISTRY_VARS if c != "pH (Field)"]
X = df_pca[pca_vars].apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]
X = X.fillna(X.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
pca_df["Distance Group"] = df_pca["Distance Group"].values
pca_df["Lake"] = df_pca["Lake"].values

pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100

# Table 3.1.1
variance_df = pd.DataFrame({
    "Component": ["PC1", "PC2", "Cumulative"],
    "Explained Variance (%)": [
        round(pc1_var, 2),
        round(pc2_var, 2),
        round(pc1_var + pc2_var, 2)
    ]
})
variance_df.to_csv(OUTPUT_DIR / "table_3_1_1_pca_variance.csv", index=False)

# Figure 3.1.1
fig, ax = plt.subplots(figsize=(7, 4.8))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Distance Group",
    hue_order=DISTANCE_ORDER,
    palette=PALETTE,
    s=40,
    ax=ax
)
ax.set_title(f"PCA of Water Chemistry by Distance Group (PC1: {pc1_var:.2f}%, PC2: {pc2_var:.2f}%)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
save_fig("figure_3_1_1_pca_scatter.png")

# Table 3.1.2
loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=X.columns
).reset_index()
loadings.columns = ["Variable", "PC1", "PC2"]
loadings["abs_max"] = loadings[["PC1", "PC2"]].abs().max(axis=1)
loadings_top = loadings.sort_values("abs_max", ascending=False).drop(columns="abs_max")
loadings_top.to_csv(OUTPUT_DIR / "table_3_1_2_pca_loadings.csv", index=False)

# Figure 3.1.2
loadings_pc1 = loadings.sort_values("PC1", key=lambda s: s.abs(), ascending=False).head(15)

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.barplot(
    data=loadings_pc1,
    x="PC1",
    y="Variable",
    color="#7EAEDB",
    ax=ax
)
ax.set_title("Top Variable Loadings for PC1")
ax.set_xlabel("PC1 Loading")
ax.set_ylabel("Variable")
save_fig("figure_3_1_2_pca_loadings_pc1.png")


# ANOVA
anova_results = []

for col in CHEMISTRY_VARS:
    temp = df[["Distance Group", col]].copy()
    temp[col] = pd.to_numeric(temp[col], errors="coerce")

    grouped = []
    for g in DISTANCE_ORDER:
        vals = temp[temp["Distance Group"] == g][col].dropna()
        if len(vals) > 1:
            grouped.append(vals)

    if len(grouped) >= 2:
        f_stat, p_val = f_oneway(*grouped)
        anova_results.append({
            "Variable": col,
            "F Statistic": round(f_stat, 4),
            "p-value": p_val,
            "Unit": UNIT_MAP.get(col, ""),
            "Significant": "Yes" if p_val < 0.05 else "No"
        })

anova_df = pd.DataFrame(anova_results).sort_values("F Statistic", ascending=False)
anova_df.to_csv(OUTPUT_DIR / "table_3_2_1_anova_summary.csv", index=False)

paired_boxplots = [
    ("Calcium Dissolved", "Calcium Total"),
    ("Nickel Dissolved", "Nickel Total Recoverable"),
    ("Vanadium Dissolved", "Vanadium Total Recoverable"),
    ("Nitrogen Kjeldahl Total", "Nitrogen Kjeldahl Dissolved"),
    ("Phosphorus Total", "Phosphorus Total Dissolved"),
    ("pH", "pH (Field)"),
]

single_boxplots = [
    "Turbidity",
    "Mercury Total",
]

fig_num = 1

for var1, var2 in paired_boxplots:
    if var1 in df.columns and var2 in df.columns:
        p1 = anova_df.loc[anova_df["Variable"] == var1, "p-value"].values
        p2 = anova_df.loc[anova_df["Variable"] == var2, "p-value"].values
        p1_text = f"{p1[0]:.4f}" if len(p1) else "N/A"
        p2_text = f"{p2[0]:.4f}" if len(p2) else "N/A"

        make_dual_boxplot(
            df,
            var1,
            var2,
            p1_text,
            p2_text,
            f"figure_3_2_{fig_num}_anova_{var1.lower().replace(' ', '_')}_{var2.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        )
        fig_num += 1

for var in single_boxplots:
    if var in df.columns:
        p = anova_df.loc[anova_df["Variable"] == var, "p-value"].values
        p_text = f"{p[0]:.4f}" if len(p) else "N/A"
        make_boxplot(
            df,
            var,
            p_text,
            f"figure_3_2_{fig_num}_anova_{var.lower().replace(' ', '_')}.png"
        )
        fig_num += 1


# RANDOM FOREST
rf_vars = [c for c in CHEMISTRY_VARS if c in df.columns]
X_rf = df[rf_vars].apply(pd.to_numeric, errors="coerce")
X_rf = X_rf.dropna(axis=1, how="all")
X_rf = X_rf.loc[:, X_rf.nunique() > 1]
y_rf = df["Distance Group"].copy()

imputer = SimpleImputer(strategy="median")
X_rf_imputed = imputer.fit_transform(X_rf)

X_train, X_test, y_train, y_test = train_test_split(
    X_rf_imputed,
    y_rf,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_rf
)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Table 3.3.1
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
report_df.to_csv(OUTPUT_DIR / "table_3_3_1_rf_classification_report.csv")

# Figure 3.3.1
fig, ax = plt.subplots(figsize=(6.5, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred, labels=DISTANCE_ORDER),
    display_labels=DISTANCE_ORDER
)
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title("Random Forest Confusion Matrix")
save_fig("figure_3_3_1_rf_confusion_matrix.png")

# Table 3.3.2
importance_df = pd.DataFrame({
    "Feature": X_rf.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)
importance_df.to_csv(OUTPUT_DIR / "table_3_3_2_rf_feature_importance.csv", index=False)

# Figure 3.3.2
top_n = 10
importance_plot = importance_df.head(top_n).copy()
importance_plot["Feature Label"] = importance_plot["Feature"].apply(with_unit)

fig, ax = plt.subplots(figsize=(7, 4.8))
sns.barplot(
    data=importance_plot,
    x="Importance",
    y="Feature Label",
    color="#7EAEDB",
    ax=ax
)
ax.set_title(f"Top {top_n} Random Forest Feature Importances")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
save_fig("figure_3_3_2_rf_feature_importance.png")


print("\nAll figures and tables exported to:")
print(OUTPUT_DIR.resolve())