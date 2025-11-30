import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

DATA_PATH = "data/X_test_with_shap_posneg_mocked.csv"


def load_prompt_logs() -> pd.DataFrame:
    """Load prompt logs from CSV and derive fields used in the dashboard."""
    df = pd.read_csv(DATA_PATH, parse_dates=["prompt_datetime"])
    df = df.rename(
        columns={
            "prompt_datetime": "Date",
            "prompt_category": "Prompt Category",
            "department": "Department",
            "user_id": "User",
            "model_name": "Model",
            "device_type": "Device",
        }
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df["Prompt Count"] = 1
    token_sum = df["prompt_token_length"].fillna(0) + df["response_token_length"].fillna(0)
    df["Energy (Wh)"] = (token_sum * 100).round(1)
    df["Prompt Text"] = "Prompt about " + df["Prompt Category"].astype(str)
    df["Response Text"] = "Response from " + df["Model"].astype(str)
    return df


prompt_logs_df = load_prompt_logs()
USER_NAME = str(prompt_logs_df["User"].iloc[0])
USER_DEPARTMENT = str(prompt_logs_df[prompt_logs_df["User"] == USER_NAME]["Department"].iloc[0])
MIN_DATE = prompt_logs_df["Date"].min().date()
MAX_DATE = prompt_logs_df["Date"].max().date()


def aggregate_prompts(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate prompt counts by the chosen frequency."""
    freq_map = {"D": "D", "W": "W-MON", "M": "MS"}
    grouped = df.set_index("Date").resample(freq_map[freq]).agg({"Prompt Count": "sum"}).reset_index()
    grouped["Date"] = grouped["Date"].dt.date

    if freq == "D":
        dt_index = pd.to_datetime(grouped["Date"])
        grouped["MonthLabel"] = dt_index.dt.strftime("%b")
        grouped["WeekLabel"] = "W" + dt_index.dt.isocalendar().week.astype(str)
        grouped["DayLabel"] = dt_index.dt.strftime("%d")
        grouped["Label"] = grouped["DayLabel"]
    elif freq == "W":
        week_num = pd.to_datetime(grouped["Date"]).dt.isocalendar().week
        grouped["Label"] = "W" + week_num.astype(str)
    else:
        grouped["Label"] = pd.to_datetime(grouped["Date"]).dt.strftime("%b")

    return grouped.rename(columns={"Prompt Count": "Prompts"})


def filter_by_date(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Filter dataframe by date range."""
    start = pd.to_datetime(start_date) if start_date else df["Date"].min()
    end = pd.to_datetime(end_date) if end_date else df["Date"].max()
    return df[(df["Date"] >= start) & (df["Date"] <= end)]


def build_kpi_cards(filtered_df: pd.DataFrame):
    energy_mwh = filtered_df["Energy (Wh)"].sum() / 1_000_000
    prompts_served = int(filtered_df["Prompt Count"].sum())
    avg_energy_wh = filtered_df["Energy (Wh)"].mean() if len(filtered_df) else 0
    kpis = [
        {"label": "Total Energy (MWh)", "value": f"{energy_mwh:.2f}"},
        {"label": "Prompts Served", "value": f"{prompts_served:,}"},
        {"label": "Avg Energy / Prompt (Wh)", "value": f"{avg_energy_wh:.0f}"},
    ]
    return [
        html.Div(
            [
                html.Div(kpi["label"], style={"color": "#6b7280", "fontSize": 13}),
                html.Div(kpi["value"], style={"fontSize": 24, "fontWeight": 700}),
            ],
            style={
                "padding": "14px 16px",
                "backgroundColor": "white",
                "border": "1px solid #e5e7eb",
                "borderRadius": "12px",
                "textAlign": "center",
            },
        )
        for kpi in kpis
    ]


def build_department_donut(filtered_df: pd.DataFrame):
    grouped = (
        filtered_df.groupby("Department")
        .agg({"Energy (Wh)": "sum"})
        .reset_index()
        .rename(columns={"Energy (Wh)": "Energy (MWh)"})
    )
    grouped["Energy (MWh)"] = (grouped["Energy (MWh)"] / 1_000_000).round(2)
    fig = px.pie(
        grouped,
        names="Department",
        values="Energy (MWh)",
        hole=0.5,
        title="Energy Usage by Department",
    )
    fig.update_traces(
        textinfo="text",
        texttemplate="%{label}: %{value:.2f} (%{percent})",
        hovertemplate="%{label}: %{value:.2f} (%{percent})",
        showlegend=False,
        pull=0.02,
        textposition="outside",
        marker_line_color="white",
        marker_line_width=1,
    )
    fig.update_layout(annotations=[], uniformtext_minsize=10, uniformtext_mode="hide")
    return fig


def build_model_energy_fig(filtered_df: pd.DataFrame):
    grouped = (
        filtered_df.groupby(["Model", "Prompt Category"])
        .agg({"Energy (Wh)": "sum"})
        .reset_index()
        .rename(columns={"Energy (Wh)": "Energy (MWh)"})
    )
    grouped["Energy (MWh)"] = grouped["Energy (MWh)"] / 1_000_000
    model_totals = grouped.groupby("Model")["Energy (MWh)"].sum().sort_values(ascending=False)
    top_models = model_totals.head(5).index.tolist()
    grouped = grouped[grouped["Model"].isin(top_models)]
    grouped["Model"] = pd.Categorical(grouped["Model"], categories=top_models, ordered=True)
    fig = px.bar(
        grouped,
        x="Energy (MWh)",
        y="Model",
        color="Prompt Category",
        orientation="h",
        title="Top Models by Energy (stacked by prompt type)",
        color_discrete_sequence=px.colors.sequential.Blues[2:],
    )
    fig.update_layout(
        barmode="stack",
        yaxis_title="",
        xaxis_title="",
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def build_prompts_over_time_fig(filtered_df: pd.DataFrame, freq: str, title_prefix: str):
    aggregated = aggregate_prompts(filtered_df, freq)
    if freq == "D":
        fig = px.bar(
            aggregated,
            x="Date",
            y="Prompts",
            title=f"{title_prefix} (Daily)",
        )
        dt_series = pd.to_datetime(aggregated["Date"])
        month_groups = dt_series.groupby(dt_series.dt.to_period("M"))
        tickvals = []
        ticktext = []
        for period, dates in month_groups:
            midpoint = dates.min() + (dates.max() - dates.min()) / 2
            tickvals.append(midpoint)
            ticktext.append(period.strftime("%b"))
        fig.update_xaxes(
            type="date",
            title="",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
        )
    else:
        label = "Weekly" if freq == "W" else "Monthly"
        fig = px.line(aggregated, x="Label", y="Prompts", markers=True, title=f"{title_prefix} ({label})")
        fig.update_xaxes(categoryorder="array", categoryarray=aggregated["Label"])
    fig.update_layout(yaxis_title="", xaxis_title="")
    return fig


def build_user_category_donut(filtered_df: pd.DataFrame):
    grouped = (
        filtered_df.groupby("Prompt Category")
        .agg({"Energy (Wh)": "sum"})
        .reset_index()
        .rename(columns={"Energy (Wh)": "Energy (MWh)"})
    )
    grouped["Energy (MWh)"] = (grouped["Energy (MWh)"] / 1_000_000).round(2)
    fig = px.pie(grouped, names="Prompt Category", values="Energy (MWh)", hole=0.45, title="Energy by Prompt Category")
    fig.update_traces(
        textinfo="text",
        texttemplate="%{label}: %{value:.2f} (%{percent})",
        hovertemplate="%{label}: %{value:.2f} (%{percent})",
        showlegend=False,
        pull=0.02,
        textposition="outside",
        marker_line_color="white",
        marker_line_width=1,
    )
    fig.update_layout(annotations=[], uniformtext_minsize=10, uniformtext_mode="hide")
    return fig


def build_user_prompts_fig(user_df: pd.DataFrame, freq: str):
    user_agg = aggregate_prompts(user_df, freq)
    if freq == "D":
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=user_agg["Date"],
                y=user_agg["Prompts"],
                name="User",
                marker_color=px.colors.sequential.Blues[4],
            )
        )
        dt_series = pd.to_datetime(user_agg["Date"])
        month_groups = dt_series.groupby(dt_series.dt.to_period("M"))
        tickvals = []
        ticktext = []
        for period, dates in month_groups:
            midpoint = dates.min() + (dates.max() - dates.min()) / 2
            tickvals.append(midpoint)
            ticktext.append(period.strftime("%b"))
        fig.update_layout(title="Prompts over time", yaxis_title="", xaxis_title="")
        fig.update_xaxes(
            type="date",
            title="",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
        )
    else:
        fig = px.line(
            user_agg,
            x="Label",
            y="Prompts",
            markers=True,
            title="Prompts over time",
            labels={"Prompts": "User"},
        )
        fig.update_xaxes(categoryorder="array", categoryarray=user_agg["Label"])
    fig.update_layout(yaxis_title="", xaxis_title="")
    return fig


def build_user_table_data(user_df: pd.DataFrame):
    sorted_df = user_df.sort_values(by="Energy (Wh)", ascending=False)
    display_df = sorted_df[
        ["Date", "Model", "Device", "Prompt Text", "Response Text", "Energy (Wh)"]
    ].copy()
    display_df["Date"] = display_df["Date"].dt.date
    display_df["Energy (kWh)"] = (display_df["Energy (Wh)"] / 1000).round(2)
    display_df = display_df.drop(columns=["Energy (Wh)"])
    return display_df.to_dict("records")


def gradient_colors(values: pd.Series, scale: list[str], reverse: bool = False) -> list[str]:
    """Map values to a color gradient."""
    if values.empty:
        return []
    vals = values.to_numpy()
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        return [scale[0]] * len(values)
    norm = (vals - vmin) / (vmax - vmin)
    if reverse:
        norm = 1 - norm
    return [px.colors.sample_colorscale(scale, float(n))[0] for n in norm]


app = Dash(__name__)
app.title = "Company & User Dashboard"
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("Performance Pulse"),
                html.P("Company and user reports side by side in a single Dash app."),
            ],
            style={"padding": "16px 24px", "borderBottom": "1px solid #e1e5ee", "backgroundColor": "white"},
        ),
        dcc.Tabs(
            children=[
                dcc.Tab(
                    label="Company Report",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Date range", style={"fontWeight": 600}),
                                                dcc.DatePickerRange(
                                                    id="company-date-range",
                                                    start_date=MIN_DATE,
                                                    end_date=MAX_DATE,
                                                    min_date_allowed=MIN_DATE,
                                                    max_date_allowed=MAX_DATE,
                                                ),
                                            ],
                                            style={"marginBottom": "12px"},
                                        ),
                                        html.Div(id="company-kpis", style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"}),
                                        html.Div(
                                            [
                                                html.Div(
                                                    dcc.Graph(id="dept-energy-graph"),
                                                    style={"backgroundColor": "white", "borderRadius": "12px", "padding": "4px"},
                                                ),
                                                html.Div(
                                                    dcc.Graph(id="model-energy-graph"),
                                                    style={"backgroundColor": "white", "borderRadius": "12px", "padding": "4px"},
                                                ),
                                            ],
                                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.H4("Prompts over time", style={"marginBottom": "6px"}),
                                                        dcc.RadioItems(
                                                            id="company-prompt-agg",
                                                            options=[
                                                                {"label": "Daily", "value": "D"},
                                                                {"label": "Weekly", "value": "W"},
                                                                {"label": "Monthly", "value": "M"},
                                                            ],
                                                            value="D",
                                                            labelStyle={
                                                                "display": "inline-block",
                                                                "padding": "6px 12px",
                                                                "marginRight": "10px",
                                                                "border": "1px solid #d7dce5",
                                                                "borderRadius": "10px",
                                                                "cursor": "pointer",
                                                            },
                                                            inputStyle={"marginRight": "6px"},
                                                            style={"marginBottom": "10px"},
                                                        ),
                                                    ],
                                                    style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                                ),
                                                dcc.Graph(id="company-prompts-graph"),
                                            ],
                                            style={
                                                "backgroundColor": "white",
                                                "borderRadius": "12px",
                                                "border": "1px solid #e5e7eb",
                                                "padding": "10px",
                                                "marginTop": "12px",
                                            },
                                        ),
                                    ],
                                    style={"display": "flex", "flexDirection": "column", "gap": "16px", "padding": "18px 22px"},
                                )
                            ],
                            style={"padding": "18px 22px"},
                        )
                    ],
                ),
                dcc.Tab(
                    label="User Report",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Date range", style={"fontWeight": 600}),
                                                dcc.DatePickerRange(
                                                    id="user-date-range",
                                                    start_date=MIN_DATE,
                                                    end_date=MAX_DATE,
                                                    min_date_allowed=MIN_DATE,
                                                    max_date_allowed=MAX_DATE,
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Comparison baseline", style={"fontWeight": 600, "marginRight": "10px"}),
                                                dcc.RadioItems(
                                                    id="user-comparison-mode",
                                                    options=[
                                                        {"label": "Department", "value": "Department"},
                                                        {"label": "Company", "value": "Company"},
                                                    ],
                                                    value="Department",
                                                    labelStyle={"display": "inline-block", "marginRight": "12px"},
                                                ),
                                            ],
                                            style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "18px", "marginBottom": "16px"},
                                ),
                                html.Div(
                                    [
                                        html.Div(dcc.Graph(id="user-negative-bar")),
                                        html.Div(dcc.Graph(id="user-positive-bar")),
                                    ],
                                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            dcc.Graph(id="user-category-donut"),
                                            style={"backgroundColor": "white", "borderRadius": "12px", "border": "1px solid #e5e7eb", "padding": "4px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Aggregation", style={"fontWeight": 600, "marginRight": "10px"}),
                                                        dcc.RadioItems(
                                                            id="user-prompt-agg",
                                                            options=[
                                                                {"label": "Daily", "value": "D"},
                                                                {"label": "Weekly", "value": "W"},
                                                                {"label": "Monthly", "value": "M"},
                                                            ],
                                                            value="D",
                                                            labelStyle={"display": "inline-block", "marginRight": "12px"},
                                                        ),
                                                    ],
                                                    style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "8px"},
                                                ),
                                                dcc.Graph(id="user-prompts-graph"),
                                            ],
                                            style={"backgroundColor": "white", "borderRadius": "12px", "border": "1px solid #e5e7eb", "padding": "10px"},
                                        ),
                                    ],
                                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "12px"},
                                ),
                                html.Div(
                                    [
                                        html.H4("Prompt log (sorted by energy used)"),
                                        dash_table.DataTable(
                                            id="user-prompts-table",
                                            columns=[
                                                {"name": "Date", "id": "Date"},
                                                {"name": "Model", "id": "Model"},
                                                {"name": "Device", "id": "Device"},
                                                {"name": "Prompt Text", "id": "Prompt Text"},
                                                {"name": "Response Text", "id": "Response Text"},
                                                {"name": "Energy (kWh)", "id": "Energy (kWh)"},
                                            ],
                                            data=[],
                                            style_as_list_view=True,
                                            style_cell={"padding": "8px", "fontSize": 13},
                                            style_header={"fontWeight": "600", "backgroundColor": "#f5f7fb"},
                                            page_size=10,
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "white",
                                        "borderRadius": "12px",
                                        "border": "1px solid #e5e7eb",
                                        "padding": "10px",
                                        "marginTop": "12px",
                                    },
                                ),
                            ],
                            style={"padding": "18px 22px"},
                        )
                    ],
                ),
            ],
            style={"padding": "0 16px"},
        ),
    ],
    style={"fontFamily": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", "backgroundColor": "#f6f8fb"},
)


@app.callback(
    [
        Output("company-kpis", "children"),
        Output("dept-energy-graph", "figure"),
        Output("model-energy-graph", "figure"),
        Output("company-prompts-graph", "figure"),
    ],
    [Input("company-date-range", "start_date"), Input("company-date-range", "end_date"), Input("company-prompt-agg", "value")],
)
def update_company_dashboard(start_date, end_date, freq):
    filtered = filter_by_date(prompt_logs_df, start_date, end_date)
    kpi_children = build_kpi_cards(filtered)
    dept_fig = build_department_donut(filtered) if not filtered.empty else px.pie(title="Energy Usage by Department")
    model_fig = build_model_energy_fig(filtered) if not filtered.empty else px.bar(title="Top Models by Energy")
    prompts_fig = build_prompts_over_time_fig(filtered, freq, "Company prompts")
    return kpi_children, dept_fig, model_fig, prompts_fig


@app.callback(
    [
        Output("user-category-donut", "figure"),
        Output("user-negative-bar", "figure"),
        Output("user-positive-bar", "figure"),
        Output("user-prompts-graph", "figure"),
        Output("user-prompts-table", "data"),
    ],
    [
        Input("user-date-range", "start_date"),
        Input("user-date-range", "end_date"),
        Input("user-prompt-agg", "value"),
    ],
)
def update_user_dashboard(start_date, end_date, freq):
    filtered = filter_by_date(prompt_logs_df, start_date, end_date)
    user_df = filtered[filtered["User"] == USER_NAME]

    if user_df.empty:
        empty_fig = px.bar(title="No user prompts in range")
        return empty_fig, empty_fig, empty_fig, empty_fig, []

    def extract_shap(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        rows = []
        for i in range(1, 6):
            feat_col = f"{kind}{i}_feature"
            shap_col = f"{kind}{i}_shap"
            if feat_col in df and shap_col in df:
                rows.append(df[[feat_col, shap_col]].rename(columns={feat_col: "feature", shap_col: "shap"}))
        if not rows:
            return pd.DataFrame(columns=["feature", "shap"])
        return pd.concat(rows, ignore_index=True).dropna()

    # Convert SHAP to watt-hours (assuming shap is in kWh-equivalent scale)
    shap_to_wh = lambda s: s * 1000

    pos_agg = extract_shap(user_df, "pos").groupby("feature")["shap"].mean().reset_index()
    pos_agg["shap_wh"] = shap_to_wh(pos_agg["shap"])
    pos_agg = pos_agg.sort_values(by="shap_wh", ascending=True).head(5)  # lowest to highest

    neg_agg = extract_shap(user_df, "neg").groupby("feature")["shap"].mean().reset_index()
    neg_agg["shap_wh"] = shap_to_wh(neg_agg["shap"])
    neg_agg = neg_agg.sort_values(by="shap_wh", ascending=False).head(5)  # highest (most negative) to lowest

    neg_fig = px.bar(
        neg_agg,
        x="shap_wh",
        y="feature",
        orientation="h",
        title="User strong sides",
        color="shap_wh",
        color_continuous_scale=px.colors.sequential.Blues[::-1],
    )
    neg_fig.update_traces(text=neg_agg["shap_wh"].round(2), textposition="outside")
    neg_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="", coloraxis_showscale=False)

    pos_fig = px.bar(
        pos_agg,
        x="shap_wh",
        y="feature",
        orientation="h",
        title="Areas to improve",
        color="shap_wh",
        color_continuous_scale=px.colors.sequential.Reds,
    )
    pos_fig.update_traces(text=pos_agg["shap_wh"].round(2), textposition="outside")
    pos_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="", coloraxis_showscale=False)

    donut_fig = build_user_category_donut(user_df)
    prompts_fig = build_user_prompts_fig(user_df, freq)
    table_data = build_user_table_data(user_df)
    return donut_fig, neg_fig, pos_fig, prompts_fig, table_data


if __name__ == "__main__":
    app.run_server(debug=False)
