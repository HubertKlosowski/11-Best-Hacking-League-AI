import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html


def build_prompt_logs(num_rows: int = 320) -> pd.DataFrame:
    """Simulated prompt-level logs with energy consumption."""
    np.random.seed(12)
    start = pd.to_datetime("2024-09-01")
    dates = start + pd.to_timedelta(np.random.randint(0, 75, size=num_rows), unit="D")
    departments = ["Engineering", "Data Science", "Product", "GTM", "Operations"]
    users = {
        "Engineering": ["Alex Rivera", "Priya Shah", "Sam Taylor"],
        "Data Science": ["Chen Li", "Olivia Stone"],
        "Product": ["Jamie Brooks", "Morgan Lee"],
        "GTM": ["Casey Ward", "Jordan Fox"],
        "Operations": ["Riley Cruz", "Taylor Kim"],
    }
    models = ["Orion-70B", "Nimbus-34B", "Atlas-12B", "Quill-9B", "Focus-7B"]
    categories = ["Chat", "Code", "Retrieval", "Vision"]
    devices = ["Laptop", "Desktop", "Mobile", "Edge GPU"]

    records = []
    for i in range(num_rows):
        dept = np.random.choice(departments)
        user = np.random.choice(users[dept])
        category = np.random.choice(categories, p=[0.45, 0.2, 0.25, 0.1])
        model = np.random.choice(models, p=[0.28, 0.24, 0.2, 0.16, 0.12])
        device = np.random.choice(devices, p=[0.45, 0.25, 0.2, 0.1])
        energy_wh = np.random.uniform(5000, 15000)  # per prompt, larger to make MWh totals readable
        prompt_text = f"Prompt #{i+1} about {category.lower()} on {model}"
        response_text = f"Response for {category.lower()} with {model}"
        records.append(
            {
                "Date": dates[i],
                "Department": dept,
                "User": user,
                "Model": model,
                "Device": device,
                "Prompt Category": category,
                "Prompt Text": prompt_text,
                "Response Text": response_text,
                "Prompt Count": 1,
                "Energy (Wh)": round(energy_wh, 1),
            }
        )
    return pd.DataFrame(records)


prompt_logs_df = build_prompt_logs()
USER_NAME = "Alex Rivera"
USER_DEPARTMENT = "Engineering"
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


def build_user_prompts_fig(user_df: pd.DataFrame, baseline_df: pd.DataFrame, freq: str, comparison_label: str):
    user_agg = aggregate_prompts(user_df, freq)
    baseline_agg = aggregate_prompts(baseline_df, freq)
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
        fig.add_trace(
            go.Scatter(
                x=baseline_agg["Date"],
                y=baseline_agg["Prompts"],
                mode="lines",
                name=comparison_label,
                line={"shape": "spline", "color": "#b0b7c3"},
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
        fig.update_layout(title=f"Prompts over time (User vs {comparison_label})", yaxis_title="", xaxis_title="")
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
            title=f"Prompts over time (User vs {comparison_label})",
            labels={"Prompts": "User"},
        )
        fig.add_scatter(x=baseline_agg["Label"], y=baseline_agg["Prompts"], mode="lines+markers", name=comparison_label)
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
        Input("user-comparison-mode", "value"),
    ],
)
def update_user_dashboard(start_date, end_date, freq, comparison_mode):
    filtered = filter_by_date(prompt_logs_df, start_date, end_date)
    user_df = filtered[filtered["User"] == USER_NAME]
    baseline_df = filtered[filtered["Department"] == USER_DEPARTMENT] if comparison_mode == "Department" else filtered
    comparison_label = "Department baseline" if comparison_mode == "Department" else "Company baseline"

    if user_df.empty:
        empty_fig = px.bar(title="No user prompts in range")
        return empty_fig, empty_fig, empty_fig, []

    # First row bars
    baseline_cat = (
        baseline_df.groupby("Prompt Category").agg({"Energy (Wh)": "sum"}).reset_index().rename(columns={"Energy (Wh)": "Energy (kWh)"})
    )
    baseline_cat["Energy (kWh)"] = (baseline_cat["Energy (kWh)"] / 1000).round(1)
    baseline_cat["Energy (kWh)"] = baseline_cat["Energy (kWh)"] * -1  # negative for left bars
    baseline_cat = baseline_cat.sort_values(by="Energy (kWh)")  # most negative at top
    neg_colors = gradient_colors(baseline_cat["Energy (kWh)"], px.colors.sequential.Reds, reverse=False)

    user_cat = (
        user_df.groupby("Prompt Category").agg({"Energy (Wh)": "sum"}).reset_index().rename(columns={"Energy (Wh)": "Energy (kWh)"})
    )
    user_cat["Energy (kWh)"] = (user_cat["Energy (kWh)"] / 1000).round(1)
    user_cat = user_cat.sort_values(by="Energy (kWh)", ascending=False)  # largest at top
    pos_colors = gradient_colors(user_cat["Energy (kWh)"], px.colors.sequential.Blues, reverse=False)

    negative_fig = px.bar(
        baseline_cat,
        x="Energy (kWh)",
        y="Prompt Category",
        orientation="h",
        title=f"{comparison_label} energy (kWh, negative)",
    )
    negative_fig.update_traces(
        marker_color=neg_colors,
        showlegend=False,
        text=[f"{abs(v):.1f}" for v in baseline_cat["Energy (kWh)"]],
        textposition="outside",  # value to the left of the bar end
    )
    min_val = baseline_cat["Energy (kWh)"].min()
    negative_fig.update_xaxes(range=[min_val * 1.15, 0], showticklabels=False)

    neg_annotations = []
    for _, row in baseline_cat.iterrows():
        label_x = 0  # right edge of plot
        neg_annotations.append(
            dict(
                x=label_x,
                y=row["Prompt Category"],
                xref="x",
                yref="y",
                text=row["Prompt Category"],
                showarrow=False,
                xanchor="left",
                align="left",
                xshift=8,
            )
        )
    negative_fig.update_yaxes(showticklabels=False)
    negative_fig.update_layout(yaxis_title="", xaxis_title="", showlegend=False, annotations=neg_annotations)

    positive_fig = px.bar(
        user_cat,
        x="Energy (kWh)",
        y="Prompt Category",
        orientation="h",
        title="User energy (kWh)",
    )
    positive_fig.update_traces(
        marker_color=pos_colors,
        showlegend=False,
        text=[f"{v:.1f}" for v in user_cat["Energy (kWh)"]],
        textposition="outside",  # value to the right of the bar end
    )
    pos_padding = user_cat["Energy (kWh)"].max() * 0.15
    pos_annotations = []
    for _, row in user_cat.iterrows():
        pos_annotations.append(
            dict(
                x=-0.02,
                xref="paper",
                y=row["Prompt Category"],
                yref="y",
                text=row["Prompt Category"],
                showarrow=False,
                xanchor="right",
                align="right",
            )
        )
    positive_fig.update_yaxes(showticklabels=False)
    positive_fig.update_xaxes(
        showticklabels=False,
        range=[0, user_cat["Energy (kWh)"].max() * 1.05],
        zeroline=False,
    )
    positive_fig.update_layout(yaxis_title="", xaxis_title="", showlegend=False, annotations=pos_annotations)

    donut_fig = build_user_category_donut(user_df)
    prompts_fig = build_user_prompts_fig(user_df, baseline_df, freq, comparison_label)
    table_data = build_user_table_data(user_df)
    return donut_fig, negative_fig, positive_fig, prompts_fig, table_data


if __name__ == "__main__":
    app.run_server(debug=True)
