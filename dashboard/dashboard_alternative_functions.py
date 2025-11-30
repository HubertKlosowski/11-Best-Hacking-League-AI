import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Minimal elegant color palette
COLORS = {
    'primary': '#10b981',      # Green
    'primary_light': '#34d399',
    'primary_dark': '#059669',
    'dark': '#1f2937',
    'gray': '#6b7280',
    'light_gray': '#f3f4f6',
    'border': '#e5e7eb',
    'white': '#ffffff',
}


def load_prompt_logs(data_path: str) -> pd.DataFrame:
    """Load prompt logs from CSV and derive fields used in the dashboard."""
    df = pd.read_csv(data_path, parse_dates=["prompt_datetime"])
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


def style_plotly_figure(fig):
    """Apply consistent minimal styling to plotly figures."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, -apple-system, sans-serif", color=COLORS['dark'], size=12),
        title_font=dict(size=16, color=COLORS['dark']),
        title_x=0,
        margin=dict(t=50, b=40, l=40, r=40),
        hovermode='closest',
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor=COLORS['border']
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=COLORS['border'],
        zeroline=False
    )
    return fig


def build_department_donut(filtered_df: pd.DataFrame):
    """Create a donut chart showing energy usage by department."""
    grouped = (
        filtered_df.groupby("Department")
        .agg({"Energy (Wh)": "sum"})
        .reset_index()
        .rename(columns={"Energy (Wh)": "Energy (MWh)"})
    )
    grouped["Energy (MWh)"] = (grouped["Energy (MWh)"] / 1_000_000).round(2)
    
    # Use greyscale with green accent
    colors = ['#10b981', '#6b7280', '#9ca3af', '#d1d5db']
    
    fig = px.pie(
        grouped,
        names="Department",
        values="Energy (MWh)",
        hole=0.6,
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        textinfo="label+percent",
        texttemplate="%{label}<br>%{percent}",
        hovertemplate="<b>%{label}</b><br>%{value:.2f} MWh<br>%{percent}<extra></extra>",
        marker=dict(line=dict(color='white', width=2)),
        textposition="outside",
        textfont=dict(size=12, family="Inter, sans-serif")
    )
    
    total_energy = grouped["Energy (MWh)"].sum()
    fig.add_annotation(
        text=f"<b>{total_energy:.1f}</b><br><span style='font-size:12px; color:{COLORS['gray']}'>MWh Total</span>",
        x=0.5, y=0.5,
        font=dict(size=20, color=COLORS['dark'], family="Inter, sans-serif"),
        showarrow=False
    )
    
    fig.update_layout(
        title="Energy Usage by Department",
        showlegend=False,
    )
    
    return style_plotly_figure(fig)


def build_model_energy_fig(filtered_df: pd.DataFrame):
    """Create a stacked bar chart of top models by energy usage."""
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
    
    # Color-code by prompt category (these need color coding)
    colors = ['#10b981', '#34d399', '#6ee7b7', '#a7f3d0', '#d1fae5']
    
    fig = px.bar(
        grouped,
        x="Energy (MWh)",
        y="Model",
        color="Prompt Category",
        orientation="h",
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        barmode="stack",
        title="Top Models by Energy Usage",
        yaxis_title="",
        xaxis_title="Energy (MWh)",
        legend=dict(
            title_text="Prompt Category",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=11, family="Inter, sans-serif")
        ),
    )
    
    return style_plotly_figure(fig)


def build_prompts_over_time_fig(filtered_df: pd.DataFrame, freq: str, title: str):
    """Create a time series chart of prompts."""
    aggregated = aggregate_prompts(filtered_df, freq)
    
    if freq == "D":
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=aggregated["Date"],
                y=aggregated["Prompts"],
                marker=dict(
                    color=COLORS['primary'],
                    line=dict(width=0)
                ),
                hovertemplate="<b>%{x}</b><br>Prompts: %{y:,}<extra></extra>"
            )
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
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=aggregated["Label"],
                y=aggregated["Prompts"],
                mode="lines+markers",
                line=dict(color=COLORS['primary'], width=2),
                marker=dict(size=6, color=COLORS['primary']),
                hovertemplate="<b>%{x}</b><br>Prompts: %{y:,}<extra></extra>"
            )
        )
        fig.update_xaxes(categoryorder="array", categoryarray=aggregated["Label"])
    
    label = {"D": "Daily", "W": "Weekly", "M": "Monthly"}[freq]
    fig.update_layout(
        title=f"{title} - {label}",
        yaxis_title="Prompts",
        xaxis_title="",
        showlegend=False,
        hovermode="x unified"
    )
    
    return style_plotly_figure(fig)


def build_user_category_donut(filtered_df: pd.DataFrame):
    """Create a donut chart of energy by prompt category for user."""
    grouped = (
        filtered_df.groupby("Prompt Category")
        .agg({"Energy (Wh)": "sum"})
        .reset_index()
        .rename(columns={"Energy (Wh)": "Energy (MWh)"})
    )
    grouped["Energy (MWh)"] = (grouped["Energy (MWh)"] / 1_000_000).round(2)
    
    # Color-code categories (these need color coding)
    colors = ['#10b981', '#34d399', '#6ee7b7', '#a7f3d0', '#d1fae5']
    
    fig = px.pie(
        grouped,
        names="Prompt Category",
        values="Energy (MWh)",
        hole=0.6,
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        textinfo="label+percent",
        texttemplate="%{label}<br>%{percent}",
        hovertemplate="<b>%{label}</b><br>%{value:.2f} MWh<br>%{percent}<extra></extra>",
        marker=dict(line=dict(color='white', width=2)),
        textposition="outside",
        textfont=dict(size=12, family="Inter, sans-serif")
    )
    
    fig.update_layout(
        title="Energy by Prompt Category",
        showlegend=False,
    )
    
    return style_plotly_figure(fig)


def build_user_shap_bars(user_df: pd.DataFrame):
    """Build SHAP value bar charts showing user strengths and improvement areas."""
    def extract_shap(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        rows = []
        for i in range(1, 6):
            feat_col = f"{kind}{i}_feature"
            shap_col = f"{kind}{i}_shap"
            if feat_col in df.columns and shap_col in df.columns:
                rows.append(df[[feat_col, shap_col]].rename(columns={feat_col: "feature", shap_col: "shap"}))
        if not rows:
            return pd.DataFrame(columns=["feature", "shap"])
        return pd.concat(rows, ignore_index=True).dropna()

    shap_to_wh = lambda s: s * 1000

    pos_agg = extract_shap(user_df, "pos").groupby("feature")["shap"].mean().reset_index()
    pos_agg["shap_wh"] = shap_to_wh(pos_agg["shap"])
    pos_agg = pos_agg.sort_values(by="shap_wh", ascending=False).head(5)  # highest to lowest

    neg_agg = extract_shap(user_df, "neg").groupby("feature")["shap"].mean().reset_index()
    neg_agg["shap_wh"] = shap_to_wh(neg_agg["shap"])
    neg_agg = neg_agg.sort_values(by="shap_wh", ascending=True).head(5)  # lowest (most negative) to highest

    # Strengths (negative SHAP reduces energy) - blue gradient reversed, values in Wh with 2 decimals
    neg_fig = px.bar(
        neg_agg,
        x="shap_wh",
        y="feature",
        orientation="h",
        title="User Strong Sides",
        color="shap_wh",
        color_continuous_scale=px.colors.sequential.Blues_r,
    )
    neg_fig.update_traces(
        text=neg_agg["shap_wh"].round(2),
        texttemplate="%{text} Wh",
        textposition="outside",
    )
    neg_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="", coloraxis_showscale=False)
    neg_fig = style_plotly_figure(neg_fig)

    # Areas to improve (positive SHAP increases energy) - red gradient, values in Wh with 2 decimals
    pos_fig = px.bar(
        pos_agg,
        x="shap_wh",
        y="feature",
        orientation="h",
        title="Areas to Improve",
        color="shap_wh",
        color_continuous_scale=px.colors.sequential.Reds,
    )
    pos_fig.update_traces(
        text=pos_agg["shap_wh"].round(2),
        texttemplate="%{text} Wh",
        textposition="outside",
    )
    pos_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="", coloraxis_showscale=False)
    pos_fig = style_plotly_figure(pos_fig)

    return neg_fig, pos_fig


def build_user_table_data(user_df: pd.DataFrame):
    """Prepare user prompt log data for table display."""
    sorted_df = user_df.sort_values(by="Energy (Wh)", ascending=False)
    display_df = sorted_df[
        ["Date", "Model", "Device", "Prompt Text", "Response Text", "Energy (Wh)"]
    ].copy()
    display_df["Date"] = display_df["Date"].dt.date
    display_df["Energy (kWh)"] = (display_df["Energy (Wh)"] / 1000).round(2)
    display_df = display_df.drop(columns=["Energy (Wh)"])
    return display_df.to_dict("records")
