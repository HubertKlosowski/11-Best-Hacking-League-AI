from dash import Dash, Input, Output, dash_table, dcc, html
from dashboard_alternative_functions import *

DATA_PATH = "data/X_test_with_shap_posneg_mocked.csv"

# Load data
prompt_logs_df = load_prompt_logs(DATA_PATH)
USER_NAME = str(prompt_logs_df["User"].iloc[0])
USER_DEPARTMENT = str(prompt_logs_df[prompt_logs_df["User"] == USER_NAME]["Department"].iloc[0])
MIN_DATE = prompt_logs_df["Date"].min().date()
MAX_DATE = prompt_logs_df["Date"].max().date()


def build_kpi_cards(filtered_df):
    """Create beautiful KPI cards with gradients."""
    energy_mwh = filtered_df["Energy (Wh)"].sum() / 1_000_000
    prompts_served = int(filtered_df["Prompt Count"].sum())
    avg_energy_wh = filtered_df["Energy (Wh)"].mean() if len(filtered_df) else 0
    
    kpis = [
        {
            "label": "Total Energy",
            "value": f"{energy_mwh:.2f}",
            "unit": "MWh",
            "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "icon": "⚡"
        },
        {
            "label": "Prompts Served",
            "value": f"{prompts_served:,}",
            "unit": "",
            "gradient": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            "icon": "💬"
        },
        {
            "label": "Avg Energy",
            "value": f"{avg_energy_wh:.0f}",
            "unit": "Wh",
            "gradient": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "icon": "📊"
        },
    ]
    
    return [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            kpi["icon"],
                            style={
                                "fontSize": "36px",
                                "marginBottom": "12px",
                                "filter": "drop-shadow(0 2px 4px rgba(0,0,0,0.1))"
                            }
                        ),
                        html.Div(
                            [
                                html.Span(
                                    kpi["value"],
                                    style={
                                        "fontSize": "36px",
                                        "fontWeight": "700",
                                        "background": kpi["gradient"],
                                        "WebkitBackgroundClip": "text",
                                        "WebkitTextFillColor": "transparent",
                                        "backgroundClip": "text",
                                        "display": "inline-block"
                                    }
                                ),
                                html.Span(
                                    f" {kpi['unit']}" if kpi['unit'] else "",
                                    style={
                                        "fontSize": "18px",
                                        "color": "#94a3b8",
                                        "fontWeight": "600",
                                        "marginLeft": "4px"
                                    }
                                ) if kpi['unit'] else None
                            ],
                            style={"marginBottom": "8px"}
                        ),
                        html.Div(
                            kpi["label"],
                            style={
                                "color": "#64748b",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "letterSpacing": "0.3px"
                            }
                        )
                    ],
                    style={"textAlign": "center"}
                )
            ],
            style={
                "padding": "28px 24px",
                "background": "linear-gradient(135deg, #ffffff 0%, #fafafa 100%)",
                "border": "1px solid #e2e8f0",
                "borderRadius": "20px",
                "boxShadow": "0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03)",
                "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                "cursor": "pointer",
                "position": "relative",
                "overflow": "hidden"
            },
            className="kpi-card"
        )
        for kpi in kpis
    ]


# Create Dash app
app = Dash(__name__)
app.title = "⚡ Performance Pulse"
server = app.server

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }
            .kpi-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.12) !important;
            }
            .kpi-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                opacity: 0;
                transition: opacity 0.3s;
            }
            .kpi-card:hover::before {
                opacity: 1;
            }
            .custom-tab {
                padding: 14px 28px !important;
                border: none !important;
                background: transparent !important;
                color: #64748b !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                transition: all 0.3s !important;
                border-bottom: 3px solid transparent !important;
            }
            .custom-tab:hover {
                color: #6366f1 !important;
                background: rgba(99, 102, 241, 0.05) !important;
            }
            .custom-tab--selected {
                color: #6366f1 !important;
                border-bottom: 3px solid #6366f1 !important;
                background: rgba(99, 102, 241, 0.08) !important;
            }
            input[type="radio"] {
                accent-color: #6366f1;
            }
            .chart-container {
                background: white;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
                border: 1px solid #e2e8f0;
                transition: all 0.3s;
            }
            .chart-container:hover {
                box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.08);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "⚡ Performance Pulse",
                            style={
                                "margin": "0",
                                "fontSize": "32px",
                                "fontWeight": "800",
                                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)",
                                "WebkitBackgroundClip": "text",
                                "WebkitTextFillColor": "transparent",
                                "backgroundClip": "text",
                                "letterSpacing": "-0.5px"
                            }
                        ),
                        html.P(
                            "Real-time insights into AI usage and performance metrics",
                            style={
                                "margin": "6px 0 0 0",
                                "fontSize": "15px",
                                "color": "#64748b",
                                "fontWeight": "500"
                            }
                        )
                    ],
                    style={"maxWidth": "1400px", "margin": "0 auto", "padding": "0 32px"}
                )
            ],
            style={
                "padding": "28px 0",
                "background": "white",
                "borderBottom": "1px solid #e2e8f0",
                "boxShadow": "0 1px 3px 0 rgba(0, 0, 0, 0.05)",
                "position": "sticky",
                "top": 0,
                "zIndex": 1000
            }
        ),
        
        # Main content
        html.Div(
            [
                dcc.Tabs(
                    id="main-tabs",
                    className="custom-tabs",
                    children=[
                        # Company Report Tab
                        dcc.Tab(
                            label="🏢 Company Report",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            children=[
                                html.Div(
                                    [
                                        # Date Range
                                        html.Div(
                                            [
                                                html.Label(
                                                    "📅 Date Range",
                                                    style={
                                                        "fontWeight": "600",
                                                        "color": "#334155",
                                                        "fontSize": "14px",
                                                        "marginBottom": "10px",
                                                        "display": "block"
                                                    }
                                                ),
                                                dcc.DatePickerRange(
                                                    id="company-date-range",
                                                    start_date=MIN_DATE,
                                                    end_date=MAX_DATE,
                                                    min_date_allowed=MIN_DATE,
                                                    max_date_allowed=MAX_DATE,
                                                    display_format="MMM DD, YYYY"
                                                )
                                            ],
                                            style={"marginBottom": "28px"}
                                        ),
                                        
                                        # KPI Cards
                                        html.Div(
                                            id="company-kpis",
                                            style={
                                                "display": "grid",
                                                "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                                                "gap": "20px",
                                                "marginBottom": "28px"
                                            }
                                        ),
                                        
                                        # Charts Row
                                        html.Div(
                                            [
                                                html.Div(
                                                    dcc.Graph(id="dept-energy-graph", config={"displayModeBar": False}),
                                                    className="chart-container"
                                                ),
                                                html.Div(
                                                    dcc.Graph(id="model-energy-graph", config={"displayModeBar": False}),
                                                    className="chart-container"
                                                )
                                            ],
                                            style={
                                                "display": "grid",
                                                "gridTemplateColumns": "repeat(auto-fit, minmax(450px, 1fr))",
                                                "gap": "24px",
                                                "marginBottom": "28px"
                                            }
                                        ),
                                        
                                        # Time Series
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Aggregation",
                                                            style={
                                                                "fontWeight": "600",
                                                                "color": "#334155",
                                                                "fontSize": "14px",
                                                                "marginRight": "16px"
                                                            }
                                                        ),
                                                        dcc.RadioItems(
                                                            id="company-prompt-agg",
                                                            options=[
                                                                {"label": " Daily", "value": "D"},
                                                                {"label": " Weekly", "value": "W"},
                                                                {"label": " Monthly", "value": "M"}
                                                            ],
                                                            value="D",
                                                            labelStyle={
                                                                "display": "inline-block",
                                                                "padding": "10px 20px",
                                                                "marginRight": "10px",
                                                                "border": "2px solid #e2e8f0",
                                                                "borderRadius": "12px",
                                                                "cursor": "pointer",
                                                                "fontSize": "13px",
                                                                "fontWeight": "500",
                                                                "transition": "all 0.2s"
                                                            },
                                                            inputStyle={"marginRight": "8px"}
                                                        )
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "alignItems": "center",
                                                        "marginBottom": "20px"
                                                    }
                                                ),
                                                dcc.Graph(id="company-prompts-graph", config={"displayModeBar": False})
                                            ],
                                            className="chart-container"
                                        )
                                    ],
                                    style={"padding": "32px", "maxWidth": "1400px", "margin": "0 auto"}
                                )
                            ]
                        ),
                        
                        # User Report Tab
                        dcc.Tab(
                            label="👤 User Report",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            children=[
                                html.Div(
                                    [
                                        # Date Range
                                        html.Div(
                                            [
                                                html.Label(
                                                    "📅 Date Range",
                                                    style={
                                                        "fontWeight": "600",
                                                        "color": "#334155",
                                                        "fontSize": "14px",
                                                        "marginRight": "12px"
                                                    }
                                                ),
                                                dcc.DatePickerRange(
                                                    id="user-date-range",
                                                    start_date=MIN_DATE,
                                                    end_date=MAX_DATE,
                                                    min_date_allowed=MIN_DATE,
                                                    max_date_allowed=MAX_DATE,
                                                    display_format="MMM DD, YYYY"
                                                )
                                            ],
                                            style={
                                                "display": "flex",
                                                "alignItems": "center",
                                                "justifyContent": "center",
                                                "padding": "20px",
                                                "background": "white",
                                                "borderRadius": "16px",
                                                "border": "1px solid #e2e8f0",
                                                "marginBottom": "28px",
                                                "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.05)"
                                            }
                                        ),
                                        
                                        # SHAP bars
                                        html.Div(
                                            [
                                                html.Div(
                                                    dcc.Graph(id="user-negative-bar", config={"displayModeBar": False}),
                                                    className="chart-container"
                                                ),
                                                html.Div(
                                                    dcc.Graph(id="user-positive-bar", config={"displayModeBar": False}),
                                                    className="chart-container"
                                                )
                                            ],
                                            style={
                                                "display": "grid",
                                                "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
                                                "gap": "24px",
                                                "marginBottom": "28px"
                                            }
                                        ),
                                        
                                        # Category donut and time series
                                        html.Div(
                                            [
                                                html.Div(
                                                    dcc.Graph(id="user-category-donut", config={"displayModeBar": False}),
                                                    className="chart-container"
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Aggregation",
                                                                    style={
                                                                        "fontWeight": "600",
                                                                        "color": "#334155",
                                                                        "fontSize": "14px",
                                                                        "marginRight": "16px"
                                                                    }
                                                                ),
                                                                dcc.RadioItems(
                                                                    id="user-prompt-agg",
                                                                    options=[
                                                                        {"label": " Daily", "value": "D"},
                                                                        {"label": " Weekly", "value": "W"},
                                                                        {"label": " Monthly", "value": "M"}
                                                                    ],
                                                                    value="D",
                                                                    labelStyle={
                                                                        "display": "inline-block",
                                                                        "padding": "8px 16px",
                                                                        "marginRight": "8px",
                                                                        "border": "2px solid #e2e8f0",
                                                                        "borderRadius": "12px",
                                                                        "cursor": "pointer",
                                                                        "fontSize": "13px",
                                                                        "fontWeight": "500"
                                                                    },
                                                                    inputStyle={"marginRight": "8px"}
                                                                )
                                                            ],
                                                            style={
                                                                "display": "flex",
                                                                "alignItems": "center",
                                                                "marginBottom": "16px"
                                                            }
                                                        ),
                                                        dcc.Graph(id="user-prompts-graph", config={"displayModeBar": False})
                                                    ],
                                                    className="chart-container"
                                                )
                                            ],
                                            style={
                                                "display": "grid",
                                                "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
                                                "gap": "24px",
                                                "marginBottom": "28px"
                                            }
                                        ),
                                        
                                        # Table
                                        html.Div(
                                            [
                                                html.H4(
                                                    "📋 Prompt Log",
                                                    style={
                                                        "margin": "0 0 16px 0",
                                                        "fontSize": "18px",
                                                        "fontWeight": "600",
                                                        "color": "#1e293b"
                                                    }
                                                ),
                                                dash_table.DataTable(
                                                    id="user-prompts-table",
                                                    columns=[
                                                        {"name": "Date", "id": "Date"},
                                                        {"name": "Model", "id": "Model"},
                                                        {"name": "Device", "id": "Device"},
                                                        {"name": "Prompt", "id": "Prompt Text"},
                                                        {"name": "Response", "id": "Response Text"},
                                                        {"name": "Energy (kWh)", "id": "Energy (kWh)"}
                                                    ],
                                                    data=[],
                                                    style_as_list_view=True,
                                                    style_cell={
                                                        "padding": "12px",
                                                        "fontSize": "13px",
                                                        "fontFamily": "Inter, sans-serif",
                                                        "textAlign": "left"
                                                    },
                                                    style_header={
                                                        "fontWeight": "600",
                                                        "backgroundColor": "#f8fafc",
                                                        "color": "#334155",
                                                        "borderBottom": "2px solid #e2e8f0"
                                                    },
                                                    style_data={
                                                        "backgroundColor": "white",
                                                        "color": "#475569"
                                                    },
                                                    style_data_conditional=[
                                                        {
                                                            "if": {"row_index": "odd"},
                                                            "backgroundColor": "#fafafa"
                                                        }
                                                    ],
                                                    page_size=10
                                                )
                                            ],
                                            className="chart-container"
                                        )
                                    ],
                                    style={"padding": "32px", "maxWidth": "1400px", "margin": "0 auto"}
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)


# Callbacks
@app.callback(
    [
        Output("company-kpis", "children"),
        Output("dept-energy-graph", "figure"),
        Output("model-energy-graph", "figure"),
        Output("company-prompts-graph", "figure"),
    ],
    [
        Input("company-date-range", "start_date"),
        Input("company-date-range", "end_date"),
        Input("company-prompt-agg", "value")
    ]
)
def update_company_dashboard(start_date, end_date, freq):
    filtered = filter_by_date(prompt_logs_df, start_date, end_date)
    
    kpi_children = build_kpi_cards(filtered)
    dept_fig = build_department_donut(filtered) if not filtered.empty else go.Figure()
    model_fig = build_model_energy_fig(filtered) if not filtered.empty else go.Figure()
    prompts_fig = build_prompts_over_time_fig(filtered, freq, "Company Prompts")
    
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
        Input("user-prompt-agg", "value")
    ]
)
def update_user_dashboard(start_date, end_date, freq):
    filtered = filter_by_date(prompt_logs_df, start_date, end_date)
    user_df = filtered[filtered["User"] == USER_NAME]
    
    if user_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available for selected date range")
        return empty_fig, empty_fig, empty_fig, empty_fig, []
    
    donut_fig = build_user_category_donut(user_df)
    neg_fig, pos_fig = build_user_shap_bars(user_df)
    prompts_fig = build_prompts_over_time_fig(user_df, freq, "Your Prompts")
    table_data = build_user_table_data(user_df)
    
    return donut_fig, neg_fig, pos_fig, prompts_fig, table_data


if __name__ == "__main__":
    app.run_server(debug=True)