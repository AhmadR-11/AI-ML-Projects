from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from analysis import load_and_clean_data, get_encoded_data, train_all_models

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title='Car Price Prediction Dashboard',
    suppress_callback_exceptions=True,
)

# Load data at startup
_df = load_and_clean_data()
df_encoded = get_encoded_data(_df)
models, X_test, y_test, results_df, feature_names = train_all_models(df_encoded)

best_model_name = results_df.loc[0, 'Model']
best_model_r2 = results_df.loc[0, 'R2_Score']

brand_options = [
    {'label': brand, 'value': brand}
    for brand in sorted(_df['Brand'].dropna().unique())
]
fuel_options = [
    {'label': fuel, 'value': fuel}
    for fuel in sorted(_df['Fuel_Type'].dropna().unique())
]
seller_options = [
    {'label': seller, 'value': seller}
    for seller in sorted(_df['Seller_Type'].dropna().unique())
] if 'Seller_Type' in _df.columns else [
    {'label': 'Individual', 'value': 'Individual'},
    {'label': 'Dealer', 'value': 'Dealer'},
    {'label': 'Trustmark Dealer', 'value': 'Trustmark Dealer'},
]
transmission_options = [
    {'label': transmission, 'value': transmission}
    for transmission in ['Manual', 'Automatic']
]

numeric_defaults = {
    col: _df[col].median() for col in _df.select_dtypes(include=[np.number]).columns
    if col != 'Selling_Price'
}
max_price = float(_df['Selling_Price'].max())

summary_cards = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6('Total Cars', className='card-title text-muted'),
                        html.H3(f"{len(_df):,}"),
                    ]
                ),
                className='mb-3 shadow',
            ),
            md=3,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6('Average Selling Price', className='card-title text-muted'),
                        html.H3(f"${_df['Selling_Price'].mean():,.2f}"),
                    ]
                ),
                className='mb-3 shadow',
            ),
            md=3,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6('Best Model', className='card-title text-muted'),
                        html.H3(best_model_name),
                    ]
                ),
                className='mb-3 shadow',
            ),
            md=3,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6('Best Model R²', className='card-title text-muted'),
                        html.H3(f"{best_model_r2:.4f}"),
                    ]
                ),
                className='mb-3 shadow',
            ),
            md=3,
        ),
    ],
    className='mb-4',
)


def render_eda_tab(df):
    """Render the EDA tab with Plotly charts."""
    price_dist_fig = px.histogram(
        df,
        x='Selling_Price',
        nbins=40,
        color_discrete_sequence=['#00b4d8'],
        marginal='box',
        title='Distribution of Selling Price',
        template='plotly_dark',
    )
    price_dist_fig.update_layout(
        xaxis_title='Selling Price (Lakhs)',
        yaxis_title='Count',
    )

    price_fuel_fig = px.box(
        df,
        x='Fuel_Type',
        y='Selling_Price',
        color='Fuel_Type',
        color_discrete_sequence=px.colors.qualitative.Set2,
        title='Selling Price by Fuel Type',
        template='plotly_dark',
    )
    price_fuel_fig.update_layout(showlegend=False)

    if 'Car_Age' in df.columns:
        price_age_fig = px.scatter(
            df,
            x='Car_Age',
            y='Selling_Price',
            color='Fuel_Type',
            size='Selling_Price',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            trendline='ols',
            title='Selling Price vs Car Age',
            template='plotly_dark',
        )
    else:
        price_age_fig = None

    brand_df = (
        df.groupby('Brand', as_index=False)['Selling_Price']
        .mean()
        .sort_values('Selling_Price', ascending=False)
        .head(10)
        .rename(columns={'Selling_Price': 'Avg_Price'})
    )
    brand_fig = px.bar(
        brand_df,
        x='Avg_Price',
        y='Brand',
        orientation='h',
        color='Avg_Price',
        color_continuous_scale='Viridis',
        title='Top 10 Brands by Average Selling Price',
        template='plotly_dark',
    )
    brand_fig.update_layout(yaxis={'categoryorder': 'total ascending'})

    if 'Transmission' in df.columns:
        price_trans_fig = px.violin(
            df,
            x='Transmission',
            y='Selling_Price',
            color='Transmission',
            box=True,
            points='all',
            title='Selling Price Distribution by Transmission',
            template='plotly_dark',
        )
    else:
        price_trans_fig = None

    return dbc.Container(
        [
            html.H4('Price Distribution', className='text-light mt-3 mb-2'),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(figure=price_dist_fig)),
                            className='mb-3 shadow',
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(figure=price_fuel_fig)),
                            className='mb-3 shadow',
                        ),
                        width=6,
                    ),
                ]
            ),
            html.H4('Price Relationships', className='text-light mt-3 mb-2'),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Graph(figure=price_age_fig) if price_age_fig is not None else html.Div(
                                    html.P('Car_Age is not available in the dataset.', className='text-light'),
                                    className='text-center',
                                )
                            ),
                            className='mb-3 shadow',
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(figure=brand_fig)),
                            className='mb-3 shadow',
                        ),
                        width=6,
                    ),
                ]
            ),
            html.H4('Transmission Analysis', className='text-light mt-3 mb-2'),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(figure=price_trans_fig) if price_trans_fig is not None else html.Div(
                                html.P('Transmission is not available in the dataset.', className='text-light'),
                                className='text-center',
                            )
                        ),
                        className='mb-3 shadow',
                    ),
                    width=12,
                )
            ),
        ],
        fluid=True,
    )


def render_correlation_tab(df):
    """Render the correlations tab with heatmap and relationship charts."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    heatmap_fig = go.Figure(
        data=[
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(title='Correlation'),
                text=corr_matrix.round(2).astype(str),
                hovertemplate='Feature: %{x}<br>Feature: %{y}<br>Corr: %{z:.2f}<extra></extra>',
            )
        ]
    )
    heatmap_fig.update_layout(
        title='Feature Correlation Heatmap',
        template='plotly_dark',
        height=600,
    )

    kms_columns = [col for col in ['Driven_kms', 'kms_driven'] if col in df.columns]
    if kms_columns:
        kms_col = kms_columns[0]
        km_scatter_fig = px.scatter(
            df,
            x=kms_col,
            y='Selling_Price',
            color='Fuel_Type' if 'Fuel_Type' in df.columns else None,
            trendline='ols',
            title='Selling Price vs KMs Driven',
            template='plotly_dark',
        )
    else:
        km_scatter_fig = None

    if 'Car_Age' in df.columns:
        year_series = datetime.now().year - df['Car_Age']
        year_df = (
            pd.DataFrame({'Year': year_series, 'Selling_Price': df['Selling_Price']})
            .groupby('Year', as_index=False)
            .mean()
            .sort_values('Year')
        )
        year_fig = px.line(
            year_df,
            x='Year',
            y='Selling_Price',
            markers=True,
            color_discrete_sequence=['#f72585'],
            title='Average Selling Price by Year',
            template='plotly_dark',
        )
    else:
        year_fig = None

    return dbc.Container(
        [
            dbc.Alert(
                "💡 Insight: Features with high absolute correlation to Selling_Price are the most useful predictors.",
                color='info',
                className='mb-3 shadow',
            ),
            dbc.Card(
                dbc.CardBody(dcc.Graph(figure=heatmap_fig)),
                className='mb-3 shadow',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Graph(figure=km_scatter_fig) if km_scatter_fig is not None else html.Div(
                                    html.P('No KMs driven column found in the dataset.', className='text-light'),
                                    className='text-center',
                                )
                            ),
                            className='mb-3 shadow',
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Graph(figure=year_fig) if year_fig is not None else html.Div(
                                    html.P('Car_Age is not available, so Year cannot be derived.', className='text-light'),
                                    className='text-center',
                                )
                            ),
                            className='mb-3 shadow',
                        ),
                        width=6,
                    ),
                ]
            ),
        ],
        fluid=True,
    )


def render_model_tab(models, X_test, y_test, results_df):
    """Render the model performance tab with metrics and visualizations."""
    metrics_table = dbc.Table.from_dataframe(
        results_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className='mb-0',
    )

    mae_fig = px.bar(
        results_df,
        x='Model',
        y='MAE',
        color='Model',
        color_discrete_sequence=px.colors.qualitative.Set1,
        title='MAE by Model (Lower is Better)',
        template='plotly_dark',
    )
    lowest_mae = results_df.loc[results_df['MAE'].idxmin()]
    mae_fig.add_annotation(
        x=lowest_mae['Model'],
        y=lowest_mae['MAE'],
        text='Lowest MAE',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color='white'),
    )

    rmse_fig = px.bar(
        results_df,
        x='Model',
        y='RMSE',
        color='Model',
        color_discrete_sequence=px.colors.qualitative.Set1,
        title='RMSE by Model (Lower is Better)',
        template='plotly_dark',
    )
    lowest_rmse = results_df.loc[results_df['RMSE'].idxmin()]
    rmse_fig.add_annotation(
        x=lowest_rmse['Model'],
        y=lowest_rmse['RMSE'],
        text='Lowest RMSE',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color='white'),
    )

    r2_fig = px.bar(
        results_df,
        x='Model',
        y='R2_Score',
        color='Model',
        color_discrete_sequence=px.colors.qualitative.Set2,
        title='R² Score by Model (Higher is Better)',
        template='plotly_dark',
    )
    highest_r2 = results_df.loc[results_df['R2_Score'].idxmax()]
    r2_fig.add_annotation(
        x=highest_r2['Model'],
        y=highest_r2['R2_Score'],
        text='Highest R²',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color='white'),
    )

    actual_pred_cards = []
    for name, model in models.items():
        predictions = model.predict(X_test)
        scatter_fig = px.scatter(
            x=y_test,
            y=predictions,
            opacity=0.6,
            title=f'Actual vs Predicted — {name}',
            template='plotly_dark',
        )
        perfect_line = go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False,
        )
        scatter_fig.add_trace(perfect_line)
        scatter_fig.update_layout(
            xaxis_title='Actual',
            yaxis_title='Predicted',
        )
        actual_pred_cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(dcc.Graph(figure=scatter_fig)),
                    className='mb-3 shadow',
                ),
                width=4,
            )
        )

    return dbc.Container(
        [
            dbc.Alert(
                f"🏆 Best Model: {results_df.loc[0, 'Model']} with R² = {results_df.loc[0, 'R2_Score']:.4f}",
                color='success',
                className='mb-3 shadow',
            ),
            dbc.Card(
                [
                    dbc.CardHeader(html.H5('📋 Model Metrics Summary')),
                    dbc.CardBody(metrics_table),
                ],
                className='mb-3 shadow',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(dbc.CardBody(dcc.Graph(figure=mae_fig)), className='mb-3 shadow'),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Card(dbc.CardBody(dcc.Graph(figure=rmse_fig)), className='mb-3 shadow'),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Card(dbc.CardBody(dcc.Graph(figure=r2_fig)), className='mb-3 shadow'),
                        width=4,
                    ),
                ]
            ),
            dbc.Row(actual_pred_cards),
        ],
        fluid=True,
    )


def render_feature_tab(models, feature_names):
    """Render the feature importance comparison for tree-based models."""
    rf_model = models.get('Random Forest')
    gb_model = models.get('Gradient Boosting')

    rf_importance_df = pd.DataFrame(
        {
            'Feature': feature_names,
            'RF_Importance': rf_model.feature_importances_ if rf_model is not None else np.zeros(len(feature_names)),
        }
    )
    rf_importance_df = rf_importance_df.sort_values('RF_Importance', ascending=False).head(15)

    gb_importance_df = pd.DataFrame(
        {
            'Feature': feature_names,
            'GB_Importance': gb_model.feature_importances_ if gb_model is not None else np.zeros(len(feature_names)),
        }
    )
    gb_importance_df = gb_importance_df.sort_values('GB_Importance', ascending=False).head(15)

    rf_fig = px.bar(
        rf_importance_df.sort_values('RF_Importance', ascending=True),
        x='RF_Importance',
        y='Feature',
        orientation='h',
        color='RF_Importance',
        color_continuous_scale='YlOrRd',
        title='Top 15 Features — Random Forest',
        template='plotly_dark',
        height=500,
    )

    gb_fig = px.bar(
        gb_importance_df.sort_values('GB_Importance', ascending=True),
        x='GB_Importance',
        y='Feature',
        orientation='h',
        color='GB_Importance',
        color_continuous_scale='YlGnBu',
        title='Top 15 Features — Gradient Boosting',
        template='plotly_dark',
        height=500,
    )

    merged_df = pd.merge(
        rf_importance_df[['Feature', 'RF_Importance']],
        gb_importance_df[['Feature', 'GB_Importance']],
        on='Feature',
        how='inner',
    )
    merged_df = merged_df.sort_values(['RF_Importance', 'GB_Importance'], ascending=False).head(10)

    comparison_fig = px.bar(
        merged_df,
        x='Feature',
        y=['RF_Importance', 'GB_Importance'],
        barmode='group',
        color_discrete_map={'RF_Importance': '#f72585', 'GB_Importance': '#4cc9f0'},
        title='Feature Importance: Random Forest vs Gradient Boosting',
        template='plotly_dark',
    )

    return dbc.Container(
        [
            dbc.Alert(
                "⚠️ Feature importance is only available for \n   tree-based models: Random Forest and Gradient Boosting",
                color='warning',
                className='mb-3 shadow',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(dbc.CardBody(dcc.Graph(figure=rf_fig)), className='mb-3 shadow'),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(dbc.CardBody(dcc.Graph(figure=gb_fig)), className='mb-3 shadow'),
                        width=6,
                    ),
                ]
            ),
            dbc.Card(
                dbc.CardBody(dcc.Graph(figure=comparison_fig)),
                className='mb-3 shadow',
            ),
        ],
        fluid=True,
    )


def render_predictor_tab(df, models, feature_names):
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4('🔧 Enter Car Details', className='text-white'),
                                    html.Hr(className='border-light'),
                                    dbc.Label('Brand', html_for='input-brand', className='text-light'),
                                    dcc.Dropdown(
                                        id='input-brand',
                                        options=brand_options,
                                        placeholder='Select Brand',
                                        clearable=True,
                                        className='mb-3',
                                        style={'color': 'black'},
                                    ),
                                    dbc.Label('Fuel Type', html_for='input-fuel', className='text-light'),
                                    dcc.Dropdown(
                                        id='input-fuel',
                                        options=fuel_options,
                                        placeholder='Select Fuel Type',
                                        clearable=True,
                                        className='mb-3',
                                        style={'color': 'black'},
                                    ),
                                    dbc.Label('Transmission', html_for='input-transmission', className='text-light'),
                                    dcc.Dropdown(
                                        id='input-transmission',
                                        options=transmission_options,
                                        placeholder='Select Transmission',
                                        clearable=True,
                                        className='mb-3',
                                        style={'color': 'black'},
                                    ),
                                    dbc.Label('Seller Type', html_for='input-seller', className='text-light'),
                                    dcc.Dropdown(
                                        id='input-seller',
                                        options=seller_options,
                                        placeholder='Select Seller Type',
                                        clearable=True,
                                        className='mb-3',
                                        style={'color': 'black'},
                                    ),
                                    dbc.Label('Car Age', html_for='input-age', className='text-light'),
                                    dcc.Slider(
                                        id='input-age',
                                        min=0,
                                        max=20,
                                        step=1,
                                        value=5,
                                        marks={i: str(i) for i in range(0, 21, 5)},
                                    ),
                                    html.Div('Selected Age: 5 years', id='age-value', className='text-light mb-3 mt-2'),
                                    dbc.Label('KMs Driven', html_for='input-kms', className='text-light'),
                                    dcc.Slider(
                                        id='input-kms',
                                        min=0,
                                        max=300000,
                                        step=5000,
                                        value=50000,
                                        marks={0: '0', 100000: '1L', 200000: '2L', 300000: '3L'},
                                    ),
                                    html.Div('Selected KMs: 50,000', id='kms-value', className='text-light mb-3 mt-2'),
                                    dbc.Button('🔮 Predict Price', id='predict-btn', color='primary', className='w-100 mt-3'),
                                ]
                            ),
                            className='mb-3 shadow',
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4('📊 Prediction Results', className='text-white'),
                                    html.Hr(className='border-light'),
                                    html.Div(id='prediction-output'),
                                ]
                            ),
                            className='mb-3 shadow',
                        ),
                        width=8,
                    ),
                ],
                className='g-4',
            ),
        ],
        fluid=True,
    )

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span('🚗 Car Price Prediction Dashboard', className='navbar-brand mb-0 h1'),
                ]
            ),
            html.Div('ML Analysis & Insights', className='text-muted'),
        ]
    ),
    color='dark',
    dark=True,
    className='mb-4 shadow',
)

app.layout = dbc.Container(
    [
        navbar,
        summary_cards,
        dbc.Tabs(
            [
                dbc.Tab(label='📊 EDA', tab_id='tab-eda'),
                dbc.Tab(label='🔥 Correlations', tab_id='tab-correlation'),
                dbc.Tab(label='🤖 Model Performance', tab_id='tab-model'),
                dbc.Tab(label='🎯 Feature Importance', tab_id='tab-features'),
                dbc.Tab(label='🔮 Price Predictor', tab_id='tab-predictor'),
            ],
            id='main-tabs',
            active_tab='tab-eda',
            className='mb-4 shadow',
        ),
        dcc.Loading(
            id='loading',
            type='cube',
            color='#00b4d8',
            children=html.Div(id='tab-content'),
        ),
        html.Div(id='scroll-dummy', style={'display': 'none'}),
    ],
    fluid=True,
    className='px-4 py-3',
)

@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'active_tab'))
def render_tab(active_tab):
    if active_tab == 'tab-eda':
        return render_eda_tab(_df)
    elif active_tab == 'tab-correlation':
        return render_correlation_tab(_df)
    elif active_tab == 'tab-model':
        return render_model_tab(models, X_test, y_test, results_df)
    elif active_tab == 'tab-features':
        return render_feature_tab(models, feature_names)
    elif active_tab == 'tab-predictor':
        return render_predictor_tab(_df, models, feature_names)
    return html.Div()


app.clientside_callback(
    "function(tab) { window.scrollTo(0, 0); return ''; }",
    Output('scroll-dummy', 'children'),
    Input('main-tabs', 'active_tab'),
)


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('input-brand', 'value'),
    State('input-fuel', 'value'),
    State('input-transmission', 'value'),
    State('input-seller', 'value'),
    State('input-age', 'value'),
    State('input-kms', 'value'),
    prevent_initial_call=True,
)
def predict_price(n_clicks, brand, fuel, transmission, seller, age, kms):
    if not all([brand, fuel, transmission, seller, age is not None, kms is not None]):
        return dbc.Alert(
            '⚠️ Please fill all fields before predicting',
            color='danger',
            className='shadow',
        )

    input_data = {
        'Fuel_Type': fuel,
        'Brand': brand,
        'Transmission': transmission,
        'Seller_Type': seller,
        'Car_Age': age,
        'Driven_kms': kms,
    }

    df_input = pd.DataFrame([input_data])

    categorical_cols = [
        'Fuel_Type',
        'Transmission',
        'Seller_Type',
        'Brand',
    ]
    df_encoded_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True, dtype=int)
    df_encoded_input = df_encoded_input.reindex(columns=feature_names, fill_value=0)

    predictions = {}
    for name, model in models.items():
        try:
            predicted = model.predict(df_encoded_input)[0]
        except Exception:
            predicted = None
        predictions[name] = predicted

    alert_boxes = []
    color_map = {
        'Linear Regression': 'info',
        'Random Forest': 'success',
        'Gradient Boosting': 'warning',
    }
    for name, pred in predictions.items():
        if pred is None:
            alert_boxes.append(
                dbc.Alert(
                    f'{name}: Prediction failed for the given input.',
                    color='danger',
                    className='shadow mb-2',
                )
            )
        else:
            alert_boxes.append(
                dbc.Alert(
                    f'{name} Predicted Price: ₹ {pred:.2f} Lakhs',
                    color=color_map.get(name, 'secondary'),
                    className='shadow mb-2',
                )
            )

    best_name = max(predictions, key=lambda k: predictions[k] if predictions[k] is not None else -np.inf)
    best_prediction = predictions[best_name]
    gauge_fig = go.Figure(
        go.Indicator(
            mode='gauge+number',
            value=best_prediction,
            title={'text': f'Best Prediction — {best_name}'},
            gauge={
                'axis': {'range': [0, max_price]},
                'bar': {'color': '#4cc9f0'},
                'steps': [
                    {'range': [0, max_price * 0.5], 'color': '#1b4332'},
                    {'range': [max_price * 0.5, max_price * 0.8], 'color': '#2a6f97'},
                    {'range': [max_price * 0.8, max_price], 'color': '#7209b7'},
                ],
            },
        )
    )
    gauge_fig.update_layout(template='plotly_dark', height=400)

    alert_boxes.append(
        dbc.Card(
            dbc.CardBody(dcc.Graph(figure=gauge_fig)),
            className='shadow',
        )
    )

    return html.Div(alert_boxes)


if __name__ == '__main__':
    app.run(debug=True)
