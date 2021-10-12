import numpy as np
import pandas as pd

import re

import plotly.express as px
import plotly.graph_objs as go

# import dash.dcc as dcc
# import dash.html as html
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from utils import *
# imported functions:
#   importusgssite
#   fill_small_gaps, shorten_time_series, convert_time_series
#   splityear, calculate_hydrometric
#   run_mktest, simple_linear_regression, sens_slope_lub


gauges = pd.read_excel('data/Station_Info_HCDN.xlsx', index_col=0)
sid_options = [{'label': sid, 'value': sid} for sid in gauges.index]

sel_hym = [
    'mean', 'median', 'std', 'skew', 'range',
    '5p', '10p', '25p', '75p', '90p', '95p',
    'max', 'min', 'min7d', 'max7d',
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'jan_p', 'feb_p', 'mar_p', 'apr_p', 'may_p', 'jun_p',
    'jul_p', 'aug_p', 'sep_p', 'oct_p', 'nov_p', 'dec_p',
    'cen_jd', '25p_jd', '50p_jd', '75p_jd',
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'si']
hym_tab = pd.read_excel('data/Hydrometric_Name_Table.xlsx', index_col=0)
hym_tab = hym_tab.loc[sel_hym]
hym_options = []
for sub, name in hym_tab['Name'].iteritems():
    hym_options.append({'label': name, 'value': sub})

data_summary_frame = """
    |Data Summary||
    | ----------- | ----------- |
    |Start Date: {}|End Date: {}|
    |\\# Records: {:d}|\\# NaNs: {:d}|
    |Mean: {:.2f}| Median: {:.2f}|
    |Max: {:.2f}|90th Percentile: {:.2f}|
    |Min: {:.2f}|10th Percentile: {:.2f}|
"""

trend_summary_frame = """
    |Trend Analysis|
    | ----------- | ----------- |
    |\\# Samples: {:d}|
    |MK-Test P-Value: {:.3f}|
    |Sen's Slope: {:.5f}|
    |Sen's Slope Upper Bound: {:.5f}|
    |Sen's Slope Lower Bound: {:.5f}|
    |OLS Slope: {:.5f}|
    |R^2: {:.3f}|
"""


# App Layout -----------------------------------------------------------------
# ----------------------------------------------------------------------------
def create_layout(app):
    return html.Div(
        className='row',
        children=[
            html.Div(
                className='row header',
                id='app-header',
                style={'background-color': '#f9f9f9'},
                children=[
                    html.Div(
                        [
                            html.Img(
                                src=app.get_asset_url('dash-logo.png'),
                                className='logo',
                                id='plotly-image',
                                style={'height': '150px'},
                            )
                        ],
                        className='four columns header_img',
                    ),
                    html.Div(
                        [
                            html.H2(
                                'Annual-Based Hydrometrics Trend Analysis Toolkit',
                                className='header_title',
                                id='app-title',
                            )
                        ],
                        className='eight columns header_title_container',
                    ),
                ],
            ),
            html.Div(
               className='row',
               children=[
                    html.Div(
                        className='three columns',
                        children=[
                            html.H6(
                                children='Select USGS Station',
                                style={'width': '95%'},
                            ),
                            dcc.Dropdown(
                                id='select-sid',
                                style={'width': '100%'},
                                placeholder='Select SID',
                                options=sid_options,
                                value='WY-09210500',
                            ),
                            html.Div(id='div-graph-map'),
                        ],
                    ),
                    html.Div(
                        className='three columns',
                        children=[
                            dcc.Markdown(
                                id='data-summary',
                                style={
                                    'width': '100%',
                                    'background-color': 'white',
                                    'margin-top': '20px'},
                                children=[
                                    re.sub('{(.*)}', '', data_summary_frame)
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        className='six columns',
                        children=[
                            dcc.Graph(id='gts-plot'),
                        ]
                    ),
                ],
            ),
            html.Div(
                className='row',
                children=[
                    dcc.Graph(id='flow-plot'),
                    dcc.Store(id='flow-data', data=None, storage_type='session'),
                    dcc.Store(id='hys-data', data=None, storage_type='session'),
                ],
            ),
            html.Div(
                className='row',
                style={'margin-top': '60px'},
                children=[
                    html.Div(
                        className='three columns',
                        children=[
                            html.H6(
                                children='Select Hydrometric',
                                style={'width': '95%'},
                            ),
                            dcc.Dropdown(
                                id='select-hydrometric',
                                placeholder='Select Hydrometric',
                                options=hym_options,
                            ),
                            html.H6(
                                children='Select Mann-Kendall Test',
                                style={'width': '95%'},
                            ),
                            dcc.Dropdown(
                                id='select-mktest',
                                placeholder='Select Mann-Kendall Test',
                                options=[
                                    {
                                        'label': 'Original Mann-Kendall Test',
                                        'value': 'original'},
                                    {
                                        'label': 'Hamed and Rao Modified MK Test',
                                        'value': 'rao'},
                                    {
                                        'label': 'Yue and Wang Modified MK Test',
                                        'value': 'yue'},
                                    {
                                        'label': 'Modified MK Test using Pre-Whitening Method',
                                        'value': 'prewhiten'},
                                    {
                                        'label': 'Modified MK Test using Trend Free Pre-Whitening Method',
                                        'value': 'trendfree'},
                                ],
                            ),
                            html.Button(
                                id='run-button',
                                style={
                                    'width': '100%',
                                    'margin-top': '40px',
                                    'font-size': 14
                                },
                                n_clicks=0,
                                children='RUN',
                            ),
                        ]
                    ),
                    html.Div(
                        className='three columns',
                        children=[
                            dcc.Markdown(
                                id='trend-summary',
                                style={'background-color': 'white'},
                                children=[
                                    re.sub('{(.*)}', '', trend_summary_frame)
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className='six columns',
                        children=[
                            dcc.Graph(id='trendplot'),
                            dcc.Store(id='trend-data', data=None, storage_type='session'),
                            dcc.Checklist(
                                id='trend-line-type',
                                options=[
                                    {'label': 'Sens Slope', 'value': 'sens'},
                                    {'label': 'Linear Regression', 'value': 'linear'},
                                    {'label': 'Upper Bound', 'value': 'upper'},
                                    {'label': 'Lower Bound', 'value': 'lower'},
                                ],
                                value=['sens'],
                                labelStyle={
                                    'display': 'block',
                                    'float': 'left',
                                    'margin-right': '24px',
                                    'margin-bottom': '10px'},
                                # inputStyle={'margin-right': '30px'},
                                style={
                                    'display': 'inline-block',
                                    'margin-left': '30px',
                                    'margin-top': '10px',
                                    'font-size': 18,
                                },
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def demo_callbacks(app):

    @app.callback(
        [Output('flow-data', 'data'), Output('hys-data', 'data')],
        Input('select-sid', 'value'),
    )
    def download_flow_data(sid):
        try:
            rflow = importusgssite(sid[3:])
        except Exception as e:
            print(e)
            print('Can not download data at {}!'.format(sid))

        ts = pd.Series(rflow['Q'].values, index=rflow.index)
        thr = np.timedelta64(1, 'D')
        tf = pd.date_range(min(ts.index), max(ts.index), freq='D')
        flow = convert_time_series(tf, ts, thr)
        flow[flow < -9999] = np.nan

        df_flow = flow.to_frame()
        df_flow.index = df_flow.index.strftime('%Y-%m-%d')
        js_flow = df_flow.to_json(orient='columns')

        df_hys = splityear(flow)
        df_hys = df_hys[np.isnan(df_hys).sum(axis=1) == 0]
        js_hys = df_hys.to_json(orient='columns')

        return js_flow, js_hys

    @app.callback(
        Output('div-graph-map', 'children'),
        Input('select-sid', 'value'),
    )
    def show_map(sid):
        df_map = gauges.loc[[sid]]
        fig = px.scatter_mapbox(
            df_map, lat='LATITUDE', lon='LONGITUDE',
            hover_name='STATION NAME', hover_data=['DRAINAGE AREA'],
            color_discrete_sequence=['red'], zoom=3)
        fig.update_traces(marker_size=8)
        fig.update_layout(
            mapbox_style='open-street-map',
            margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
        na_map = dcc.Graph(
            id='graph-map',
            figure=fig,
            style={'height': '30vh', 'margin-top': '10px'},
            config={'displayModeBar': False},
        )
        return na_map

    @app.callback(
        Output('flow-plot', 'figure'),
        Input('flow-data', 'data'),
    )
    def show_hydrograph(js_flow):

        flow = pd.read_json(js_flow, orient='columns')[0]
        flow.index = pd.DatetimeIndex(flow.index)

        t = flow.index
        y = flow.values

        data = [
            go.Scatter(x=t, y=y, mode='lines', showlegend=False)
        ]

        layout = {
            'title': 'Long-Term Hydrograph',
            'font': dict(size=16),
            'hovermode': 'closest',
            'showlegend': True,
            'autosize': False,
            'xaxis': dict(
                title='Time',
                zeroline=False,
                domain=[0., .98],
                showgrid=False,
                automargin=True
            ),
            'yaxis': dict(
                title='Flow',
                zeroline=True,
                domain=[0., .98],
                showgrid=False,
                automargin=True
            ),
            'paper_bgcolor': '#F2F2F2',
            # 'width': 800,
            'height': 450,
            'margin': dict(l=2, r=2, t=50, b=2),
        }

        figure = go.Figure(data=data, layout=layout)
        return figure

    @app.callback(
        Output('gts-plot', 'figure'),
        Input('hys-data', 'data'),
    )
    def show_gts(js_hys):

        df_hys = pd.read_json(js_hys, orient='columns')
        nvy = df_hys.shape[0]

        x = np.arange(365)
        y2 = df_hys.median(axis=0).values

        data = []
        for year in df_hys.index:
            y = df_hys.loc[year].values
            data.append(
                go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line_width=1.,
                    marker_color='rgba(100,100,100,0.3)',
                    hovertemplate=str(year))
            )
        data.append(
            go.Scatter(
                x=x, y=y2,
                mode='lines',
                marker_color='rgba(0,0,256,.9)',
                line_width=2.,
                hovertemplate='Average')
        )

        layout = {
            'title': 'Annual Daily Hydrographs ({})'.format(nvy),
            'font': dict(size=16),
            'hovermode': 'closest',  # closest
            'plot_bgcolor': 'white',
            'showlegend': False,
            'autosize': False,
            'xaxis': dict(
                title='Day of Year',
                zeroline=False,
                domain=[0., .98],
                showgrid=False,
                automargin=True
            ),
            'yaxis': dict(
                title='Flow',
                zeroline=True,
                domain=[0., .98],
                showgrid=False,
                automargin=True
            ),
            'paper_bgcolor': '#F2F2F2',
            # 'width': 800,
            'height': 400,
            'margin': dict(l=2, r=2, t=50, b=2),
        }

        figure = go.Figure(data=data, layout=layout)
        return figure

    @app.callback(
        Output('data-summary', 'children'),
        [Input('flow-data', 'data'), Input('hys-data', 'data')],
    )
    def show_data_summary(js_flow, js_hys):

        flow = pd.read_json(js_flow, orient='columns')[0]
        flow.index = pd.DatetimeIndex(flow.index)

        sdate = flow.index[0].strftime('%Y-%m-%d')
        edate = flow.index[-1].strftime('%Y-%m-%d')
        n = len(flow.values)
        n_nan = np.sum(np.isnan(flow.values))
        flow_avg = np.nanmean(flow.values)
        flow_med = np.nanmedian(flow.values)
        flow_max = np.nanmax(flow.values)
        flow_min = np.nanmin(flow.values)
        flow_10p = np.nanpercentile(flow.values, 10)
        flow_90p = np.nanpercentile(flow.values, 90)

        data_summary = data_summary_frame.format(
            sdate, edate, n, n_nan,
            flow_avg, flow_med, flow_max, flow_90p, flow_min, flow_10p)
        return data_summary

    @app.callback(
        Output('trend-data', 'data'),
        [Input('select-hydrometric', 'value')],
        [State('hys-data', 'data')]
    )
    def load_hydrometric(name, js_hys):

        if name is None:
            raise PreventUpdate

        else:
            df_hys = pd.read_json(js_hys, orient='columns')

            hym = calculate_hydrometric(df_hys, name)
            df_hym = pd.DataFrame(hym, index=df_hys.index)
            js_hym = df_hym.to_json(orient='columns')
            return js_hym

    @app.callback(
        [
            Output('trendplot', 'figure'),
            Output('trend-summary', 'children')
        ],
        [
            Input('run-button', 'n_clicks'),
            Input('trend-line-type', 'value')
        ],
        [
            State('trend-data', 'data'),
            State('select-mktest', 'value'),
        ]
    )
    def perform_trend_analysis(n_clicks, trlines, js_hym, method):

        if (js_hym is None) | (method is None):
            raise PreventUpdate

        else:

            ts = pd.read_json(js_hym, orient='columns')[0]
            t = ts.index.values
            y = ts.values
            x = t - t[0]

            # MK-test and Sen's slope
            slp, intp, pvalue, pvalue_d, n = run_mktest(x, y, method=method)
            y2 = slp * x + intp

            # linear regression
            slp_lr, intp_lr, rsq = simple_linear_regression(x, y)
            y3 = slp_lr * x + intp_lr

            # upper and lower bound of Sen's slope
            _, slp_up, slp_lo = sens_slope_lub(y, alpha=0.05)
            y4 = slp_up * x + intp
            y5 = slp_lo * x + intp

            data = [
                go.Scatter(
                    x=t, y=y, name='Time Series', mode='markers', text=t,
                    showlegend=False, marker={'color': 'grey', 'size': 6},
                    hovertemplate='Year: %{text}' + '<br>Value: %{y:.3f}</br>'
                )
            ]
            if 'sens' in trlines:
                data.append(
                    go.Scatter(
                        x=t, y=y2, name='Sens Slope', mode='lines',
                        showlegend=True, line={'color': 'green', 'width': 2})
                )
            if 'linear' in trlines:
                data.append(
                    go.Scatter(
                        x=t, y=y3, name='Linear Regression', mode='lines',
                        showlegend=True, line={'color': 'blue', 'width': 2})
                )
            if 'upper' in trlines:
                data.append(
                    go.Scatter(
                        x=t, y=y4, name='Upper Bound', mode='lines',
                        showlegend=True,
                        line={'color': 'green', 'dash': 'dot', 'width': 2})
                )
            if 'lower' in trlines:
                data.append(
                    go.Scatter(
                        x=t, y=y5, name='Lower Bound', mode='lines',
                        showlegend=True,
                        line={'color': 'green', 'dash': 'dot', 'width': 2})
                )

            if pvalue > 0.1:
                bgcolor = 'rbga(0,0,0,0)'
            else:
                if slp > 0:
                    bgcolor = 'rgba(256,0,0,0.2)'
                else:
                    bgcolor = 'rgba(0,0,256,.2)'

            layout = {
                'title': 'Trend Line',
                'font': dict(size=14),
                'hovermode': 'closest',  # closest
                'plot_bgcolor': bgcolor,
                'showlegend': True,
                'autosize': False,
                'xaxis': dict(
                    title='Year',
                    zeroline=False,
                    domain=[0., .98],
                    showgrid=False,
                    automargin=True
                ),
                'yaxis': dict(
                    title='',
                    zeroline=True,
                    domain=[0., .98],
                    showgrid=False,
                    automargin=True
                ),
                'paper_bgcolor': '#F2F2F2',
                # 'width': 800,
                'height': 360,
                'margin': dict(l=2, r=2, t=50, b=2),
            }

            figure = {'data': data, 'layout': layout}

            trend_summary = trend_summary_frame.format(
                n, pvalue, slp, slp_up, slp_lo, slp_lr, rsq)

            return figure, trend_summary
