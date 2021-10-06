import numpy as np
import pandas as pd

import os, sys

import glob
from sklearn.linear_model import LinearRegression

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
import plotly.express as px

from utils import run_mktest, sens_slope_lub, simple_linear_regression

file_list = glob.glob('data/*')
file_list = sorted([item for item in file_list if item.split('.')[-1] in ['csv', 'xlsx', 'xls']])

# App Layout -----------------------------------------------------------------
# ----------------------------------------------------------------------------

# adjust position of elements
# sytle = {
# 	'width': '80%', 'height': '20%',
# 	'margin-left': '10px', 'margin-right': '10px', 
# 	'margin-top': '50px', 'margin-bottom': '50px',
#	'tex tAlign': 'center',
# }

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div([

	html.Div([

		html.Div([
			
			html.H6(
				children = 'Select File',
				style = {'width': '95%', 'display': 'inline-block'},
			),

			dcc.Dropdown(
				id = 'select-file', 
				options = [{'label': '{}'.format(item), 'value': item} for item in file_list],
				placeholder = 'select data',
			)
		], style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'}), 


		html.Div([
			
			html.H6(
				children = 'Select Variable',
				style = {'width': '95%'},
			),

			dcc.Dropdown(
				id = 'select-variable', 
				placeholder = 'select variables', 
			)
		], style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'}), 


		html.Div([
			
			html.H6(
				children = 'Select Mann-Kendall Test',
				style = {'width': '95%'},
			),

			dcc.Dropdown(
				id = 'select-mktest', 
				options = [
					{'label': 'Original Mann-Kendall Test', 'value': 'original'},
					{'label': 'Hamed and Rao Modified MK Test', 'value': 'rao'},
					{'label': 'Yue and Wang Modified MK Test', 'value': 'yue'},
					{'label': 'Modified MK Test using Pre-Whitening Method', 'value': 'prewhiten'}, 
					{'label': 'Modified MK Test using Trend free Pre-Whitening Method', 'value': 'trendfree'}
				], 
				value = 'trendfree', 
			)
		], style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'}), 

		html.Div([

			html.H6(
				children = 'Trend Line Type',
				style = {'width': '95%'},
			),

			html.Div([

				dcc.Checklist(
					id = 'trend-line-type-1',
					options = [
						{'label': 'Sens Slope', 'value': 'sens'},
						{'label': 'Linear Regression', 'value': 'linear'},
					], 
					value = ['sens'], 
					labelStyle={'display': 'block', 'float': 'left'},
					style = {'display': 'inline-block'}
				),

				dcc.Checklist(
					id = 'trend-line-type-2',
					options = [
						{'label': 'Upper Bound', 'value': 'upper'},
						{'label': 'Lower Bound', 'value': 'lower'},
					], 
					labelStyle={'display': 'block', 'float': 'left'},
					style = {'display': 'inline-block'}
				),
			]),


		], style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'}),

		html.Div([
			html.Button(
				id = 'run-button', 
				n_clicks = 0, 
				children = 'RUN',
	 		),
		], style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'}),

		html.Div([
			dcc.Markdown(
				id = 'mktest-output',
				style = {'background-color': 'white'}
				# 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding-left': '10%', 
			),
			
		], style={'width': '85%', 'padding-left': '5%', 'margin-top': '50px'}),

	], className='four columns', style = {'height': 1200, 'background-color': '#f7f7f7'}),

	html.Div([
		dcc.Graph(
			id = 'trendplot'
		),
	], className='eight columns', style = {'height': 1200, 'background-color': '#f7f7f7',})
], className='row')


@app.callback(
	Output('select-variable', 'options'), 
	[Input('select-file', 'value')],
)
def list_variables(data_path):

	global df

	extension = data_path.split('.')[-1]
	if extension in ['csv']: 
		df = pd.read_csv(data_path, index_col=0)
	if extension in ['xlsx', 'xls']:
		df = pd.read_excel(data_path, index_col=0)

	varname_list = df.columns.tolist()
	return [{'label': item, 'value': item} for item in varname_list]

@app.callback(
	[Output('trendplot', 'figure'), 
	 Output('mktest-output', 'children')],
	[Input('run-button', 'n_clicks')],
	[State('select-variable', 'value'),
	 State('select-mktest', 'value'), 
	 State('trend-line-type-1', 'value'),
	 State('trend-line-type-2', 'value')]
)
def perform_trend_analysis(n_clicks, varname, method, *trlines):

	trlines = [item if item is not None else [] for item in trlines]
	trlines = sum(trlines, [])

	ts = df.loc[:, varname]
	t = ts.index.values
	y = ts.values
	x = np.arange(len(y))

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

	# hovertemplate key words:
	# italic: <i></i>
	# bold: <b></b>
	# break line: <br></br>

	data = [
		go.Scatter(x=t, y=y, mode='markers', text=t, name='Time Series', showlegend=False, 
			marker={'color': 'grey', 'size': 6}, 
			hovertemplate='Time: %{text}'+'<br>Value: %{y:.3f}</br>'
		)
	]

	if 'sens' in trlines:
		data.append(
			go.Scatter(x=t, y=y2, mode='lines', name='Sens Slope', showlegend=True, 
				line={'color': 'green', 'width': 2})
		)

	if 'linear' in trlines:
		data.append(
			go.Scatter(x=t, y=y3, mode='lines', name='Linear Regression', showlegend=True, 
				line={'color': 'blue', 'width': 2})
		)

	if 'upper' in trlines:
		data.append(
			go.Scatter(x=t, y=y4, mode='lines', name='Upper Bound', showlegend=True, 
				line={'color': 'green', 'dash': 'dot', 'width': 2})
		)

	if 'lower' in trlines:
		data.append(
			go.Scatter(x=t, y=y5, mode='lines', name='Lower Bound', showlegend=True, 
				line={'color': 'green', 'dash': 'dot', 'width': 2})
		)

	if pvalue < 0.1:
		bgcolor = 'rgba(256, 0, 0, 0.2)' if slp > 0 else 'rgba(0, 0, 256, 0.2)'
	else:
		bgcolor = 'rbga(0, 0, 0, 0)'

	xticks = np.linspace(x[0], x[-1], 5).astype(int)
	xticks_label = t[xticks]

	layout = {
		'title': 'Trend: {}'.format(varname),
		'font': dict(size=18),
		'hovermode': 'closest', # closest
		'plot_bgcolor': bgcolor,
		'showlegend': True,
		'autosize': False,
		'xaxis': dict(title='Time', zeroline=False, domain=[0., .98], showgrid=False, automargin=True), 
		'yaxis': dict(title='', zeroline=True, domain=[0., .98], showgrid=False, automargin=True),
		'paper_bgcolor': '#F2F2F2',
		# 'width': 800, 
		'height': 600, 
		'margin': dict(l=2, r=2, t=50, b=2),
	}

	text = '''

		#### Summary

		Sen's Slope: {:.8f}

		MK-Test P-Value: {:.3f}

		Number of Samples: {:d}

		OLS Slope: {:.8f}

		R^2: {:.3f}

		Sen's Slope Upper Bound: {:.8f}

		Sen's Slope Lower Bound: {:.8f}

	'''.format(slp, pvalue, n, slp_lr, rsq, slp_up, slp_lo)

	return {'data': data, 'layout': layout,}, text

# How to Run
# 1. source mybash/activate-venv-dash
# 2. cd Dropbox/Python/plotly_dash_trend_analysis/
# 3. python app.py
# 4. open browser, type "localhost:6920"

if __name__ == '__main__':
	# default port: 127.0.0.1:8050
	app.run_server(debug=True, port=9200, host='0.0.0.0')