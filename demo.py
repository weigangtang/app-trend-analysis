import numpy as np
import pandas as pd

import os, sys

import base64
import io

import dash
import dash.dcc as dcc
import dash.html as html
# import dash_core_components as dcc
# import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
import plotly.express as px

from utils import *

# App Layout -----------------------------------------------------------------
# ----------------------------------------------------------------------------

def create_layout(app):
	return html.Div(
		className="row",
		children = [

			html.Div(
				className='four columns',
				style = {'height': 1200, 'background-color': '#f7f7f7'},
				children=[

					html.Div(
						style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'}, 
						children=[
							dcc.Upload(
								id='upload-data',
								children=html.Div(
									children=[
										'Drag and Drop or ', 
										html.A('Select File')
									]
								),
								style={
									'width': '100%',
									'height': '60px',
									'lineHeight': '60px',
									'borderWidth': '1px',
									'borderStyle': 'dashed',
									'borderRadius': '5px',
									'textAlign': 'center',
									'margin': '10px'
								}
							), 

							html.Label(id='upload-filename'),
						]
					),

					html.Div(
						style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'},
						children=[
						
							html.H6(
								children = 'Select Variable',
								style = {'width': '95%'},
							),

							dcc.Dropdown(
								id = 'select-variable', 
								placeholder = 'select variables', 
							)
						]
					), 

					html.Div(
						style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'},
						children=[
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
						]
					), 

					html.Div(
						style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'},
						children=[
							html.H6(
								children = 'Trend Line Type',
								style = {'width': '95%'},
							),
							dcc.Checklist(
								id = 'trend-line-type',
								options = [
									{'label': 'Sens Slope', 'value': 'sens'},
									{'label': 'Linear Regression', 'value': 'linear'},
									{'label': 'Upper Bound', 'value': 'upper'},
									{'label': 'Lower Bound', 'value': 'lower'},
								], 
								value = ['sens'], 
								labelStyle={'display': 'block', 'float': 'left', 'margin-right': '24px', 'margin-bottom': '10px'},
								# inputStyle={'margin-right': '30px'},
								style = {'display': 'inline-block', 'margin-left': '30px'}
							),
						]
					),

					html.Div(
						style={'width': '96%', 'textAlign': 'center', 'margin-top': '20px'}, 
						children = [
							html.Button(
								id = 'run-button', 
								n_clicks = 0, 
								children = 'RUN',
					 		),
					 	]
					),

					html.Div(
						style={'width': '85%', 'padding-left': '5%', 'margin-top': '50px'}, 
						children=[
							dcc.Markdown(
								id = 'mktest-output',
								style = {'background-color': 'white'}
								# 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding-left': '10%', 
							),
						]
					),
				]
			),

			html.Div(
				className='eight columns', 
				style={'height': 1200, 'background-color': '#f7f7f7'},
				children=[
					dcc.Graph(id = 'trendplot'),
					dcc.Store(id = 'input-data', data = None), 
				]
			)
		], 
	)

def demo_callbacks(app):

	@app.callback(
		[
			Output('select-variable', 'options'), 
			Output('input-data', 'data'), 
			Output('upload-filename', 'children')
		], 
		[Input('upload-data', 'contents')],
		[State('upload-data', 'filename')]
	)
	def load_data(contents, filename):

		if contents:

			content_type, content_string = contents.split(',')
			decoded = base64.b64decode(content_string)

			fext = filename.split('.')[-1]

			if fext in ['csv', 'xls', 'xlsx']:

				if fext == 'csv':
					df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
				else:
					df = pd.read_excel(io.BytesIO(decoded))

				df = df.set_index(df.columns[0])
				js = df.to_json(orient='columns') # if not set orient, index will be lost

				varname_list = df.columns.tolist()
				options = [{'label': item, 'value': item} for item in varname_list]

				return options, js, filename

			else:
				print('Unacceptable Data Format!')
				raise PreventUpdate
		else: 
			raise PreventUpdate

	@app.callback(
		[Output('trendplot', 'figure'), Output('mktest-output', 'children')],
		[Input('run-button', 'n_clicks'), Input('input-data', 'data')],
		[
			State('select-variable', 'value'),
			State('select-mktest', 'value'), 
			State('trend-line-type', 'value')
		]
	)
	def perform_trend_analysis(n_clicks, data, varname, method, trlines):

		if (data is None) | (varname is None):
			raise PreventUpdate

		else:

			df = pd.read_json(data, orient='columns')

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

			if pvalue > 0.1:
				bgcolor = 'rbga(0, 0, 0, 0)'
			else:
				bgcolor = 'rgba(256, 0, 0, 0.2)' if slp > 0 else 'rgba(0, 0, 256, 0.2)'

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

			figure = {'data': data, 'layout': layout}

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

			return figure, text
