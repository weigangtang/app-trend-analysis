import dash
from demo import create_layout, demo_callbacks

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
)

server = app.server
app.layout = create_layout(app)
demo_callbacks(app)

if __name__ == '__main__':
    # app.run_server(debug=True, port=9100, host='0.0.0.0')
    app.run_server(debug=True)
