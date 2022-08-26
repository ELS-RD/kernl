import pickle

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)
with open(r"benchmarks.pickle", "rb") as input_file:
    v = pickle.load(input_file)
    df = pd.json_normalize([s.to_dict() for s in v])

charts = []
grouped = df.groupby(['fullfunc'])
for func, group in grouped:
    id = func.replace(".", "-")
    @app.callback(
        Output(id, 'figure'),
        Input(id + '-measure', 'value'))
    def update_y_timeseries(measure, group=group):
        y = "data_gpu.median" if measure == "GPU" else "data_full.median"
        fig = px.bar(group, x="group",
                     y=y,
                     color="params.implementation", barmode='group')
        fig.update_xaxes(type='category', categoryorder='min ascending')
        return fig


    charts.append(html.H2(children=func))
    charts.append(html.P(children="Measure"))
    charts.append(
        dcc.RadioItems(
            ['GPU', 'Full'],
            'GPU',
            id=id + '-measure',
            labelStyle={'display': 'inline-block', 'marginTop': '5px'}
        )
    )
    charts.append(dcc.Graph(
        id=id
    ))

app.layout = html.Div(children=[
    *charts
])

if __name__ == '__main__':
    app.run_server(debug=True, port="8080")
