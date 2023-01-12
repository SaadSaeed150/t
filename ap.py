from flask import Flask, send_from_directory
from dash import Dash, dcc, html, Input, Output
import dash_auth

# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'saad': '1212'
}


import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

server = Flask(__name__)

app.layout = html.Div([

    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,  # in milliseconds
        n_intervals=0
    ),

    html.H3('*************** Order In process ************'),
    html.Div(id='output-data-upload'),

    html.H3('*************** BTC PRICE ************'),
    html.Div(id='output-data-upload3'),

    html.H3('*************** Grid Information ************'),
    html.Div(id='output-data-upload2'),

    html.H3('*************** Last 5 data columns ************'),
    html.Div(id='output-data-upload4'),

    html.H3('*************** All Orders ************'),
    html.Div(id='output-data-upload1'),

])


@app.callback(Output('output-data-upload', 'children'),
              Input('interval-component', 'n_intervals'))
def parse_contents(n_intervals):
    df = pd.read_csv("order_list.csv", index_col=[0])

    df = df[df['status'] != 'queued']
    return html.Div([
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line
        html.Div('Raw Content'),
    ])


@app.callback(Output('output-data-upload1', 'children'),
              Input('interval-component', 'n_intervals'))
def parse_contents(n_intervals):
    df = pd.read_csv("order_list.csv", index_col=[0])

    return html.Div([
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line
        html.Div('Raw Content'),
    ])


@app.callback(Output('output-data-upload2', 'children'),
              Input('interval-component', 'n_intervals'))
def parse_contents2(n_intervals):
    df = pd.read_csv("grids_informatio.csv", index_col=[0])

    return html.Div([
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line
        html.Div('Raw Content'),
    ])


@app.callback(Output('output-data-upload3', 'children'),
              Input('interval-component', 'n_intervals'))
def parse_contents2(n_intervals):
    df = pd.read_csv("btc_data.csv", index_col=[0])

    df = df[-20:]

    return html.Div([
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line
        html.Div('Raw Content'),
    ])


@app.callback(Output('output-data-upload4', 'children'),
              Input('interval-component', 'n_intervals'))
def parse_contents2(n_intervals):
    df = pd.read_csv("df_final.csv", index_col=[0])

    df = df[-5:]

    return html.Div([
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line
        html.Div('Raw Content'),
    ])


if __name__ == '__main__':
    app.run_server(debug=True, port=8051, host='0.0.0.0')
