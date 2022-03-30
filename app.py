# use dotEnv
import os
from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import pandas as pd
from dash.dependencies import Input, Output, State
from dotenv import load_dotenv
from dash import dash_table

from binance_requester import BinanceFuturesRequester

# load the environment vars
load_dotenv()

# initialize application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server

app.config["suppress_callback_exceptions"] = True  # set this because we render different components in different tabs
app.title = "Binance Futures Dashboard"

app.layout = html.Div(className="container-fluid", id="trade-main-div", children=[
    html.Div([
        html.Div("Current Positions", className="h3"),
        html.Div(["used API weight:", html.Span(id="weight-info")], className="h3"),
        dbc.Row(
            [
                dbc.Col([dbc.Label("Reduce Amount")]),
                dbc.Col([dbc.Input(type="number", id="reduceAmt", value="10")]),
                dbc.Col([dbc.FormText("Suggests reduction values for specified Amount of USDT")]),
            ]),
    ], className="d-flex flex-row mt-2 mx-2 justify-content-between"),
    html.Div(children=[
        dcc.Interval("load-interval-cmp", interval=10 * 1000, n_intervals=0),
        dcc.Store(id='session-store', storage_type='session'),
        html.Div(id="testpos")
    ], className="container-fluid"),
    html.Hr(),
    html.Div(id="upnl", children=[], className="d-flex d-flex-row justify-content-around flex-wrap h3"),
    html.Div(id="gain-line", children=[], className="d-flex d-flex-row justify-content-around flex-wrap h3"),
    html.Div(id="gain-line2", children=[], className="d-flex d-flex-row justify-content-around flex-wrap h3"),
    html.Hr(),
    dcc.Graph(id="income_chart", className="d-flex flex-row justify-content-around"),
    html.Hr(),
    dbc.Table(id="gain-table", bordered=True, hover=True, responsive=True, striped=True)
])


def checkKeyExistence(keyNum):
    if not os.getenv("API_KEY" + keyNum):
        return 0
    else:
        if not os.getenv("API_SECRET" + keyNum):
            return 0
        else:
            return 1


global flag2
global flag3
flag2 = checkKeyExistence('2')
flag3 = checkKeyExistence('3')


def calc_roe(row):
    # calulate Margin = current position size / leverage
    row['entryMargin'] = (row.entryPrice * row.positionAmt) / row.leverage
    row['markMargin'] = (row.markPrice * row.positionAmt) / row.leverage
    # calculate pnl = current Total - entry Total
    row['pnl'] = (row.markPrice * row.positionAmt) - (row.entryPrice * row.positionAmt)
    row.markMargin = abs(row.markMargin)
    row.entryMargin = abs(row.entryMargin)
    # ROE = PNL / Margin
    row['markRoe'] = row['pnl'] / row.markMargin
    row['entryRoe'] = row['pnl'] / row.entryMargin
    # row['markRoe'] = row.unRealizedProfit / row.markMargin
    # row['entryRoe'] = row.unRealizedProfit / row.entryMargin
    return row


def calc_reduction(row, total_amt=10):
    # reduction asset amt = (bag weight * totalAmt) / markPrice
    if row['unRealizedProfit'] < 0:
        row['redSuggestion'] = (row['pnlPerc'] * total_amt) / row['markPrice']
        row['redSuggestionQuote'] = row['pnlPerc'] * total_amt
        row['redSuggestionMargin'] = (row['pnlPerc'] * total_amt * row['leverage']) / row['markPrice']
        row['redSuggestionQuoteMargin'] = row['pnlPerc'] * total_amt * row['leverage']
    else:
        row['redSuggestion'] = 0
        row['redSuggestionQuote'] = 0
        row['redSuggestionMargin'] = 0
        row['redSuggestionQuoteMargin'] = 0
    return row


def get_liq_color(liq_dist):
    """
    Get some colors for out Liquidation distance
    :param liq_dist:
    :return:
    """
    if liq_dist > 0.5:
        return " text-success"
    elif 0.25 <= liq_dist <= 0.50:
        return " text-warning"
    elif 0 < liq_dist < 0.25:
        return " text-danger"
    else:
        return " text-body"


def create_gain_line(balance, gain, gain_yesterday):
    yesterday = balance - gain
    gain_perc = (gain / yesterday) if balance > 0 else 0
    # difference correct?
    diff = gain - gain_yesterday
    delta_gain = diff / abs(gain_yesterday)
    gl1 = [
        html.Div([
            html.Div("PnL today"),
            html.Div(children=[
                html.Span("{:.2f} USDT".format(gain), className="ml-2 font-weight-bold"),
            ])
        ], className="mx-2 d-flex"),
        html.Div([
            html.Div("Gain today"),
            html.Div(children=[
                html.Span("{:.2%}".format(gain_perc), className="ml-2 font-weight-bold"),
            ])
        ], className="mx-2 d-flex"),
        html.Div([
            html.Div("Current balance"),
            html.Div(children=[
                html.Span("{:.2f} USDT".format(balance), className="ml-2 font-weight-bold"),
            ])
        ], className="mx-2 d-flex"),
        html.Div([
            html.Div("Yesterday balance"),
            html.Div(children=[
                html.Span("{:.2f} USDT".format(yesterday), className="ml-2 font-weight-bold"),
            ])
        ], className="mx-2 d-flex"),
    ]
    gl2 = [
        html.Div([
            html.Div("PnL yesterday"),
            html.Div(children=[
                html.Span("{:.2f} USDT".format(gain_yesterday), className="ml-2 font-weight-bold"),
            ])
        ], className="mx-2 d-flex"),
        html.Div([
            html.Div("Diff."),
            html.Div(children=[
                html.Span("{:.2f} USDT".format(diff), className="ml-2 font-weight-bold"),
            ])
        ], className="mx-2 d-flex"),
        html.Div([
            html.Div(children=[
                html.Span("{:.2%}".format(delta_gain), className="ml-2 font-weight-bold"),
            ])
        ], className="mx-2 d-flex")
    ]
    return gl1, gl2


@app.callback([Output('income_chart', 'figure'),
               Output('gain-line', 'children'),
               Output('gain-line2', 'children'),
               Output("gain-table", "children"),
               ], [
                  Input("load-interval-cmp", "n_intervals"),
                  State("session-store", "data")
              ])
def update_income_stats(n, store):
    bf = BinanceFuturesRequester(os.getenv("API_KEY"), os.getenv("API_SECRET"))
    income_data, used_weight = bf.get_income_data('REALIZED_PNL')
    if flag2 == 1:
        bf2 = BinanceFuturesRequester(os.getenv("API_KEY2"), os.getenv("API_SECRET2"))
        income_data2, used_weight2 = bf2.get_income_data('REALIZED_PNL')
        id_df2 = pd.DataFrame(income_data2.json())
    if flag3 == 1:
        bf3 = BinanceFuturesRequester(os.getenv("API_KEY3"), os.getenv("API_SECRET3"))
        income_data3, used_weight3 = bf3.get_income_data('REALIZED_PNL')
        id_df3 = pd.DataFrame(income_data3.json())

    print("Income-Data call used weight: {} ({:.2%})".format(used_weight, int(used_weight) / 2400))

    # income data as frame...
    id_df = pd.DataFrame(income_data.json())

    if flag2 == 1 & flag3 == 1:
        id_df = pd.concat([id_df, id_df2, id_df3], ignore_index=True, sort=False)
    elif flag2 == 1 & flag3 == 0:
        id_df = pd.concat([id_df, id_df2], ignore_index=True, sort=False)

    id_df["time2"] = id_df["time"].apply(lambda x: pd.to_datetime(x / 1000, unit='s', utc=True))
    id_df["date"] = id_df["time2"].dt.date
    id_df['income'] = id_df["income"].astype(float)

    income_stats = id_df.groupby(['date', 'symbol'], as_index=False).agg(
        {'income': ['sum', 'mean', 'min', 'max', 'count']})
    income_stats.columns = ['_'.join(x) if isinstance(x, tuple) and x[1] != '' else x[0] for x in
                            income_stats.columns.ravel()]

    traces = []

    # find records for today and yesterday to compare totals
    yesterday_date = pd.to_datetime(datetime.utcnow() - timedelta(1))
    yesterdays_income = id_df[id_df['date'] == yesterday_date]
    yesterday_total = float(yesterdays_income[['income']].sum())
    latest_income = id_df[id_df['date'] == id_df['date'].max()]  # .agg({'income':['sum','median','mean','min', 'max']})
    total_income = float(latest_income[['income']].sum())

    # generate a line with pnl today, gain , gain % and current balance
    balance = float(store['balance']) if store is not None else 0
    gain_line, gain_line2 = create_gain_line(balance, total_income, yesterday_total)

    symbols = income_stats.symbol.unique()
    for s in symbols:
        # filter...
        f = income_stats[income_stats['symbol'] == s]
        # f = f[f['incomeType'] == 'REALIZED_PNL']
        # append traces
        traces.append(dict(
            y=f['date'],
            x=f['income_sum'],
            type="bar",
            name=s,
            orientation="h"
        ))

    gt_tbl = [
        html.Thead([
            html.Tr([
                html.Th("Date"),
                html.Th("Trades"),
                html.Th("PnL"),
                html.Th('PnL per Trade')
            ])
        ]),
        html.Tbody([

        ])
    ]

    tbl_df = id_df[['income', 'date']].groupby(['date']).agg(["sum", "count"])
    tbl_df = tbl_df.sort_index(ascending=False)
    for index, row in tbl_df.iterrows():
        count = "{:d}".format(int(row[('income', 'count')]))
        sum = "{:.2f} USDT".format(float(row[('income', 'sum')]))
        per_trade = "{:.2f} USDT".format(row['income', 'sum'] / row['income', 'count'])
        gt_tbl[1].children.append(
            html.Tr([
                html.Td(index),
                html.Td(count),
                html.Td(sum),
                html.Td(per_trade),
            ])
        )

    return {
               "data": traces,
               "layout": dict(
                   barmode="relative",
                   yaxis={"type": "date", "title": "Date"},

                   xaxis={"title": "PnL"},
                   transition={'duration': 500},
                   title="Realized PnL",
               )
           }, gain_line, gain_line2, gt_tbl


@app.callback([
    Output("testpos", "children"),
    Output("upnl", "children"),
    Output("weight-info", "children"),
    Output("session-store", 'data')
],
    [
        Input("load-interval-cmp", "n_intervals"),
        Input("reduceAmt", "value")
    ]
)
def update_position_stats(n, reduceValue):
    global upnl, values_to_store
    method_weight2 = 0
    method_weight3 = 0
    bf = BinanceFuturesRequester(os.getenv("API_KEY"), os.getenv("API_SECRET"))
    if flag2 == 1:
        bf2 = BinanceFuturesRequester(os.getenv("API_KEY2"), os.getenv("API_SECRET2"))
        userdata2, user_weight2 = bf2.get_position_risc()
        balance_data2, balance_weight2 = bf2.get_balance()
        method_weight2 = balance_weight2 + user_weight2
        print("Update-Current call used weight (acc2): {} ({:.2%})".format(method_weight2, int(method_weight2) / 2400))
    if flag3 == 1:
        bf3 = BinanceFuturesRequester(os.getenv("API_KEY3"), os.getenv("API_SECRET3"))
        userdata3, user_weight3 = bf3.get_position_risc()
        balance_data3, balance_weight3 = bf3.get_balance()
        method_weight3 = balance_weight3 + user_weight3
        print("Update-Current call used weight (acc3): {} ({:.2%})".format(method_weight3, int(method_weight3) / 2400))
    userdata, user_weight = bf.get_position_risc()
    balance_data, balance_weight = bf.get_balance()

    method_weight = balance_weight + user_weight
    if flag2 == 1 & flag3 == 1:
        method_weight = (method_weight + method_weight2 + method_weight3) / 3
    elif flag2 == 1 & flag3 == 0:
        method_weight = (method_weight + method_weight2) / 2

    print("Update-Current call used weight: {} ({:.2%})".format(method_weight, int(method_weight) / 2400))
    used_weight = " {} ({:.2%})".format(method_weight, int(method_weight) / 2400)

    ud_df = pd.DataFrame(userdata.json()).astype({
        "symbol": str,
        "positionAmt": float,
        "entryPrice": float,
        "markPrice": float,
        "unRealizedProfit": float,
        "liquidationPrice": float,
        "leverage": float,
        "maxNotionalValue": float,
        "marginType": str,
        "isolatedMargin": bool,
        "isAutoAddMargin": bool,
        "positionSide": str
    })
    if flag2 == 1:
        ud_df2 = pd.DataFrame(userdata2.json()).astype({
            "symbol": str,
            "positionAmt": float,
            "entryPrice": float,
            "markPrice": float,
            "unRealizedProfit": float,
            "liquidationPrice": float,
            "leverage": float,
            "maxNotionalValue": float,
            "marginType": str,
            "isolatedMargin": bool,
            "isAutoAddMargin": bool,
            "positionSide": str
        })
        if flag3 == 0:
            ud_df = pd.concat([ud_df, ud_df2], ignore_index=False)
        else:
            ud_df3 = pd.DataFrame(userdata3.json()).astype({
                "symbol": str,
                "positionAmt": float,
                "entryPrice": float,
                "markPrice": float,
                "unRealizedProfit": float,
                "liquidationPrice": float,
                "leverage": float,
                "maxNotionalValue": float,
                "marginType": str,
                "isolatedMargin": bool,
                "isAutoAddMargin": bool,
                "positionSide": str
            })
            ud_df = pd.concat([ud_df, ud_df2, ud_df3], ignore_index=False)

    total_pnl = 0
    total_margin = 0
    balance = 0
    margin_balance = 0
    wallet_allocation = 0
    card_div = html.Div(html.H3("It seems you have no open positions for now..."))
    ud_df = ud_df[ud_df.positionAmt != 0]  # filter for open positions
    ud_df['pnlPerc'] = ud_df['unRealizedProfit'] / ud_df['unRealizedProfit'].sum()
    rAmnt = int(reduceValue) if reduceValue is not None else 0

    # set here with empty values
    upnl = html.Div()
    values_to_store = {
        'balance': 0
    }

    if len(ud_df):
        ud_df = ud_df.apply(calc_roe, axis=1)  # calculate roe and margin -> axis = 1 for rows instead of columns
        ud_df = ud_df.apply(calc_reduction, args=(rAmnt,), axis=1)
        # print(ud_df[['pnl', 'pnlPerc', 'redSuggestion']].head())
        balance_data = balance_data.json()[0]
        balance = float(balance_data['balance'])
        if flag2 == 1:
            balance_data2 = balance_data2.json()[0]
            balance2 = float(balance_data2['balance'])
            if flag3 == 1:
                balance_data3 = balance_data3.json()[0]
                balance3 = float(balance_data3['balance'])
                balance = balance + balance2 + balance3
            else:
                balance = balance + balance2
        ud_df["walletAllocation"] = ud_df['markMargin'].apply(lambda x: (x / balance) if balance > 0 else 0)
        ud_df["walletDrain"] = ud_df['pnl'].apply(lambda x: (x / balance) if balance > 0 else 0)
        ud_df["dollarCost"] = ud_df[['positionAmt', 'entryPrice']].apply(lambda x: abs(x[0] * x[1]), axis=1)
        mark_margin = ud_df["markMargin"].sum()
        wallet_allocation = (mark_margin / balance) if balance > 0 else 1
        total_pnl = ud_df['pnl'].sum()
        total_margin = ud_df['markMargin'].sum()
        margin_balance = balance + total_pnl  # is this correct here?

        ud_df.sort_values(by='symbol', inplace=True)

        card_div = html.Div(children=[],
                            className="d-flex flex-row justify-content-around flex-wrap")  # dbc.CardDeck(children=[])
        for index, row in ud_df.iterrows():
            l_dist = (1 - (row['entryPrice'] / row['liquidationPrice'])) if row[
                                                                                'liquidationPrice'] > 0 else 0  # for short
            if row['liquidationPrice'] > 0 and row['positionAmt'] > 0:
                l_dist = (1 - (row['liquidationPrice'] / row['entryPrice']))
            l_dist_m = (1 - (row['markPrice'] / row['liquidationPrice'])) if row[
                                                                                 'liquidationPrice'] > 0 else 0  # for short
            if row['liquidationPrice'] > 0 and row['positionAmt'] > 0:
                l_dist_m = (1 - (row['liquidationPrice'] / row['markPrice']))
            # print(row['symbol'], row['entryPrice'], row['liquidationPrice'], '{:.2%}'.format(l_dist))
            l_color = get_liq_color(l_dist)
            l_color_m = get_liq_color(l_dist_m)
            # as long as both liqs are in one string, we take l_color_m if it differs
            if l_color != l_color_m:
                l_color = l_color_m

            card = dbc.Card([
                dbc.CardHeader([
                    "{}  x{:.0f}".format(row['symbol'], row.leverage),
                    html.Span('SHORT' if row['positionAmt'] < 0 else 'LONG',
                              className='badge '
                                        + ('badge-primary' if row.positionAmt == 0
                                           else 'badge-success' if row.positionAmt > 0 else 'badge-danger')
                                        + ' float-right')
                ],
                    className='font-weight-bold'),
                dbc.CardBody([
                    html.Div(className="d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("Size"),
                        html.Div(children=html.Span('{0:.3f}'.format(row['positionAmt']),
                                                    className="font-weight-bold")
                                 )
                    ]),
                    html.Div(className="mb-1 d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("uPnL"),
                        html.Div(children=html.Span('{0:.2f} USDT'.format(row['pnl']),
                                                    className="font-weight-bold"
                                                              + (" text-danger" if row['pnl'] < 0 else " text-success"))
                                 )
                    ]),
                    html.Div(className="d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("Liq-Distance"),
                        html.Div(children=html.Span('E: {:.2%} / M: {:.2%}'.format(l_dist, l_dist_m),
                                                    className="font-weight-bold" + l_color)
                                 ),
                    ]) if l_dist > 0 else None,
                    html.Div(className="mb-1 d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("Liq-Price"),
                        html.Div(children=html.Span("{0:.2f} USDT".format(row['liquidationPrice']),
                                                    className="font-weight-bold")
                                 )
                    ]) if row['liquidationPrice'] > 0 else None,
                    html.Div(className="d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("ROE"),
                        html.Div(children=
                                 html.Span("{:.2%}".format(row['markRoe']),
                                           className="p-1 font-weight-bold" + (
                                               " text-danger" if row['markRoe'] < 0 else " text-success"))
                                 )
                    ]),
                    html.Div(className="d-flex flex-row justify-content-between  align-items-baseline", children=[
                        html.Div("Margin"),
                        html.Div(children=
                                 html.Span("{:.2f} USDT".format(row['markMargin']), className="font-weight-bold")
                                 )
                    ]),
                    html.Div(className="d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("allocated"),
                        html.Div(children=
                                 html.Span("{:.2%}".format(row['walletAllocation']), className="p-1 font-weight-bold")
                                 )
                    ]),
                    html.Div(className="d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("loss/profit"),
                        html.Div(children=
                                 html.Span("{:.2%}".format(row['walletDrain']), className="p-1 font-weight-bold"
                                                                                          + (" text-danger" if row[
                                                                                                                   'markRoe'] < 0 else " text-success"))
                                 )
                    ]),
                    html.Div(className="d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("Cost"),
                        html.Div(children=
                                 html.Span("{:.2f} USDT".format(row['dollarCost']), className="p-1 font-weight-bold")
                                 )
                    ]),
                    html.Hr(),
                    html.Div(className="d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("Reduce margin by"),
                        html.Div(children=
                        html.Span("{:.2f} USDT".format(
                            row['redSuggestionQuote']),
                            className="p-1 font-weight-bold")
                        )
                    ]) if row['redSuggestionQuote'] > 0 else None,
                    html.Div(className="d-flex flex-row justify-content-between align-items-baseline", children=[
                        html.Div("SELL" if row['positionAmt'] > 0 else "BUY"),
                        html.Div(children=
                        html.Span("{:.3f} {}".format(
                            row['redSuggestionMargin'], row['symbol']),
                            className="p-1 font-weight-bold")
                        )
                    ]) if row['redSuggestionQuote'] > 0 else None,
                ]),
                dbc.CardFooter([
                    html.Div(className="d-flex flex-row justify-content-between  align-items-baseline", children=[
                        html.Div("Entry Price"),
                        html.Div(children=
                                 html.Span("{:.8f} USDT".format(row['entryPrice']), className="font-weight-bold")
                                 )
                    ]),
                    html.Div(className="d-flex flex-row justify-content-between  align-items-baseline", children=[
                        html.Div("Mark Price"),
                        html.Div(children=
                                 html.Span("{:.8f} USDT".format(row['markPrice']),
                                           className="font-weight-bold"
                                                     + (" text-danger" if row['markRoe'] < 0 else " text-success"))
                                 )
                    ])
                ])
            ], className='m-1 flex-fill ' + ('border-success' if row['pnl'] >= 0 else 'border-danger'))
            card_div.children.append(card)
        upnl = [
            html.Div([
                html.Div("uPnL"),
                html.Div(children=[
                    html.Span("{:.2f} USDT".format(total_pnl), className="ml-2 font-weight-bold"),
                    html.Span("({:.2%})".format(total_pnl / balance if balance > 0 else 1),
                              className="ml-2 font-weight-bold")
                ])
            ], className="mx-2 d-flex"),
            html.Div([
                html.Div("used margin"),
                html.Div(children=
                         html.Span("{:.2f} USDT".format(total_margin), className="ml-2 font-weight-bold")
                         )
            ], className="mx-2 d-flex"),
            html.Div([
                html.Div("margin balance"),
                html.Div(children=
                         html.Span("{:.2f} USDT".format(margin_balance), className="ml-2 font-weight-bold")
                         )
            ], className="mx-2 d-flex"),
            html.Div([
                html.Div("balance"),
                html.Div(children=
                         html.Span("{:.2f} USDT".format(balance), className="ml-2 font-weight-bold")
                         )
            ], className="mx-2 d-flex"),
            html.Div([
                html.Div("wallet allocation"),
                html.Div(children=
                         html.Span("{:.2%}".format(wallet_allocation), className="ml-2 font-weight-bold")
                         )
            ], className="mx-2 d-flex"),
        ]

        values_to_store = {
            'balance': balance
        }
        print("Update Data {} times".format(n))
    return card_div, upnl, used_weight, values_to_store


if __name__ == "__main__":
    app.run_server(debug=True, port=int(os.getenv("SERVER_PORT")), host=os.getenv("HOST"))
