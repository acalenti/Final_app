import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import xlrd
import numpy as np
import pandas as pd
from dash.dependencies import State, Input, Output
import time
import matplotlib
from matplotlib import cm

from dash.exceptions import PreventUpdate
import datetime
from datetime import datetime as dt
import os
import json

from pandas.io.json import json_normalize

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 3500)
pd.set_option('display.max_colwidth', 5000)
pd.set_option('display.width', 15000)









import base64




image_filename = 'assets/blue_logo.jpg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


app = dash.Dash(
    __name__,
)

server = app.server

app.config.suppress_callback_exceptions = True

starter_dataframe = pd.read_csv('sample_data2.csv')
#starter_dataframe = pd.read_csv('Sample_Data.csv')


def base_drop_down_options():
    base_data = pd.read_csv('Sample_Data.csv')


    """Dont think you will need this becasue you are not putting you 
    data in excel after pulling from the DB"""

    def read_date(date):
        return xlrd.xldate.xldate_as_datetime(date, 0)

    base_data['StartDate'] =pd.to_datetime(base_data['StartDate'].apply(read_date), errors='coerce')
    base_data['EndDate'] =pd.to_datetime(base_data['EndDate'].apply(read_date), errors='coerce')

    """_____________**************************************************________________"""


    start_date_time_frame = base_data['StartDate'].min()
    end_data_time_frame = base_data['EndDate'].max()

    unique_plans = base_data.Plan.unique()
    unique_plan_dict = []
    unique_plan_list = []
    for plan in unique_plans:
        add_line = {'label': plan, 'value': plan}
        unique_plan_dict.append(add_line)
        unique_plan_list.append(plan)
    counties = base_data.County.unique()
    counties_dict = []
    counties_list = []
    for county in counties:
        add_line = {'label': county, 'value': county}
        counties_dict.append(add_line)
        counties_list.append(county)
    return start_date_time_frame, end_data_time_frame, unique_plan_dict, counties_dict, unique_plan_list, counties_list


base_drop_down_options()

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div([
                    html.H4("Workbook Title"),
                ], className='row'),
                html.Div([
                    dcc.Markdown('''
                    Welcome to the workbook.  This is where we 
                    will tell a little about
                    what this thing does, and how to use it, 
                    maybe who to contact with issues.  
                    not sure what else you would want to put 
                    in here but enough to move the options down to start at around
                    one third of the way down, ideally'''),
                ], className='row'),

                html.Div([
                    html.H5("Select Date Range"),
                    dcc.DatePickerRange(
                        id="date-picker-range",
                        start_date=base_drop_down_options()[0],
                        end_date=base_drop_down_options()[1],
                        start_date_placeholder_text='Select Start Date',
                        end_date_placeholder_text='Select End Date'
                    ),
                ], className='row chooser_margins'),
                html.Div([
                    html.H5("Group By"),
                    dcc.RadioItems(
                        id="group-by",
                        # options={'label':'test','value':'teser'}
                        options=[{'label': 'Age', 'value': 'Age'},
                                 {'label': 'County', 'value': 'County'},
                                 {'label': 'Plan', 'value': 'Plan'},
                                 {'label': 'Subsidy', 'value': 'Subsidy'}],
                        value='County',
                        labelStyle={'display': 'inline-block'}

                    ),
                ],
                    className="row chooser_margins",
                ),
                html.Div([
                    html.H5("Select Age"),
                    dcc.RangeSlider(
                        marks={i: '{}'.format(i) for i in range(0, 100, 10)},
                        id="slider-age",
                        className='slider'
                    ),
                ],
                    className="row",
                ),
                html.Div([
                    html.Div([
                        html.H5("Select County",
                                style={'margin-top': '20px',
                                       'margin-left': '5px'}),
                        dcc.Checklist(
                            id="checklist-county",
                            options=base_drop_down_options()[3],
                            value=base_drop_down_options()[5],
                            style={'width': '150px',
                                   'margin-left': '5px'}
                        ),
                    ], className="three-col",
                    ),
                    html.Div([
                        html.H5("Select Plan",
                                style={'margin-top': '20px'}),
                        dcc.Checklist(
                            id="dropdown-plan",
                            options=base_drop_down_options()[2],
                            value=base_drop_down_options()[4],
                            # className='move_up'
                        ),
                    ], className="three-col move_up",
                    ),
                ], className="row",
                ),
            ], className="three columns div-left-area"),
        html.Div(
            [
                html.Div([

                    html.Div(
                        id='arizona_county_map_outer',
                        className="county columns div-left-panel",
                        children=dcc.Loading(children=dcc.Graph(id='arizona_county_map',
                                                                config={'displayModeBar': False}),
                                             type="default",
                                             ),
                    ),
                    html.Div(
                        id='arizona_zip_map_outer',
                        className="zip columns div-left-panel",
                        children=dcc.Loading(children=dcc.Graph(id='arizona_zip_map',
                                                                config={'displayModeBar': False}),
                                             type="default",
                                             ),

                    ),
                ], className="row"
                ),
                html.Div([

                            html.Div(
                                id='table_outer',
                                className="grouped columns div-left-panel margin_setter",
                                children=dcc.Loading(children=dcc.Graph(id='table',
                                                                        config={'displayModeBar': False}),
                                                     type='default',
                                                     ),
                            ),

                            html.Div(
                                id='pie_outer',
                                className="pie columns div-left-panel ",
                                children=dcc.Loading(children=dcc.Graph(id='pie_chart',
                                                                        config={'displayModeBar': False}),
                                                     type="default",
                                                     ),
                            ),
                            html.Div(
                                id='histogram_outer',
                                className="histo columns div-left-panel",
                                children=dcc.Loading(children=dcc.Graph(id='histogram',
                                                                        config={'displayModeBar': False}),
                                                     type="default",
                                                     ),
                            ),
                ], className="row")

            ], className='twelve columns div-move-over'),
        html.Div(id='intermediate-value', style={'display': 'none'})
    ]
)
@app.callback(
    Output('intermediate-value', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')])
def determine_days(start_date, end_date):
    df = starter_dataframe
    input_daterange = pd.date_range(start=start_date, end=end_date, freq='D')

    df['count_of_match'] = 0
    for index, row in df.iterrows():
        start = row['StartDate']
        end = row['EndDate']
        user_daterange = pd.date_range(start=start, end=end, freq='D')
        set_list = set(user_daterange) & set(input_daterange)
        df.at[index, 'count_of_match'] = len(set_list)


    return df.to_json(date_format='iso', orient='split')

@app.callback(
    Output('table','figure'),
    [Input('checklist-county', 'value'),
     Input('group-by', 'value'),
     Input('dropdown-plan', 'value'),
     Input('slider-age', 'value'),
     Input('intermediate-value', 'children')])
def all_item_output(county, groupby, plan, age, df):

    df = pd.read_json(df, orient='split')
    df = df[df.count_of_match != 0]
    df = df[df['County'].isin(county)]
    df = df[df['Plan'].isin(plan)]
    if age is not None:
        df = df[(df['Age'] >= age[0]) & (df['Age'] <= age[1])]
    df = df.groupby(groupby).size().reset_index(name='Count')
    values = list(df.columns)
    datas = []
    for column in values:
        groups = df[column].tolist()
        datas.append(groups)

    figure = {
        "data": [go.Table(
            columnwidth=[40, 17],
            header=dict(
                values=values,
                line=dict(
                    color='#ffffff'),
                fill=dict(
                    color='#ffffff'),
                align=['left'] * 5,
                #font=dict(color='rgb(255,255,255)'),
            ),
            cells=dict(
                values=datas,
                fill=dict(
                    color='#22252b'),
                line=dict(
                    color='#ffffff'
                ),
                font=dict(color='#ffffff'),
                align=['left'] * 5,

            )
        )
        ],
        "layout": go.Layout(
            paper_bgcolor='#22252b',
            bargap=.3,
            #plot_bgcolor='rgb(0,0,0)',
            title='Grouped',
            width=304,
            titlefont=dict(
               size=18,
               color='#ffffff'),
            height=640,
            margin=dict(t=42, r=4, l=25, b=0, pad=0)
        )
    }
    return figure

@app.callback(
    Output('pie_chart','figure'),
    [Input('checklist-county', 'value'),
     Input('group-by', 'value'),
     Input('dropdown-plan', 'value'),
     Input('slider-age', 'value'),
     Input('intermediate-value', 'children')])
def pie_chart(county, groupby, plan, age, df):
    # if county == None:
    #     county = base_drop_down_options()[5]
    # if plan == None:
    #     plan = base_drop_down_options()[4]

    df = pd.read_json(df, orient='split')
    df = df[df.count_of_match != 0]
    df = df[df['County'].isin(county)]
    df = df[df['Plan'].isin(plan)]
    if age is not None:
        df = df[(df['Age'] >= age[0]) & (df['Age'] <= age[1])]

    # print(county)
    # print(groupby)
    # print(plan)
    # print(age)


    figure = {
        "data": [go.Pie(
            labels=(['Plan A','Plan B','Plan C','Plan D']),
            values=[df.loc[df['Plan'] == 'A', 'count_of_match'].sum(),
                    df.loc[df['Plan'] == 'B', 'count_of_match'].sum(),
                    df.loc[df['Plan'] == 'C', 'count_of_match'].sum(),
                    df.loc[df['Plan'] == 'D', 'count_of_match'].sum(),
                    ],
            hole=.70
        )
        ],
        "layout": go.Layout(
            paper_bgcolor='#1f2630',
            plot_bgcolor='#1f2630',
            title='Plan Breakdown',
            width=470,
            titlefont=dict(
               size=24,
               color='rgb(255,255,255)'),
            height=470,
            margin=dict(t=70, r=10, l=40, b=40)
        )
    }
    return figure

@app.callback(
    Output('histogram','figure'),
    [Input('checklist-county', 'value'),
     Input('group-by', 'value'),
     Input('dropdown-plan', 'value'),
     Input('slider-age', 'value'),
     Input('intermediate-value', 'children')])
def histogram(county, groupby, plan, age, df):
    # if county == None:
    #     county = base_drop_down_options()[5]
    # if plan == None:
    #     plan = base_drop_down_options()[4]

    df = pd.read_json(df, orient='split')
    df = df[df.count_of_match != 0]
    df = df[df['County'].isin(county)]
    df = df[df['Plan'].isin(plan)]
    if age is not None:
        df = df[(df['Age'] >= age[0]) & (df['Age'] <= age[1])]

    figure = {
        "data": [go.Histogram(
            x=df['Age']
            )
        ],
        "layout": go.Layout(
            paper_bgcolor='#1f2630',
            plot_bgcolor='#1f2630',
            bargap=.2,
            orientation=45,
            title='Age Histogram',
            width=450,
            titlefont=dict(
               size=24,
               color='rgb(255,255,255)'),
            height=450,
            xaxis=dict(
                tickfont=dict(
                    size=14,
                    color='rgb(255,255,255)',
                ),
              linecolor='rgb(255,255,255)',
              #gridcolor='rgb(255,255,255)',
            ),
            yaxis=dict(
                linecolor='rgb(255,255,255)',
                gridcolor='rgb(255,255,255)',
                tickfont=dict(
                    size=14,
                    color='rgb(255,255,255)',
                ),
            ),
            margin=dict(t=70, r=40, l=40, b=40)
        )
    }
    return figure

@app.callback(
    Output('arizona_zip_map','figure'),
    [Input('dropdown-plan', 'value'),
     Input('slider-age', 'value'),
     Input('intermediate-value', 'children'),
     Input('arizona_county_map','clickData')])
def arizona_zip_map(plan, age, df, clickData):
    # if county == None:
    #     county = base_drop_down_options()[5]
    # if plan == None:
    #     plan = base_drop_down_options()[4]
    if clickData is not None:
        clickData = clickData['points'][0]['curveNumber']
        find_county = pd.read_csv('converted_counties.csv')
        county = find_county.iloc[clickData,1]
        print(county)
    else:
        county = 'Maricopa County'
        print(county)

    df = pd.read_json(df, orient='split')
    df = df[df.count_of_match != 0]
    df = df[df['County'].isin([county])]
    df = df[df['Plan'].isin(plan)]
    if age is not None:
        df = df[(df['Age'] >= age[0]) & (df['Age'] <= age[1])]

    sum_df_to_merge = df.groupby('BillingZipCode')['count_of_match'].sum()
    sum_df_to_merge = sum_df_to_merge.to_frame()
    shape_file = pd.read_csv('az_state_new.csv')

    final_df = pd.merge(shape_file, sum_df_to_merge, on='BillingZipCode', how='right')
    index_list = final_df.index
    final_df['count_of_match'] = final_df['count_of_match'].fillna(0)


    vmin, vmax = final_df['count_of_match'].min(), final_df['count_of_match'].max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.get_cmap('Blues')
    #median = final_df['count_of_match'].median

    cdata = []
    for index in index_list:
        x = final_df['x_cords'][index]
        y = final_df['y_cords'][index]
        x = x.split(',')
        y = y.split(',')

        fill_color = 'rgb' + str(cmap(norm(final_df['count_of_match'][index]))[0:3])

        # equip_num = map_df['EQUIP_NO'][index]
        # status_code = map_df['EQUIP_STATUS_NM'][index]
        # ng_num = map_df['NG_NUM'][index]

        try:
            text_infor = "Days: " + str(final_df['count_of_match'][index]) + '<br>' + \
                         "Zip Code: " + str(final_df['BillingZipCode'][index]) + '<br>'

        except TypeError:
            text_infor = "No Data"

        region = go.Scatter(  # choropleth trace
            showlegend=False,
            mode='lines',
            line=dict(color='rgb(255,255,255)', width=0.75),
            x=x,
            y=y,
            text=text_infor,
            marker=dict(
                color='white'),
            fill='toself',
            fillcolor=fill_color,
            name="")
        # hoverinfo=text_infor

        cdata.append(region)
    figure = {
        'data': cdata,

        "layout": go.Layout(
            showlegend=False,
            height=640,
            # length=1200,
            bargap=.3,
            #title=('GA01 As of ' + datetime.datetime.now().strftime('%I:%M:%S')),
            margin=dict(t=20, r=20, l=20, b=20, pad=0),
            #titlefont=dict(
             #   size=24,
             #   color='rgb(255,255,255)'
            #),
            legend=dict(
                x=.9,
                y=1,
                traceorder='normal',
                font=dict(
                    size=14,
                    color='rgb(255,255,255)'),
                bgcolor='rgb(0,0,0)',
                bordercolor='#FFFFFF',
                orientation="h"),

            plot_bgcolor='#1f2630',
            paper_bgcolor='#1f2630',
            xaxis=dict(
                linecolor='rgb(255,255,255)',
                tickangle=65,
                tickfont=dict(
                    size=14,
                    color='rgb(255,255,255)',
                ),
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                gridcolor='rgb(255,255,255)',
                titlefont=dict(
                    size=18,
                    color='rgb(255,255,255)'
                )
            ),
            yaxis=dict(
                linecolor='rgb(255,255,255)',
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                gridcolor='rgb(255,255,255)',
                tickfont=dict(
                    size=14,
                    color='rgb(255,255,255)',
                ),
                titlefont=dict(
                    size=18,
                    color='rgb(255,255,255)'
                ),
            ),
            # yaxis2=dict(
            #     overlaying = 'y',
            #     side='right',
            #     linecolor='rgb(255,255,255)',
            #     tickfont=dict(
            #         size=14,
            #         color='rgb(255,255,255)',
            #     ),
            #     showgrid=False,
            #     gridcolor='rgb(255,255,255)',
            #     titlefont=dict(
            #         size=18,
            #         color='rgb(255,255,255)'
            #     )
            # )
        )
    }
    return figure


@app.callback(
    Output('arizona_county_map','figure'),
    [Input('checklist-county', 'value'),
     Input('group-by', 'value'),
     Input('dropdown-plan', 'value'),
     Input('slider-age', 'value'),
     Input('intermediate-value', 'children')])
def arizona_county_map(county, groupby, plan, age, df):

    df = pd.read_json(df, orient='split')
    df = df[df.count_of_match != 0]
    df = df[df['County'].isin(county)]
    df = df[df['Plan'].isin(plan)]
    if age is not None:
        df = df[(df['Age'] >= age[0]) & (df['Age'] <= age[1])]

    sum_df_to_merge = df.groupby('County')['count_of_match'].sum()
    sum_df_to_merge = sum_df_to_merge.to_frame()
    shape_file = pd.read_csv('converted_counties.csv')

    final_df = pd.merge(shape_file, sum_df_to_merge, on='County', how='left')

    index_list = final_df.index
    final_df['count_of_match'] = final_df['count_of_match'].fillna(0)


    vmin, vmax = final_df['count_of_match'].min(), final_df['count_of_match'].max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.get_cmap('Blues')
    #median = final_df['count_of_match'].median

    cdata = []
    for index in index_list:
        x = final_df['x_cords'][index]
        y = final_df['y_cords'][index]
        x = x.split(',')
        y = y.split(',')

        fill_color = 'rgb' + str(cmap(norm(final_df['count_of_match'][index]))[0:3])

        try:
            text_infor = "Days: " + str(final_df['count_of_match'][index]) + '<br>' + \
                         "County: " + str(final_df['County'][index]) + '<br>'

        except TypeError:
            text_infor = "No Data"

        region = go.Scatter(  # choropleth trace
            showlegend=False,
            mode='lines',
            line=dict(color='rgb(255,255,255)', width=0.75),
            x=x,
            y=y,
            text=text_infor,
            marker=dict(
                color='white'),
            fill='toself',
            #hoveron='fills',
            fillcolor=fill_color,
            name="")
        # hoverinfo=text_infor

        cdata.append(region)
    figure = {
        'data': cdata,

        "layout": go.Layout(
            clickmode='event+select',
            showlegend=True,
            height=640,
            # length=1200,
            bargap=.3,
            #title=('GA01 As of ' + datetime.datetime.now().strftime('%I:%M:%S')),
            margin=dict(t=20, r=20, l=20, b=20, pad=0),
            #titlefont=dict(
             #   size=24,
             #   color='rgb(255,255,255)'
            #),
            legend=dict(
                x=.9,
                y=1,
                traceorder='normal',
                font=dict(
                    size=14,
                    color='rgb(255,255,255)'),
                bgcolor='rgb(146,46,63)',
                bordercolor='#FFFFFF',
                orientation="h"),

            plot_bgcolor='#1f2630',
            paper_bgcolor='#1f2630',
            xaxis=dict(
                linecolor='rgb(255,255,255)',
                tickangle=65,
                tickfont=dict(
                    size=14,
                    color='rgb(255,255,255)',
                ),
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                gridcolor='rgb(255,255,255)',
                titlefont=dict(
                    size=18,
                    color='rgb(255,255,255)'
                )
            ),
            yaxis=dict(
                linecolor='rgb(255,255,255)',
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                gridcolor='rgb(255,255,255)',
                tickfont=dict(
                    size=14,
                    color='rgb(255,255,255)',
                ),
                titlefont=dict(
                    size=18,
                    color='rgb(255,255,255)'
                ),
            ),
            # yaxis2=dict(
            #     overlaying = 'y',
            #     side='right',
            #     linecolor='rgb(255,255,255)',
            #     tickfont=dict(
            #         size=14,
            #         color='rgb(255,255,255)',
            #     ),
            #     showgrid=False,
            #     gridcolor='rgb(255,255,255)',
            #     titlefont=dict(
            #         size=18,
            #         color='rgb(255,255,255)'
            #     )
            # )
        )
    }
    return figure

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True,) #dev_tools_ui=False)
