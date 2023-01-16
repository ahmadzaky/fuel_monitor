import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import time
from plotly.subplots import make_subplots


st.set_page_config(layout="wide")
st.title("Vessel Fuel Monitoring")
st.sidebar.image('assets/LOGO_JAW.png', width=200, use_column_width=False, output_format='PNG')
st.sidebar.title("Vessel Fuel Monitoring Menu")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style_sum.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


selection = [f for f in glob.glob("*/")]
foldername = st.sidebar.selectbox('select folder',selection,1)

mylist = [f for f in glob.glob(foldername+"*.csv")]
filename = st.sidebar.selectbox('select file',mylist,1)

DATA_URL = filename#('0922/001.csv')
DATA_URL_ = ('https://docs.google.com/spreadsheets/d/13xOuT81vLy5T7RE5Fb8UFIOt6IaqP5AfMA5X7v4MfY4/edit#gid=149084843')

colors = {
    'background': '#f5f5f6',
    'bgchart': '#2d3038',
    'bgfig'  : '#2d3038',#'#0091D5',
    'text'   : '#a8a9ac',
    'volt'   : '#ffd15f', 
    'cur'    : '#6ab187',
    'curAvg' : '#ac3e31',
    'RPM'    : '#CCCCCC',
    'temp'   : '#e45756',
    'bmax'   : '#4c78a8',
    'bar'   : '#ffd15f',
    'batt'   : '#f58518',
    
}

colcard_W = 360
colcard_H = 480
colmap_W = 1080
colmap_H = 480
col2_W = 720
col2_H = 280
col3_W = 480
col3_H = 340


np_ncut_count = np.zeros(0)
np_time_max = np.zeros(0)
np_bat_max = np.zeros(0)
np_bat_min = np.zeros(0)
np_amp_max = np.zeros(0)
np_amp_mean = np.zeros(0)
np_amp_q1 = np.zeros(0)
np_amp_q3 = np.zeros(0)
total_cut = 0
total_runtime = 0

for n in mylist :
    df = pd.read_csv(n)
    df['avg_current'] = df.iloc[:,1].rolling(window=50).mean()
    df['running'] = ''
    df['attemp_count'] = ''
    df['attemp'] = ''
    df.loc[df.avg_current <= 500, 'running'] = 0
    df.loc[df.avg_current > 500, 'running'] = 1
    df.loc[df.avg_current <= 500, 'standby'] = 1
    df.loc[df.avg_current > 500, 'standby'] = 0
    df.loc[df["running"].shift() != df["running"], 'attemp'] = 1
    df.loc[df["running"].shift() == df["running"], 'attemp'] = 0
    df.loc[df["running"] == 0, 'attemp'] = 0
    df['attemp_count']  = df["attemp"].cumsum()
    aMax = df['avg_current'].max()
    aMean= df[df['running']==1]['avg_current'].mean(skipna=False)
    aQ1  = df[df['running']==1]['avg_current'].quantile(q=0.25,interpolation='midpoint')
    aQ3  = df[df['running']==1]['avg_current'].quantile(q=0.8,interpolation='midpoint')
    bmax = (((df[df['avg_current']==0]['voltage'].rolling(window=1000).mean().max()-3400)*0.08)+10)
    bmin = (((df[df['avg_current']==0]['voltage'].rolling(window=1000).mean().min()-3400)*0.08)+10)
    np_ncut_count = np.append(np_ncut_count,  df['attemp_count'].max())
    np_bat_max  = np.append(np_bat_max, bmax)    
    np_bat_min  = np.append(np_bat_min, bmin)
    np_amp_max  = np.append(np_amp_max, aMax)
    np_amp_mean = np.append(np_amp_mean, aMean)
    np_amp_q1   = np.append(np_amp_q1, aQ1)
    np_amp_q3   = np.append(np_amp_q3, aQ3)
    np_time_max = np.append(np_time_max,  df['time'].max())

total_cut = np.sum(np_ncut_count)
total_runtime = np.sum(np_time_max) 

aMean = list(np_amp_mean)
aMax = list(np_amp_max)
aQ1  = list(np_amp_q1)
aQ3  = list(np_amp_q3)
bmax = list(np_bat_max)
bmin = list(np_bat_min)
ncut = list(np_ncut_count)

traceSummary = go.Bar(x=mylist, 
                    y=ncut,
                    marker_color=colors['bar'],
                    name='Attemp',
                    )



#labels = ["Cutting Attemp", "Pro"]
#traceTotalCut = go.Pie(label = labels, value = [total_cut, 0]

go.Bar(x=mylist, 
                    y=ncut,
                    marker_color=colors['bar'],
                    name='Attemp',
                    )

traceBmax = go.Bar(x=mylist, 
                    y=bmax,
                    marker_color='rgb(26, 118, 255)',
                    name='Capacity Start',
                    )


traceBmin = go.Bar(x=mylist, 
                    y=bmin,
                    marker_color='rgb(55, 83, 109)',
                    name='Capacity End',
                    )

layout0 = go.Layout(font=dict(family='Arial',color=colors['text']),
                    yaxis = dict(title = dict(text='Number of Cutting Attemp',font=dict(family='Arial',color=colors['text']))),
                    paper_bgcolor=colors['bgfig'],
                    plot_bgcolor= colors['bgchart'],
                    hovermode ='x',
                    barmode='overlay',
                    bargap=0.2,
                    width=col2_W,
                    height=col2_H
)


layout1 = go.Layout(font=dict(family='Arial',color=colors['text']),
                    yaxis = dict(title = dict(text='Battery Capacity (%)',font=dict(family='Arial',color=colors['text']))),
                    paper_bgcolor=colors['bgfig'],
                    plot_bgcolor= colors['bgchart'],
                    hovermode ='x',
                    barmode='overlay',
                    bargap=0.2,
                    width=col2_W,
                    height=col2_H
                    )

figcard = go.Figure()

figcard.add_trace(go.Indicator(
    mode = "number+delta",
    value = total_cut,    
    title = {"text": "Total Cuts<br><span style='font-size:0.8em;color:gray'>"},
    title_font_color = 'grey',
    number_font = dict(color = 'white'),
    domain = {'x': [0, 0.5], 'y': [0.7, 1]},
    delta = {'reference': 1705, 'relative': True, 'position' : "bottom"}))

figcard.add_trace(go.Indicator(
    mode = "number+delta",
    value = total_cut/4,
    title = {"text": "Number of Tree<br><span style='font-size:0.8em;color:gray'>"},
    title_font_color = 'grey',
    number_font = dict(color = 'white'),
    delta = {'reference': 400, 'relative': True, 'position' : "bottom"},
    domain = {'x': [0, 0.5], 'y': [0.4, 0.6]}))

figcard.add_trace(go.Indicator(
    mode = "number",
    title_font_size = 20,
    title_font_color = 'grey',
    title = {"text": "Total Running Time"},
    delta = {'reference': 400, 'relative': True},
    number_font = dict(size = 6),
    domain = {'x': [0, 0.5], 'y': [0.2, 0.3]}))
    

figcard.add_trace(go.Indicator(
    mode = "number",
    title_font_size = 40,
    title_font_color = 'white',
    title = dict( text = time.strftime('%H:%M:%S',time.gmtime(total_runtime/1000))),
    delta = {'reference': 400, 'relative': True},
    number_font = dict(size = 6),
    domain = {'x': [0, 0.5], 'y': [0, 0.1]}))

figcard.update_layout(paper_bgcolor = colors['bgfig'],    width=colcard_W, height=colcard_H)

#fig.show()

#fig_sum = go.Figure([traceSummary],layout0);
fig_sum_total = go.Figure([traceSummary],layout0);

fig_sum1 = go.Figure([traceBmax,traceBmin],layout1);
fig_sum1_total = go.Figure([traceBmax,traceBmin],layout1);





df = pd.read_csv(DATA_URL)

traceCurrent = go.Scatter(x=df['time'], 
                    y=df['current'],
                    name='Current',
                    fill='tozeroy',
                    line=dict(color=colors['cur']))

             
df['avg_current'] = df.iloc[:,1].rolling(window=50).mean()


df['running'] = ''
df['attemp_count'] = ''
df['attemp'] = ''
df.loc[df.avg_current <= 500, 'running'] = 0
df.loc[df.avg_current > 500, 'running'] = 1
df.loc[df.avg_current <= 500, 'standby'] = 1
df.loc[df.avg_current > 500, 'standby'] = 0


df.loc[df["running"].shift() != df["running"], 'attemp'] = 1
df.loc[df["running"].shift() == df["running"], 'attemp'] = 0
df.loc[df["running"] == 0, 'attemp'] = 0

df['attemp_count']  = df["attemp"].cumsum()
run_count  = np.ones(shape=(1,df[df["running"]==1].shape[0])).sum() 
sb_count  = np.ones(shape=(1,df[df["running"]==0].shape[0])).sum()



traceACurrent = go.Scatter(x=df['time']/60000,
                    y=df['avg_current'],
                    name='AVG Current',
                    line=dict(color=colors['curAvg'],width=2))
                    
                    

traceACurrentOn = go.Scatter(x=df['time']/60000,
                    y=df['running'],
                    name='AVG Current',
                    line=dict(color=colors['volt'],width=2))

traceVoltage = go.Scatter(x=df['time']/60000,
                          y=df[df['avg_current']==0]['voltage'].rolling(window=1000).mean()*0.934,
                          name='Voltage',
                          fill='tozeroy',
                          line=dict(color=colors['volt']))
                   

traceBatteryCapacity = go.Box(x=((((df[df['avg_current']==0]['voltage'].rolling(window=10000).mean())-3400)*0.08)+10), name = 'Battery level',marker_color = 'rgb(26, 118, 255)')

dfTav = df.iloc[:,3].rolling(window=100).mean()       

dfRun = df[df['rpm']>0]  

dfTav = df.iloc[:,3].rolling(window=100).mean()       
 

layout0 = go.Layout(title=dict(text='Battery Capacity',font=dict(family='Arial',color=colors['text'])),
                    font=dict(family='Arial',color=colors['text']),
                    #yaxis = dict(title = dict(text='time(minutes)',font=dict(family='Arial',color=colors['text']))),
                    xaxis = dict(title = dict(text='Capacity Distribution (%)',font=dict(family='Arial',color=colors['text']))),
                    paper_bgcolor=colors['bgfig'],
                    plot_bgcolor= colors['bgchart'],
                    hovermode ='x',
                    bargap=0, height=col3_H, width=col3_W)
                    

fig_battery = go.Figure([traceBatteryCapacity],layout0);




layoutCurrent = go.Layout(title=dict(text='Current Graph',font=dict(family='Arial',color=colors['text'])),
                    font=dict(family='Arial',color=colors['text']),
                    xaxis = dict(title = dict(text='Minute',font=dict(family='Arial',color=colors['text']))),
                    yaxis = dict(title = dict(text='Amp(mA)',font=dict(family='Arial',color=colors['text']))),
                    paper_bgcolor=colors['bgfig'],
                    plot_bgcolor= colors['bgchart'],
                    hovermode ='closest',   height=col2_H, width=col2_W)

layoutVoltage = go.Layout(title=dict(text='Voltage Graph',font=dict(family='Arial',color=colors['text'])),
                    font=dict(family='Arial',color=colors['text']),
                    xaxis = dict(title = dict(text='Minute',font=dict(family='Arial',color=colors['text']))),
                    yaxis = dict(title = dict(text='Voltage(mV)',font=dict(family='Arial',color=colors['text']))),
                    paper_bgcolor=colors['bgfig'],
                    plot_bgcolor= colors['bgchart'],
                    hovermode ='closest',   height=col3_H, width=col3_W)
        

fig_current = go.Figure([traceACurrent],layoutCurrent)
fig_current.update_xaxes(rangeslider_visible=True)
        
fig_volt = go.Figure([traceVoltage],layoutVoltage)
fig_volt.update_xaxes(rangeslider_visible=True)



layout3 = go.Layout(title=dict(text='Cutting Intensity and Temperature Comparation Graph',xanchor='auto',font=dict(family='Arial',color=colors['text'])),
                    font=dict(family='Arial',color=colors['text']),
                    xaxis = dict(title = dict(text='Minute',font=dict(family='Arial',color=colors['text']))),
                    yaxis = dict(title = 'Temperature (C)'),
                    paper_bgcolor=colors['bgfig'],
                    plot_bgcolor= colors['bgchart'],
                    hovermode ='x',  height=col3_H, width=col3_W)
                    
fig_temperature = go.Figure([go.Scatter(x=df['time']/60000,y=df["running"]*80,name='Cutting',fill='tozeroy',line=dict(color=colors['bmax'])), 
                  go.Scatter(x=df['time']/60000,y=dfTav/10,name='Temperature',fill='tozeroy',line=dict(color=colors['temp'])),
                  ],layout3)
            


fig_temperature.update_xaxes(rangeslider_visible=True)

layoutCurrentDistribution = go.Layout(title=dict(text='Current Consumption Distribution',xanchor='auto',font=dict(family='Arial',color=colors['text'])),
                    font=dict(family='Arial',color=colors['text']),
                    yaxis = dict(title = 'Current (mA)'),
                    paper_bgcolor=colors['bgfig'],
                    plot_bgcolor= colors['bgchart'],
                    hovermode ='x',  height=500, width=1200)

fig_CurrentDistribution = go.Figure([go.Scatter(x=mylist,y=aMax,name='Max',fill = 'none',line=dict(shape="spline",color=colors['curAvg'])),
                  go.Scatter(x=mylist,y=aMean,name='Mean',line=dict(shape="spline",color=colors['volt'])), 
                  go.Scatter(x=mylist,y=aQ1,name='Q1',fill='tozeroy',line=dict(shape="spline",color="#444444")),
                  go.Scatter(x=mylist,y=aQ3,name='Q3',fill='tonexty',line=dict(shape="spline",color="#6ece58")),
                  ],layoutCurrentDistribution)




piecolor = ['#1f9e89','#6ece58','#F0F921']
labels = ["Stand By", "Motor Running", "Cutting"]


# Create subplots: use 'domain' type for Pie subplot
fig_activity = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig_activity.add_trace(go.Pie(labels=labels, values=[sb_count, run_count, 0], name="GHG Emissions"), 1, 1)
fig_activity.add_trace(go.Pie(labels=labels, values=[0,100,0], name="Cutting Attemp"), 1, 2)

# Use `hole` to create a donut-like pie chart
fig_activity.update_traces(hole=.4, hoverinfo="label+percent+name",marker=dict(colors=piecolor, line=dict(color='#000000', width=0)))

fig_activity.update_layout(
    title_text="Activity Trace",
    font=dict(family='Arial',color=colors['text']),
    plot_bgcolor='red',
    paper_bgcolor=colors['bgfig'],
    height=col2_H, width=col2_W,
    annotations=[dict(text= ("Total Duration: " + time.strftime('%H:%M:%S',time.gmtime((df["time"].max())/1000))), x=0.01, y=-0.5, font_size=16, showarrow=False),
                 dict(text= "Cutting attemp: " + str(df['attemp_count'].max()-1)+ "/"+ str(total_cut) , x=0.98, y=-0.5, font_size=16, showarrow=False)])
                 

         



us_cities = pd.read_csv("maps.csv")

fig_map_main = px.line_mapbox(us_cities, lat="lat", lon="lon", hover_name="Vessel", hover_data=["State", "Production"],
                         color="Vessel",
                       zoom=5,  height=colmap_H, width=colmap_W)
                       
fig_map_main.add_trace = px.scatter_mapbox(us_cities, lat="lat1", lon="lon1", hover_name="Vessel", hover_data=["State", "Production"],
                         color="Vessel",size="Production",size_max=30,
                       zoom=5,  height=colmap_H, width=colmap_W)
fig_map_main.update_layout(mapbox_style="open-street-map")
fig_map_main.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


fig_map = px.scatter_mapbox(us_cities, lat="lat", lon="lon", hover_name="Vessel", hover_data=["State", "Production"],
                        color_discrete_sequence=["fuchsia"], color="Production", size="Production",
                        color_continuous_scale="Bluered",size_max=30,zoom=5,  height=col2_H, width=col2_W)
fig_map.update_layout(mapbox_style="open-street-map")
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



# LAYOUT ARRANGEMENT



col1, col2 = st.columns((1,3))
col1.header("Activity Summary")
col1.plotly_chart(figcard)
col2.header("Harvesting Distribution Map")
col2.plotly_chart(fig_map_main)


col1, col2 = st.columns(2)
col1.header("Daily Productivity")
col1.plotly_chart(fig_sum_total)
col2.header("Battery Consumption")
col2.plotly_chart(fig_sum1_total)

st.plotly_chart(fig_CurrentDistribution)
st.header(" ")
st.subheader("Detail information for: " + filename)


col1, col2 = st.columns(2)
col1.plotly_chart(fig_activity)
col2.plotly_chart(fig_current)


col1, col2, col3 = st.columns([1,1,1])
col1.plotly_chart(fig_battery)
col2.plotly_chart(fig_volt)
col3.plotly_chart(fig_temperature)

