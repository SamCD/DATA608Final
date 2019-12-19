intro = 'This project is meant to address dual issues that have been in the news, especially local news, \
often as of late. On the one hand is the issue if school choice and zoning, and on the other hand the issue of \
standardized testing for elite high schools (i.e. SHSAT). \
\
Using open data from NYC.gov, this project attempts to gain more insight into the specialized admissions process \
through data aggregation and visualization. Notably, it is meant to give support to the possibility of expanding the \
specialized high school system within NYC by highlighting the different types of possible specializations as well as \
identify underserved areas of the city that would benefit from an expansion of this system. '

import csv
import pandas as pd
import numpy as np
from itertools import chain
import plotly.express as px
import geopy.distance
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.offline as pyo
pyo.init_notebook_mode()

"""
#Data Acquisition

In this data set, many fields are broken up into smaller sections, I am guessing for efficiency on the provider's end, however I am going to aggregate the similar columns for data anlysis purposes

Here I also identify base fields that can be used, such as dbn, which is used as a primary key, and latitude and longitude for mapping purposes.
"""

baseCols = [
'dbn'
,'school_name'
,'borocode'
,'overview_paragraph'
,'borough'
,'latitude'
,'longitude'
,'bin'
,'bbl'
,'nta'
,'total_students'
,'neighborhood'
]

specialCols = [
'academicopportunities1'
,'academicopportunities2'
,'academicopportunities3'
,'academicopportunities4'
,'academicopportunities5'
,'academicopportunities6'
,'advancedplacement_courses'
,'psal_sports_boys'
,'psal_sports_girls'
,'psal_sports_coed'
,'school_sports'
,'interest1'
,'interest2'
,'interest3'
,'interest4'
,'interest5'
,'interest6'
,'interest7'
,'interest8'
,'interest9'
,'interest10'
,'common_audition1'
,'common_audition2'
,'common_audition3'
,'common_audition4'
,'common_audition5'
,'common_audition6'
,'common_audition7'
,'common_audition8'
,'common_audition9'
,'common_audition10'
,'specialized'
]

df = pd.read_csv('https://data.cityofnewyork.us/resource/23z9-6uk9.csv')

baseDF = df[baseCols] #id, name and location of high schools

MSDirDF = pd.read_csv('https://data.cityofnewyork.us/resource/uqxv-h2se.csv')
MSDirDF = MSDirDF[['schooldbn','latitude','longitude']] #id and location of middle schools (additional info in feeder data)

"""
In this step, I am coalescing similar columns into one for efficiency and later processing, as well as creating a subset of specialized schools, using the specialized flag

#Data Preparation and Cleansing
"""

specDF = df[baseCols + specialCols]

specDF['academicopportunities'] = specDF['academicopportunities1'].replace(np.nan, '', regex=True).astype(str)\
+ ',' + specDF['academicopportunities2'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['academicopportunities3'].replace(np.nan, '', regex=True).astype(str) \
+ ','+ specDF['academicopportunities4'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['academicopportunities5'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['academicopportunities6'].replace(np.nan, '', regex=True).astype(str)

specDF['interests'] = specDF['interest1'].replace(np.nan, '', regex=True).astype(str)\
+ ',' + specDF['interest2'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['interest3'].replace(np.nan, '', regex=True).astype(str) \
+ ','+ specDF['interest4'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['interest5'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['interest6'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['interest7'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['interest8'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['interest9'].replace(np.nan, '', regex=True).astype(str)\
+ ','+ specDF['interest10'].replace(np.nan, '', regex=True).astype(str)

specDF = specDF.drop(['academicopportunities1'
                      ,'academicopportunities2'
                      ,'academicopportunities3'
                      ,'academicopportunities4'
                      ,'academicopportunities5'
                      ,'academicopportunities6'
                     ,'interest1'
                     ,'interest2'
                     ,'interest3'
                     ,'interest4'
                     ,'interest5'
                     ,'interest6'
                     ,'interest7'
                     ,'interest8'
                     ,'interest9'
                     ,'interest10']
                     ,axis=1)

specSchools = specDF[specDF['specialized'].notna() == True][['school_name'
                                                             ,'overview_paragraph'
                                                             ,'borough'
                                                             ,'latitude'
                                                             ,'longitude'
                                                             ,'total_students'
                                                            ,'neighborhood']]
specSchools['total_students'] = pd.to_numeric(specSchools['total_students'])

"""
In this section, I am defining a few functions for data processing. The first function takes a field of comma-separated values and expands them into distinct records.
"""

#https://stackoverflow.com/questions/50731229/split-cell-into-multiple-rows-in-pandas-dataframe

# return list from series of comma-separated strings
def chainer(s):
    return list(chain.from_iterable(s.str.split(',')))

# calculate lengths of splits
lens = specDF['interests'].str.split(',').map(len)

# create new dataframe, repeating or chaining as appropriate
df3 = pd.DataFrame({'school_name': np.repeat(specDF['school_name'], lens),
                    'nta': np.repeat(specDF['nta'], lens),
                    'interests': chainer(specDF['interests'])
                   }
                  )

#df3.head(25)
dfg3 = df3.groupby('interests').filter(lambda x: len(x) > 1)
dfg3 = dfg3.groupby('interests').size()\
                             .reset_index(name='count') \
                             .sort_values(['count'], ascending=False)

dfg3.replace('', np.nan, inplace=True)
dfg3.dropna(subset=['interests'], inplace=True)

#This function is used to create a cartesian product between feeder schools and specialized schools for comparison.

#https://stackoverflow.com/questions/53699012/performant-cartesian-product-cross-join-with-pandas
def cartesian_product_basic(left, right):
    return (
       left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))
	   
par1 = 'Specifically, we want to look at the distance between feeder schools and specialized schools, \
including getting data on the closest specialized school to a feeder school. Also doing some data manipulation \
to create numerical data. Offer count and tester count both had a bucket for 0-5. We want to be able to use them in \
visuals, and also to calculate an acceptance rate. Use a random integer between 0 and 5, then do some correcting to make \
sure that there are no records with offers > testers.'

dfSHSAT = pd.read_csv('https://data.cityofnewyork.us/resource/uf53-ree9.csv')

dfSHSAT=MSDirDF.set_index('schooldbn').join(dfSHSAT.set_index('feeder_school_dbn')).dropna()

dfSHSAT.loc[dfSHSAT['number_of_offers'] == '0-5','number_of_offers'] = \
dfSHSAT['number_of_offers'].apply(lambda x: np.random.randint(0,5))

dfSHSAT.loc[dfSHSAT['count_of_testers'] == '0-5','count_of_testers'] = \
dfSHSAT['count_of_testers'].apply(lambda x: np.random.randint(0,5))

dfSHSAT['number_of_offers'] = dfSHSAT['number_of_offers'].astype('int32')
dfSHSAT['count_of_testers'] = dfSHSAT['count_of_testers'].astype('int32')

dfSHSAT[dfSHSAT['number_of_offers'] > dfSHSAT['count_of_testers']]['number_of_offers'] = \
dfSHSAT[dfSHSAT['number_of_offers'] > dfSHSAT['count_of_testers']]['count_of_testers']

feederDF = cartesian_product_basic(dfSHSAT,specSchools[['school_name','latitude','longitude']])

feederDF['distance'] = [
    geopy.distance.distance((a, b), (c, d)).km
    for a, b, c, d in feederDF[['latitude_x', 'longitude_x', 'latitude_y', 'longitude_y']].values
]

feederDF['acceptance_rate'] = feederDF['number_of_offers']/feederDF['count_of_testers']
feederDF['acceptance_rate'] = feederDF['acceptance_rate'].replace(np.inf, 0).replace(np.NaN, 0)

feed2 = feederDF.groupby(['feeder_school_name'
                          , 'count_of_testers'
                          , 'number_of_offers'
                          , 'latitude_x'
                          , 'longitude_x'
                          , 'acceptance_rate'
                         ], as_index=False)['distance'].min()

#Using MapBox (via Plotly plugin) for city-focused geo visuals. Default Plotly maps are only available at more macro levels (globe, country)

MBToken = 'pk.eyJ1Ijoic2NvaGVuZGUiLCJhIjoiY2szemMxczZoMXJhajNrcGRsM3cxdGdibiJ9.2oazpPgLvgJGF9EBOYa9Wg'
px.set_mapbox_access_token(MBToken)

specText = 'Below is a map of all schools marked as "Specialized", broken up by borough (color). The size of the circle \
denotes enrollment size. The four largest circles are the oldest specialized schools: Stuyvesant, Bronx Science and Brooklyn Tech, \
which specialize in STEM; and Laguardia, which specializes in the arts). These are also the oldest specialized schools. The \
smaller circles are relative newcomers to the specialized admissions process, and show that an expansion of the system is \
already underway'

specFig = px.scatter_mapbox(specSchools.dropna()
                        , lat="latitude"
                        , lon="longitude"
                        , color="borough"
                        , size="total_students"
                        , text="school_name"
                        , hover_name="school_name"
                        , hover_data=["neighborhood"]
                        , size_max=15
                        , zoom=9
                       )

intText = 'Taking a look at the breakdown in popular interests in non-specialized schools. Science/Tech takes up a lot of \
space, but there is a lot of interest in Humanities. As of now, there are no schools which specialize in social sciences.'

intFig = go.Figure(data=[go.Pie(labels=dfg3['interests'], values=dfg3['count'])])

distText = 'Scatter showing min distance from any specialized school using color gradient, and number of specialized test \
takers as marker size. The large, dark circles indicate a school with a large number of students interested in attending \
one of the specialized schools, but that are situated inconveniently far away.'

distFig = px.scatter_mapbox(feed2.dropna()
                        , lat="latitude_x"
                        , lon="longitude_x"
                        , color="distance"
                        , size="count_of_testers"
                        , color_continuous_scale='Greens'#px.colors.cyclical.IceFire
                        , size_max=15
                        , zoom=10
                       )

trafficText1 = 'Here we are using scattermapbox lines as a sort of traffic map, with line thickness determined by the number of test \
takers (which in theory indicates an interest in attending a specialized school). A large circle denotes a high acceptance rate. \
To reduce clutter, we are only looking at the 40 most remote schools.'

fig2 = go.Figure()

stuyCoord = {'lon': feederDF[feederDF['school_name'] == 'Stuyvesant High School']['longitude_y'].iloc[1]
            ,'lat': feederDF[feederDF['school_name'] == 'Stuyvesant High School']['latitude_y'].iloc[1]
            }

stuyDF = feederDF[feederDF['school_name'] == 'Stuyvesant High School'].sort_values('distance',ascending = False).head(40)
stuyDF2 = feederDF[feederDF['school_name'] == 'Stuyvesant High School'].sort_values('distance',ascending = True).head(40)

for i in range(len(stuyDF)):
    fig2.add_trace(go.Scattermapbox(
        mode = "markers+lines",
        lon = [stuyDF.iloc[i]['longitude_x'],stuyDF.iloc[i]['longitude_y']],
        lat = [stuyDF.iloc[i]['latitude_x'],stuyDF.iloc[i]['latitude_y']],
        line = {'width': stuyDF.iloc[i]['count_of_testers']/20.0},
        marker = {'size': stuyDF.iloc[i]['acceptance_rate']*30.0},
        hovertext = stuyDF.iloc[i]['feeder_school_name'] + ' <-> ' + stuyDF.iloc[i]['school_name'],
        #legendgroup = feederDF.iloc[i]['school_name']
        showlegend = False
    )
                  )

fig2.update_layout(
    margin ={'l':0,'t':0,'b':0,'r':0},
    mapbox = {
        'center': stuyCoord,
        'style': "stamen-terrain",
        #'center': {'lon': stuyDF['longitude_x'].mean(), 'lat': stuyDF['latitude_x'].mean()},
        'zoom': 11})

trafficText2 = 'Doing the same for Laguardia High School for the arts, although it is unclear from this data whether \
"test takers" includes auditioners.'

fig3 = go.Figure()

lagCoord = {'lon': feederDF[feederDF['school_name'] == 'Fiorello H. LaGuardia High School of Music & Art and Performing Arts']['longitude_y'].iloc[1]
            ,'lat': feederDF[feederDF['school_name'] == 'Fiorello H. LaGuardia High School of Music & Art and Performing Arts']['latitude_y'].iloc[1]
            }

lagDF = feederDF[feederDF['school_name'] == 'Fiorello H. LaGuardia High School of Music & Art and Performing Arts'].sort_values('distance',ascending = False).head(40)
lagDF2 = feederDF[feederDF['school_name'] == 'Fiorello H. LaGuardia High School of Music & Art and Performing Arts'].sort_values('distance',ascending = True).head(40)

for i in range(len(lagDF)):
    fig3.add_trace(go.Scattermapbox(
        mode = "markers+lines",
        lon = [lagDF.iloc[i]['longitude_x'],lagDF.iloc[i]['longitude_y']],
        lat = [lagDF.iloc[i]['latitude_x'],lagDF.iloc[i]['latitude_y']],
        line = {'width': lagDF.iloc[i]['count_of_testers']/20.0},
        marker = {'size': lagDF.iloc[i]['acceptance_rate']*30.0},
        hovertext = lagDF.iloc[i]['feeder_school_name'] + ' <-> ' + lagDF.iloc[i]['school_name'],
        #legendgroup = feederDF.iloc[i]['school_name']
        showlegend = False
    )
                  )

fig3.update_layout(
    margin ={'l':0,'t':0,'b':0,'r':0},
    mapbox = {
        'center': lagCoord,
        'style': "stamen-terrain",
        #'center': {'lon': stuyDF['longitude_x'].mean(), 'lat': stuyDF['latitude_x'].mean()},
        'zoom': 11})

app = dash.Dash()

app.layout = html.Div([
    html.Div([html.H1('NYC Specialized Schools')], style={'textAlign': 'center'}),
    html.Div([html.H2('Sam Cohen-Devries, CUNY SPS DATA608, Fall 2019')], style={'textAlign': 'left'}),
    html.Div(children=intro),
    html.Div(children=specText),
    dcc.Graph(children=specFig),
    html.Div(children=intText),
    dcc.Graph(figure=intFig),
    html.Div(children=distText),
    dcc.Graph(figure=distFig),
    html.Div(children=trafficText1),
    dcc.Graph(figure=fig2),
    html.Div(children=trafficText2),
    dcc.Graph(figure=fig3),
])


if __name__ == '__main__':
    app.run_server(debug=False)