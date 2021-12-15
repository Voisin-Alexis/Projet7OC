import dash
from dash import dcc
from dash import html
import dash_split_pane
import plotly.express as px
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns

#!pip install dash_daq
import dash_daq as daq

import shap
import sklearn
from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error, accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, fbeta_score, confusion_matrix, auc

import plotly.figure_factory as ff

import joblib
from joblib import load

import base64
from PIL import Image

from shap.plots._force_matplotlib import draw_additive_plot
from dash import dash_table

import flask
from flask import Flask, render_template, jsonify
import json
import requests
from gevent.pywsgi import WSGIServer

cheminFichierJoblib = './fichierJoblib/'

#X_test = joblib.load(cheminFichierJoblib + 'X_test2.joblib')
y_test = joblib.load(cheminFichierJoblib + 'y_test2.joblib')
#target_train = joblib.load(cheminFichierJoblib + 'target_train.joblib')

X_testID2 = joblib.load(cheminFichierJoblib + 'X_testID2.joblib')
dataframeInfoXTest2 = joblib.load(cheminFichierJoblib + 'dataframeInfoXTest2.joblib')
listeIndexSort = joblib.load(cheminFichierJoblib + 'listeIndexSort.joblib')
dfIdClientIndex = joblib.load(cheminFichierJoblib + 'dfIdClientIndex.joblib')

#dataframeInfoXTest = joblib.load(cheminFichierJoblib + 'dataframeInfoXTest.joblib')

predictionStreamlit = joblib.load(cheminFichierJoblib + 'predictionStreamlit.joblib')
predictionProbaStreamlit = joblib.load(cheminFichierJoblib + 'predictionProbaStreamlit.joblib')
y_predProba_lgbmHPSeuil = joblib.load(cheminFichierJoblib + 'y_predProba_lgbmHPSeuil.joblib')

#lgbmHPSeuil = joblib.load(cheminFichierJoblib + 'lgbmHPSeuil.joblib')

image_filenameG = 'featureglobal.png'
encoded_imageG = base64.b64encode(open(image_filenameG, 'rb').read())

image_filename = 'featurelocal.png'

def ouvertureImageFilename(image_filename):    
    with open(image_filename, 'rb') as f: image = f.read()    
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

fpr, tpr, thresholds = roc_curve(y_test, y_predProba_lgbmHPSeuil)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = 1 - thresholds[optimal_idx]

print("Le seuil optimal est égal à", round(optimal_threshold, 3))

app = dash.Dash(__name__)
app.title = "Dashboard Projet 7" #Assigning title to be displayed on tab
server = app.server

app.layout = html.Div(
    children=[
        html.Div(
            children = [
                 html.H1(
                     children='Dashboard pour le Projet 7: Implémentez un modèle de scoring', 
                     style={'color': '#FFFFFF',
                            'fontSize': '48px',
                            'font-weight': 'bold',
                            'text-align': 'center',
                            'margin': '0 auto'},
                 ),
                 html.P(
                    children='Pour ce projet nous avons étudié différents modèles, ' 
                             'tous basés sur les arbres de décision qui donnent de meilleurs résultats '
                             'de manière générale.',
                    style={'color': '#CFCFCF',                          
                           'max-width': '500px',
                           'text-align': 'center',
                           'margin': '4px auto'},
                ),
            ],
             style={'background-color': '#222222',
                    'height': '105px',
                    'padding': '16px 0 0 0'},
        ), 
       
        html.Div(
            children = [
                 html.H6(
                     children='Prédiction pour les différents clients', 
                     style={'color': '#FFFFFF',
                            'fontSize': '24px',
                            'font-weight': 'bold',
                            'text-align': 'center',
                            'margin': '0 auto'},
                 ),
                 html.P(
                    children='Après avoir récupéré un modèle pour la classification des différents ' 
                             'clients, nous pouvons maintenant les classifier, le but ici sera donc, pour '
                             'chaque client à notre disposition, de définir si nous pouvons lui donner'
                             'son crédit ou non. ',
                    style={'color': '#CFCFCF',                          
                           'max-width': '500px',
                           'text-align': 'center',
                           'margin': '4px auto'},
                ),
                 html.P(
                    children='Si le score est de 0 le client aura tendance à ne pas rembourser le crédit, '
                             'si le score est de 1 il aura tendande a rembourser.',
                    style={'color': '#CFCFCF',                          
                           'max-width': '500px',
                           'text-align': 'center',
                           'margin': '4px auto'},
                ),
            ],
             style={'background-color': '#222222',
                    'height': '250px',
                    'padding': '16px 0 0 0'},
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Identifiant du client", className="menu-title"),
                        dcc.Dropdown(
                            id="id-client",
                            options=[
                                {"label": IDclient, "value": IDclient}
                                for IDclient in listeid1500
                            ],
                            value=listeIndexSort[0],
                            clearable=False,
                            searchable=False,
                            style={'width': '275px'}, 
                        ),
                    ],
                ),
        
            ],
            style={'background-color': '#FFFFFF',
                   'height': '100px',
                   'width': '550px',
                   'display': 'flex',
                   'justify-content': 'space-evenly',
                   'margin': '-80px auto 0 auto',
                   'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)',
                   'padding-top': '24px'},
        ),
        html.Div(
            children=[
                html.H6(
                    children = 'Information sur le client',
                    style={'color': '#000000',
                           'fontSize': '24px',
                           'font-weight': 'bold',
                           'text-align': 'left',
                           'margin': '0 auto'},
                ),
            ],
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Table([
                            html.Tr([html.Td(['Nombre d\'enfant: ']), html.Td(id='cntchildren')]),
                            html.Tr([html.Td(['Revenu: ']), html.Td(id='amtincometotal')]),
                            html.Tr([html.Td(['Montant du crédit: ']), html.Td(id='amtcredit')]),
                            html.Tr([html.Td(['Annuité de prêt: ']), html.Td(id='amtannuity')]),
                            html.Tr([html.Td(['Durée du remboursement: ']), html.Td(id='dureeremboursement')]),                       
                        ]),
                    ],                            
                ),
            ],
        ),
        html.Div(
            children=[
                html.H6(
                    children = 'Score de la prédiction',
                    style={'color': '#000000',
                           'fontSize': '24px',
                           'font-weight': 'bold',
                           'text-align': 'left',
                           'display': 'flex',
                           'justify-content': 'space-evenly',
                           'text-align': 'center',
                           'margin': '25px auto 0 auto'},
                ),
            ],
        ),
        html.Div(
            children=[
                html.Table([
                    html.Tr([html.Td(['Le client reçoit son crédit: ']), html.Td(id='crédit')]),
                    html.Tr([html.Td(['(Le seuil optimal est égal à ', round(optimal_threshold, 3), ')'])]),
                ]),                
            ],
            style={'color': '#000000',                          
                   'display': 'flex',
                   'justify-content': 'space-evenly',
                   'text-align': 'center',
                   'margin': '25px auto 0 auto'},                            
        ),
        html.Div(
            children=[
                 dcc.Graph(id = 'figureGauge')             
            ],
            style={'color': '#000000',                          
                   'display': 'flex',
                   'justify-content': 'space-evenly',
                   'text-align': 'center',
                   'margin': '25px auto 0 auto'},                            
        ),
        html.Div(
            children=[
                html.H6(
                    children = 'Les features les plus importantes',
                    style={'color': '#000000',
                           'fontSize': '24px',
                           'font-weight': 'bold',
                           'text-align': 'left',
                           'display': 'flex',
                           'justify-content': 'space-evenly',
                           'text-align': 'center',
                           'margin': '25px auto 0 auto'},
                ),
            ],
        ),
        html.Div(
            children=[
                html.Table([
                    html.Tr([html.Td(['Les meilleurs features globales']), html.Td(id='featureglobal')]),
                ]),                
            ],                
            style={'color': '#000000',                          
                   'display': 'flex',
                   'justify-content': 'space-evenly',
                   'text-align': 'center',
                   'margin': '25px auto 0 auto'},       
                                   
        ),
        html.Div(
            html.Img(src='data:image/png;base64,{}'.format(encoded_imageG.decode()), height='500', width='500'),
            style={'color': '#000000',                          
                   'display': 'flex',
                   'justify-content': 'space-evenly',
                   'text-align': 'center',
                   'margin': '25px auto 0 auto'}, 
        ),        
        #html.Div(
        #    children=[
        #        html.Table([
        #            html.Tr([html.Td(['Les meilleurs features Locales']), html.Td(id='featurelocal')]),
        #        ]),                
        #    ],                
        #    style={'color': '#000000',                          
        #           'display': 'flex',
        #           'justify-content': 'space-evenly',
        #           'text-align': 'center',
        #           'margin': '25px auto 0 auto'},       
        #                           
        #),
        #html.Div(
        #    children=[
        #         html.Img(id="figurefeaturelocal", src = ouvertureImageFilename(image_filename), height='150', width='750'),
        #    ],
        #    style={'color': '#000000',                          
        #           'display': 'flex',
        #           'justify-content': 'space-evenly',
        #           'text-align': 'center',
        #           'margin': '25px auto 0 auto'}, 
        #),   
       html.Div(
            children = [
                 html.H6(
                     children='Graphique des différentes features', 
                     style={'color': '#FFFFFF',
                            'fontSize': '24px',
                            'font-weight': 'bold',
                            'text-align': 'center',
                            'margin': '0 auto'},
                 ),
                 html.P(
                    children='Nous allons tracer les graphes de distribution des 2 features selectionnées, '
                             'un graphe d\'analyse bivariée entre ces 2 features, ainsi que la distibution '
                             'de la variable globale.',
                    style={'color': '#CFCFCF',                          
                           'max-width': '500px',
                           'text-align': 'center',
                           'margin': '4px auto'},
                ),
            ],
             style={'background-color': '#222222',
                    'height': '200px',
                    'padding': '16px 0 0 0'},
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Feature1", className="menu-title"),
                        dcc.Dropdown(
                            id="modeleFeature1",
                            options=[
                                {"label": feature1, "value": feature1}
                                for feature1 in listeClassementVariable
                            ],
                            value='CNT_CHILDREN',
                            clearable=False,
                            style={'width': '300px'}, 
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(children="Feature2", className="menu-title"),
                        dcc.Dropdown(
                            id="modeleFeature2",
                            options=[
                                {"label": feature2, "value": feature2}
                                for feature2 in listeClassementVariable
                            ],
                            value='AMT_INCOME_TOTAL',
                            clearable=False,
                            searchable=False,
                            style={'width': '300px'}, 
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(children="Identifiant du client", className="menu-title"),
                        dcc.Dropdown(
                            id="id-client1500",
                            options=[
                                {"label": IDclient, "value": IDclient}
                                for IDclient in listeid1500
                            ],
                            value=listeid1500[0],
                            clearable=False,
                            searchable=False,
                            style={'width': '275px'}, 
                        ),
                    ],
                ),
            ],
            style={'background-color': '#FFFFFF',
                   'height': '100px',
                   'width': '975px',
                   'display': 'flex',
                   'justify-content': 'space-evenly',
                   'margin': '-80px auto 0 auto',
                   'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)',
                   'padding-top': '24px'},
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H6(children = 'La distribution de la feature1',
                                style={'color': '#000000',
                                       'fontSize': '15px',                                   
                                       'text-align': 'center'                                       
                                      },
                        ),
                        html.Div(
                            children=[
                                 dcc.Graph(id="figuredistributionfeature1"),
                            ],
                        ),
                    ], 
                ),                
                html.Div(
                    children=[
                        html.H6(children = 'La distribution de la feature2',
                                style={'color': '#000000',
                                       'fontSize': '15px',                                     
                                       'text-align': 'center'
                                      },
                        ),
                        html.Div(
                            children=[
                                 dcc.Graph(id="figuredistributionfeature2"),
                            ], 
                        ),
                    ],
                ), 
                html.Div(
                    children=[
                        html.H6(children = 'Analyse bi-variée entre les 2 features selectionnées',
                                style={'color': '#000000',
                                       'fontSize': '15px',
                                       'text-align': 'center'
                                      },
                        ),
                        html.Div(
                            children=[
                                dcc.Graph(id="figureanalysebivariee"),
                            ], 
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.H6(children = 'Distribution de la feature globale',
                                style={'color': '#000000',
                                       'fontSize': '15px',
                                       'text-align': 'center'
                                      },
                        ),
                        html.Div(
                            children=[
                                dcc.Graph(id="figuredistributionglobale"),
                            ], 
                        ),
                    ],
                ),
            ],
            style={'display': 'flex', 
                   'justify-content': 'space-evenly', 
                   'flex-direction': 'row',
                   'margin': '0px auto 0 auto'
                   },
        ),
        html.Div(
            children = [
                 html.H6(
                     children='Prédiction de prêt de crédit', 
                     style={'color': '#FFFFFF',
                            'fontSize': '24px',
                            'font-weight': 'bold',
                            'text-align': 'center',
                            'margin': '0 auto'},
                 ),
                 html.P(
                    children='Dans cette partie nous allons faire appel à l\'API, qui fera appel à son tour '
                             'au modèle selectionné pour nous renseigner.',
                    style={'color': '#CFCFCF',                          
                           'max-width': '500px',
                           'text-align': 'center',
                           'margin': '4px auto'},
                ),
                html.P(
                    children='(Avec modèle optimisé et changement du seuil optimal (0.10))',
                    style={'color': '#CFCFCF',                          
                           'max-width': '500px',
                           'text-align': 'center',
                           'margin': '4px auto'},
                ),
            ],
             style={'background-color': '#222222',
                    'height': '200px',
                    'padding': '16px 0 0 0'},
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Identifiant du client", className="menu-title"),
                        dcc.Dropdown(
                            id="id-clientAPI",
                            options=[
                                {"label": IDclientAPI, "value": IDclientAPI}
                                for IDclientAPI in listeid1500
                            ],
                            value=listeid1500[0],
                            clearable=False,
                            searchable=False,
                            style={'width': '275px'}, 
                        ),
                    ],
                ),
            ],
            style={'background-color': '#FFFFFF',
                   'height': '100px',
                   'width': '550px',
                   'display': 'flex',
                   'justify-content': 'space-evenly',
                   'margin': '-80px auto 0 auto',
                   'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)',
                   'padding-top': '24px'
                 },
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Table([
                            #html.Tr([html.Td(['Confidence: ']), html.Td(id='confidenceAPI')]),
                            html.Tr([html.Td(['Prediction: ']), html.Td(id='predictionAPI')])                            
                        ]),
                    ], 
                    style={'color': '#000000',                          
                   'display': 'flex',
                   'justify-content': 'space-evenly',
                   'text-align': 'center',
                   'margin': '25px auto 0 auto'},
                ),
            ],
        ),
    ],
)


@app.callback(
    [
         Output('crédit', 'children')
    ],
    [
        Input("id-client", "value")
    ]
)

def prediction(id):

    crédit = X_testID2[X_testID2['ID'] == id]['Prediction'].values #####
 
    return [crédit]


@app.callback(
    [
         Output('cntchildren', 'children'),
         Output('amtincometotal', 'children'),
         Output('amtcredit', 'children'),
         Output('amtannuity', 'children'),
         Output('dureeremboursement', 'children')
    ],
    [
        Input("id-client", "value")
    ]
)

def prediction(id):
    
    cntchildren = dataframeInfoXTest2[dataframeInfoXTest2['SK_ID_CURR'] == id]['CNT_CHILDREN'].values  #####
    amtincometotal = dataframeInfoXTest2[dataframeInfoXTest2['SK_ID_CURR'] == id]['AMT_INCOME_TOTAL'].values
    amtcredit = dataframeInfoXTest2[dataframeInfoXTest2['SK_ID_CURR'] == id]['AMT_CREDIT'].values
    amtannuity = dataframeInfoXTest2[dataframeInfoXTest2['SK_ID_CURR'] == id]['AMT_ANNUITY'].values
    dureeRemboursement = amtcredit/amtannuity
  
    return cntchildren, amtincometotal, amtcredit, amtannuity, dureeRemboursement 

@app.callback(
    [
         Output('figureGauge', 'figure')
    ],
    [
        Input("id-client", "value")
    ]
)

def plotfiguregauge(id):
    
    values =  X_testID2[X_testID2['ID'] == id]['Score prediction proba'].values #######

    fpr, tpr, thresholds = roc_curve(y_test, y_predProba_lgbmHPSeuil)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = 1 - thresholds[optimal_idx]
    
    figureGauge = go.Figure(
                      go.Indicator(
                          mode = "gauge+number+delta",
                          value = float(1 - values),
                          domain = {'x': [0., 1.], 'y': [0., 1.]},
                          title = {'text': "Prediction"},
                          delta = {'reference': float(optimal_threshold), 'increasing': {'color': "RebeccaPurple"}},
                          gauge = {'axis': {'range': [None, 1]},
                                   'steps' : 
                                       [
                                            {'range': [1, float(1 - values)], 'color': "red"},
                                            {'range': [float(1 - values), 0], 'color': "green"}
                                       ],
                                   'threshold': 
                                       {
                                            'line': {'color': "yellow", 'width': 4},
                                            'thickness': 1,
                                            'value': float(optimal_threshold),
                                       },
                          },
                      ),
                  )
    
    figureGauge.update_layout(#autosize = True,
                              font_size = 10,
                              width = 350,
                              height = 350)
    
    
    return [go.Figure(data = figureGauge)]




#@app.callback(
#    [
#         Output('figurefeaturelocal', 'src')
#    ],
#    [
#        Input("id-client", "value")
#    ]
#)
#
#def plotfigurefeature(id):
#    
#    values =  X_testID2[X_testID2['ID'] == id]['index'].values #####
#    
#    shap.force_plot(explainer.expected_value[1], shap_values[1][int(values)], X_test.iloc[[int(values)]], show=False, matplotlib=True)
#    plt.savefig(image_filename)
#    src = ouvertureImageFilename(image_filename)
#    
#    return [src]
@app.callback(
    [
        Output('figuredistributionfeature1', 'figure'),
        Output('figuredistributionfeature2', 'figure'),
        Output('figuredistributionglobale', 'figure'),
    ],
    [
        Input("id-client1500", "value"),
        Input("modeleFeature1", "value"),
        Input("modeleFeature2", "value"),
    ]
)

def plotfiguremodelefeature(id, modeleFeature1, modeleFeature2):
    
    
    data1 = px.histogram(train_dataWNaN50F1500L, x = modeleFeature1, color = "TARGET",
                   marginal = "box", 
                   hover_data=train_dataWNaN50F1500L.columns)
    
    data2 = px.histogram(train_dataWNaN50F1500L, x = modeleFeature2, color = "TARGET",
                   marginal = "box", 
                   hover_data=train_dataWNaN50F1500L.columns)
    
    dataglobale = px.histogram(train_dataWNaN50F1500L, x = 'TARGET_NEIGHBORS_500_MEAN', color = "TARGET",
                   marginal = "box", 
                   hover_data=train_dataWNaN50F1500L.columns)
    
    data1.update_layout(width = 350, height = 350)
    
    data2.update_layout(width = 350, height = 350)
    
    dataglobale.update_layout(width = 350, height = 350)
    
    return data1, data2, dataglobale

@app.callback(
    [
        Output('figureanalysebivariee', 'figure')
    ],
    [
        Input("id-client1500", "value"),
        Input("modeleFeature1", "value"),
        Input("modeleFeature2", "value"),
    ]
)

def plotfiguremodelefeature(id, modeleFeature1, modeleFeature2):

    x1 = train_dataWNaN50F1500L[modeleFeature1]
    y1 = train_dataWNaN50F1500L[modeleFeature2]

    x2 =  train_dataWNaN50F1500L[train_dataWNaN50F1500L['SK_ID_CURR'] == id][modeleFeature1].values
    y2 =  train_dataWNaN50F1500L[train_dataWNaN50F1500L['SK_ID_CURR'] == id][modeleFeature2].values
    
    fig = go.Figure(data=go.Scatter(x = x1,
                                y = y1,
                                mode = 'markers',
                                marker_color = train_dataWNaN50F1500L['Score prediction'],
                                name = 'Bleu => 0 et Jaune => 1'))

    fig.add_trace(go.Scatter(x = x2, y = y2,
                        mode = 'markers',
                        marker_color = 'red',
                        name = 'Client'))
    
    fig.update_layout(width = 450, height = 350)
    
    return [fig]

@app.callback(
    [
         #Output('confidenceAPI', 'children'),
         Output('predictionAPI', 'children')
    ],
    [
        Input("id-clientAPI", "value")
    ]
)

def prediction(id):
    
    API_URL = 'https://apimodeleprojet7.herokuapp.com/'
    data_mean = API_URL + "/predictionModeleClient?id=" + str(id)
    #print(data_mean)
    response = requests.get(data_mean)
    content = json.loads(response.content.decode('utf-8'))
    
    #confidence = content[0]['confidence']
    prediction = content[0]['prediction']

    return [prediction]

#==========================================================================================================================#

if __name__ == "__main__":
    app.run_server(debug = True)
