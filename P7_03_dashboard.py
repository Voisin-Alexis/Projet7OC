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

y_test = joblib.load(cheminFichierJoblib + 'y_test2.joblib')

dataframeInfoXTest = joblib.load(cheminFichierJoblib + 'dataframeInfoXTest.joblib')
importance_dfclassement = joblib.load(cheminFichierJoblib + 'importance_dfclassement.joblib')
X_testID1500 = joblib.load(cheminFichierJoblib + 'X_testID1500.joblib')
lgbmHPSeuil = joblib.load(cheminFichierJoblib + 'lgbmHPSeuil.joblib')
listeidSort = joblib.load(cheminFichierJoblib + 'listeidSort.joblib')
train_dataWNaN50F1500L = joblib.load(cheminFichierJoblib + 'train_dataWNaN50F1500L.joblib')

y_predProba_lgbmHPSeuil = joblib.load(cheminFichierJoblib + 'y_predProba_lgbmHPSeuil.joblib')

image_filenameG = 'featuresglobaleTOP5.png'
encoded_imageG = base64.b64encode(open(image_filenameG, 'rb').read())

fpr, tpr, thresholds = roc_curve(y_test, y_predProba_lgbmHPSeuil)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = 1 - thresholds[optimal_idx]

print("Le seuil optimal est égal à", round(optimal_threshold, 3))

listeClassementVariable = joblib.load(cheminFichierJoblib + 'listeClassementVariable.joblib')
listeid1500 = joblib.load(cheminFichierJoblib + 'listeid1500.joblib')
train_dataWNaN50F1500L = joblib.load(cheminFichierJoblib + 'train_dataWNaN50F1500L.joblib')

train_dataWNaN50F1500L['Prediction'] = train_dataWNaN50F1500L['TARGET']

map_prediction = {0 : 'Positif',
                  1 : 'Négatif'} 

train_dataWNaN50F1500L['Prediction'] = train_dataWNaN50F1500L['Prediction'].map(map_prediction)

map_prediction = {'Oui' : 'Positif',
                  'Non' : 'Négatif'} 

X_testID1500['Prediction'] = X_testID1500['Prediction'].map(map_prediction)

train_dataWNaN50F1500L.rename(columns = {'TARGET_NEIGHBORS_500_MEAN':'Clients similaires (durée de remboursement du crédit)', 'EXT_SOURCE_MEAN':'Moyenne des différents scores de sources extérieures', 'DAYS_PAYMENT_RATIO_MAX_MEAN':'Délai de paiement', 'INTEREST_SHARE_MAX_ALL':'Intérêt du crédit', 'EXT_SOURCE_MUL':'Multiplication des différents scores de sources extérieures', 'CNT_CHILDREN':'Nombre d\'enfant', 'AMT_INCOME_TOTAL':'Revenus du client'}, inplace=True)
listeVariableARemplacer = ['TARGET_NEIGHBORS_500_MEAN', 'EXT_SOURCE_MEAN', 'DAYS_PAYMENT_RATIO_MAX_MEAN', 'INTEREST_SHARE_MAX_ALL', 'EXT_SOURCE_MUL', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL']
listeVariableRemplacement = ['Clients similaires (durée de remboursement du crédit)', 'Moyenne des différents scores de sources extérieures', 'Délai de paiement', 'Intérêt du crédit', 'Multiplication des différents scores de sources extérieures', 'Nombre d\'enfant', 'Revenus du client']

for i in range(len(listeVariableARemplacer)):
    for j in range(len(listeClassementVariable)):
        if listeClassementVariable[j] == listeVariableARemplacer[i]:
            listeClassementVariable[j] = listeVariableRemplacement[i]    

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
                    children='(Avec modèle optimisé et changement du seuil optimal (0.09))',
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
                html.Table([
                    #html.Tr([html.Td(['Confidence: ']), html.Td(id='confidenceAPI')]),
                    html.Tr([html.Td(['Prediction: '])])                            
                ]),
            ],
            style={'display': 'flex',
                   'textAlign': 'center',
                   'justify-content': 'space-evenly', 
                   #'flex-direction': 'row',
                   'margin': '15px auto 0 auto'
                   },
        ),
        html.Div(
            children=[
                html.P(id = 'predictionAPI',
                                style={'textAlign': 'center',
                                       'fontSize': 40,
                                       'font-weight': 'bold',
                                       #'margin': '0px auto 100px auto'
                                      }
                ),
            ], 
            style={'display': 'flex',
                   'textAlign': 'center',
                   'justify-content': 'space-evenly', 
                   #'flex-direction': 'row',
                   'margin': '-40px auto 0 auto'
                   },
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
                             'si le score est de 1 il aura tendance a rembourser.',
                    style={'color': '#CFCFCF',                          
                           'max-width': '500px',
                           'text-align': 'center',
                           'margin': '4px auto'},
                ),
            ],
             style={'background-color': '#222222',
                    'height': '250px',
                    'padding': '25px 0 0 0',
                    'margin': '10px auto'},
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
                                for IDclient in listeidSort
                            ],
                            value=listeidSort[0],
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
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.H6(
                                    children = 'Information sur le client',
                                    style={'color': '#000000',
                                           'fontSize': '24px',
                                           'font-weight': 'bold',
                                           'text-align': 'left',
                                           'display': 'flex',
                                           'justify-content': 'space-evenly',
                                           'text-align': 'center',
                                           'margin': '25px auto 0 auto'},
                                ),
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
                                    children = 'Prédiction client',
                                    style={'color': '#000000',
                                           'fontSize': '24px',
                                           'font-weight': 'bold',
                                           'text-align': 'left',
                                           'margin': '25px auto 0 auto'},
                                ),
                            ],
                        ),
                        html.Div(
                            children=[
                                html.Table([
                                    html.Tr([html.Td(['Le client reçoit son crédit: '])]),
                                    #html.Tr([html.Td(['(Le seuil optimal est égal à ', round(optimal_threshold, 3), ')'])]),
                                ]),
                                html.P(id = 'crédit',
                                       style={'textAlign': 'center',
                                              'font-weight': 'bold',
                                              'fontSize': 40}
                               ),  
                            ],
                            style={'display': 'flex', 
                                   'justify-content': 'space-evenly', 
                                   'flex-direction': 'row',
                                   'margin': '-20px auto 0 auto'
                                   },
                        ),
                        html.Div(
                            children=[
                                 dcc.Graph(id = 'figureGauge')             
                            ],
                            style={'display': 'flex', 
                                   'justify-content': 'space-evenly', 
                                   'margin': '-45px auto 0 auto'
                                   },
                        ),
                    ],
                ),            
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.H6(
                                    children = 'Les caractéristiques les plus importantes',
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
                                html.H6(children = 'Les meilleurs caractéristiques globales',
                                    style={'color': '#000000',
                                           'fontSize': '15px',                         
                                           'display': 'flex',
                                           'justify-content': 'space-evenly',
                                           'text-align': 'center'                         
                                          },
                                ),     
                            ],                
                            style={'color': '#000000',                          
                                   'display': 'flex',
                                   'justify-content': 'space-evenly',
                                   'text-align': 'center',
                                   'margin': '25px auto 0 auto'},       
                                           
                        ),
                        html.Div(
                            html.Img(src='data:image/png;base64,{}'.format(encoded_imageG.decode()), height='250', width='700'),
                            style={'color': '#000000',                          
                                   'display': 'flex',
                                   'justify-content': 'space-evenly',
                                   'text-align': 'center',
                                   'margin': '25px auto 0 auto'}, 
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
                     children='Graphique des différentes caractéristiques', 
                     style={'color': '#FFFFFF',
                            'fontSize': '24px',
                            'font-weight': 'bold',
                            'text-align': 'center',
                            'margin': '0 auto'},
                 ),
                 html.P(
                    children='Nous allons tracer les graphes de distribution des 2 caractéristiques selectionnées, '
                             'un graphe d\'analyse bivariée entre ces 2 caractéristiques, ainsi que la distibution '
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
                            value='Nombre d\'enfant',
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
                            value='Revenus du client',
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
                        html.H6(children = 'Distribution de la 1ere caractéristique',
                                style={'color': '#000000',
                                       'fontSize': '15px',                                   
                                       'text-align': 'center'                                       
                                      },
                        ),
                        html.Div(
                            children=[
                                html.Table([
                                    html.Tr([html.Td(['Valeur du client: ']), html.Td(id='positionclientfeature1')]),
                                ]),
                            ],  
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
                        html.H6(children = 'Distribution de la 2eme caractéristique',
                                style={'color': '#000000',
                                       'fontSize': '15px',                                     
                                       'text-align': 'center'
                                      },
                        ),
                        html.Div(
                            children=[
                                html.Table([
                                    html.Tr([html.Td(['Valeur du client: ']), html.Td(id='positionclientfeature2')]),
                                ]),
                            ], 
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
            ],
            style={'display': 'flex', 
                   'justify-content': 'space-evenly', 
                   'flex-direction': 'row',
                   'margin': '0px auto 0 auto'
                   },
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H6(children = 'Analyse bi-variée entre les 2 caractéristiques selectionnées',
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
                        html.H6(children = 'Distribution de la caractéristique globale',
                                style={'color': '#000000',
                                       'fontSize': '15px',
                                       'text-align': 'center'
                                      },
                        ),
                        html.Div(
                            children=[
                                html.Table([
                                    html.Tr([html.Td(['Valeur du client: ']), html.Td(id='positionclientglobale')]),
                                ]),
                            ], 
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
    ],
)


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


@app.callback(
    [
         Output('crédit', 'children')
    ],
    [
        Input("id-client", "value")
    ]
)

def prediction(id):

    crédit = X_testID1500[X_testID1500['ID'] == id]['Prediction'].values
    
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
    
    cntchildren = dataframeInfoXTest[dataframeInfoXTest['SK_ID_CURR'] == id]['CNT_CHILDREN'].values    
    amtincometotal = dataframeInfoXTest[dataframeInfoXTest['SK_ID_CURR'] == id]['AMT_INCOME_TOTAL'].values    
    amtcredit = dataframeInfoXTest[dataframeInfoXTest['SK_ID_CURR'] == id]['AMT_CREDIT'].values    
    amtannuity = dataframeInfoXTest[dataframeInfoXTest['SK_ID_CURR'] == id]['AMT_ANNUITY'].values
    
    dureeRemboursement = amtcredit/amtannuity
    dureeRemboursementFloor = math.floor(dureeRemboursement)
    dureeRemboursementDiff = dureeRemboursement - dureeRemboursementFloor
    dureeRemboursementDiff12 = dureeRemboursementDiff * 12
    dureeRemboursementDiff12Min = math.floor(dureeRemboursementDiff12)
    dureeRemboursementDiff12Max = math.ceil(dureeRemboursementDiff12)
    
    return cntchildren, str(amtincometotal)[1:-1] + ' €', str(amtcredit)[1:-1] + ' €', str(amtannuity)[1:-1] + ' €', str(dureeRemboursementFloor) + ' ans et ' + str(dureeRemboursementDiff12Min) + ' - ' + str(dureeRemboursementDiff12Max) + ' mois'

@app.callback(
    [
         Output('figureGauge', 'figure')
    ],
    [
        Input("id-client", "value")
    ]
)

def plotfiguregauge(id):
    
    values =  X_testID1500[X_testID1500['ID'] == id]['Score prediction proba'].values

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
                              width = 450,
                              height = 450)
    
    
    return [go.Figure(data = figureGauge)]

@app.callback(
    [        
        Output('positionclientfeature1', 'children')
    ],
    [
        Input("id-client1500", "value"),
        Input("modeleFeature1", "value")
    ]
)

def positionclient1(id, modeleFeature1):  
    
    positionclientfeature1 = train_dataWNaN50F1500L[train_dataWNaN50F1500L['SK_ID_CURR'] == id][modeleFeature1].values
        
    return [positionclientfeature1]

@app.callback(
    [        
        Output('positionclientfeature2', 'children')
    ],
    [
        Input("id-client1500", "value"),
        Input("modeleFeature2", "value")
    ]
)

def positionclient2(id, modeleFeature2):  
    
    positionclientfeature2 = train_dataWNaN50F1500L[train_dataWNaN50F1500L['SK_ID_CURR'] == id][modeleFeature2].values
        
    return [positionclientfeature2]

@app.callback(
    [        
        Output('positionclientglobale', 'children')
    ],
    [
        Input("id-client1500", "value")
    ]
)

def positionclientglobale(id):  
    
    positionclientfeatureglobale = train_dataWNaN50F1500L[train_dataWNaN50F1500L['SK_ID_CURR'] == id]['Clients similaires (durée de remboursement du crédit)'].values
        
    return [positionclientfeatureglobale]

@app.callback(
    [        
        Output('figuredistributionfeature1', 'figure')
    ],
    [
        Input("id-client1500", "value"),
        Input("modeleFeature1", "value")
    ]
)

def plotfiguremodelefeature1(id, modeleFeature1):  
    
    fig1_0 = train_dataWNaN50F1500L[train_dataWNaN50F1500L['TARGET'] == 0][modeleFeature1].values
    fig1_1 = train_dataWNaN50F1500L[train_dataWNaN50F1500L['TARGET'] == 1][modeleFeature1].values
    
    fig1_client =  train_dataWNaN50F1500L[train_dataWNaN50F1500L['SK_ID_CURR'] == id][modeleFeature1].values    
    
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x = fig1_0, name = 'Positif'))
    fig1.add_trace(go.Histogram(x = fig1_1, name = 'Négatif'))
    fig1.add_trace(go.Scatter(x = fig1_client, y = [0],
                         mode = 'markers',
                         marker_color = 'red',
                         name = 'Client'))
    
    fig1.update_layout(barmode='stack', xaxis_title_text= modeleFeature1, yaxis_title_text='Nombre de Client', width = 550, height = 550)
    
    return [fig1]

@app.callback(
    [        
        Output('figuredistributionfeature2', 'figure')       
    ],
    [
        Input("id-client1500", "value"),
        Input("modeleFeature2", "value")
    ]
)

def plotfiguremodelefeature2(id, modeleFeature2):  
    
    fig2_0 = train_dataWNaN50F1500L[train_dataWNaN50F1500L['TARGET'] == 0][modeleFeature2].values
    fig2_1 = train_dataWNaN50F1500L[train_dataWNaN50F1500L['TARGET'] == 1][modeleFeature2].values
    
    fig2_client =  train_dataWNaN50F1500L[train_dataWNaN50F1500L['SK_ID_CURR'] == id][modeleFeature2].values 
    
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x = fig2_0, name = 'Positif'))
    fig2.add_trace(go.Histogram(x = fig2_1, name = 'Négatif'))
    fig2.add_trace(go.Scatter(x = fig2_client, y = [0],
                         mode = 'markers',
                         marker_color = 'red',
                         name = 'Client'))
    
    fig2.update_layout(barmode='stack', xaxis_title_text= modeleFeature2, yaxis_title_text='Nombre de Client', width = 550, height = 550)
    
    
    return [fig2]

@app.callback(
    [        
        Output('figuredistributionglobale', 'figure')
    ],
    [
        Input("id-client1500", "value")
    ]
)

def plotfiguremodelefeatureglobale(id):  
    
    figglobale_0 = train_dataWNaN50F1500L[train_dataWNaN50F1500L['TARGET'] == 0]['Clients similaires (durée de remboursement du crédit)'].values
    figglobale_1 = train_dataWNaN50F1500L[train_dataWNaN50F1500L['TARGET'] == 1]['Clients similaires (durée de remboursement du crédit)'].values
    
    figglobale_client =  train_dataWNaN50F1500L[train_dataWNaN50F1500L['SK_ID_CURR'] == id]['Clients similaires (durée de remboursement du crédit)'].values    
    
    figglobale = go.Figure()
    figglobale.add_trace(go.Histogram(x = figglobale_0, name = 'Positif'))
    figglobale.add_trace(go.Histogram(x = figglobale_1, name = 'Négatif'))
    
    figglobale.add_trace(go.Scatter(x = figglobale_client, y = [0],
                         mode = 'markers',
                         marker_color = 'red',
                         name = 'Client'))
    
    figglobale.update_layout(barmode='stack', xaxis_title_text= 'Clients similaires (durée de remboursement du crédit)', yaxis_title_text='Nombre de Client', width = 550, height = 550)
    
    return [figglobale]

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
                                    name = 'B=>Pos et J=>Neg'))

    fig.add_trace(go.Scatter(x = x2, y = y2,
                             mode = 'markers',
                             marker_color = 'red',
                             name = 'Client'))
    
    fig.update_layout(width = 550, height = 550)
    
    return [fig]

#==========================================================================================================================#

if __name__ == "__main__":
    app.run_server(debug = True)
