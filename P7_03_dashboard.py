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
                                for IDclient in listeIndexSort
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
        html.Div(
            children=[
                html.Table([
                    html.Tr([html.Td(['Les meilleurs features Locales']), html.Td(id='featurelocal')]),
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
                 html.Img(id="figurefeaturelocal", src = ouvertureImageFilename(image_filename), height='150', width='750'),
            ],
            style={'color': '#000000',                          
                   'display': 'flex',
                   'justify-content': 'space-evenly',
                   'text-align': 'center',
                   'margin': '25px auto 0 auto'}, 
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




@app.callback(
    [
         Output('figurefeaturelocal', 'src')
    ],
    [
        Input("id-client", "value")
    ]
)

def plotfigurefeature(id):
    
    values =  X_testID2[X_testID2['ID'] == id]['index'].values #####
    
    shap.force_plot(explainer.expected_value[1], shap_values[1][int(values)], X_test.iloc[[int(values)]], show=False, matplotlib=True)
    plt.savefig(image_filename)
    src = ouvertureImageFilename(image_filename)
    
    return [src]

#==========================================================================================================================#

if __name__ == "__main__":
    app.run_server(debug = True)
