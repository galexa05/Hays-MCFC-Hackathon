import streamlit as st
from src.passing_network_analysis.functions import *
import os
import pandas as pd
import numpy as np
import json
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import glob
import os
import sys
from visualization.passing_network import draw_pitch, draw_pass_map
import warnings
warnings.filterwarnings("ignore")


## Graph Visualization from Eigenvector Centrality

def _statsbomb_to_point(location, max_width=120, max_height=80, pos=0):
    '''
    Convert a point's coordinates from a StatsBomb's range to 0-1 range.
    '''
    if pos == 0:
        return np.round(location / max_width,2)
    else:
        return np.round(1-(location / max_height),2)


def get_passes_matrix(df_passes, players):
    # Get in Matrix format
    player_pass_count = df_passes.groupby(["player_name_1","player_name_2"]).size().to_frame("num_passes").reset_index()
    player_pass_count = player_pass_count.pivot_table(values='num_passes', index="player_name_1", columns='player_name_2')

    # Get passing matrix for all players in team (all no matter if did not play or did not pass)
    passing_matrix = pd.DataFrame(player_pass_count,columns=players,index=players).fillna(0).astype(int)
    return passing_matrix   

def plot_passing_network_eigenvector(df,team_players,plot_title,plot_legend,config_team):
    graph_passes = df.copy()
    graph_passes.rename(columns={'player_name':'player_name_1','pass_recipient_name':'player_name_2'},inplace=True)

    players = graph_passes['player_name_1'].unique()
    matrix_all = get_passes_matrix(graph_passes,players) 

    # Get graphs
    graph_passes_all = nx.from_pandas_adjacency(matrix_all, create_using = nx.DiGraph)

    # Get centralities for each graph
    eigen_cent_all = list(nx.eigenvector_centrality(graph_passes_all,weight='weight').values())

    # Get num of passes done and received
    passes_done_all = matrix_all.sum(axis = 1)
    passes_received_all = matrix_all.sum()

    # Get the stats for that team in the match
    stats_match = pd.DataFrame(list(zip(players,passes_done_all,passes_received_all,
                                        eigen_cent_all)), 
                               columns = ['Player','Passes done All','Passes received All','eigen_cent_all'])



    passer_names = graph_passes['player_name_1'].unique().tolist()
    recipient_names = graph_passes['player_name_2'].unique().tolist()
    receivers = []
    for player in recipient_names:
        if player not in passer_names:
            receivers.append(player)

    if (len(receivers)!=0):
        receive_passes = graph_passes[graph_passes['player_name_2'].isin(receivers)]
        receive_passes['origin_pos_x'] = receive_passes.end_x.apply(lambda x: _statsbomb_to_point(x,pos=0))
        receive_passes['origin_pos_y'] = receive_passes.end_y.apply(lambda x: _statsbomb_to_point(x,pos=1))
        player_position_rec = receive_passes.groupby("player_name_2").agg({"origin_pos_x": "mean", "origin_pos_y": "mean"})
        player_pass_count_rec = receive_passes.groupby("player_name_2").size().to_frame("num_passes");player_pass_count_rec['num_passes']=0
        player_pass_value_rec = receive_passes.groupby("player_name_2").size().to_frame("pass_value");player_pass_value_rec['pass_value']=0

    graph_passes['pair_key'] = graph_passes['player_name_1']+'_'+graph_passes['player_name_2']

    graph_passes['origin_pos_x'] = graph_passes.x.apply(lambda x: _statsbomb_to_point(x,pos=0))
    graph_passes['origin_pos_y'] = graph_passes.y.apply(lambda x: _statsbomb_to_point(x,pos=1))
    player_position = graph_passes.rename(columns={'player_name_1':'player_name'}).groupby('player_name').agg({"origin_pos_x": "mean", "origin_pos_y": "mean"})

    player_pass_count = pd.DataFrame()
    player_pass_count['num_passes'] = stats_match['eigen_cent_all']
    player_pass_count.index = stats_match.rename(columns={'Player':'player_name'})['player_name']

    player_pass_value = graph_passes.groupby("player_name_1").size().to_frame("pass_value")

    pair_pass_count = graph_passes.groupby("pair_key").size().to_frame("num_passes")
    pair_pass_value = graph_passes.groupby("pair_key").size().to_frame("pass_value")
    
    if (len(receivers)!=0):
        # import pdb;pdb.set_trace()
        player_position = pd.concat([player_position,player_position_rec])
        player_pass_count = pd.concat([player_pass_count,player_pass_count_rec])
        player_pass_value = pd.concat([player_pass_value,player_pass_value_rec])


    ax = draw_pitch(config_team=config_team)
    ax = draw_pass_map(ax, player_position, player_pass_count, player_pass_value,
                       pair_pass_count, pair_pass_value, plot_title, plot_legend,debug=False,
                       config_team=config_team,team_players=team_players)