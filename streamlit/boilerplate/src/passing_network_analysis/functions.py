import streamlit as st
from src.passing_network_analysis.functions import *
import os
import pandas as pd
import numpy as np
import json
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import networkx as nx
import glob
import os
import sys
from visualization.passing_network import draw_pitch, draw_pass_map
import warnings
warnings.filterwarnings("ignore")


# Get all match games as a dictionary format 
def get_all_match_games():
    match_games = {}
    # Define the wildcard pattern to match the file names
    pattern = 'data/StatsBomb/Data/*_lineups.json'
    # Use glob to find all files that match the pattern
    lineups_path = glob.glob(pattern)

    df_list = []
    for file in lineups_path:
        # Open the file and load its contents into a dictionary
        with open(file, 'r') as f:
            lineups = json.load(f)
            home_players = []
            away_players = []
            # Get all players from home team who started the match
            for player in lineups[0]['lineup']:
                if len(player['positions'])!=0:
                    if player['positions'][0]['start_reason']=='Starting XI':
                        home_players.append(player['player_name'])
            # Get all players from away team who started the match
            for player in lineups[1]['lineup']:
                if len(player['positions'])!=0:
                    if player['positions'][0]['start_reason']=='Starting XI':
                        away_players.append(player['player_name'])
   
            team_home = lineups[0]['team_name']
            team_away = lineups[1]['team_name']
            match_title = f'{team_home} vs {team_away}'
            match_game_code = '_'.join(file.split('/')[-1].split('_')[:2])
            match_games[match_title] = \
                {'home_name':team_home,
                 'home_id':lineups[0]['team_id'],
                 'home_players':home_players,
                 'away_name':team_away,
                 'away_id':lineups[1]['team_id'],
                 'away_players':away_players,
                 'match_game':match_game_code
                 }
    return match_games

# Get all events from all matches and returns a dataframe 
def get_events_all_matches():
    # Define the wildcard pattern to match the file names
    pattern = 'data/StatsBomb/Data/*_events.json'
    # Use glob to find all files that match the pattern
    file_list = glob.glob(pattern)

    match_game = '_'.join(file_list[0].split('/')[-1].split('_')[:2])

    df_list = []
    for file in file_list:
        # Open the file and load its contents into a dictionary
        with open(file, 'r') as f:
            data = json.load(f)
            match_game = '_'.join(file.split('/')[-1].split('_')[:2])
            df_game = pd.json_normalize(data,sep='_')
            df_game['match_game'] = match_game
            # Convert the list of JSON values to a DataFrame
            df_list.append(df_game)
    df = pd.concat(df_list,ignore_index = True)
    return df
    
# Get a match from a selected match
def get_match_events(df,match_game):
    return df[df['match_game']==match_game]

# Prepare data and keeping only passes
def prepare_data_pass(df,team_name,sub=True):
    #check for index of first sub
    if sub:
        sub = df.loc[df["type_name"] == "Substitution"].loc[df["team_name"] == team_name].iloc[0]["index"]

    #make df with successfull passes by Manchester City until the first substitution
    if sub:
        mask_mancity = (df['type_name'] == 'Pass')\
        & (df['team_name']==team_name) \
        & (df.index < sub) \
        &(df['substitution_replacement_name'].isnull())\
        &(~df['pass_recipient_name'].isnull())
    else:
        mask_mancity = (df['type_name'] == 'Pass')\
        & (df['team_name']==team_name) \
        &(df['substitution_replacement_name'].isnull())\
        &(~df['pass_recipient_name'].isnull()) 

    #taking necessary columns
    df_pass_details = df.loc[mask_mancity, ['location','pass_end_location','player_name','pass_recipient_name','minute']]

    df_pass_details['x'] = [axes[0] for axes in df_pass_details['location'].values]
    df_pass_details['y'] = [axes[1] for axes in df_pass_details['location'].values]

    df_pass_details['end_x'] = [axes[0] for axes in df_pass_details['pass_end_location'].values]
    df_pass_details['end_y'] = [axes[1] for axes in df_pass_details['pass_end_location'].values]

    df_pass = df_pass_details[['x', 'y', 'end_x', 'end_y','player_name','pass_recipient_name','minute']]

    #adjusting that only the surname of a player is presented.
    df_pass["player_name"] = df_pass["player_name"].apply(lambda x: str(x).split()[-1])
    df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])
    return df_pass


# Calucalte and return passing information for every player
def calculate_vertices(df_pass):
    scatter_df = pd.DataFrame()
    for i, name in enumerate(df_pass["player_name"].unique()):
        passx = df_pass.loc[df_pass["player_name"] == name]["x"].to_numpy()
        recx = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_x"].to_numpy()
        passy = df_pass.loc[df_pass["player_name"] == name]["y"].to_numpy()
        recy = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_y"].to_numpy()
        scatter_df.at[i, "player_name"] = name
        #make sure that x and y location for each circle representing the player is the average of passes and receptions
        scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
        #calculate number of passes
        scatter_df.at[i, "no"] = df_pass.loc[df_pass["player_name"] == name].count().iloc[0]

    #adjust the size of a circle so that the player who made more passes 
    scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)
    return scatter_df

#counting passes between players
def calculate_edges_width(df_temp):
    df_pass = df_temp.copy()
    df_pass["pair_key"] = df_pass.apply(lambda x: "_".join(sorted([x["player_name"], x["pass_recipient_name"]])), axis=1)
    lines_df = df_pass.groupby(["pair_key"]).x.count().reset_index()
    lines_df.rename({'x':'pass_count'}, axis='columns', inplace=True)
    #setting a treshold. You can try to investigate how it changes when you change it.
    lines_df = lines_df[lines_df['pass_count']>2]
    scatter_df = calculate_vertices(df_pass)
    return df_pass,lines_df,scatter_df

# get a passing network from a team 
def get_passing_network_data(df,team_name,team_players,node_color,edge_color,sub=True):
    df_pass_home = prepare_data_pass(df,team_name=team_name,sub=sub)

    df_pass_edges_home,lines_df_home,scatter_df_home = calculate_edges_width(df_pass_home)

    home_dict = \
    {
        'df_pass' : df_pass_home,
        'lines_df' : lines_df_home,
        'scatter_df' : scatter_df_home,
        'color': node_color,
        'color2': edge_color,
        'team_name' : team_name,
        'team_starters' : team_players
    }
    return home_dict


#plot once again pitch and vertices
def plotting_edges(team_dict,debug=False):
    if debug:
        import pdb;pdb.set_trace()
    # get the information regarding the passes between the home team     
    df_pass = team_dict['df_pass']
    lines_df = team_dict['lines_df']
    team_color = team_dict['color']
    team_name = team_dict['team_name']
    scatter_df = team_dict['scatter_df']
    
    pitch = Pitch(line_color='grey')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0, endnote_space=0)
    
    # Plot the node per player     
    pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color=team_color, edgecolors='grey', linewidth=1, alpha=1, ax=ax["pitch"], zorder = 3)
    for i, row in scatter_df.iterrows():
        pitch.annotate(row['player_name'], xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax["pitch"], zorder = 7)

    for i, row in lines_df.iterrows():
            player1 = row["pair_key"].split("_")[0]
            player2 = row['pair_key'].split("_")[1]
            #take the average location of players to plot a line between them 
            player1_x = scatter_df.loc[scatter_df["player_name"] == player1]['x'].iloc[0]
            player1_y = scatter_df.loc[scatter_df["player_name"] == player1]['y'].iloc[0]
            player2_x = scatter_df.loc[scatter_df["player_name"] == player2]['x'].iloc[0]
            player2_y = scatter_df.loc[scatter_df["player_name"] == player2]['y'].iloc[0]
            num_passes = row["pass_count"]
            #adjust the line width so that the more passes, the wider the line
            line_width = (num_passes / lines_df['pass_count'].max() * 10)
            #plot lines on the pitch
            pitch.lines(player1_x, player1_y, player2_x, player2_y,
                            alpha=1, lw=line_width, zorder=2, color="#c7d5cc", ax = ax["pitch"])

    fig.suptitle(f"{team_name} Passing Network", fontsize = 30)
    # plt.show()


def _statsbomb_to_point(location, max_width=120, max_height=80, pos=0):
    '''
    Convert a point's coordinates from a StatsBomb's range to 0-1 range.
    '''
    if pos == 0:
        return np.round(location / max_width,2)
    else:
        return np.round(1-(location / max_height),2)

# Plot passing network graph
def plot_passing_network_graph(df,team_players,plot_title,plot_legend,config_team):
    graph_passes = df.copy()  
    graph_passes['origin_pos_x'] = graph_passes.x.apply(lambda x: _statsbomb_to_point(x,pos=0))
    graph_passes['origin_pos_y'] = graph_passes.y.apply(lambda x: _statsbomb_to_point(x,pos=1))
    
    passer_names = graph_passes['player_name'].unique().tolist()
    recipient_names = graph_passes['pass_recipient_name'].unique().tolist()
    receivers = []
    for player in recipient_names:
        if player not in passer_names:
            receivers.append(player)

    if (len(receivers)!=0):
        receive_passes = graph_passes[graph_passes['pass_recipient_name'].isin(receivers)]
        receive_passes['origin_pos_x'] = receive_passes.end_x.apply(lambda x: _statsbomb_to_point(x,pos=0))
        receive_passes['origin_pos_y'] = receive_passes.end_y.apply(lambda x: _statsbomb_to_point(x,pos=1))
        player_position_rec = receive_passes.groupby("pass_recipient_name").agg({"origin_pos_x": "mean", "origin_pos_y": "mean"})
        player_pass_count_rec = receive_passes.groupby("pass_recipient_name").size().to_frame("num_passes");player_pass_count_rec['num_passes']=0
        player_pass_value_rec = receive_passes.groupby("pass_recipient_name").size().to_frame("pass_value");player_pass_value_rec['pass_value']=0
        
    player_position = graph_passes.groupby("player_name").agg({"origin_pos_x": "mean", "origin_pos_y": "mean"})
    player_pass_count = graph_passes.groupby("player_name").size().to_frame("num_passes")
    player_pass_value = graph_passes.groupby("player_name").size().to_frame("pass_value")

    if (len(receivers)!=0):
        # import pdb;pdb.set_trace()
        player_position = pd.concat([player_position,player_position_rec])
        player_pass_count = pd.concat([player_pass_count,player_pass_count_rec])
        player_pass_value = pd.concat([player_pass_value,player_pass_value_rec])

    graph_passes["pair_key"] = graph_passes.apply(lambda x: "_".join([x["player_name"], x["pass_recipient_name"]]), axis=1)
    pair_pass_count = graph_passes.groupby("pair_key").size().to_frame("num_passes")
    pair_pass_value = graph_passes.groupby("pair_key").size().to_frame("pass_value")

    ax = draw_pitch(config_team=config_team)
    ax = draw_pass_map(
        ax, player_position, player_pass_count, player_pass_value,
        pair_pass_count, pair_pass_value, plot_title, plot_legend,
        config_team=config_team,team_players=team_players
        )
