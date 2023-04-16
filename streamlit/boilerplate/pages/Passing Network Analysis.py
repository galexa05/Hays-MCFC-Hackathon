# Contents of ~/my_app/pages/page_2.py
import streamlit as st
from src.passing_network_analysis.functions import *
from src.passing_network_analysis.eigenvector_score import *
from src.passing_network_analysis.passing_maps import *
from src.passing_network_analysis.danger_passes import *
import os
import pandas as pd
import numpy as np
import json
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import networkx as nx
import glob
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="Passing Network Analysis", layout="wide", initial_sidebar_state="collapsed")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("# Passing Network Analysis")

match_dict = get_all_match_games()
df_events = get_events_all_matches()

# Plase the sidebar at the left side of the webapp giving the option to select your match
match_title = st.sidebar.selectbox(
    "Select a match",
    ([title for title in match_dict.keys()])
)

# Retrieve the events from the selected match game
home_name = match_dict[match_title]['home_name']
home_starters = match_dict[match_title]['home_players']
away_name = match_dict[match_title]['away_name']
away_starters = match_dict[match_title]['away_players']
df_match_events = get_match_events(df_events,match_dict[match_title]['match_game'])

# Prepare data for passing networks from home and away teams
home_dict = get_passing_network_data(
        df_match_events,
        team_name=home_name,
        team_players = home_starters,
        node_color='red',
        edge_color='white', 
        sub=False
    )
away_dict = get_passing_network_data(
        df_match_events,
        team_name=away_name,
        team_players = away_starters,
        node_color='blue',
        edge_color='white',
        sub=False
    )

# Visualize the passing network graph
st.markdown(f"{home_name} vs {away_name}")


col_time,col_network = st.columns(2)
with col_time:
    slider_range = st.slider('Specify the period of time:',
                             value=[min(df_match_events['minute']),max(df_match_events['minute'])],
                             key='slider_pass_network')
    from_time = slider_range[0]
    to_time = slider_range[1]   
    # st.write('Values:', values)

with col_network:
    option = st.selectbox(
        'Select your type of network',
        ('All Passes Network', 'Forward Passes Network','Lost Passes Network','Homogeinity Network'))

st.markdown(f'## **{option} Analysis**')
# Based on the selected option, perform different functionality
if option == "All Passes Network":
    col_home_network,col_away_network = st.columns(2)
    with col_home_network:
        # st.markdown(f"##### Passing Network from {home_name}")
        plot_title_home =f"{home_name}'s passing network"
        plot_legend_home = "Location: pass origin\nSize: number of passes\nColor: number of passes"
        
        df_period_home = home_dict['df_pass'].copy()
        df_period_home = df_period_home[(df_period_home['minute']>=from_time) & (df_period_home['minute']<=to_time)]

        plot_passing_network_graph(df_period_home,home_dict['team_starters'],plot_title_home,plot_legend_home,config_team='home')
        st.pyplot()

    with col_away_network:
        plot_title_home =f"{away_name}'s passing network"
        plot_legend_home = "Location: pass origin\nSize: number of passes\nColor: number of passes"
        
        df_period_away = away_dict['df_pass'].copy()
        print(f"Away team Minute Max: {max(away_dict['df_pass']['minute'])}")
        df_period_away = df_period_away[(df_period_away['minute']>=from_time) & (df_period_away['minute']<=to_time)]
        if len(df_period_away)==0:
            print(f"Away team Minute Max: {max(away_dict['df_pass']['minute'])}")
        plot_passing_network_graph(df_period_away,away_dict['team_starters'],plot_title_home,plot_legend_home,config_team='away')
        st.pyplot()

elif option == "Forward Passes Network":
    col_home_network_forward,col_away_network_forward = st.columns(2)
    with col_home_network_forward:
        plot_title_home =f"{home_name}'s passing network"
        plot_legend_home = "Location: pass origin\nSize: number of passes\nColor: number of passes"
        
        df_period_home = home_dict['df_pass'].copy()
        df_period_home = df_period_home[(df_period_home['minute']>=from_time) & (df_period_home['minute']<=to_time)]
        df_period_home = df_period_home[df_period_home['x']<df_period_home['end_x']]
        plot_passing_network_graph(df_period_home,home_dict['team_starters'],plot_title_home,plot_legend_home,config_team='home')
        st.pyplot()
    with col_away_network_forward:
        plot_title_home =f"{away_name}'s passing network"
        plot_legend_home = "Location: pass origin\nSize: number of passes\nColor: number of passes"
        
        df_period_away = away_dict['df_pass'].copy()
        df_period_away = df_period_away[(df_period_away['minute']>=from_time) & (df_period_away['minute']<=to_time)]
        df_period_away = df_period_away[df_period_away['x']<df_period_away['end_x']]
        plot_passing_network_graph(df_period_away,away_dict['team_starters'],plot_title_home,plot_legend_home,config_team='away')
        st.pyplot()

elif option == "Lost Passes Network":
    df_lost_passes = df_match_events[(df_match_events['type_name']=='Pass') & (~df_match_events['pass_outcome_name'].isna()) & (~df_match_events['pass_recipient_name'].isna())] 
    home_dict_lost_passes = get_passing_network_data(
                df_lost_passes,
                team_name=home_name,
                team_players = home_starters,
                node_color='red',
                edge_color='white', 
                sub=False
    )

    away_dict_lost_passes = get_passing_network_data(
            df_lost_passes,
            team_name=away_name,
            team_players = away_starters,
            node_color='blue',
            edge_color='white',
            sub=False
    )
    col_home_network_lost,col_away_network_lost = st.columns(2)
    with col_home_network_lost:
        plot_title_home =f"{home_name}'s passing network"
        plot_legend_home = "Location: pass origin\nSize: number of lost passes\nColor: number of lost passes"
        # Prepare data for passing networks from home and away teams
        
        df_period_home = home_dict_lost_passes['df_pass'].copy()
        df_period_home = df_period_home[(df_period_home['minute']>=from_time) & (df_period_home['minute']<=to_time)]
        plot_passing_network_graph(df_period_home,home_dict['team_starters'],plot_title_home,plot_legend_home,config_team='home')
        # plot_passing_network_eigenvector(df_period_home,home_dict['team_starters'],plot_title=plot_title_home,plot_legend=plot_legend_home,config_team='home')    
        st.pyplot()
    with col_away_network_lost:
        plot_title_away =f"{away_name}'s passing network"
        plot_legend_away = "Location: pass origin\nSize: number of lost passes\nColor: number of lost passes"

        df_period_away = away_dict_lost_passes['df_pass'].copy()
        df_period_away = df_period_away[(df_period_away['minute']>=from_time) & (df_period_away['minute']<=to_time)]
        plot_passing_network_graph(df_period_away,away_dict['team_starters'],plot_title_home,plot_legend_home,config_team='away')
        # plot_passing_network_eigenvector(df_period_away,away_dict['team_starters'],plot_title=plot_title_away,plot_legend=plot_legend_away,config_team='away')    
        st.pyplot()

elif option == "Homogeinity Network":
    col_home_network_eigen,col_away_network_eigen = st.columns(2)
    with col_home_network_eigen:
        plot_title_home =f"{home_name}'s passing network"
        plot_legend_home = "Location: pass origin\nNode size: eigenvector centrality\nNode color: number of passes"
        
        df_period_home = home_dict['df_pass'].copy()
        df_period_home = df_period_home[(df_period_home['minute']>=from_time) & (df_period_home['minute']<=to_time)]
        plot_passing_network_eigenvector(df_period_home,home_dict['team_starters'],plot_title=plot_title_home,plot_legend=plot_legend_home,config_team='home')    
        st.pyplot()

    with col_away_network_eigen:
        plot_title_away =f"{away_name}'s passing network"
        plot_legend_away = "Location: pass origin\nNode size: eigenvector centrality\nNode color: number of passes"
        
        df_period_away = away_dict['df_pass'].copy()
        df_period_away = df_period_away[(df_period_away['minute']>=from_time) & (df_period_away['minute']<=to_time)]
        plot_passing_network_eigenvector(df_period_away,away_dict['team_starters'],plot_title=plot_title_away,plot_legend=plot_legend_away,config_team='away')    
        st.pyplot()


else:
    st.write("You did not select any network")


# Visualize the passing network graph
st.markdown(f"## Passing Map Analysis")
slider_range_passmap = st.slider(
    'Specify the period of time:',
    value=[min(df_match_events['minute']),max(df_match_events['minute'])],
    key='slider_passmap'
    )
from_time_map = slider_range_passmap[0]
to_time_map = slider_range_passmap[1]   
col_home_pass_maps,col_away_pass_maps = st.columns(2)
df_period_passmap = df_match_events[(df_match_events['minute']>=from_time_map) & (df_match_events['minute']<=to_time_map)]
with col_home_pass_maps:
    df_passes_map_home = prepare_passes_each_player(df=df_period_passmap,team_name=home_name)
    plot_pitch_passes(df_passes_map_home,team_name=home_name,color_node='blue')
    st.pyplot()
with col_away_pass_maps:
    df_passes_map_away = prepare_passes_each_player(df=df_period_passmap,team_name=away_name)
    plot_pitch_passes(df_passes_map_away,team_name=away_name,color_node='red')
    st.pyplot()

# Visualize the danger passes
st.markdown(f"## Danger Passes Analysis")
col_home_danger_pass,col_away_danger_pass = st.columns(2)
with col_home_danger_pass:
    danger_passes_home_df = get_danger_passes_team(df_match_events,home_name)
    plot_danger_passes(danger_passes=danger_passes_home_df,team_name=home_name,color_node='blue')
    st.pyplot()
    players_danger_passes_home = danger_passes_home_df.groupby(["player_name"]).agg({'index':'count'}).rename(columns={'index':'Number of Danger Passes'})
    # st.bar_chart(players_danger_passes_home)
    fig_home = plot_danger_passes_bar_chart(players_danger_passes_home,home_name,color_node='blue')
    st.plotly_chart(fig_home, use_container_width=True)
    
    # Plot the directed danger passes per selected player for the home team
    home_option = st.selectbox(
        f"Select a {home_name}'s player",
        danger_passes_home_df['player_name'].unique()
    )
    home_player = danger_passes_home_df[danger_passes_home_df['player_name']==home_option]
    plot_danger_passes(danger_passes=home_player,team_name=home_name,color_node='blue')
    st.pyplot()
    
with col_away_danger_pass:
    danger_passes_away_df = get_danger_passes_team(df_match_events,away_name)
    plot_danger_passes(danger_passes=danger_passes_away_df,team_name=away_name,color_node='red')
    st.pyplot()
    players_danger_passes_away = danger_passes_away_df.groupby(["player_name"]).agg({'index':'count'}).rename(columns={'index':'Number of Danger Passes'})
    # st.bar_chart(players_danger_passes_away)
    fig_away = plot_danger_passes_bar_chart(players_danger_passes_away,away_name,color_node='red')
    st.plotly_chart(fig_away, use_container_width=True)

    # Plot the directed danger passes per selected player for the away team
    away_option = st.selectbox(
        f"Select a {away_name}'s player",
        danger_passes_away_df['player_name'].unique()
    )
    away_player = danger_passes_away_df[danger_passes_away_df['player_name']==away_option]
    plot_danger_passes(danger_passes=away_player,team_name=away_name,color_node='red')
    st.pyplot()
