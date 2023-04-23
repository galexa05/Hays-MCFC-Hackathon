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
import plotly.express as px



#taking necessary columns
def get_mask_df(df,mask_pass):
    df_pass_details = df.loc[mask_pass, ['location','pass_end_location','player_name','pass_recipient_name','minute', 'second', 'index']]

    df_pass_details['x'] = [axes[0] for axes in df_pass_details['location'].values]
    df_pass_details['y'] = [axes[1] for axes in df_pass_details['location'].values]

    df_pass_details['end_x'] = [axes[0] for axes in df_pass_details['pass_end_location'].values]
    df_pass_details['end_y'] = [axes[1] for axes in df_pass_details['pass_end_location'].values]

    df_pass = df_pass_details[['x', 'y', 'end_x', 'end_y','player_name','pass_recipient_name','minute', 'second','index']]
    return df_pass

#declare an empty dataframe
def get_danger_passes_team(df,team):
    danger_passes = pd.DataFrame()
    for period in [1, 2]:
        mask_pass = \
        (df['team_name'] == team) \
        & (df['type_name'] == "Pass") \
        & (~df['pass_recipient_name'].isnull()) \
        & (df['period'] == period) \
        # & (df.sub_type_name.isnull())
        #keep only necessary columns
        passes = get_mask_df(df,mask_pass)

        
        #keep only Shots by England in this period
        mask_shot = (df.team_name == team) & (df.type_name == "Shot") & (df.period == period)
        #keep only necessary columns
        shots = df.loc[mask_shot, ["minute", "second"]]
        #convert time to seconds
        shot_times = shots['minute']*60+shots['second']
        shot_window = 15
        #find starts of the window
        shot_start = shot_times - shot_window
        #condition to avoid negative shot starts
        shot_start = shot_start.apply(lambda i: i if i>0 else (period-1)*45)
        #convert to seconds
        pass_times = passes['minute']*60+passes['second']
        #check if pass is in any of the windows for this half
        pass_to_shot = pass_times.apply(lambda x: True in ((shot_start < x) & (x < shot_times)).unique())

        #keep only danger passes
        danger_passes_period = passes.loc[pass_to_shot]
        #concatenate dataframe with a previous one to keep danger passes from the whole tournament
        danger_passes = pd.concat([danger_passes, danger_passes_period], ignore_index = True)
    return danger_passes


def plot_danger_passes(danger_passes,team_name,color_node='blue'):
    #plot pitch
    pitch = Pitch(line_zorder=2,line_color='black')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    
    # Create bins for the football pitch and draw the bins based on the number of located passes
    #get the 2D histogram
    bin_statistic = pitch.bin_statistic(danger_passes.x, danger_passes.y, statistic='count', bins=(6, 5), normalize=False)
    #normalize by number of games
    bin_statistic["statistic"] = bin_statistic["statistic"]
    pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax['pitch'])

    #scatter the location on the pitch
    pitch.scatter(danger_passes.x, danger_passes.y, s=100, color=color_node, edgecolors='grey', linewidth=1, alpha=0.2, ax=ax["pitch"])
    #uncomment it to plot arrows
    pitch.arrows(danger_passes.x, danger_passes.y, danger_passes.end_x, danger_passes.end_y, color = color_node, ax=ax['pitch'])
    #add title
    fig.suptitle('Location of danger passes by ' + team_name, fontsize = 30)


def plot_danger_passes_bar_chart(danger_passes,team_name,color_node='blue'):
    # Create bar chart with custom colors using Plotly
    # import pdb;pdb.set_trace()
    danger_passes = danger_passes.reset_index()
    fig = px.bar(danger_passes, x='player_name', y='Number of Danger Passes', color='player_name')
    fig.update_xaxes(title='Player Name')
    fig.update_yaxes(title='Number of Danger Passes')
    fig.update_layout(title=f'Location of danger passes by {team_name}',
                    xaxis_tickangle=-45,
                    margin=dict(t=50, b=50, l=50, r=50))
    # Set the same color for all bars
    fig.update_traces(marker=dict(color=color_node))
    return fig