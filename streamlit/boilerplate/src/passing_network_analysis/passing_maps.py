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
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

#prepare the dataframe of passes by the team that were no-throw ins
def prepare_passes_each_player(df,team_name):
    mask_team = (df['type_name'] == 'Pass') & (df['team_name'] == team_name)
    #taking necessary columns
    df_pass_details = df.loc[mask_team, ['location','pass_end_location','player_name','pass_recipient_name']]
    #get the list of all players who made a pass
    names = df_pass_details['player_name'].unique()
    names = [name.split(" ")[-1] for name in names]

    df_pass_details['player_name'] = df_pass_details['player_name'].apply(lambda x: str(x).split(" ")[-1])
    df_pass_details['pass_recipient_name'] = df_pass_details['pass_recipient_name'].apply(lambda x: str(x).split(" ")[-1])

    df_pass_details['x'] = [axes[0] for axes in df_pass_details['location'].values]
    df_pass_details['y'] = [axes[1] for axes in df_pass_details['location'].values]

    df_pass_details['end_x'] = [axes[0] for axes in df_pass_details['pass_end_location'].values]
    df_pass_details['end_y'] = [axes[1] for axes in df_pass_details['pass_end_location'].values]

    df_passes = df_pass_details[['x', 'y', 'end_x', 'end_y','player_name','pass_recipient_name']]
    return df_passes 

#draw 4x4 pitches
def plot_pitch_passes(df_passes,team_name=None,color_node='blue'):
    pitch = Pitch(line_color='black', pad_top=20)
    names = df_passes['player_name'].unique()
    fig, axs = pitch.grid(ncols = 4, nrows = int(len(names)/4), grid_height=0.85, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0.04, endnote_space=0.01)

    #for each player
    for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
        # #put player name over the plot
        ax.text(60, -10, name,
                ha='center', va='center', fontsize=14)
        #take only passes by this player
        player_df = df_passes.loc[df_passes["player_name"] == name]
        #scatter
        pitch.scatter(player_df.x, player_df.y, alpha = 0.2, s = 50, color = color_node, ax=ax)
        #plot arrow
        pitch.arrows(player_df.x, player_df.y,
                player_df.end_x, player_df.end_y, color = color_node, ax=ax, width=1)

    #We have more than enough pitches - remove them
    for ax in axs['pitch'][-1, 16:]:
        ax.remove()

    #Another way to set title using mplsoccer
    axs['title'].text(0.5, 0.5, f'{team_name} passes per player', ha='center', va='center', fontsize=20)
    # plt.show()