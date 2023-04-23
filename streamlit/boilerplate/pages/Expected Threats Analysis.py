# Contents of ~/my_app/pages/page_2.py
import streamlit as st
from src.expected_threat_analysis.threat_actions import *
from src.passing_network_analysis.functions import *
import os
import pandas as pd
import numpy as np
import json
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import networkx as nx
import glob
import warnings
import plotly.graph_objects as go
from pickle import dump, load
import gzip
# from joblib import dump, load
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
warnings.filterwarnings("ignore")
# import streamlit.secrets as secrets


# Set page configuration
st.set_page_config(page_title="Expected Threat (xT) Analysis", layout="wide", initial_sidebar_state="collapsed")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("# Expected Threat (xT) Analysis")

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

# Get the events from a specified match
df_match_events = get_match_events(df_events,match_dict[match_title]['match_game'])
next_event = df_match_events.shift(-1, fill_value=0)

# Create a column with 1 if the ball was kicked out - to mark when the chain should be stopped.
df_match_events['nextEvent'] = next_event['type_name']
df_match_events['kickedOut'] = np.where(next_event['out']==True,1,0)

# Keep Events from the specified match
keepEvents = \
[
    'Carry',
    'Pass',
    'Duel',
    'Dribble',
    'Clearance',
    'Shot',
    'Foul',
    'Foul Committed',
    'Injury Stoppage',
    'Foul',
    'Substitution',
    'Offside',
    'Interception',
    'Ball Recovery',
    'Ball Receipt*'
]
df_match_events_ = df_match_events[df_match_events['type_name'].isin(keepEvents)]

# Isolate possession chains of events
df_isolated = isolateChains(df_match_events_)
# Get all the events from possession chains that last 15 seconds before the shot
df_isolated = get_possession_cains_before_shot(df_isolated,seconds=15)

# Retrieve all possession Chain that lead to a shot per game
chains_index, shot, shot_patterns_df = get_retrieve_all_possessions_to_shot(df_isolated)
# import pdb;pdb.set_trace()
# Preparing data for modeling
df_isolated_chains = prepareChains(df_isolated)
shot_patterns_df = prepare_coordinates(shot_patterns_df)


st.markdown(f'## {home_name}')
st.markdown(
    f"""
    ### Possession chains leading to a shot
    - Below is a list of all shots made by **{home_name}**, which indicates: 
        - Corresponding Chain ID 
        - Minute in which the chance was made
        - Player who made the shot
        - Halftime period 
        - Number of passes that were made before the shot

    - By selecting a specific record from the list, you can visualize the entire activity of the corresponding chain leading up to the shot. This provides a comprehensive view of the buildup to the chance, allowing for a more detailed analysis of the team's performance.
    """
)
col_df_home_df,col_df_home_plot = st.columns(2)
with col_df_home_df:
    # Present a dataframe with the possession chains that ended to a shot
    home_possession_df = get_team_patterns(shot_patterns_df,home_name)
    home_chain_table,grid_response_home = structure_chains_table(home_possession_df)

with col_df_home_plot:
    # Plot chain of event before shot
    response = get_details_from_chain(shot_patterns_df,grid_response_home,home_name)
    # import pdb;pdb.set_trace()
    shot = response['shot']
    passes = response['passes']
    not_pass = response['not_pass']
    chain_id = response['chain_id']
    plot_passes_before_shot(shot,passes,not_pass,chain_id)
    st.pyplot()   
    # import pdb;pdb.set_trace()

    # line_home_df = shot_patterns_df[(shot_patterns_df['type_name']=='Shot') & (shot_patterns_df['team_name']==home_name)].copy()
    # line_home_df['minute'] = shot_patterns_df['seconds'].apply(lambda x: int(x/60))
    # line_home_df['full_minutes'] = np.where(line_home_df['period']==2,line_home_df['minute']+45, line_home_df['minute'])    

    # min_minute = line_home_df['full_minutes'].min()
    # max_minute = line_home_df['full_minutes'].max()
    # bins = range(min_minute, max_minute + 6, 5)
    # line_home_df['interval'] = pd.cut(line_home_df['full_minutes'], bins=bins, labels=bins[:-1])
    # line_home_df['interval'] = np.where(line_home_df['interval'].isna(),0,line_home_df['interval'])
    # # line_home_df[['interval','full_minutes','shot_statsbomb_xg']]
    # grouped = line_home_df.groupby('interval').agg({'shot_statsbomb_xg':'sum','index':'count'}).reset_index()
    # # Create figure with secondary y-axis
    # fig = go.Figure()

    # fig.add_trace(
    #     go.Bar(x=grouped['interval'], y=grouped['index'], name='Number of shots', marker_color='#1f77b4')
    # )

    # fig.add_trace(
    #     go.Scatter(x=grouped['interval'], y=grouped['shot_statsbomb_xg'], name='Total xGoal score', yaxis='y2', line_color='#ff7f0e')
    # )

    # # Set layout properties
    # fig.update_layout(
    #     title='Combined total xGoal score with the number of shots',
    #     yaxis=dict(
    #         title='Number of shots',
    #         titlefont=dict(color='#1f77b4'),
    #         tickfont=dict(color='#1f77b4')
    #     ),
    #     yaxis2=dict(
    #         title='Total xGoal score',
    #         titlefont=dict(color='#ff7f0e'),
    #         tickfont=dict(color='#ff7f0e'),
    #         overlaying='y',
    #         side='right'
    #     )
    # )

    # # Display plot with 100% width
    # st.plotly_chart(fig, width="100%")


st.markdown(
    f"""
    ### xT(Expected Threat) score per player
    - Below is a list of players from **{home_name}** that summarizes their performance based on their xThreat score.
    - The list displays each player's:
        - Total xThreat score for the match
        - Average xThreat score based on their created attempts
        - Total attempts at creating chances
        - Rank based on their total xThreat score
    - By selecting a specific record from the list, you can visualize the entire activity of the corresponding chain leading up to the chance creation.
    - This allows for a more detailed analysis of the player's performance and the team's overall strategy.    """
)
col_summary_df_home,col_summary_plot_home = st.columns(2)
with col_summary_df_home:
    df_coordinates,var = prepare_data_modeling_xgb(df_isolated_chains,home_name,home_possession_df)
    ### TRAINING, it's not perfect ML procedure, but results in AUC 0.2 higher than Logistic Regression ###
    passes = df_coordinates.loc[ df_coordinates["type_name"].isin(["Pass"])]
    
    X = passes[var]
    y = passes["end_shot"]

    xgb_model, ols_model =  load_ml_models(passes,X,y)
    
    # Predict the shooting probability
    y_pred_proba = xgb_model.predict_proba(X)[::,1]
    # Prepare data for the expected goal (xGoal) probability
    passes["shot_prob"] = y_pred_proba
    
    # predict the xg goal 
    y_pred = ols_model.predict(X)

    passes["xG_pred"] = y_pred
    #calculate xThreat score
    passes["xT"] = passes["xG_pred"]*passes["shot_prob"]

    tab_player, tab_possession_chain = st.tabs(["Player Rank", "Select a possession chain"])
    with tab_player:
        # import pdb;pdb.set_trace()
        # extract an option dataframe of each plaeyr in the home team
        rank_xT = calculate_xt_score(passes,home_name)
        rank_xT.sort_values(by='Rank',inplace=True)
        rank_xT,xt_player_resposne = structure_rank_xT(rank_xT)
        # extract the instances regarding the xThreat scores from the selected player
        selected_player_passes_df = get_passes_xt(passes,xt_player_resposne,team_name=home_name)
        selected_player_passes_df,xt_player_selected_resposne = structure_rank_xT(selected_player_passes_df)
        try:
            selected_chain = xt_player_selected_resposne['selected_rows'][0]['Chain ID']
        except:
            selected_chain = xt_player_selected_resposne['data']['Chain ID'].iloc[0]
    with tab_possession_chain:
        type_chain = st.text_input("Type a possession chain", key="text")
        def clear_text():
            st.session_state["text"] = ""
        st.button("Clear text", on_click=clear_text)
with col_summary_plot_home:
    if type_chain!='':
        try:
            option_chain = int(type_chain)
        except:
            option_chain = selected_chain
    else:
        option_chain = selected_chain
    # option_chain = str(option_chain)
    chain = df_coordinates.loc[(df_coordinates["possession_chain"] == option_chain)]
    #get passes
    passes_in = passes.loc[ 
        (passes["possession_chain"] == option_chain) 
    ]

    max_value = passes_in["xT"].max()
    #get events different than pass
    not_pass = chain.loc[chain["type_name"] != "Pass"].iloc[:-1]
    #shot is the last event of the chain (or should be)
    shot = chain.iloc[-1]
    # import pdb;pdb.set_trace()

    #plot
    pitch = Pitch(line_color='black',pitch_type='custom', pitch_length=120, pitch_width=80, line_zorder = 2)
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #add size adjusted arrows
    for i, row in passes_in.iterrows():
        value = row["xT"]
        #adjust the line width so that the more passes, the wider the line
        line_width = (value / max_value * 10)
        #get angle
        angle = np.arctan((row.y1-row.y0)/((row.x1+0.01)-row.x0)*180/np.pi)
        #plot lines on the pitch
        pitch.arrows(row.x0, row.y0, row.x1, row.y1,
                            alpha=0.6, width=line_width, zorder=2, color="blue", ax = ax["pitch"])
        #annotate text
        ax["pitch"].text((row.x0+row.x1-8)/2, (row.y0+row.y1-4)/2, str(value)[:5], fontweight = "bold", color = "blue", zorder = 4, fontsize = 20, rotation = int(angle))

    #shot
    pitch.arrows(shot.x0, shot.y0,
                shot.x1, shot.y1, width=line_width, color = "red", ax=ax['pitch'], zorder =  3)
    #other passes like arrows
    pitch.lines(not_pass.x0, not_pass.y0, not_pass.x1, not_pass.y1, color = "grey", lw = 1.5, ls = 'dotted', ax=ax['pitch'])
    ax['title'].text(0.5, 0.5, f'Attempts leading to a shot (Chain ID: {option_chain})', ha='center', va='center', fontsize=20)
    st.write('\n')
    st.write('\n') 
    st.write('\n')    
    st.pyplot()



st.markdown(f'## {away_name}')
st.markdown(
f"""
### Possession chains leading to a shot
- Below is a list of all shots made by **{away_name}**, which indicates: 
    - Corresponding Chain ID 
    - Minute in which the chance was made
    - Player who made the shot
    - Halftime period 
    - Number of passes that were made before the shot

- By selecting a specific record from the list, you can visualize the entire activity of the corresponding chain leading up to the shot. This provides a comprehensive view of the buildup to the chance, allowing for a more detailed analysis of the team's performance.
"""
)

col_df_away_df,col_df_away_plot = st.columns(2)
with col_df_away_df:
    away_possession_df = get_team_patterns(shot_patterns_df,away_name)
    away_chain_table,grid_response_away = structure_chains_table(away_possession_df)
with col_df_away_plot:
    response = get_details_from_chain(shot_patterns_df,grid_response_away,away_name)
    shot = response['shot']
    passes = response['passes']
    not_pass = response['not_pass']
    chain_id = response['chain_id']
    plot_passes_before_shot(shot,passes,not_pass,chain_id)
    st.write('\n')
    st.write('\n')
    st.pyplot()

st.markdown(
    f"""
    ### xT(Expected Threat) score per player
    - Below is a list of players from **{away_name}** that summarizes their performance based on their xThreat score.
    - The list displays each player's:
        - Total xThreat score for the match
        - Average xThreat score based on their created attempts
        - Total attempts at creating chances
        - Rank based on their total xThreat score
    - By selecting a specific record from the list, you can visualize the entire activity of the corresponding chain leading up to the chance creation.
    - This allows for a more detailed analysis of the player's performance and the team's overall strategy.    """
)
col_summary_df_away,col_summary_plot_away = st.columns(2)
with col_summary_df_away:
    df_coordinates_away,var = prepare_data_modeling_xgb(df_isolated_chains,away_name,away_possession_df)
    ### TRAINING, it's not perfect ML procedure, but results in AUC 0.2 higher than Logistic Regression ###
    passes_away = df_coordinates_away.loc[ df_coordinates_away["type_name"].isin(["Pass"])]
    
    X = passes_away[var]
    y = passes_away["end_shot"]
    xgb_model, ols_model =  load_ml_models(passes_away,X,y)
    
    # Predict the shooting probability
    y_pred_proba = xgb_model.predict_proba(X)[::,1]
    # Prepare data for the expected goal (xGoal) probability
    passes_away["shot_prob"] = y_pred_proba
    
    # predict the xg goal 
    y_pred_away = ols_model.predict(X)
    passes_away["xG_pred"] = y_pred_away
    #calculate xThreat score
    passes_away["xT"] = passes_away["xG_pred"]*passes_away["shot_prob"]
    # extract an option dataframe of each player in the home team
    tab_player, tab_possession_chain = st.tabs(["Player Rank", "Select a possession chain"])
    with tab_player:
        rank_xT_away = calculate_xt_score(passes_away,away_name)
        rank_xT_away.sort_values(by='Rank',inplace=True)
        rank_xT_away,xt_player_response_away = structure_rank_xT(rank_xT_away,key='rank_players')
        # import pdb;pdb.set_trace()
        # extract the instances regarding the xThreat scores from the selected player
        selected_player_passes_df_away = get_passes_xt(passes_away,xt_player_response_away,team_name=away_name)
        selected_player_passes_df_away,xt_player_selected_resposne_away = structure_rank_xT(selected_player_passes_df_away,key=None)
        try:
            # import pdb;pdb.set_trace()
            selected_chain_away = xt_player_selected_resposne_away['selected_rows'][0]['Chain ID']
        except:
            selected_chain_away = xt_player_selected_resposne_away['data']['Chain ID'].iloc[0]
    with tab_possession_chain:
        type_chain_away = st.text_input("Type a possession chain...", key="chain_away")
        def clear_text():
            st.session_state["chain_away"] = ""
        st.button("Clear text", on_click=clear_text, key="button_clear_away")
with col_summary_plot_away:
    if type_chain_away!='':
        option_chain = int(type_chain_away)
    else:
        option_chain = selected_chain_away
    chain = df_coordinates.loc[(df_coordinates["possession_chain"] == option_chain)]
    #get passes
    passes_in = passes_away.loc[ 
        (passes_away["possession_chain"] == option_chain) 
    ]

    max_value = passes_in["xT"].max()
    #get events different than pass
    not_pass = chain.loc[chain["type_name"] != "Pass"].iloc[:-1]
    #shot is the last event of the chain (or should be)
    shot = chain.iloc[-1]

    #plot
    pitch = Pitch(line_color='black',pitch_type='custom', pitch_length=120, pitch_width=80, line_zorder = 2)
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #add size adjusted arrows
    for i, row in passes_in.iterrows():
        value = row["xT"]
        #adjust the line width so that the more passes, the wider the line
        line_width = (value / max_value * 10)
        #get angle
        angle = np.arctan((row.y1-row.y0)/((row.x1+0.01)-row.x0)*180/np.pi)
        #plot lines on the pitch
        pitch.arrows(row.x0, row.y0, row.x1, row.y1,
                            alpha=0.6, width=line_width, zorder=2, color="blue", ax = ax["pitch"])
        #annotate text
        ax["pitch"].text((row.x0+row.x1-8)/2, (row.y0+row.y1-4)/2, str(value)[:5], fontweight = "bold", color = "blue", zorder = 4, fontsize = 20, rotation = int(angle))

    #shot
    pitch.arrows(shot.x0, shot.y0,
                shot.x1, shot.y1, width=line_width, color = "red", ax=ax['pitch'], zorder =  3)
    #other passes like arrows
    pitch.lines(not_pass.x0, not_pass.y0, not_pass.x1, not_pass.y1, color = "grey", lw = 1.5, ls = 'dotted', ax=ax['pitch'])
    ax['title'].text(0.5, 0.5, f'Attempts leading to a shot (Chain ID: {option_chain})', ha='center', va='center', fontsize=20)
    st.write('\n')
    st.write('\n') 
    st.write('\n')    
    st.pyplot()


    # line_home_df = shot_patterns_df[(shot_patterns_df['type_name']=='Shot') & (shot_patterns_df['team_name']==away_name)].copy()
    # line_home_df['minute'] = shot_patterns_df['seconds'].apply(lambda x: int(x/60))
    # line_home_df['full_minutes'] = np.where(line_home_df['period']==2,line_home_df['minute']+45, line_home_df['minute'])    

    # min_minute = line_home_df['full_minutes'].min()
    # max_minute = line_home_df['full_minutes'].max()
    # bins = range(min_minute, max_minute + 6, 5)
    # line_home_df['interval'] = pd.cut(line_home_df['full_minutes'], bins=bins, labels=bins[:-1])
    # line_home_df['interval'] = np.where(line_home_df['interval'].isna(),0,line_home_df['interval'])
    # # line_home_df[['interval','full_minutes','shot_statsbomb_xg']]
    # grouped = line_home_df.groupby('interval').agg({'shot_statsbomb_xg':'sum','index':'count'}).reset_index()
    # # Create figure with secondary y-axis
    # fig = go.Figure()

    # fig.add_trace(
    #     go.Bar(x=grouped['interval'], y=grouped['index'], name='Number of shots', marker_color='red')
    # )

    # fig.add_trace(
    #     go.Scatter(x=grouped['interval'], y=grouped['shot_statsbomb_xg'], name='Total xGoal score', yaxis='y2', line_color='green')
    # )

    # # Set layout properties
    # fig.update_layout(
    #     title='Combined total xGoal score with the number of shots',
    #     yaxis=dict(
    #         title='Number of shots',
    #         titlefont=dict(color='red'),
    #         tickfont=dict(color='red')
    #     ),
    #     yaxis2=dict(
    #         title='Total xGoal score',
    #         titlefont=dict(color='green'),
    #         tickfont=dict(color='green'),
    #         overlaying='y',
    #         side='right'
    #     )
    # )

    # # Display plot with 100% width
    # st.plotly_chart(fig, width="100%")

