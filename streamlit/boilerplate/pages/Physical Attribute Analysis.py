# Contents of ~/my_app/pages/page_2.py
import streamlit as st
from src.physical_attribute_analysis.functions import *
import os
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import json
from mplsoccer import Pitch, VerticalPitch, FontManager, Sbopen
import ruptures as rpt


# Set page configuration
st.set_page_config(page_title="Physical Attribute Analysis", layout="wide", initial_sidebar_state="collapsed")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("# Physical Attribute Analysis")

match_dict = get_all_match_games()
df_events = get_events_all_matches()

match_mapping_dict = {
"Manchester City WFC vs Arsenal WFC": "2312135",
"Manchester City WFC vs Aston Villa":"0",
"Manchester City WFC vs Brighton & Hove Albion WFC":"2312183",
"Manchester City WFC vs Leicester City WFC":"2312152",
"Manchester City WFC vs Liverpool WFC":"2312166",
"Manchester City WFC vs Tottenham Hotspur Women":"2312213",
}

statsbomb_match_mapping_dict = {
"Manchester City WFC vs Arsenal WFC": "ManCity_Arsenal_events.json",
"Manchester City WFC vs Aston Villa":"ManCity_AstonVilla_events.json",
"Manchester City WFC vs Brighton & Hove Albion WFC":"ManCity_Brighton_events.json",
"Manchester City WFC vs Leicester City WFC":"ManCity_Leicester_events.json",
"Manchester City WFC vs Liverpool WFC":"ManCity_Liverpool_events.json",
"Manchester City WFC vs Tottenham Hotspur Women":"ManCity_Tottenham_events.json",
}

team_mapping_dict = {
    "Manchester City WFC": "edfee15e-0dd7-42bc-be2a-289870187ddc",
    "Arsenal WFC":"a11d34f3-da66-4219-9332-85421e44692f",
    "Brighton & Hove Albion WFC": "36e9ec32-d3ef-429f-b1bf-4cf654f83e70",
    "Leicester City WFC": "e6cbf28a-7fc6-443a-9d14-c24fdde711d0",
    "Liverpool WFC": "4ecd00b0-0b70-4db4-8c56-e402a8785424",
    "Tottenham Hotspur Women" : "cddcbcf6-0c84-435b-90bb-074e72537ad7"
}



# Plase the sidebar at the left side of the webapp giving the option to select your match
match_title = st.sidebar.selectbox(
    "Select a match",
    ([title for title in match_dict.keys()])
)

# Retrieve the events from the selected match game
home_name = match_dict[match_title]['home_name']
away_name = match_dict[match_title]['away_name']

game_id = match_mapping_dict[match_title]


# Visualize the passing network graph
st.markdown(f"{home_name} vs {away_name}")
print("#############")
print("home_name: "  , home_name)
print("away_name: " , away_name)



if(
    (os.path.exists(f"final_data/final_{home_name}_{game_id}.csv")) and (os.path.exists(f"final_data/final_{away_name}_{game_id}.csv"))
   ):

    st.markdown("## Distance Covered by each team member")
    st.markdown("""Knowing the distance covered by each team member can provide valuable insights into their work rate and fitness levels, 
                helping the coaching staff to tailor training programs and make tactical decisions during matches.""")

    col_home_network,col_away_network = st.columns(2)
    with col_home_network:
        st.markdown(f"### {home_name}")
        path_home = f"final_data/final_{home_name}_{game_id}.csv"
        tracking_home, players_name_and_pos_conc_home, second_half_idx_home = read_data_1(path_home)
        home_summary = dist_1(tracking_home, players_name_and_pos_conc_home)
        st.dataframe(home_summary.set_index('Name_Pos'))
        st.bar_chart(home_summary.set_index('Name_Pos')[['Walking','Jogging','Running','Sprinting']])
        ax = home_summary.set_index('Name_Pos')[['Walking','Jogging','Running','Sprinting']].plot.bar( colormap='coolwarm', figsize=(10, 5))
        ax.set_xlabel('Player', fontsize = 14)
        ax.set_ylabel('Distance covered [km]', fontsize = 14)
        ax.set_title(f"Distance Covered At Various Velocity Bands - {home_name}", fontsize = 16)
        st.pyplot()
    
    with col_away_network:
        st.markdown(f"### {away_name}")
        path_away = f"final_data/final_{away_name}_{game_id}.csv"
        tracking_away, players_name_and_pos_conc_away, second_half_idx_away = read_data_1(path_away)
        away_summary = dist_1(tracking_away, players_name_and_pos_conc_away)
        st.dataframe(away_summary.set_index('Name_Pos'))
        st.bar_chart(away_summary.set_index('Name_Pos')[['Walking','Jogging','Running','Sprinting']])
        ax = away_summary.set_index('Name_Pos')[['Walking','Jogging','Running','Sprinting']].plot.bar( colormap='coolwarm', figsize=(10, 5))
        ax.set_xlabel('Player', fontsize = 14)
        ax.set_ylabel('Distance covered [km]', fontsize = 14)
        ax.set_title(f"Distance Covered At Various Velocity Bands - {away_name}", fontsize = 16)
        st.pyplot()




    df_pressure_home, df_pass_home = read_statsbomb(statsbomb_match_mapping_dict[match_title], home_name)
    df_pressure_away, df_pass_away = read_statsbomb(statsbomb_match_mapping_dict[match_title], away_name)


    # setup pitch
    pitch = Pitch(pitch_type='statsbomb', line_zorder=2,
                pitch_color='#22312b', line_color='#efefef')
    
    
    # fontmanager for google font (robotto)
    robotto_regular = FontManager()

    # path effects
    path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()]

    
    st.markdown("## Heatmap for whole team and individual player")
    st.markdown(""" A heatmap for the whole team or where each player moves can help identify patterns in their positioning and movement, 
    allowing the coaching staff to fine-tune tactics and improve team coordination.""")
    col_home_network,col_away_network = st.columns(2)

    with col_home_network:
        st.markdown(f"### {home_name}")
        # draw
        fig, ax = pitch.draw(figsize=(6.6, 4.125))
        fig.set_facecolor('#22312b')
        bin_statistic = pitch.bin_statistic(df_pressure_home.x, df_pressure_home.y, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        # Add the colorbar and format off-white
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
        cbar.outline.set_edgecolor('#efefef')
        cbar.ax.yaxis.set_tick_params(color='#efefef')
        ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')
        st.pyplot()

             # Heatmap for specific team member
        st.markdown(f"### Heatmap for specific team member for {home_name}")

        player_selector = st.selectbox(
        "Select a Player",
        df_pressure_home["player.name"].unique().tolist()
        )
        
        df_pressure_specific = df_pressure_home[df_pressure_home["player.name"] == player_selector]
        fig, ax = pitch.draw(figsize=(6.6, 4.125))
        fig.set_facecolor('#22312b')
        bin_statistic = pitch.bin_statistic(df_pressure_specific.x, df_pressure_specific.y, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        # Add the colorbar and format off-white
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
        cbar.outline.set_edgecolor('#efefef')
        cbar.ax.yaxis.set_tick_params(color='#efefef')
        ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')
        st.pyplot()

        



    with col_away_network:
        st.markdown(f"### {away_name}")
        # draw
        fig, ax = pitch.draw(figsize=(6.6, 4.125))
        fig.set_facecolor('#22312b')
        bin_statistic = pitch.bin_statistic(df_pressure_away.x, df_pressure_away.y, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        # Add the colorbar and format off-white
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
        cbar.outline.set_edgecolor('#efefef')
        cbar.ax.yaxis.set_tick_params(color='#efefef')
        ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')
        st.pyplot()

             # Heatmap for specific team member
        st.markdown(f"### Heatmap for specific team member for {away_name}")

        player_selector = st.selectbox(
        "Select a Player",
        df_pressure_away["player.name"].unique().tolist()
        )
        
        df_pressure_specific = df_pressure_away[df_pressure_away["player.name"] == player_selector]
        fig, ax = pitch.draw(figsize=(6.6, 4.125))
        fig.set_facecolor('#22312b')
        bin_statistic = pitch.bin_statistic(df_pressure_specific.x, df_pressure_specific.y, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        # Add the colorbar and format off-white
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
        cbar.outline.set_edgecolor('#efefef')
        cbar.ax.yaxis.set_tick_params(color='#efefef')
        ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')
        st.pyplot()






    st.markdown("## Heatmap for all pressure events")
    st.markdown(""" A heatmap showing where the team pressured the opposing team more can help identify areas of strength and weakness in the team's defensive strategy, 
    allowing the coaching staff to make adjustments to better control the game.""")
    col_home_network,col_away_network = st.columns(2)

    with col_home_network:       
        st.markdown(f"### {home_name}") 
        pitch_v = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#f4edf0')
        fig, ax = pitch_v.draw(figsize=(4.125, 6))
        fig.set_facecolor('#f4edf0')
        bin_x = np.linspace(pitch_v.dim.left, pitch_v.dim.right, num=7)
        bin_y = np.sort(np.array([pitch_v.dim.bottom, pitch_v.dim.six_yard_bottom,
                                pitch_v.dim.six_yard_top, pitch_v.dim.top]))
        bin_statistic = pitch_v.bin_statistic(df_pressure_home.x, df_pressure_home.y, statistic='count',
                                            bins=(bin_x, bin_y), normalize=True)
        pitch_v.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9')
        labels2 = pitch_v.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                                    ax=ax, ha='center', va='center',
                                    str_format='{:.0%}', path_effects=path_eff)

        st.pyplot()



    with col_away_network:   
        st.markdown(f"### {away_name}")
        pitch_v = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#f4edf0')
        fig, ax = pitch_v.draw(figsize=(4.125, 6))
        fig.set_facecolor('#f4edf0')
        bin_x = np.linspace(pitch_v.dim.left, pitch_v.dim.right, num=7)
        bin_y = np.sort(np.array([pitch_v.dim.bottom, pitch_v.dim.six_yard_bottom,
                                pitch_v.dim.six_yard_top, pitch_v.dim.top]))
        bin_statistic = pitch_v.bin_statistic(df_pressure_away.x, df_pressure_away.y, statistic='count',
                                            bins=(bin_x, bin_y), normalize=True)
        pitch_v.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9')
        labels2 = pitch_v.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                                    ax=ax, ha='center', va='center',
                                    str_format='{:.0%}', path_effects=path_eff)

        st.pyplot()

    # # Acc/Decc ratio
    st.markdown("## Acceleration/Deceleration ratio")
    st.markdown("""
    Understanding the acceleration/deceleration ratio of each player can help identify those who are particularly explosive and quick off the mark, 
    which could be beneficial in certain positions or game situations.
    """)
        

    col_home_network,col_away_network = st.columns(2)
    with col_home_network:
        st.markdown(f"### {home_name}")
        with st.spinner('Wait for it...'):
            home_summary = acc_1(tracking_home, home_summary, players_name_and_pos_conc_home)
            fig, ax = plt.subplots()
            ax.scatter(home_summary['Distance [km]'], home_summary['AccDec'], color = "red", s = 50)
            for i in home_summary.index:
                ax.annotate(str(home_summary[home_summary.index==i]['Name_Pos'].tolist()[0]), 
                            (home_summary[home_summary.index==i]['Distance [km]']+ 0.1, 
                            home_summary[home_summary.index==i]['AccDec'] + 0.005), fontsize = 10)
            ax.set_xlabel("Distance [km]")
            ax.set_ylabel("AccDec Ratio")
            plt.grid()
            plt.title("Acceleration - Deceleration Ratio")
            st.pyplot()
    
    with col_away_network:
        st.markdown(f"### {away_name}")
        with st.spinner('Wait for it...'):
            away_summary = acc_1(tracking_away, away_summary, players_name_and_pos_conc_away)
            fig, ax = plt.subplots()
            ax.scatter(away_summary['Distance [km]'], away_summary['AccDec'], color = "red", s = 50)
            for i in away_summary.index:
                ax.annotate(str(away_summary[away_summary.index==i]['Name_Pos'].tolist()[0]), 
                            (away_summary[away_summary.index==i]['Distance [km]']+ 0.1, 
                            away_summary[away_summary.index==i]['AccDec'] + 0.005), fontsize = 10)
            ax.set_xlabel("Distance [km]")
            ax.set_ylabel("AccDec Ratio")
            plt.grid()
            plt.title("Acceleration - Deceleration Ratio")
            st.pyplot()



    # Metabolic
    st.markdown("## Metabolic power output")
    st.markdown("""
    Measuring the metabolic power output of players can provide valuable information about their energy expenditure and intensity levels, 
    helping the coaching staff to optimise training and match preparation strategies.
    """)


    col_home_network,col_away_network = st.columns(2)
    with col_home_network:
        st.markdown(f"### {home_name}")
        player_selector_metabolic_home = st.selectbox(
        "Select a Player",
        list(players_name_and_pos_conc_home.values())[::-1]
        )
        player_selector_metabolic_mapped = list(filter(lambda x: players_name_and_pos_conc_home[x] == player_selector_metabolic_home, players_name_and_pos_conc_home))[0]
        player_selector_metabolic_mapped_mod = f"Home_{player_selector_metabolic_mapped}"
        print("MET for: ", player_selector_metabolic_mapped_mod)
        print("Half: " , second_half_idx_home)

        with st.spinner('Wait for it...'):
            team = tracking_home
            fig, ax = plt.subplots(figsize = (10, 6))
            # player = 'Home_8c34093f-843c-41a4-b02d-1ed289de00a3'
            #calculate metabolic cost
            mc_temp = list(map(lambda x: metabolic_cost(team[player_selector_metabolic_mapped_mod + '_Acc'][x]), range(0, len(team[player_selector_metabolic_mapped_mod + '_Acc']))))
            #multiply it by speed
            mp_temp = mc_temp * team[player_selector_metabolic_mapped_mod+'_speed']
            print(len(mp_temp))
            #calculate rolling average
            test_mp = mp_temp.rolling(second_half_idx_home,min_periods=1).apply(lambda x : np.nansum(x)) #Use Changepoint Detection Here
            signal = np.array(test_mp[second_half_idx_home:len(test_mp)]).reshape((len(test_mp[second_half_idx_home:len(test_mp)]),1))
            algo = rpt.Binseg(model="l2").fit(signal)  ##potentially finding spot where substitution should happen
            result = algo.predict(n_bkps=1)  # big_seg
            result[0] = np.round(result[0]*0.04/60,2)+int(second_half_idx_home * 0.04 / 60) 
            result[1] = np.round(result[1]*0.04/60,2)+int(second_half_idx_home * 0.04 / 60) 
            diff = int(second_half_idx_home * 0.04 / 60) - 45
            test_mp.index = np.array((test_mp.index)*0.04/60) - diff

            ax.plot(test_mp[int(second_half_idx_home * 0.04 / 60):])
            ax.axvspan(result[0]-diff,result[1]-diff, alpha=0.5, color='red')
            ax.set_title('Metabolic Power Output')
            ax.set_ylabel("Metabolic Power")
            ax.set_xlabel("Time [minutes]")
            st.pyplot()


    with col_away_network:
        st.markdown(f"### {away_name}")
        player_selector_metabolic_away = st.selectbox(
        "Select a Player",
        list(players_name_and_pos_conc_away.values())
        )

        player_selector_metabolic_mapped = list(filter(lambda x: players_name_and_pos_conc_away[x] == player_selector_metabolic_away, players_name_and_pos_conc_away))[0]
        player_selector_metabolic_mapped_mod = f"Home_{player_selector_metabolic_mapped}"
        print("MET for: ", player_selector_metabolic_mapped_mod)
        print("Half: " , second_half_idx_away)

        with st.spinner('Wait for it...'):
            team = tracking_away
            fig, ax = plt.subplots(figsize = (10, 6))
            #calculate metabolic cost
            mc_temp = list(map(lambda x: metabolic_cost(team[player_selector_metabolic_mapped_mod + '_Acc'][x]), range(0, len(team[player_selector_metabolic_mapped_mod + '_Acc']))))
            #multiply it by speed
            mp_temp = mc_temp * team[player_selector_metabolic_mapped_mod+'_speed']
            print(len(mp_temp))
            #calculate rolling average
            test_mp = mp_temp.rolling(second_half_idx_away,min_periods=1).apply(lambda x : np.nansum(x)) #Use Changepoint Detection Here
            signal = np.array(test_mp[second_half_idx_away:len(test_mp)]).reshape((len(test_mp[second_half_idx_away:len(test_mp)]),1))
            algo = rpt.Binseg(model="l2").fit(signal)  ##potentially finding spot where substitution should happen
            result = algo.predict(n_bkps=1)  # big_seg
            result[0] = np.round(result[0]*0.04/60,2)+int(second_half_idx_away * 0.04 / 60) 
            result[1] = np.round(result[1]*0.04/60,2)+int(second_half_idx_away * 0.04 / 60) 
            diff = int(second_half_idx_away * 0.04 / 60) - 45
            test_mp.index = np.array((test_mp.index)*0.04/60) - diff

            ax.plot(test_mp[int(second_half_idx_away * 0.04 / 60):])
            ax.axvspan(result[0]-diff,result[1]-diff, alpha=0.5, color='red')
            ax.set_title('Metabolic Power Output')
            ax.set_ylabel("Metabolic Power")
            ax.set_xlabel("Time [minutes]")
            st.pyplot()

   


else:
    st.markdown("# Generating Data")
    with st.spinner('Wait for it...'):
        convert_data(game_id,team_mapping_dict, home_name, away_name)

    st.success('Done! Please refresh the page')

# df = pd.read_csv(f"final_home_{game_id}.csv")




