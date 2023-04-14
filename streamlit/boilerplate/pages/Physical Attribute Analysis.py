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



st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("# Physical Attribute Analysis")


game_id = st.sidebar.selectbox(
    "Select a match",
    ("2312135", "2312152", "2312166", "2312183", "2312213")
)


if(os.path.exists(f"final_data/final_home_{game_id}.csv")):
    st.markdown("## Distance Covered by each team member")
    path = f"final_data/final_home_{game_id}.csv"

    # Distance DF    
    tracking_home, players_name_and_pos_conc = read_preprocessed_data(path)
    home_summary = calc_dist_df(tracking_home, players_name_and_pos_conc)
    st.dataframe(home_summary.set_index('Name_Pos'))


    # Distance plot
    home_summary = dist_plot(tracking_home, home_summary, players_name_and_pos_conc)
    st.bar_chart(home_summary.set_index('Name_Pos')[['Walking','Jogging','Running','Sprinting']])

    fig, ax = plt.subplots()
    ax = home_summary.set_index('Name_Pos')[['Walking','Jogging','Running','Sprinting']].plot.bar( colormap='coolwarm', figsize=(10, 5))
    ax.set_xlabel('Player', fontsize = 14)
    ax.set_ylabel('Distance covered [km]', fontsize = 14)
    ax.set_title('Distance Covered At Various Velocity Bands - Home Team', fontsize = 16)
    st.pyplot()


    # Acc/Decc ratio

    # home_summary = acc_decc_ratio(tracking_home, home_summary, players_name_and_pos_conc)
    # print("here")
    # print(home_summary)

    # fig, ax = plt.subplots()
    # ax.scatter(home_summary['Distance [km]'], home_summary['AccDec'], color = "red", s = 50)
    # for i in home_summary.index:
    #     ax.annotate(str(home_summary[home_summary.index==i]['Name_Pos'].tolist()[0]), 
    #                 (home_summary[home_summary.index==i]['Distance [km]']+ 0.1, 
    #                 home_summary[home_summary.index==i]['AccDec'] + 0.005), fontsize = 10)
    # ax.set_xlabel("Distance [km]")
    # ax.set_ylabel("AccDec Ratio")
    # plt.grid()
    # plt.title("Acceleration - Deceleration Ratio")
    # st.pyplot()

    # print("here2")


    # Metabolic
   

    # Heatmap for whole team
    

    df_pressure, df_pass = read_statsbomb()
    # setup pitch
    pitch = Pitch(pitch_type='statsbomb', line_zorder=2,
                pitch_color='#22312b', line_color='#efefef')
    

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## Heatmap for whole team")
        # draw
        fig, ax = pitch.draw(figsize=(6.6, 4.125))
        fig.set_facecolor('#22312b')
        bin_statistic = pitch.bin_statistic(df_pressure.x, df_pressure.y, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        # Add the colorbar and format off-white
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
        cbar.outline.set_edgecolor('#efefef')
        cbar.ax.yaxis.set_tick_params(color='#efefef')
        ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')
        st.pyplot()

    with col2:
        # Heatmap for specific team member
        st.markdown("## Heatmap for specific team member")

        player_selector = st.selectbox(
        "Select a Player",
        df_pressure["player.name"].unique().tolist()
        )
        
        df_pressure_specific = df_pressure[df_pressure["player.name"] == player_selector]
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



    # Bins


    # fontmanager for google font (robotto)
    robotto_regular = FontManager()

    # path effects
    path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'),
                path_effects.Normal()]


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Heatmap to bins")
        # col 1
        # see the custom colormaps example for more ideas on setting colormaps
        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#15242e', '#4393c4'], N=10)

        pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#f4edf0')
        fig, ax = pitch.draw(figsize=(4.125, 6))
        fig.set_facecolor('#f4edf0')
        bin_statistic = pitch.bin_statistic(df_pressure.x, df_pressure.y, statistic='count', bins=(6, 5), normalize=True)
        pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9')
        labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                                    ax=ax, ha='center', va='center',
                                    str_format='{:.0%}', path_effects=path_eff)
        st.pyplot()

    with col2:
        st.markdown("### Heatmap to larger bins")
        pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#f4edf0')
        fig, ax = pitch.draw(figsize=(4.125, 6))
        fig.set_facecolor('#f4edf0')
        bin_x = np.linspace(pitch.dim.left, pitch.dim.right, num=7)
        bin_y = np.sort(np.array([pitch.dim.bottom, pitch.dim.six_yard_bottom,
                                pitch.dim.six_yard_top, pitch.dim.top]))
        bin_statistic = pitch.bin_statistic(df_pressure.x, df_pressure.y, statistic='count',
                                            bins=(bin_x, bin_y), normalize=True)
        pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9')
        labels2 = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                                    ax=ax, ha='center', va='center',
                                    str_format='{:.0%}', path_effects=path_eff)

        st.pyplot()

else:
    st.markdown("# not here")
    with st.spinner('Wait for it...'):
        convert_data(game_id)

    st.success('Done!')

# df = pd.read_csv(f"final_home_{game_id}.csv")




