import pandas as pd
import numpy as np
import warnings
import os
import scipy.signal as signal
import json
import glob


warnings.filterwarnings('ignore')

###### before

def convert_data(game_id,team_mapping_dict, home_name, away_name):
    # print(os.getcwd())
    game_id = int(game_id)
    teams = [home_name, away_name]

    for team in teams:
        print(team)
        path = "final_data/tracking_player.csv"
        df = pd.read_csv(path)

        df['new_time'] = 0
        df.loc[(df['period'] == 1), 'new_time'] = df['time'] 
        df.loc[(df['period'] == 2), 'new_time'] = round(df['time'] + round(df[df['period'] == 1]["time"].tolist()[-1],2),2) # 2829.24
        df["time"] = df['new_time']

        player_home = pd.read_csv("final_data/player.csv")
        player_home = player_home[player_home["teamSsiId"] == team_mapping_dict[team]]
        player_ids_home = player_home["ssiId"].tolist()



        cols_lst_home = []
        for i in player_ids_home:
            col_x = "Home_"+i+"_x"
            col_y = "Home_"+i+"_y"
            cols_lst_home.append(col_x)
            cols_lst_home.append(col_y)
        cols_lst_away = []


        final = pd.DataFrame(columns=['Frame', 'Period', 'Time [s]'] + cols_lst_home) 
        first_half = len(df[(df['period'] == 1) & (df['game_id'] == game_id) ]["time"].unique())
        second_half = len(df[(df['period'] == 1) & (df['game_id'] == game_id) ]["time"].unique()) + 1
        final["Time [s]"] = np.round(np.arange(0, 6600, 0.04),2)#df.time.unique()
        final["Period"].iloc[0:first_half] =1
        final["Period"].iloc[second_half:] =2

        for tt in range(0,len(player_ids_home)):
            count = tt
            i = player_ids_home[tt]
            print(i)
            col_x = "Home_"+i+"_x"
            col_y = "Home_"+i+"_y"
            p_df = df[(df["game_id"] == game_id) & (df["player_id"] == i) ]
            p_df = df[(df["game_id"] == game_id) & (df["player_id"] ==i) ]
            p_df = p_df[['period', 'time','loc_x','loc_y']]
            if( len(p_df) == 0 ):
                continue
            p_df = p_df.rename(columns={
                'time': 'Time [s]',
                'loc_x' : col_x ,
                'loc_y': col_y
            })
            print('here')
            p_df = p_df.drop_duplicates(subset=["Time [s]"])

            final = final.merge(p_df, how='left', left_on=["Time [s]"], right_on=["Time [s]"])
            final[col_x] = final[col_x +'_y'].fillna(final[col_x+'_x'])
            final = final.drop([col_x+'_x', col_x+'_y'], axis=1)
            final[col_y] = final[col_y +'_y'].fillna(final[col_y+'_x'])
            final = final.drop([col_y+'_x', col_y+'_y'], axis=1)


        final = final[final.columns.drop(list(final.filter(regex='period')))]
        final.to_csv(f"final_data/final_{team}_{game_id}.csv")


def remove_player_velocities(team):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
    team = team.drop(columns=columns)
    return team

def calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
    """ calc_player_velocities( tracking_data )
    
    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
    
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returrns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)
    
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['Home'] ] )

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()
    
    # index of first frame in second half
    second_half_idx = len(team[team['Period'] == 1]["Time [s]"].tolist()) + 1
    
    # estimate velocities for players in team
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[player+"_x"].diff() / dt
        vy = team[player+"_y"].diff() / dt

        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan
            
        if smoothing:
            if filter_=='Savitzky-Golay':
                # calculate first half velocity
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)        
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window 
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' ) 
                vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )      
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' ) 
                vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' ) 
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

    return team


def to_single_playing_direction(home):
    '''
    Flip coordinates in second half so that each team always shoots in the same direction through the match.
    '''
    for team in [home]:
        print(team.Period)
        second_half_idx = len(team[team['Period'] == 1]["Time [s]"].tolist()) + 1
        columns = [c for c in team.columns if c[-1].lower() in ['x','y']]
        team.loc[second_half_idx:,columns] *= -1
    return home


def read_preprocessed_data(path):
    tracking_home = pd.read_csv(path)

    tracking_home = tracking_home.loc[:, ~tracking_home.isnull().all()]
    tracking_home = to_single_playing_direction(tracking_home)
    player_info = pd.read_csv("final_data/player.csv") # map id to name
    player_meta = pd.read_csv("final_data/players_match_meta.csv") # map id to position
    players_in_match = []
    players_name_and_pos = {}
    players_name_and_pos_conc = {}
    for i in tracking_home.columns[3:].tolist():
        p_val = i.split("_")[1]
        if(p_val not in players_in_match):
            players_in_match.append(p_val)
        else:
            continue


    for i in players_in_match:
        players_name_and_pos[i] = [player_info[player_info["ssiId"] == i]["name"].tolist()[0],
                                player_meta[player_meta["ssiId"] == i]["position"].tolist()[0]]
        
        


    for i in players_name_and_pos:
        players_name_and_pos_conc[i] = players_name_and_pos[i][0] + " - " +players_name_and_pos[i][1]

    tracking_home = calc_player_velocities(tracking_home, smoothing=True)
    return tracking_home, players_name_and_pos_conc


def calc_dist_df(tracking_home, players_name_and_pos_conc):
    # get home players
    home_players = players_name_and_pos_conc.keys()
    home_summary = pd.DataFrame(index=home_players) 

    #calculating minutes played
    minutes_home = []
    for player in home_players:
        # search for first and last frames that we have a position observation for each player (when a player is not on the pitch positions are NaN)
        column = 'Home_' + player + '_x' # use player x-position coordinate
        player_minutes = (tracking_home[column].last_valid_index() - tracking_home[column].first_valid_index() + 1) / 25 / 60. # convert to minutes
        minutes_home.append( player_minutes )
    home_summary['Minutes Played'] = minutes_home
    home_summary = home_summary.sort_values(['Minutes Played'], ascending=False)


    #calculating distance covered
    distance_home = []
    for player in home_summary.index:
        column = 'Home_' + player + '_speed'
        player_distance = tracking_home[
                            column].sum() / 25. / 1000  # this is the sum of the distance travelled from one observation to the next (1/25 = 40ms) in km.
        distance_home.append(player_distance)
    home_summary['Distance [km]'] = distance_home
    home_summary['Name_Pos'] = home_summary.index
    home_summary = home_summary.replace({"Name_Pos": players_name_and_pos_conc})


    return home_summary


def dist_plot(tracking_home, home_summary, players_name_and_pos_conc):
    home_summary['Team'] = 'Home'

    #create summary dataframe to make a plot
    game_summary = pd.concat([home_summary])
    game_summary['isSub'] = np.where(game_summary['Minutes Played']== max(game_summary['Minutes Played']),0,1)
    game_summary_sorted = game_summary.sort_values(['Team', 'Distance [km]'], ascending=[False, False])
    game_summary_sorted['Player'] = game_summary_sorted.index
    #star mean that player was subbed in or of
    game_summary_sorted["Name_Pos"] = game_summary_sorted["Player"]
    game_summary_sorted = game_summary_sorted.replace({"Name_Pos": players_name_and_pos_conc})
    game_summary_sorted['Name_Pos'] = np.where(game_summary_sorted['isSub']==0, game_summary_sorted['Name_Pos'], game_summary_sorted['Name_Pos']+ r" $\bf{SUB}$")

    walking = []
    jogging = []
    running = []
    sprinting = []
    for player in home_summary.index:
        column = 'Home_' + player + '_speed'
        # walking (less than 2 m/s)
        player_distance = tracking_home.loc[tracking_home[column] < 2, column].sum() / 25. / 1000
        walking.append(player_distance)
        # jogging (between 2 and 4 m/s)
        player_distance = tracking_home.loc[
                            (tracking_home[column] >= 2) & (tracking_home[column] < 4), column].sum() / 25. / 1000
        jogging.append(player_distance)
        # running (between 4 and 7 m/s)
        player_distance = tracking_home.loc[
                            (tracking_home[column] >= 4) & (tracking_home[column] < 7), column].sum() / 25. / 1000
        running.append(player_distance)
        # sprinting (greater than 7 m/s)
        player_distance = tracking_home.loc[tracking_home[column] >= 7, column].sum() / 25. / 1000
        sprinting.append(player_distance)

    home_summary['Walking'] = walking
    home_summary['Jogging'] = jogging
    home_summary['Running'] = running
    home_summary['Sprinting'] = sprinting
    home_summary['Name_Pos'] = home_summary.index
    home_summary = home_summary.replace({"Name_Pos": players_name_and_pos_conc})
    return home_summary


def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])

#function to calculate metabolic cost
def metabolic_cost(acc): #https://jeb.biologists.org/content/221/15/jeb182303
    if acc > 0:
        cost = 0.102 * ((acc ** 2 + 96.2) ** 0.5) * (4.03 * acc + 3.6 * np.exp(-0.408 * acc))
    elif acc < 0:
        cost = 0.102 * ((acc ** 2 + 96.2) ** 0.5) * (-0.85 * acc + 3.6 * np.exp(1.33 * acc))
    else:
        cost = 0
    return cost



def acc_decc_ratio(tracking_home, home_summary, players_name_and_pos_conc):
    print("start")
    maxacc = 6
    home_acc_dict = {}
    home_players = players_name_and_pos_conc.keys()
    dt = tracking_home['Time [s]'].diff()

    for player in home_players:
        print(player)
        #calculate acceleration

        tracking_home['Home_' + player + '_Acc'] = tracking_home['Home_' + player + '_speed'].diff() / dt
        #set acceleration condition
        tracking_home['Home_' + player + '_Acc'].loc[np.absolute(tracking_home['Home_' + player + '_Acc']) > maxacc] = np.nan
        ##check if acceleration was high or low
        tracking_home['Home_' + player + '_Acc_type'] = np.where(np.absolute(tracking_home['Home_' + player + '_Acc']) >= 2,
                                                                "High", "Low")
        tracking_home['Home_' + player + '_Acc_g'] = tracking_home['Home_' + player + '_Acc_type'].ne(
            tracking_home['Home_' + player + '_Acc_type'].shift()).cumsum()

        #for each player
        for g in np.unique(tracking_home['Home_' + player + '_Acc_g']):
            acc_temp = tracking_home[tracking_home['Home_' + player + '_Acc_g'] == g]
            if acc_temp['Home_' + player + '_Acc_type'].iloc[0] == 'High':
                #get the acceleration period
                acc_duration = round(max(acc_temp['Time [s]']) - min(acc_temp['Time [s]']), 2)
                #check if it was acceleration or deceleration
                acc_or_dec = np.where(np.mean(acc_temp['Home_'+player+'_Acc']) > 0, "Acc", "Dec")
                #create a dictionary
                home_acc_dict[len(home_acc_dict) + 1] = {'Player': player, 'Group': g, 'Duration': acc_duration,
                                                        'Type': acc_or_dec}

    home_acc_df = pd.DataFrame.from_dict(home_acc_dict,orient='index')
    #get accelerations that were longer than 0.75 sec
    home_acc_df1 = home_acc_df[home_acc_df['Duration']>=.75]
    print("mid")

    #calculate ratio for each player fo the home team
    accdec = []
    for player in home_players:
        accs = home_acc_df1[(home_acc_df1['Player']==player) & (home_acc_df1['Type']=='Acc')].count()[0]
        decs = home_acc_df1[(home_acc_df1['Player']==player) & (home_acc_df1['Type']=='Dec')].count()[0]
        ac_ratio = accs / decs
        accdec.append(ac_ratio)
    #saving it in a dataframe
    home_summary['AccDec'] = accdec
    print("end")
    print(home_summary)
    return home_summary







############# after


def read_data_1(path):
    tracking_home = pd.read_csv(path)

    tracking_home = tracking_home.loc[:, ~tracking_home.isnull().all()]
    tracking_home = to_single_playing_direction(tracking_home)

    player_info = pd.read_csv("final_data/player.csv") # map id to name
    player_meta = pd.read_csv("final_data/players_match_meta.csv") # map id to position

    players_in_match = []
    players_name_and_pos = {}
    players_name_and_pos_conc = {}
    for i in tracking_home.columns[3:].tolist():
        p_val = i.split("_")[1]
        if(p_val not in players_in_match):
            players_in_match.append(p_val)
        else:
            continue


    for i in players_in_match:
        players_name_and_pos[i] = [player_info[player_info["ssiId"] == i]["name"].tolist()[0],
                                player_meta[player_meta["ssiId"] == i]["position"].tolist()[0]]
        
        


    for i in players_name_and_pos:
        players_name_and_pos_conc[i] = players_name_and_pos[i][0] + " - " +players_name_and_pos[i][1]

    # Calculate the Player Velocities
    player_ids = np.unique(list(c[:-2] for c in tracking_home.columns if c[:4] in ['Home']))
    #impossible to run faster than 12 m/s
    maxspeed = 12
    dt = tracking_home['Time [s]'].diff()
    #get first frame of second half
    second_half_idx = len(tracking_home[tracking_home['Period'] == 1]["Time [s]"].tolist()) + 1

    tracking_home_unsmoothed = calc_player_velocities(tracking_home, smoothing=False)

    tracking_home = calc_player_velocities(tracking_home, smoothing=True)


    return tracking_home, players_name_and_pos_conc, second_half_idx


def dist_1(tracking_home, players_name_and_pos_conc):
    # get home players
    home_players = players_name_and_pos_conc.keys()
    home_summary = pd.DataFrame(index=home_players) 

    #calculating minutes played
    minutes_home = []
    for player in home_players:
        # search for first and last frames that we have a position observation for each player (when a player is not on the pitch positions are NaN)
        column = 'Home_' + player + '_x' # use player x-position coordinate
        player_minutes = (tracking_home[column].last_valid_index() - tracking_home[column].first_valid_index() + 1) / 25 / 60. # convert to minutes
        minutes_home.append( player_minutes )
    home_summary['Minutes Played'] = minutes_home
    home_summary = home_summary.sort_values(['Minutes Played'], ascending=False)

    #calculating distance covered
    distance_home = []
    for player in home_summary.index:
        column = 'Home_' + player + '_speed'
        player_distance = tracking_home[
                            column].sum() / 25. / 1000  # this is the sum of the distance travelled from one observation to the next (1/25 = 40ms) in km.
        distance_home.append(player_distance)
    home_summary['Distance [km]'] = distance_home
    home_summary['Name_Pos'] = home_summary.index
    home_summary = home_summary.replace({"Name_Pos": players_name_and_pos_conc})


    home_summary['Team'] = 'Home'

    #create summary dataframe to make a plot
    game_summary = pd.concat([home_summary])
    game_summary['isSub'] = np.where(game_summary['Minutes Played']== max(game_summary['Minutes Played']),0,1)
    game_summary_sorted = game_summary.sort_values(['Team', 'Distance [km]'], ascending=[False, False])
    game_summary_sorted['Player'] = game_summary_sorted.index
    #star mean that player was subbed in or of
    game_summary_sorted["Name_Pos"] = game_summary_sorted["Player"]
    game_summary_sorted = game_summary_sorted.replace({"Name_Pos": players_name_and_pos_conc})
    game_summary_sorted['Name_Pos'] = np.where(game_summary_sorted['isSub']==0, game_summary_sorted['Name_Pos'], game_summary_sorted['Name_Pos']+ r" $\bf{SUB}$")

    walking = []
    jogging = []
    running = []
    sprinting = []
    for player in home_summary.index:
        column = 'Home_' + player + '_speed'
        # walking (less than 2 m/s)
        player_distance = tracking_home.loc[tracking_home[column] < 2, column].sum() / 25. / 1000
        walking.append(player_distance)
        # jogging (between 2 and 4 m/s)
        player_distance = tracking_home.loc[
                            (tracking_home[column] >= 2) & (tracking_home[column] < 4), column].sum() / 25. / 1000
        jogging.append(player_distance)
        # running (between 4 and 7 m/s)
        player_distance = tracking_home.loc[
                            (tracking_home[column] >= 4) & (tracking_home[column] < 7), column].sum() / 25. / 1000
        running.append(player_distance)
        # sprinting (greater than 7 m/s)
        player_distance = tracking_home.loc[tracking_home[column] >= 7, column].sum() / 25. / 1000
        sprinting.append(player_distance)

    home_summary['Walking'] = walking
    home_summary['Jogging'] = jogging
    home_summary['Running'] = running
    home_summary['Sprinting'] = sprinting
    home_summary['Name_Pos'] = home_summary.index
    home_summary = home_summary.replace({"Name_Pos": players_name_and_pos_conc})

    return home_summary



def acc_1(tracking_home, home_summary,players_name_and_pos_conc):
    home_players = players_name_and_pos_conc.keys()
    maxacc = 6
    home_acc_dict = {}
    dt = tracking_home['Time [s]'].diff()

    for player in home_players:
        #calculate acceleration

        tracking_home['Home_' + player + '_Acc'] = tracking_home['Home_' + player + '_speed'].diff() / dt
        #set acceleration condition
        tracking_home['Home_' + player + '_Acc'].loc[np.absolute(tracking_home['Home_' + player + '_Acc']) > maxacc] = np.nan
        ##check if acceleration was high or low
        tracking_home['Home_' + player + '_Acc_type'] = np.where(np.absolute(tracking_home['Home_' + player + '_Acc']) >= 2,
                                                                "High", "Low")
        tracking_home['Home_' + player + '_Acc_g'] = tracking_home['Home_' + player + '_Acc_type'].ne(
            tracking_home['Home_' + player + '_Acc_type'].shift()).cumsum()

        #for each player
        for g in np.unique(tracking_home['Home_' + player + '_Acc_g']):
            acc_temp = tracking_home[tracking_home['Home_' + player + '_Acc_g'] == g]
            if acc_temp['Home_' + player + '_Acc_type'].iloc[0] == 'High':
                #get the acceleration period
                acc_duration = round(max(acc_temp['Time [s]']) - min(acc_temp['Time [s]']), 2)
                #check if it was acceleration or deceleration
                acc_or_dec = np.where(np.mean(acc_temp['Home_'+player+'_Acc']) > 0, "Acc", "Dec")
                #create a dictionary
                home_acc_dict[len(home_acc_dict) + 1] = {'Player': player, 'Group': g, 'Duration': acc_duration,
                                                        'Type': acc_or_dec}

    home_acc_df = pd.DataFrame.from_dict(home_acc_dict,orient='index')
    #get accelerations that were longer than 0.75 sec
    home_acc_df1 = home_acc_df[home_acc_df['Duration']>=.75]


    #calculate ratio for each player fo the home team
    accdec = []
    for player in home_players:
        accs = home_acc_df1[(home_acc_df1['Player']==player) & (home_acc_df1['Type']=='Acc')].count()[0]
        decs = home_acc_df1[(home_acc_df1['Player']==player) & (home_acc_df1['Type']=='Dec')].count()[0]
        ac_ratio = accs / decs
        accdec.append(ac_ratio)
    #saving it in a dataframe
    home_summary['AccDec'] = accdec

    return home_summary




############## Statsbomb

def read_statsbomb(file,team):
    with open(f"data/StatsBomb/Data/{file}", 'r') as f:
        data = json.load(f)
    # Convert the list of JSON values to a DataFrame
    df = pd.json_normalize(data)

    mask_mcw_pressure = (df['team.name'] == team) & (df['type.name'] == 'Pressure')
    df_pressure  = df.loc[mask_mcw_pressure, ['location','player.name','pass.recipient.name']]
    df_pressure ['x'] = [axes[0] for axes in df_pressure ['location'].values]
    df_pressure ['y'] = [axes[1] for axes in df_pressure ['location'].values]




    mask_mcw_pressure = (df['team.name'] == team) & (df['type.name'] == 'Pass')
    df_pass  = df.loc[mask_mcw_pressure, ['location','pass.end_location','player.name','pass.recipient.name']]
    df_pass ['x'] = [axes[0] for axes in df_pass ['location'].values]
    df_pass ['y'] = [axes[1] for axes in df_pass ['location'].values]
    df_pass ['end_x'] = [axes[0] for axes in df_pass ['pass.end_location'].values]
    df_pass ['end_y'] = [axes[1] for axes in df_pass ['pass.end_location'].values]
    return df_pressure, df_pass





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