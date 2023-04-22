import pandas as pd
import numpy as np
import json
# plotting
import os
import pathlib
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import glob
import math
from itertools import combinations_with_replacement
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor,XGBClassifier
from sklearn.linear_model import LinearRegression
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from pickle import dump, load
import gzip


pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')




## Isolating possesion chain
def timestamp_to_seconds(timestamp):
    parts = timestamp.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    total_seconds = (hours * 3600) + (minutes * 60) + seconds
    return int(total_seconds)


def isolateChains(df):
    """
    Parameters
    ----------
    df : dataframe
        dataframe with StatsBomb event data.

    Returns
    -------
    df: dataframe
        dataframe with isolated possession chains

    """
    df['seconds'] = df['timestamp'].apply(lambda time: timestamp_to_seconds(time))
    df["nextTeamName"] = df.shift(-1, fill_value=0)["team_name"]
    #potential +0s
    chain_team = df.iloc[0]["team_name"]
    period = df.iloc[0]["period"]
    stop_criterion = 0
    chain = 0
    df["possession_chain"] = 0
    df["possession_chain_team"] = 0

    for i, row in df.iterrows():
        # if (row['index']==283):
        #     import pdb;pdb.set_trace()
        #add value
        df.at[i, "possession_chain"] = chain
        df.at[i, "possession_chain_team"] = chain_team

        if row['team_name'] != chain_team:
            stop_criterion+=1
        if row["type_name"] in ["Shot", "Foul Committed", "Offside",'Injury Stoppage','Substitution']:
            stop_criterion += 2
        if row['type_name'] == 'Ball Receipt*':
            if row['ball_receipt_outcome_name'] == 'Incomplete':
                stop_criterion +=2
        #if ball out of field, add 2
        if row["out"]==True:
            stop_criterion += 2
        # if the ball was properly intercepted that is, the next team is the one that made the next event, we stop the chain.         
        if row['type_name']=='Interception':
            if row['team_name']!=row['nextTeamName']:
                stop_criterion +=2
            else:
                # if the ball was only touched, but did not change possession, we treat a pass as an accurate one.
                stop_criterion = 0
  
        if stop_criterion == 1 and row['team_name'] == chain_team:
            stop_criterion = 0 
        #criterion for stopping when half ended
        
        if row["period"] != period:
                chain += 1
                stop_criterion = 0
                chain_team = row['team_name']
                period = row["period"]
                df.at[i, "possession_chain"] = chain
                df.at[i, "possession_chain_team"] = chain_team
        #possession chain ended
        if stop_criterion >= 2:
            chain += 1
            stop_criterion = 0
            chain_team = row['nextTeamName']
    return df


def prepareChains(df):
    """
    Parameters
    ----------
    df : dataframe
        dataframe with Wyscout event data.

    Returns
    -------
    xG_sum: dataframe
        dataframe with assigned values for chains

    """
    # import pdb;pdb.set_trace()
    df["shot_end"] = 0
    df['xG'] = np.where(df['shot_statsbomb_xg'].isna(),0,df['shot_statsbomb_xg'])
    #get number of chains
    no_chains = max(df["possession_chain"].unique())
    indicies = []
    for i in range(no_chains+1):
        #all events get possession chain
        possession_chain_df = df.loc[df["possession_chain"] == i]
        #check if the possession chain is not empty
        if len(possession_chain_df) > 0:
            #if ended with shot
            if possession_chain_df.iloc[-1]["type_name"] == "Shot":
                #assign values
                df.loc[df["possession_chain"] == i, "shot_end"] = 1
                xG = possession_chain_df.iloc[-1]["xG"]
                df.loc[df["possession_chain"] == i, "xG"] = xG
                #check if the previous ones did not end with foul
                k = i-1
                if k > 0:
                    try:
                        prev = df.loc[df["possession_chain"] == k]
                        #create a loop if e.g. 2 chains before and 1 chain before didn;t end with shot
                        while prev.iloc[-1]["type_name"] == "Foul":
                            #assign value for them
                            df.loc[df["possession_chain"] == k, "xG"] = xG
                            df.loc[df["possession_chain"] == k, "shot_end"] = 1
                            k = k-1
                            prev = df.loc[df["possession_chain"] == k]
                    except:
                        k = k-1
            #get indiices of events made by possession team
            team_indicies = possession_chain_df.loc[possession_chain_df["team_id"] == possession_chain_df['possession_team_id']].index.values.tolist()
            indicies.extend(team_indicies)

    df = df.loc[indicies]
    return df

def get_possession_cains_before_shot(df_isolated,seconds=15):
    df_list = []
    for possession in df_isolated['possession_chain'].unique():
        temp_df = df_isolated[df_isolated['possession_chain']==possession]
        temp_df = temp_df[temp_df['seconds']>= max(temp_df['seconds']-seconds)]
        df_list.append(temp_df)
    df_isolated = pd.concat(df_list)
    return df_isolated

def get_retrieve_all_possessions_to_shot(df_isolated):
    chains_index = df_isolated[df_isolated['type_name']=='Shot']['possession_chain'].unique() 
    shot = df_isolated[df_isolated['type_name']=='Shot']
    df_isolated['end_location'] = df_isolated['carry_end_location'].fillna(df_isolated['shot_end_location']).fillna(df_isolated['pass_end_location'])

    shot_patterns = \
        df_isolated[
            df_isolated['possession_chain'].isin(chains_index)][
            [
        'type_name','location','end_location','possession_chain',
        'team_name','index','shot_statsbomb_xg','match_game','seconds',
        'player_name','period'
            ]
        ]
    shot_patterns['shot_statsbomb_xg'] = shot_patterns['shot_statsbomb_xg'].fillna(0.0)

    shot_patterns_df = {}
    for match_game in shot_patterns.keys():
        patterns_list = [] 
        for c_index in chains_index:
            temp_df = shot_patterns[shot_patterns['possession_chain']==c_index]
            temp_df['xGoal'] = max(temp_df['shot_statsbomb_xg'])
            patterns_list.append(temp_df)
        shot_patterns_df = pd.concat(patterns_list)
    return chains_index, shot, shot_patterns_df


def prepare_coordinates(shot_patterns_df):
    #columns with coordinates
    shot_patterns_df = shot_patterns_df[~shot_patterns_df['end_location'].isna()]
    shot_patterns_df["x0"] = shot_patterns_df.location.apply(lambda cell: (cell[0]))
    shot_patterns_df["c0"] = shot_patterns_df.location.apply(lambda cell: abs(60 - cell[1]))
    shot_patterns_df["x1"] = shot_patterns_df.end_location.apply(lambda cell: (cell[0]))
    shot_patterns_df["c1"] = shot_patterns_df.end_location.apply(lambda cell: abs(60 - cell[1]))
    #assign (105, 0) to end of the shot
    shot_patterns_df.loc[shot_patterns_df["type_name"] == "Shot", "x1"] = 120
    shot_patterns_df.loc[shot_patterns_df["type_name"] == "Shot", "c1"] = 0

    #for plotting
    shot_patterns_df["y0"] = shot_patterns_df.location.apply(lambda cell: (cell[1]))
    shot_patterns_df["y1"] = shot_patterns_df.end_location.apply(lambda cell: (cell[1]))
    shot_patterns_df.loc[shot_patterns_df["type_name"] == "Shot", "y1"] = 40
    shot_patterns_df['end_shot'] = 1
    return shot_patterns_df

def prepare_coordinates_not_pass(not_pass_patterns_df):
    #columns with coordinates
    not_pass_patterns_df = not_pass_patterns_df[~not_pass_patterns_df['end_location'].isna()]
    not_pass_patterns_df["x0"] = not_pass_patterns_df.location.apply(lambda cell: (cell[0]))
    not_pass_patterns_df["c0"] = not_pass_patterns_df.location.apply(lambda cell: abs(60 - cell[1]))
    not_pass_patterns_df["x1"] = not_pass_patterns_df.end_location.apply(lambda cell: (cell[0]))
    not_pass_patterns_df["c1"] = not_pass_patterns_df.end_location.apply(lambda cell: abs(60 - cell[1]))

    not_pass_patterns_df["y0"] = not_pass_patterns_df.location.apply(lambda cell: (cell[1]))
    not_pass_patterns_df["y1"] = not_pass_patterns_df.end_location.apply(lambda cell: (cell[1]))
    not_pass_patterns_df['end_shot'] = 0 
    return not_pass_patterns_df

def get_details_from_chain(df,response):
    try:
        chain_id = response['selected_rows'][0]['chain ID']
        chain = df.loc[df["possession_chain"] == chain_id]
    except:
        chain_id = df["possession_chain"].unique()[0]
        chain = df.loc[df["possession_chain"] == df["possession_chain"].unique()[0]]

    passes = chain.loc[chain["type_name"].isin(["Pass"])]
    #get events different than pass
    not_pass = chain.loc[chain["type_name"] != "Pass"].iloc[:-1]
    #shot is the last event of the chain (or should be)
    shot = chain.iloc[-1]
    return {
        'shot':shot,
        'passes':passes,
        'not_pass':not_pass,
        'chain_id':chain_id
    }


def get_passes_xt(df,response,team_name):
    try:
        # import pdb;pdb.set_trace()
        player_name = response['selected_rows'][0]['Player Name']
        passes = df.loc[(df["player_name"] == player_name) & (df["team_name"] == team_name) & (df['end_shot']==1)]
        # passes = df.loc[(df["player_name"] == player_name) & (df['xT']>=0.05)]
        # passes = df.loc[(df["player_name"] == player_name)]
    except:
        player_name = df[df["team_name"] == team_name]["player_name"].unique()[0]
        print(player_name)
        passes = df.loc[(df["player_name"] == player_name) & (df['end_shot']==1)]
        # passes = df.loc[(df["player_name"] == player_name) & (df['xT']>=0.05)]
        # passes = df.loc[(df["player_name"] == player_name)]
    passes = passes[['player_name','possession_chain','minute','xT','xG']]
    passes = \
    passes.rename(
        columns={
        'player_name':'Player Name',
        'possession_chain' : 'Chain ID',
        'minute':'Minute',
        'xT':'xThreat(%)',
        'xG':'xGoal(%)'
        }
    )
    # import pdb;pdb.set_trace()
    passes['xThreat(%)'] = passes['xThreat(%)'].apply(lambda n: '{:.2%}'.format(n))
    passes['xGoal(%)'] = passes['xGoal(%)'].map(lambda n: '{:.2%}'.format(n))
    # import pdb;pdb.set_trace()
    return passes


def plot_passes_before_shot(shot,passes,not_pass,chain_id):
    #plot
    pitch = Pitch(line_color='black',pitch_type='custom', pitch_length=120, pitch_width=80, line_zorder = 2)
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #passes
    pitch.arrows(passes.x0, passes.y0,
                passes.x1, passes.y1, color = "blue", ax=ax['pitch'], zorder =  3)
    #shot
    pitch.arrows(shot.x0, shot.y0,
                shot.x1, shot.y1, color = "red", ax=ax['pitch'], zorder =  3)
    #other passes like arrows
    pitch.lines(not_pass.x0, not_pass.y0, not_pass.x1, not_pass.y1, color = "grey", lw = 1.5, ls = 'dotted', ax=ax['pitch'])
    ax['title'].text(0.5, 0.5, f'Passes leading to a shot (Chain ID: {chain_id})', ha='center', va='center', fontsize=30)
    return pitch,ax


def get_team_patterns(df,team_name):
    ## possession chain that ended with a shot
    home_patterns = df[df['team_name']==team_name]
    home_patterns['Minute'] = home_patterns['seconds'].apply(lambda x: int(x/60))

    # Convert float column to percentage format
    home_patterns['xGoal(%)'] = home_patterns['xGoal'].map(lambda n: '{:.2%}'.format(n))

    possession_df = home_patterns[home_patterns['type_name']=='Shot'][['possession_chain','Minute','team_name','player_name','xGoal(%)','period']]
    passes_df = home_patterns[home_patterns['type_name']=='Pass'].groupby('possession_chain').count()[['index']].rename(columns={'index':'# of passes'}).reset_index()    
    possession_df = pd.merge(possession_df,passes_df,on='possession_chain',how='left')
    possession_df.fillna({'# of passes':0},inplace=True)
    return possession_df
    
def _grid_builder(df_table,key=None):
    gb = GridOptionsBuilder.from_dataframe(df_table)
    # gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('single', use_checkbox=True, groupSelectsChildren="Group checkbox select children") 
    gridOptions = gb.build()

    grid_response = AgGrid(
        df_table,
        key=key,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        # theme='blue', #Add theme color to the table
        enable_enterprise_modules=True,
        # height=350, 
        width='100%',
        reload_data=True
    )
    return grid_response

def structure_chains_table(df):
    chain_selection_table = df.drop(columns=['team_name'])
    chain_selection_table.rename(columns={
        'possession_chain':'chain ID',
        'player_name':'Player',
        'period':'Half-time'
    },
    inplace=True)
    grid_response = _grid_builder(chain_selection_table)
    
    return chain_selection_table,grid_response

def structure_rank_xT(df,key=None):
    grid_response = _grid_builder(df,key)
    return df,grid_response

def create_combinations_coordinates(df):
    #model variables
    var = ["x0", "x1", "c0", "c1"]

    #combinations
    inputs = []
    #one variable combinations
    inputs.extend(combinations_with_replacement(var, 1))
    #2 variable combinations
    inputs.extend(combinations_with_replacement(var, 2))
    #3 variable combinations
    inputs.extend(combinations_with_replacement(var, 3))

    #make new columns
    for i in inputs:
        #columns length 1 already exist
        if len(i) > 1:
            #column name
            column = ''
            x = 1
            for c in i:
                #add column name to be x0x1c0 for example
                column += c
                #multiply values in column
                x = x*df[c]
            #create a new column in df
            df[column] = x
            #add column to model variables
            var.append(column)
    return df,var

def calculate_xt_score(passes,team_name):
    # passes_team = passes[(passes['team_name']==team_name) & (passes['xT']>=0.05)]
    passes_team = passes[(passes['team_name']==team_name) & (passes['end_shot']==1)]
    agg_df = \
    passes_team.groupby('player_name').agg({
        'xT':['sum','mean'],
        # 'shot_prob':'sum',
        # 'shot_prob':['mean','median','sum'],
        'possession_chain':'nunique'
    })
    rank_xT = pd.DataFrame()
    rank_xT['Player Name'] = agg_df.index
    rank_xT['Total xT score'] = agg_df.xT['sum'].values
    rank_xT['Average xT score'] = agg_df.xT['mean'].values
    rank_xT['Total Attempts (Passes) that ended to a shot'] = agg_df['possession_chain']['nunique'].values
    # .rename(columns={'xT.sum':'Total xT Score'})
    rank_xT['Rank'] = rank_xT['Total xT score'].rank(ascending=False)

    rank_xT['Total xT score'] = rank_xT['Total xT score'].apply(lambda n: '{:.2}'.format(n))
    rank_xT['Average xT score'] = rank_xT['Average xT score'].apply(lambda n: '{:.2}'.format(n))
    return rank_xT

def prepare_data_modeling_xgb(df_isolated_chains,team_name,possession_df):
    # Present a summary dataframe of the home players statistics regarding xThreat attempts
    not_pass_index = np.setdiff1d(df_isolated_chains[df_isolated_chains['team_name']==team_name]['possession_chain'].unique(),
                                possession_df['possession_chain'].unique())

    not_pass_patterns_df = df_isolated_chains[df_isolated_chains['possession_chain'].isin(not_pass_index)][[
        'type_name','location','end_location','possession_chain',
        'team_name','index','shot_statsbomb_xg','match_game','seconds',
        'player_name','period'
        ]]

    not_pass_patterns_df = prepare_coordinates_not_pass(not_pass_patterns_df) 

    # Data modeling and preparation for the machine learning model predicting the probability of shooting
    df_poss_chain_shot = prepare_coordinates(df_isolated_chains[df_isolated_chains['shot_end']==1])
    df_poss_chain_not_shot = prepare_coordinates_not_pass(df_isolated_chains[df_isolated_chains['shot_end']==0])
    df_coordinates = pd.concat([df_poss_chain_shot,df_poss_chain_not_shot])
    df_coordinates,var = create_combinations_coordinates(df_coordinates)
    return df_coordinates,var

def _fit_xgb_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123, stratify = y)
    xgb_model = XGBClassifier(n_estimators = 100, ccp_alpha=0, max_depth=4, min_samples_leaf=10,
                        random_state=123)

    # Set the scale_pos_weight parameter to handle class imbalance
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    xgb_model.set_params(scale_pos_weight=scale_pos_weight)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def _fit_ols_model(passes):
    shot_ended = passes.loc[passes["end_shot"] == 1]
    X2 = shot_ended[var].values
    y2 = shot_ended["xG"].values
    ols_model = LinearRegression()
    ols_model.fit(X2, y2)
    return ols_model

def load_ml_models(passes=None,X=None,y=None):
    directory = os.getcwd()+'/libs/ml-model/'
    filename = 'ml_models.pkl'
    if os.path.isfile(os.path.join(directory, filename)):
    # File exists in the directory
        print("File exists!")
        with open(os.path.join(directory, filename), 'rb') as f:
            ml_models = load(f)
        xgb_model = ml_models['xgb_model']
        lr_model = ml_models['lr_model']
        return xgb_model,lr_model
    else:
        # File does not exist in the directory
        print("File does not exist!")
        filename = 'ml_models_2.pkl'
        import pdb;pdb.set_trace()
        
        # Fit XGBoost model
        xgb_model = _fit_xgb_model(X,y)

        # Predict the shooting probability
        y_pred_proba = xgb_model.predict_proba(X)[::,1]
        # Prepare data for the expected goal (xGoal) probability
        passes["shot_prob"] = y_pred_proba

        # Fit Linear Reggression model 
        lr_model = _fit_ols_model(passes)
        ml_models = \
        {
            'xgb_model':xgb_model,
            'lr_model':lr_model
        }
        with open(os.path.join(directory, filename), 'wb') as f:
            dump(ml_models,f)
        return xgb_model,lr_model        


def prepare_data_fit_ols(passes,var):
    directory = os.getcwd()+'/libs/ml-model/'
    filename_xgoal = 'xgoal_model.pklz'
    if os.path.isfile(os.path.join(directory, filename_xgoal)) & False:
        # File exists in the directory
        print("OLS model File exists!")
        with gzip.open(os.path.join(directory, filename_xgoal), 'rb') as f:
            ols_model = load(f)
    else:
        # File does not exist in the directory
        print("File does not exist!")
        shot_ended = passes.loc[passes["end_shot"] == 1]
        X2 = shot_ended[var].values
        y2 = shot_ended["xG"].values
        ols_model = LinearRegression()
        ols_model.fit(X2, y2)
        # store the xgb model
        # Define the filename of your .pkl file
        with gzip.open(os.path.join(directory, filename_xgoal), 'wb') as f:
            # Use pickle.dump() to store the model in the file
            dump(ols_model, f)
    return ols_model
