## Statsbomb

1. **FAWSL_22_23.json - Info about the whole league**
    - **`match_id`**: The unique identifier for the match.
    - **`match_date`**: The date on which the match took place.
    - **`kick_off`**: The time at which the match began.
    - **`competition`**: A nested object containing information about the competition in which the match was played, including the competition ID, country name, and competition name.
    - **`season`**: A nested object containing information about the season in which the match was played, including the season ID and season name.
    - **`home_team`**: A nested object containing information about the home team, including the team ID, team name, gender, youth status, group, country, and manager(s).
    - **`away_team`**: A nested object containing information about the away team, including the team ID, team name, gender, youth status, group, country, and manager(s).
    - **`home_score`**: The number of goals scored by the home team in the match.
    - **`away_score`**: The number of goals scored by the away team in the match.
    - **`attendance`**: The number of people who attended the match.
    - **`behind_closed_doors`**: A boolean indicating whether the match was played behind closed doors (i.e., without spectators).
    - **`neutral_ground`**: A boolean indicating whether the match was played on a neutral ground (i.e., not the home ground of either team).
    - **`play_status`**: A string indicating the status of the match.
    - **`match_status`**: A string indicating the availability of match data.
    - **`match_status_360`**: A string indicating the availability of 360-degree match data.
    - **`last_updated`**: The date and time at which the match data was last updated.
    - **`last_updated_360`**: The date and time at which 360-degree match data was last updated.
    - **`metadata`**: A nested object containing metadata about the match, including data version, shot fidelity version, and xy fidelity version.
    - **`match_week`**: The week in which the match took place.
    - **`competition_stage`**: A nested object containing information about the stage of the competition in which the match was played, including the stage ID and stage name.
    - **`stadium`**: A nested object containing information about the stadium in which the match was played, including the stadium ID, stadium name, and country.
    - **`referee`**: A nested object containing information about the referee for the match, including the referee ID, referee name, and country.

1. **events.json - specific match events**
    - `id`: A unique identifier for this event.
    - `index`: The index of this event within the match.
    - `period`: The period of the match (e.g., first half, second half).
    - `timestamp`: The time of the event within the period.
    - `minute`: The minute of the event within the period.
    - `second`: The second of the event within the minute.
    - `type`: The type of event (e.g., starting lineup, goal, substitution).
    - `possession`: The possession index of the team (i.e., 1 or 2).
    - `possession_team`: The team in possession of the ball.
    - `play_pattern`: The play pattern of the event (e.g., regular play, corner kick).
    - `obv_for_after`: The offensive value of the team after the event.
    - `obv_for_before`: The offensive value of the team before the event.
    - `obv_for_net`: The offensive value added by the event to the team.
    - `obv_against_after`: The defensive value of the opposing team after the event.
    - `obv_against_before:` The defensive value of the opposing team before the event.
    - `obv_against_net`: The defensive value subtracted by the event from the opposing team.
    - `obv_total_net`: The total value added or subtracted by the event.
    - `team`: The team associated with the event.
    - `duration`: The duration of the event.
    - `tactics`: The tactics used by the team, including the formation and lineup of players. Each player is represented by their id, name, position, and jersey number.

1. **lineup.json - specific match lineup**
    - **`team_id`** is the ID of the team (in this case, 746 represents Manchester City WFC)
    - **`team_name`** is the name of the team ("Manchester City WFC")
    - **`lineup`** is an array of player objects representing the starting lineup of the team for a particular match.
    
    Each player object in the **`lineup`** array contains the following information:
    
    - **`player_id`** is the ID of the player.
    - **`player_name`** is the name of the player.
    - **`player_nickname`** is an optional field containing the player's nickname.
    - **`birth_date`** is the date of birth of the player.
    - **`player_gender`** is the gender of the player ("female" in this case).
    - **`player_height`** is the height of the player in centimeters.
    - **`player_weight`** is the weight of the player in kilograms.
    - **`jersey_number`** is the number on the player's jersey.
    - **`country`** is an object containing the ID and name of the country the player represents.
    - **`positions`** is an array of objects representing the positions the player played during the match, including the time the player played in that position, and the reason for leaving the position.
    - **`stats`** is an object containing the player's stats for the match, including the number of goals, assists, and penalties scored, missed, or saved.
