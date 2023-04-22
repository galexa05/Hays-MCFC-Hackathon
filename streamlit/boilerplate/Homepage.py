import streamlit as st

# Set page layout and add CSS styling
st.set_page_config(layout="wide")
css = """
.row {
  display: flex;
  flex-wrap: wrap;
  margin: 0 -10px;
}
.col-md-4 {
  flex: 0 0 33.333333%;
  max-width: 33.333333%;
  padding: 0 10px;
  box-sizing: border-box;
}
.container {
    margin-top: 20px;
    margin-bottom: 20px;
    padding-top: 20px;
    padding-bottom: 20px;
    background-color: #f5f5f5;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
}
"""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Add page title, subtitle, and introduction
st.title("Manchester City Women's Soccer Analytics")
st.subheader("Data Science Project by Giannis Alexandrou and Charalambos Ioannou")
st.write("This data science project analyzes the physical attributes, expected threat, and passing networks of the Manchester City Women's soccer team. By using advanced analytics techniques, we aim to provide insights that can help the team improve their performance and win more games.")

# Add a row of images and descriptions
with st.container():
    st.write("### Expected Threat Analysis")
    col1, col2 = st.columns([2, 3])
    col2.image("https://media.giphy.com/media/YWUpVw86AtIbe/giphy.gif", use_column_width=True)
    col1.write("This analysis involves calculating the change in the likelihood of scoring a goal based on various factors such as player positions, ball location, and time remaining in the game. By identifying the areas of the field with the highest expected threat, the team can focus their efforts on creating scoring opportunities in those areas.")
 
    st.write("### Passing Network Analysis")
    col1, col2 = st.columns([2, 3])
    col1.write("This analysis involves examining the playing patterns of the team in terms of their passes. By analyzing the frequency and success rate of passes between players, we can identify key players and playing strategies that contribute to the team's success. This analysis can also reveal areas for improvement and help the team refine their playing style.")
    col2.image("https://icdn.sempremilan.com/wp-content/uploads/2020/10/c6d8fcf60c914a77682538e4b3e0de69-1.jpg", use_column_width=True)

    st.write("### Physical Attribute Analysis")
    col1, col2 = st.columns([2, 3])
    col1.write("")
    col2.image("https://icdn.sempremilan.com/wp-content/uploads/2020/10/c6d8fcf60c914a77682538e4b3e0de69-1.jpg", use_column_width=True)
    col1.write("This analysis involves examining the physical attributes of each player throughout the match. By analyzing the distance covered, acceleration/deceleration ratio, and metabolic power output, we can gain insights into the players' fitness levels and overall performance.")
  
    st.markdown(
        """
        ### Conclusion
        This analysis involves examining the playing patterns of the team in terms of their passes. By analyzing the frequency and success rate of passes between players, we can identify key players and playing strategies that contribute to the team's success. This analysis can also reveal areas for improvement and help the team refine their playing style.
        """
        )
   
    st.markdown(
        """
        ### Future Work
        - Further analyzing xThreat score data to identify the most beneficial patterns contributing to a high score of attempts (xThreat) and continuously optimizing strategies.
        - Combining xThreat score and Passing Network Analysis to identify critical insights into player performance and game strategies, such as identifying the most centralized players that contribute to high xThreat scoring, tracking player performance throughout the season, identifying key players that are not centralized to the game, and recommending strategies to the coach to reconstruct their football tactics.
        - Combining physical attributes of players and xThreat analysis through time to track player performance on the field combined with their physical skills, and recommending which players are more dangerous to sub in to create high xThreat scoring during the match.
        """)
    
# Add paragraph with references
st.write(
    """
    ### References
    1. Soccermatics. (n.d.). Soccermatics documentation. Retrieved April 22, 2023, from https://soccermatics.readthedocs.io/en/latest/index.html
    2. MPLSoccer. (n.d.). MPLSoccer documentation. Retrieved April 22, 2023, from https://mplsoccer.readthedocs.io/en/latest/index.html
    3. Goes, K. G. (2021). Estimating the most important football player statistics using neural networks (Bachelor's thesis)
    4. Karun, S. (2019, August 26). Expected Threat. Retrieved April 22, 2023, from https://karun.in/blog/expected-threat.html
    5. Friends of Tracking Data FoTD. (n.d.). Passing Networks in Python. Retrieved April 22, 2023, from https://github.com/Friends-of-Tracking-Data-FoTD/passing-networks-in-python
    """
)

# Add paragraph with references
st.write(
    """
    ### Contact Us
    - giannisa946@gmail.com, Giannis Alexandrou
    - cioannou1997@gmail.com, Charalambos Ioannou
    """
)