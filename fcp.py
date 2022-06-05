"""
fcp.py

Dependencies:
    streamlit==1.10.0
    chess==1.9.1
    streamlit-aggrid==0.2.3.post2
    streamlit-option-menu==0.3.2
    plotly==5.8.0
"""


__version__ = '0.8'
__author__ = 'fsmosca'


import chess.pgn
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import plotly.express as px


st.set_page_config(
    page_title="FCP Tournament",
    page_icon="🧊",
    layout="wide"
)

if 'wminrating' not in st.session_state:
    st.session_state.wminrating = 3010
if 'wmaxrating' not in st.session_state:
    st.session_state.wmaxrating = 3470
if 'bminrating' not in st.session_state:
    st.session_state.bminrating = 3010
if 'bmaxrating' not in st.session_state:
    st.session_state.bmaxrating = 3470


mat_map = {1:0, 2:3, 3:3, 4:5, 5:10, 6:0}


def get_material(board):
    """
    material is the total piece value except king and pawns.
    knight=3, bishop=3, rook=5, queen=10
    This is used to identify if end position (mat <= 20) is in ending phase or not.

    Returns material
    """
    mat = 0
    pmap = board.piece_map()
    for _, v in pmap.items():
        mat += mat_map[v.piece_type]

    return mat


def get_pgn_data(fn):
    max_games = 1000000
    cnt = 0
    data = []
    players = []

    with open(fn, 'r') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            wscore, bscore = 0, 0
            event = game.headers['Event']
            round = game.headers['Round']            
            white = game.headers['White']
            black = game.headers['Black']
            welo = game.headers.get('WhiteElo', '?')
            belo = game.headers.get('BlackElo', '?')
            players.append(white)
            players.append(black)
            result = game.headers['Result']
            plycnt = game.headers['PlyCount']
            opening = game.headers['Opening']
            eco = game.headers['ECO']
            node = game
            end = node.end()
            out = end.board().outcome(claim_draw=True)

            if result == '1-0':
                wscore = 1
            elif (result == '0-1'):
                bscore = 1
            else:
                wscore += 0.5
                bscore += 0.5

            mat = get_material(end.board())

            if out is None:
                if result == '1-0' or result == '0-1':
                    data.append([event, round, white, welo, black, belo, result, wscore, bscore, eco, opening, plycnt, mat, 'WIN_ADJUDICATION'])
                elif result == '1/2-1/2':
                    data.append([event, round, white, welo, black, belo, result, wscore, bscore, eco, opening, plycnt, mat, 'DRAW_ADJUDICATION'])
                else:
                    data.append([event, round, white, welo, black, belo, result, wscore, bscore, eco, opening, plycnt, mat, 'OTHERS'])
            else:
                data.append([event, round, white, welo, black, belo, result, wscore, bscore, eco, opening, plycnt, mat, out.termination.name])

            cnt += 1
            if cnt >= max_games:
                break

        df = pd.DataFrame(data)
        df.columns = ['Event', 'Round', 'White', 'Welo', 'Black', 'Belo', 'Result', 'Wscore', 'Bscore', 'Eco', 'Opening', 'Plycnt', 'Material', 'Termination']
        df.to_csv('record.csv', index=False)

        players = list(set(players))
        df_players = pd.DataFrame(players, columns=['Name'])
        df_players = df_players.sort_values(by=['Name'])
        df_players.to_csv('players.csv', index=False)

        return df, players

def gen_data():
    fn = 'complete_fcp-tourney-2022.pgn'
    df, players = get_pgn_data(fn)
    print(df)

    data = []
    for p in players:
        dfw = df.loc[df.White == p]
        dfb = df.loc[df.Black == p]

        games = len(dfw) + len(dfb)
        score = df.loc[df.White == p]['Wscore'].sum() + df.loc[df.Black == p]['Bscore'].sum()
        score_pct = round(100 * score / games, 2)
        rating = df.loc[df.White == p].Welo.iloc[0]

        wwin = len(dfw.loc[dfw.Result == '1-0'])
        wloss = len(dfw.loc[dfw.Result == '0-1'])
        wdraw = len(dfw.loc[dfw.Result == '1/2-1/2'])

        bwin = len(dfb.loc[dfb.Result == '1-0'])
        bloss = len(dfb.loc[dfb.Result == '0-1'])
        bdraw = len(dfb.loc[dfb.Result == '1/2-1/2'])

        win = wwin + bwin
        loss = wloss + bloss
        draw = wdraw + bdraw
        draw_pct = round(100 * draw / games, 2)

        data.append([p, rating, games, score, score_pct, win, loss, draw, draw_pct])

    df_score = pd.DataFrame(data, columns=['Name', 'Rating', 'Games', 'Score', 'Score%', 'Win', 'Loss', 'Draw', 'Draw%'])
    df_score = df_score.sort_values(by=['Score%', 'Win'], ascending=[False, False])
    df_score['Rank'] = list(range(1, len(df_score) + 1))
    first_column = df_score.pop('Rank')
    df_score.insert(0, "Rank", first_column)
    df_score.to_csv('standing.csv', index=False)


@st.cache
def load_record():
    return pd.read_csv('record.csv')


@st.cache(allow_output_mutation=True)
def load_player():
    """
    Returns a dataframe of players with column Name.
    """
    return pd.read_csv('players.csv')


@st.cache
def load_standing():
    """
    Returns a dataframe of player standings.
    """
    return pd.read_csv('standing.csv')


def load_games(f, name, opp, rnd, is_white=True):
    game_offsets = []

    while True:
        offset = f.tell()
        headers = chess.pgn.read_headers(f)
        if headers is None:
            break

        if is_white:
            if name in headers.get("White", "?") and opp in headers.get("Black", "?") and rnd == headers.get("Round", "?"):
                game_offsets.append(offset)
                break
        else:
            if name in headers.get("Black", "?") and opp in headers.get("White", "?") and rnd == headers.get("Round", "?"):
                game_offsets.append(offset)
                break


    return game_offsets


def replay_grid(df, selected_player, color):
    """
    Returns the game that we are going to replay.
    """
    rgame = None
    col_opp = 'Black' if color == 'White' else 'White'
    df_s = df.loc[(df[color] == selected_player) & (df.Result != '1/2-1/2')]
    df_s = df_s.drop(['Event', 'Wscore', 'Bscore'], axis=1)

    st.write('Check the box to load the game.')
    gd = GridOptionsBuilder.from_dataframe(df_s)
    gd.configure_pagination(enabled=False)
    gd.configure_default_column(editable=True, groupable=True)
    gd.configure_selection(selection_mode='single', use_checkbox=True)
    gridoptions = gd.build()
    grid_table = AgGrid(df_s, gridOptions=gridoptions,
                        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
                        height=300,
                        allow_unsafe_jscode=True,
                        key=color,
                        reload_data=True)
    sel_row = grid_table["selected_rows"]
    df_selected = pd.DataFrame(sel_row)
    with st.container():
        if len(df_selected):
            col_selected = color
            col_opp = 'Black' if col_selected == 'White' else 'White'
            is_white = True if col_selected == 'White' else False
            opp = df_selected.iloc[0][col_opp]
            rnd = str(df_selected.iloc[0]['Round'])
            fn = 'fcp-tourney-2022.pgn'
            with open(fn, 'r') as f:
                game_offsets = load_games(f, selected_player, opp, rnd, is_white=is_white)
                for offset in game_offsets:
                    f.seek(offset)
                    rgame = chess.pgn.read_game(f)
                    break

    return rgame


def main():
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", 'Player List', 'Pairing', 'Standing', 'Statistics', 'Replay'],
            icons=['house', 'person-lines-fill', 'table', 'book', 'file-bar-graph-fill', 'file-play'],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "16px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"},
            }
        )
    if selected == 'Home':
        st.markdown(f'# Home')

        cols = st.columns([1, 1])

        with cols[0]:
            st.markdown(f"""
            ##### Introduction
            This page is all about the FCP's top 41 Chess Engine Tournament done by Frank.
            It contains pairing results, standings, statistics and game replay of
            decisive games.
            """)

            st.markdown(f"""
            ##### Game Source
            The source of the games is taken from [FCP's top 41 Chess Engines Tournament](https://www.amateurschach.de/main/_fcp-tourney-2022.htm)
            specifically the fcp-tourney-2022.pgn file.
            """)

            st.markdown(f"""
            ##### Web App Source
            The python source code of this application can be found in https://github.com/fsmosca/fcp.
            """)

            st.markdown(f"""
            ##### Credits
            [Frank's Chess Page](https://www.amateurschach.de/)  
            [Streamlit](https://streamlit.io/)  
            [Python Chess](https://python-chess.readthedocs.io/en/latest/)  
            [Chess Tempo](https://chesstempo.com/)  
            [Plotly](https://plotly.com/)  
            """)

    elif selected == 'Player List':
        st.markdown(f'# {selected}')        
        df_player = load_player()
        df_p = df_player.copy()
        df_p['Num'] = list(range(1, len(df_player)+1))
        df_p = df_p[['Num', 'Name']]

        cols = st.columns([1, 1])
        with cols[0]:
            st.write('##### Before tournament')
            AgGrid(df_p, height=1200)

        dfs = load_standing()
        with cols[1]:
            st.write('##### After tournament')
            df_s = dfs[['Rank', 'Name', 'Rating']]
            AgGrid(df_s, height=1200)

    elif selected == 'Pairing':
        st.markdown(f'# {selected}')
        df_record = load_record()
        df_r = df_record.drop(['Event', 'Wscore', 'Bscore'], axis=1)
        AgGrid(df_r)

    elif selected == 'Standing':
        st.markdown(f'# {selected}')

        df_standing = load_standing()
        with st.expander('RANKING', expanded=True):
            st.markdown(f'''
            The rating is generated using the Ordo program. Shredder 13 POPCNT x64 at 3125 is set as reference.
            Ratings are rounded to the nearest ten for visual clarity.
            ''')
            AgGrid(df_standing, height=1200)

        # Show the opponents of this selected player.
        with st.expander('OPPONENTS', expanded=True):
            df_player = load_player()
            selected_player = st.selectbox(label='Select player', options=df_player.Name, index=0)

            # Score against each opp
            df_rec = load_record()
            player = load_player()

            # Selected player as white.
            dataw = []
            for p in player['Name']:
                if p == selected_player:
                    continue
                dfrw = df_rec.loc[(df_rec.White == selected_player) & (df_rec.Black == p)]
                score = dfrw.Wscore.sum()
                games = len(dfrw)
                pct = round(100*score/games, 2)
                dataw.append([p, games, score, pct])

            dfscorew = pd.DataFrame(dataw, columns=['Opponent', 'Games', 'Score', 'Score%'])

            # Selected player as black.
            datab = []
            for p in player['Name']:
                if p == selected_player:
                    continue
                dfrb = df_rec.loc[(df_rec.Black == selected_player) & (df_rec.White == p)]
                score = dfrb.Bscore.sum()
                games = len(dfrb)
                pct = round(100*score/games, 2)
                datab.append([p, games, score, pct])
            dfscoreb = pd.DataFrame(datab, columns=['Opponent', 'Games', 'Score', 'Score%'])

            cols = st.columns([1, 1])
            with cols[0]:
                st.write(f'**{selected_player}** white score')
                AgGrid(dfscorew, height=1180)

            with cols[1]:
                st.write(f'**{selected_player}** black score')
                AgGrid(dfscoreb, height=1180)

    elif selected == 'Statistics':
        st.markdown(f'# {selected}')

        # 1. Win/Loss/Draw
        df = load_record()
        player = load_player()

        games = len(df)
        wwins = df.loc[df.Result == '1-0']
        bwins = df.loc[df.Result == '0-1']
        draws = df.loc[df.Result == '1/2-1/2']

        data = {
            'Name': ['Games', 'White Wins', 'Black Wins', 'Draws'],
            'Count': [games, len(wwins), len(bwins), len(draws)],
            'Percent': [100, round(100*len(wwins)/games, 2), round(100*len(bwins)/games, 2), round(100*len(draws)/games, 2)]
        }
        dfwld = pd.DataFrame(data)
        with st.expander('WIN/LOSS/DRAW', expanded=True):
            AgGrid(dfwld, height=160)
            fig = px.bar(dfwld, x="Percent", y="Name", orientation='h', color='Name', height=300, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        # 2. Opening
        data = []
        for o in df.Opening.unique():
            count = len(df.loc[df.Opening == o])
            wwin = df.loc[(df.Opening == o) & (df.Result == '1-0')]
            bwin = df.loc[(df.Opening == o) & (df.Result == '0-1')]
            wpct, bpct = 0, 0
            if len(wwin):
                wpct = round(100*len(wwin) / count, 2)
            if len(bwin):
                bpct = round(100*len(bwin) / count, 2)
            data.append([o, games, count, round(100*count/games, 3), wpct, bpct])
        dfo = pd.DataFrame(data, columns=['Opening', 'Games', 'Count', 'Count%', 'WhiteWin%', 'BlackWin%'])
        dfo = dfo.sort_values(by=['Count', 'Opening'], ascending=[False, True])
        with st.expander('OPENING', expanded=True):
            AgGrid(dfo)
            st.write('##### Top 20')
            dfo_top10 = dfo.head(20)
            fig = px.bar(dfo_top10, x="Count%", y="Opening", orientation='h', color='Opening', height=1000, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        # 3. Termination
        data = []
        for t in df.Termination.unique():
            count = len(df.loc[df.Termination == t])
            data.append([t, count, round(100*count/games, 3)])
        dft = pd.DataFrame(data, columns=['Termination', 'Count', 'Percent'])
        dft = dft.sort_values(by=['Count', 'Termination'], ascending=[False, True])
        with st.expander('GAME TERMINATION', expanded=True):
            AgGrid(dft, height=250)
            fig = px.bar(dft, x="Percent", y="Termination", orientation='h', color='Termination', height=400, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

            # 3.2 Ply count histogram on 3-fold repetition
            df_rep = df.loc[df.Termination == 'THREEFOLD_REPETITION']
            minv = df_rep.Plycnt.min()
            maxv = df_rep.Plycnt.max()
            mean = df_rep.Plycnt.mean()
            median = df_rep.Plycnt.median()
            mode = df_rep.Plycnt.mode()[0]
            stdev = df_rep.Plycnt.std()
            data = {
                'name': ['min', 'max', 'mean', 'median', 'mode', 'stdev'],
                'value': [int(minv), int(maxv), int(mean), int(median), int(mode), int(stdev)]
            }
            df_rep_stat = pd.DataFrame(data)
            st.markdown('''
            ##### Ply Count on Draw by 3-Fold Repetition  
            Hover on the plot to see the ply count range and frequency.
            ''')
            st.dataframe(df_rep_stat)
            fig2 = px.histogram(df_rep, x="Plycnt")
            st.plotly_chart(fig2, use_container_width=True)

        # 4. Engines that defeated opponents whose rating is higher than itself.
        with st.expander('GOOD ENGINES', expanded=True):
            rdiff = int(st.text_input('Change Rating difference', value=100))
            data = []
            ps = load_standing()
            for p in player.Name:
                prating = ps.loc[ps.Name == p].Rating.iloc[0]
                dfw = df.loc[(df.White == p) & (df.Result == '1-0') & (df.Welo + rdiff <= df.Belo)]
                dfb = df.loc[(df.Black == p) & (df.Result == '0-1') & (df.Belo + rdiff <= df.Welo)]
                num_games = len(df.loc[(df.White == p) & (df.Welo + rdiff <= df.Belo)]) + len(df.loc[(df.Black == p) & (df.Belo + rdiff <= df.Welo)])
                num_wins = len(dfw) + len(dfb)
                if num_wins:
                    pct = round(100*num_wins/num_games, 2)
                    data.append([p, prating, num_games, num_wins, pct])
            df_good = pd.DataFrame(data, columns=['Name', 'Rating', 'Games', 'Wins', 'Wins%'])
            df_good = df_good.sort_values(by=['Wins%', 'Games', 'Rating'], ascending=[False, False, False])
            df_good = df_good.reset_index(drop=True)
            st.markdown(f''' 
            ##### Engines that defeated opponents with a {rdiff} or more rating higher than itself.
            ''')
            AgGrid(df_good)
            
        # 5. Plycount
        with st.expander('PLYCOUNT', expanded=True):
            # Win Ply count table by player
            data = []
            for p in player.Name:
                dfwwin = df.loc[(df.White == p) & (df.Result == '1-0')]
                dfbwin = df.loc[(df.Black == p) & (df.Result == '0-1')]
                dfwwin_plycnt_mean = dfwwin.Plycnt.mean()
                dfbwin_plycnt_mean = dfbwin.Plycnt.mean()
                data.append([p, int(dfwwin_plycnt_mean), int(dfbwin_plycnt_mean)])
            dfplycnt = pd.DataFrame(data, columns=['Name', 'White', 'Black'])
            st.markdown(f'''
            ##### Win Plycnt Mean
            The average ply count for white and black when the player wins.
            ''')
            AgGrid(dfplycnt)

            st.write(f'##### Compare players by ply count')
            cols = st.columns([1, 1])
            with cols[0]:
                player_sel = st.selectbox('Select player', options=df.White.unique(), index=33, key=1)
                dfp = df.loc[(df.White == player_sel) | (df.Black == player_sel)]
                minv = dfp.Plycnt.min()
                maxv = dfp.Plycnt.max()
                mean = dfp.Plycnt.mean()
                median = dfp.Plycnt.median()
                mode = dfp.Plycnt.mode()[0]
                stdev = dfp.Plycnt.std()
                data = {
                    'name': ['min', 'max', 'mean', 'median', 'mode', 'stdev'],
                    'value': [int(minv), int(maxv), int(mean), int(median), int(mode), int(stdev)]
                }
                df_stat = pd.DataFrame(data)
                st.dataframe(df_stat)
                fig1 = px.histogram(dfp, x="Plycnt")
                st.plotly_chart(fig1, use_container_width=True)
            with cols[1]:
                player_sel = st.selectbox('Select player', options=df.White.unique(), index=13, key=2)
                dfp = df.loc[(df.White == player_sel) | (df.Black == player_sel)]
                minv = dfp.Plycnt.min()
                maxv = dfp.Plycnt.max()
                mean = dfp.Plycnt.mean()
                median = dfp.Plycnt.median()
                mode = dfp.Plycnt.mode()[0]
                stdev = dfp.Plycnt.std()
                data = {
                    'name': ['min', 'max', 'mean', 'median', 'mode', 'stdev'],
                    'value': [int(minv), int(maxv), int(mean), int(median), int(mode), int(stdev)]
                }
                df_stat = pd.DataFrame(data)
                st.write(df_stat)
                fig2 = px.histogram(dfp, x="Plycnt")
                st.plotly_chart(fig2, use_container_width=True)

        # 6. Ending
        with st.expander('ENDING', expanded=True):
            data = []
            for p in player.Name:
                dfw = df.loc[(df.White == p) & (df.Material <= 20)]
                dfb = df.loc[(df.Black == p) & (df.Material <= 20)]
                games = len(df)
                ending = len(dfw) + len(dfb)
                ending_score = dfw.Wscore.sum() + dfb.Bscore.sum()
                data.append([p, games, ending, round(100*ending/games, 2), round(100*ending_score/ending, 2)])
            dfmat = pd.DataFrame(data, columns=['Name', 'Games', 'NumPos', 'NumPos%', 'Score%'])
            st.markdown(f'''
            ##### Player ending data
            Ending is the count of positions where material (excluding king and pawns) at the end
            of position is 20 or less. Material map: n=b=3, r=5, q=10.
            ''')
            AgGrid(dfmat, height=200)

            # Engines with 50% or more Score
            df_etop = dfmat.loc[dfmat['Score%'] >= 50]
            df_etop = df_etop.sort_values(by=['Score%', 'NumPos'], ascending=[False, False])
            df_etop = df_etop.reset_index(drop=True)
            st.write('##### Top engines in ending')
            AgGrid(df_etop, height=200)
            st.write('##### Engines that scored 50% or more')
            fig = px.bar(df_etop, x="Score%", y="Name", orientation='h', color='Name', height=700, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        # 7. ECO
        with st.expander('ECO', expanded=True):
            data = []
            for eco in df.Eco.unique():
                dfe = df.loc[df.Eco == eco]
                count = len(dfe)
                pct = round(100*count/games, 3)
                data.append([eco, games, count, pct])
            df_eco = pd.DataFrame(data, columns=['ECO', 'Games', 'Count', 'Count%'])
            df_eco = df_eco.sort_values(by=['Count'], ascending=[False])
            df_eco = df_eco.reset_index(drop=True)
            AgGrid(df_eco, height=200)

            st.write('##### Top 20 by count')
            fig1 = px.bar(df_eco.head(20), x="Count", y="ECO", orientation='h', color='ECO', height=800, text_auto=True)
            st.plotly_chart(fig1, use_container_width=True)

            st.write('##### Last 20 by count')
            fig2 = px.bar(df_eco.tail(20), x="Count", y="ECO", orientation='h', color='ECO', height=800, text_auto=True)
            st.plotly_chart(fig2, use_container_width=True)

        # 8. 3-fold repetition interactive
        with st.expander('THREEFOLD_REPETITION INTERACTIVE', expanded=True):
            df_rep = df.loc[df.Termination == 'THREEFOLD_REPETITION']
            st.markdown(f'''
            ##### Adjust the sliders to modify the histogram
            Does the ply count mean higher if players are both stronger compared to when players
            are both weaker?
            ''')
            with st.form(key='form', clear_on_submit=False):
                cols = st.columns([2, 1, 2])
                with cols[0]:
                    st.write('### White')
                    st.slider('Minimum Rating', 3010, 3470, st.session_state.wminrating, key='wminrating')
                    st.slider('Maximum Rating', 3010, 3470, st.session_state.wmaxrating, key='wmaxrating')
                with cols[2]:
                    st.write('### Black')
                    st.slider('Minimum Rating', 3010, 3470, st.session_state.bminrating, key='bminrating')
                    st.slider('Maximum Rating', 3010, 3470, st.session_state.bmaxrating, key='bmaxrating')
                is_use_opening = st.checkbox('Use Opening')
                select_opening = st.selectbox('Select Opening', df.Opening.unique())
                is_calculate = st.form_submit_button('Generate Histogram')

            if is_calculate:
                df_rep = df_rep.loc[(df.Welo >= st.session_state.wminrating) & 
                                    (df.Welo <= st.session_state.wmaxrating) & 
                                    (df.Belo >= st.session_state.bminrating) & 
                                    (df.Belo <= st.session_state.bmaxrating)]
                if is_use_opening:
                    df_rep = df_rep.loc[df_rep.Opening == select_opening]

                if len(df_rep):
                    minv = df_rep.Plycnt.min()
                    maxv = df_rep.Plycnt.max()
                    mean = df_rep.Plycnt.mean()
                    median = df_rep.Plycnt.median()
                    mode = df_rep.Plycnt.mode()[0]
                    stdev = df_rep.Plycnt.std()
                    data = {
                        'name': ['min', 'max', 'mean', 'median', 'mode', 'stdev'],
                        'value': [int(minv), int(maxv), int(mean), int(median), int(mode), int(stdev)]
                    }
                    df_rep_stat = pd.DataFrame(data)
                    st.markdown(f'''
                    ##### Ply Count on Draw by 3-Fold Repetition
                    white minrating: {st.session_state.wminrating}, white max rating: {st.session_state.wmaxrating}  
                    black minrating: {st.session_state.bminrating}, black max rating: {st.session_state.bmaxrating}  
                    ''')
                    st.dataframe(df_rep_stat)
                    fig = px.histogram(df_rep, x="Plycnt")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info('No entries found, try to adjust the sliders!')

    elif selected == 'Replay':
        st.markdown(f'# {selected}')

        df_player = load_player()
        selected_player = st.selectbox(label='Select player', options=df_player.Name, index=0)

        st.markdown(f'### {selected_player}')
        df = load_record()

        # Black opp
        with st.expander('BLACK OPPONENTS', expanded=True):
            rgame = replay_grid(df, selected_player, 'White')
            if rgame is not None:
                st.write('##### Replay Selected Game')
                html_string = f'''
                <head>
                <link href="https://c2a.chesstempo.com/pgnviewer/v2.5/pgnviewerext.vers1.css" media="all" rel="stylesheet" crossorigin>
                <script defer language="javascript" src="https://c1a.chesstempo.com/pgnviewer/v2.5/pgnviewerext.bundle.vers1.js" crossorigin></script>
                <link
                href="https://c1a.chesstempo.com/fonts/MaterialIcons-Regular.woff2"
                rel="stylesheet" crossorigin>
                </head>
                <body>
                <ct-pgn-viewer board-size="300px" move-list-folding="true" move-list-resizable="true" board-resizable="true">
                {rgame}
                </ct-pgn-viewer>
                </body>
                '''
                components.html(html_string, width=1000, height=600, scrolling=False)  # JavaScript works

        # White opp
        with st.expander('WHITE OPPONENTS', expanded=True):
            rgame = replay_grid(df, selected_player, 'Black')
            if rgame is not None:
                st.write('##### Replay Selected Game')
                html_string = f'''
                <head>
                <link href="https://c2a.chesstempo.com/pgnviewer/v2.5/pgnviewerext.vers1.css" media="all" rel="stylesheet" crossorigin>
                <script defer language="javascript" src="https://c1a.chesstempo.com/pgnviewer/v2.5/pgnviewerext.bundle.vers1.js" crossorigin></script>
                <link
                href="https://c1a.chesstempo.com/fonts/MaterialIcons-Regular.woff2"
                rel="stylesheet" crossorigin>
                </head>
                <body>
                <ct-pgn-viewer board-size="300px" move-list-folding="true" move-list-resizable="true" board-resizable="true" flip="true">
                {rgame}
                </ct-pgn-viewer>
                </body>
                '''
                components.html(html_string, width=1000, height=600, scrolling=False)  # JavaScript works


if __name__ == '__main__':
    main()
