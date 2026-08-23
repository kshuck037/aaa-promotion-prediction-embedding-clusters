import os
import json
import random
import torch
import yaml

import numpy as np
import polars as pl
import polars.selectors as cs
import torch.nn as nn
import torch.optim as optim

from pydantic import BaseModel, Field, ValidationError
from requests_ratelimiter import LimiterSession
from sklearn.metrics import fbeta_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

# https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def save(self, path):
        torch.save(self.encoder.state_dict(), path)

    def load(self, path):
        self.encoder.load_state_dict(torch.load(path))

    def encode(self, x):
        return self.encoder(x)

class AppConfig(BaseModel):
    get_data: int
    data_dir: str
    seasons: list[int]
    levels: list[int]
    level_map : dict[int, str]
    files: dict[str, str]
    date_url: str
    game_url: str
    player_url: str
    fields: dict[str, list[str]]
    metrics: dict[str, list[str]]  
    
def load_app_config(file_path: str = "config.yaml") -> AppConfig:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
        
    with open(file_path, "r") as f:
        try:
            raw_yaml_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Invalid YAML syntax in {file_path}: {exc}")

    try:
        return AppConfig(**raw_yaml_data)
    except ValidationError as exc:
        print(f"Configuration Validation Failed! Found {exc.error_count()} error(s):")
        print(exc)
        raise SystemExit(1)

def load_game_data(config: AppConfig, session: LimiterSession) -> pl.DataFrame:
    games = []
    for l in config.levels:
        for s in config.seasons:
            #url = f'https://statsapi.mlb.com/api/v1/schedule?sportId={l}&season={s}&gameType=R&fields=dates,games,gamePk,officialDate,dates,games,status,codedGameState,dates,games,tiebreaker'
            date_url = config.date_url.format(level=l, season=s)
            dates = session.get(date_url).json()['dates']

            for day in dates:
                for game in day['games']:
                    game['season'] = s
                    game['level'] = config.level_map[l]
                    #game['url'] = f'https://statsapi.mlb.com/api/v1/game/{game["gamePk"]}/'
                    game['url'] =  config.game_url.format(gamePk=game["gamePk"])
                    if game['status']['codedGameState'] in ['F', 'C']:
                        games.append(game)
    #return games
    games = pl.DataFrame(games)
    games = games.unique()

    return games;

#@profile
def get_box_scores(games: pl.DataFrame, fields: dict[str, list[str]], onfig: AppConfig, session: LimiterSession):
    plays = []
    pbp_batting = []
    pbp_pitching = []
    boxscores = {
        'batting': [],
        'fielding': [],
        'pitching': []
    }
    for game in tqdm(games.iter_rows(named=True)):
        plays = session.get(game['url'] + 'playByPlay').json()['allPlays']
        for play in plays:
            meta = {'game':game['gamePk'], 'season':game['season'], 'level':game['level']}
            meta['inning'] = play['about']['inning']
            meta['halfInning'] = play['about']['halfInning']
            meta['atBatIndex'] = play['about']['atBatIndex']

            for event in play['playEvents']:
                pitch_count = 0
                if event['isPitch']:
                    event_data = {}
                    event_data['player_id'] = play['matchup']['pitcher']['id']
                    event_data['pitch'] = event['details'].get('type', {}).get('code', None)
                    event_data['pitchCount'] = pitch_count
                    event_data['startSpeed'] = event['pitchData'].get('startSpeed', None)
                    if 'breaks' in event['pitchData']:
                        event_data['spinRate'] = event['pitchData']['breaks'].get('spinRate', None)
                        event_data['breakAngle'] = event['pitchData']['breaks'].get('breakAngle', None)
                        event_data['breakLength'] = event['pitchData']['breaks'].get('breakLength', None)
                    else:
                        event_data['spinRate'] = None
                        event_data['breakAngle'] = None
                        event_data['breakLength'] = None
                    pitch_count += 1
                    pbp_pitching.append({**meta, **event_data})

            batting_play_data = {}
            batting_play_data['player_id'] = play['matchup']['batter']['id']
            hitData = play['playEvents'][-1].get('hitData', None)
            if hitData is not None:
                batting_play_data['launchSpeed'] = hitData.get('launchSpeed', None)
                batting_play_data['launchAngle'] = hitData.get('launchAngle', None)
                batting_play_data['totalDistance'] = hitData.get('totalDistance', None)
                if batting_play_data['launchSpeed'] is not None:
                    batting_play_data['hardHit'] = hitData['launchSpeed'] > 95
                else:
                    batting_play_data['hardHit'] = None
            else:
                batting_play_data['launchSpeed'] = None
                batting_play_data['launchAngle'] = None
                batting_play_data['totalDistance'] = None
                batting_play_data['hardHit'] = None
            batting_play_data['isHit'] = play['result']['event'] in ['Home Run', 'Single', 'Double', 'Triple']
            batting_play_data['isHR'] = play['result']['event'] == 'Home Run'
            batting_play_data['isContactOut'] = play['result']['event'] in ['Pop Out','Groundout', 'Flyout', 'Lineout']
            (batting_play_data['isHit'] | batting_play_data['isContactOut']) and pbp_batting.append({**meta, **batting_play_data})

        #print(game['url'] + 'boxscore')
        box_score = session.get(game['url'] + 'boxscore').json()['teams']
        for team in ['away', 'home']:
            for player in box_score[team]['players']:
                meta = {'game':game['gamePk'], 'season':game['season'], 'level':game['level'], 'player_id': int(player[2:])}
                for stat_type in fields:
                    if len(box_score[team]['players'][player]['stats'][stat_type]) > 0:
                        boxscores[stat_type].append({**meta, **{key:box_score[team]['players'][player]['stats'][stat_type].get(key, None) for key in fields[stat_type]}})
    return pbp_pitching, pbp_batting, boxscores

def agg_stats(session: LimiterSession, pbp_pitching: list[dict], pbp_batting: list[dict], boxscores: dict, config: AppConfig):

    
    """
    ------
    Batting Stats
    ------
    """
    boxscores_fielding = pl.DataFrame(boxscores['fielding'])
    fielding_stats = boxscores_fielding.group_by(['season', 'level', 'player_id']).agg([
        ((pl.col('assists').sum() + pl.col('putOuts').sum())/ pl.col('chances').sum()).alias('fieldingEfficiency')
    ])

    boxscores_batting = pl.DataFrame(boxscores['batting'])
    batting_pbp = pl.DataFrame(pbp_batting)

    batting_pbp = batting_pbp.select(pl.exclude(['game', 'inning', 'halfInning', 'atBatIndex']))
    batting_metrics = batting_pbp.group_by(['season', 'level', 'player_id']).agg([
        (hits := pl.col('isHit').sum()).alias('hits'),
        (outs := pl.col('isContactOut').sum()).alias('outs'),
        (pl.when(pl.col("isHit")).then(pl.col('hardHit')).sum() / hits).alias('hardHitRate'),
        (pl.when(pl.col("isContactOut")).then(pl.col('hardHit')).sum() / outs).alias('hardOutRate'),
        pl.when(pl.col('isHit')).then(pl.col("launchSpeed")).mean().alias("avgHitSpeed"),
        pl.when(pl.col('isHit')).then(pl.col("launchAngle")).mean().alias("avgHitAngle"),
        pl.when(pl.col("isHR")).then(pl.col("launchSpeed")).mean().alias("avgHRSpeed"),
        pl.when(pl.col("isHR")).then(pl.col("launchAngle")).mean().alias("avgHRAngle"),
        pl.when(pl.col("isHR")).then(pl.col("totalDistance")).mean().alias("avgHRDistance")
    ]).select(pl.exclude(['hits', 'outs'])).sort(['season', 'level', 'player_id'])

    batting_stats = boxscores_batting.select(pl.exclude('game')).group_by(['season', 'player_id', 'level']).sum()
    batting_stats = batting_stats.group_by(['season', 'level', 'player_id']).agg([
        (pa := pl.col('plateAppearances')).sum().alias('pa'),
        (ab := pl.col('atBats')).sum().alias('ab'),
        (outs := pl.col('airOuts') + pl.col('groundOuts') + pl.col('strikeOuts') ).sum().alias('outs'),
        (tb := pl.col('totalBases')).sum().alias('tb'),
        (pl.col('strikeOuts') / pa).sum().alias('strikeOutRate'),
        (pl.col('baseOnBalls') / pa).sum().alias('bbRate'),
        (pl.col('hits') / pa).sum().alias('hitRate'),
        (pl.col('groundOuts') / ab).sum().alias('groundOutRate'),
        (pl.col('airOuts') / ab).sum().alias('airOutRate'),
        (pl.col('homeRuns') / pa).sum().alias('hrRate'),
        (pl.col('rbi') / pa).sum().alias('rbiRate'),
        (pl.col('runs') / pa).sum().alias('runRate'),
        (pl.col('groundOuts') / outs).sum().alias('groundOutRatio'),
        (pl.col('airOuts') / outs).sum().alias('airOutRatio'),
        (tb / pa).sum().alias('avgBases'),
        (slg := tb / ab).sum().alias('SLG'),
        (obp := (pl.col('hits') + pl.col('baseOnBalls') + pl.col('hitByPitch')) / (pl.col('atBats') + pl.col('baseOnBalls') + pl.col('hitByPitch') + pl.col('sacFlies'))).sum().alias('OBP'),
        (slg + obp).sum().alias('OPS')
    ]).select(pl.exclude(['pa', 'outs', 'tb'])).sort(['season', 'level', 'player_id'])

    batting_stats = batting_stats.join(batting_metrics, on=['season', 'level', 'player_id'], how='left')
    batting_stats = batting_stats.join(fielding_stats,  on=['season', 'level', 'player_id'], how='left')

    """
    ------
    Pitching Stats
    ------
    """
    boxscores_pitching = pl.DataFrame(boxscores['pitching'])
    boxscores_pitching = boxscores_pitching.with_columns(pl.col('inningsPitched').cast(pl.Float64))
    pbp_pitching = pl.DataFrame(pbp_pitching, infer_schema_length=None)

    pitching_stats = boxscores_pitching.select(pl.exclude('game')).group_by(['season', 'player_id', 'level']).sum()
    pitching_stats = pitching_stats.group_by(['season', 'player_id', 'level']).agg([
        (pl.col('strikeOuts') / pl.col('outs')).sum().alias('strikeOutRate'),
        (pl.col('airOuts') / pl.col('outs')).sum().alias('airOutRate'),
        (pl.col('groundOuts') / pl.col('outs')).sum().alias('groundOutRate'),
        (pl.col('pitchesThrown') / pl.col('atBats')).sum().alias('pitchesPerAtBat'),
        (pl.col('pitchesThrown') / pl.col('inningsPitched')).sum().alias('pitchesPerAtInning'),
        (pl.col('strikes') / pl.col('pitchesThrown')).sum().alias('strikeRate'),
        (pl.col('balls') / pl.col('pitchesThrown')).sum().alias('ballRate'),
        (pl.col('hitBatsmen') / pl.col('battersFaced')).sum().alias('hitBatsmenRate'),
        (pl.col('earnedRuns') / pl.col('inningsPitched')).sum().alias('earnedRunsAverage'),
        ((pl.col('hits') + pl.col('baseOnBalls')) / pl.col('inningsPitched')).sum().alias('WHIP'),
        (pl.col('homeRuns') / pl.col('inningsPitched')).sum().alias('HRperInning'),
        (pl.col('inningsPitched')).sum().alias('inningsPitched')
    ])


    pbp_pitching = pbp_pitching.select(pl.exclude(['game', 'inning', 'halfInning', 'atBatIndex', 'pitchCount']))
    pitching_metrics = pbp_pitching.group_by(['season', 'level', 'player_id', 'pitch']).mean()
    pitching_metrics = pitching_metrics.filter(pl.col('pitch').is_in(['FF', 'SL', 'CH', 'SI', 'CU'])).pivot(values=['startSpeed', 'spinRate', 'breakAngle', 'breakLength'], index=['season', 'level', 'player_id'], on='pitch')
    pitching_stats = pitching_stats.join(pitching_metrics, on=['season', 'level', 'player_id'], how='left')
    pitching_stats = pitching_stats.join(fielding_stats,  on=['season', 'level', 'player_id'], how='left')

    player_debuts = player_debut(batting_stats, session, config)
    batting_stats = batting_stats.join(player_debuts, on=['season', 'player_id'], how='left')

    player_debuts = player_debut(pitching_stats, session, config)
    pitching_stats = pitching_stats.join(player_debuts, on=['season', 'player_id'], how='left')

    return batting_stats, pitching_stats

def save_pbp(pbp_batting: pl.DataFrame, pbp_pitching: pl.DataFrame, boxscores: dict, config: AppConfig):
    with open(os.path.join(config.data_dir,  config.files['boxscores']), "w") as file:
        json.dump(boxscores, file, indent=4)
    with open(os.path.join(config.data_dir,  config.files['pbp_batting']), "w") as file:
        json.dump(pbp_batting, file, indent=4)
    with open(os.path.join(config.data_dir,  config.files['pbp_pitching']), "w") as file:
        json.dump(pbp_pitching, file, indent=4)

def load_pbp(config: AppConfig):
    with open(os.path.join(config.data_dir,  config.files['boxscores']), "r") as file:
        boxscores = json.load(file)
    with open(os.path.join(config.data_dir,  config.files['pbp_batting']), "r") as file:
        pbp_batting = json.load(file)
    with open(os.path.join(config.data_dir,  config.files['pbp_pitching']), "r") as file:
        pbp_pitching = json.load(file)
    return pbp_batting, pbp_pitching, boxscores

def save_stats(batting_stats: pl.DataFrame, pitching_stats: pl.DataFrame, config: AppConfig):
    batting_stats.write_csv(os.path.join(config.data_dir, config.files['batting_stats']), float_precision=4)
    pitching_stats.write_csv(os.path.join(config.data_dir, config.files['pitching_stats']), float_precision=4)

def load_stats(config: AppConfig):
    batting_stats = pl.read_csv(os.path.join(config.data_dir, config.files['batting_stats']))
    pitching_stats = pl.read_csv(os.path.join(config.data_dir, config.files['pitching_stats']))
    return batting_stats, pitching_stats

def player_debut(stats: pl.DataFrame, session: LimiterSession, config: AppConfig) -> pl.DataFrame:
    player_debuts = []
    for group in stats.select(['season', 'player_id']).group_by('player_id'):
        player_id = group[0][0]
        data = group[1]['season']
        player_debut = []
        debut_date = session.get(config.player_url.format(id=player_id)).json()['people'][0].get('mlbDebutDate', None)
        for season in data:
            player_year = {}
            player_year['season'] = season
            player_year['player_id'] = player_id
            if debut_date is not None:
                if int(debut_date[:4]) <= int(season):
                    player_year['debutNext'] = 'X'
                elif int(debut_date[:4]) == int(season) + 1:
                    player_year['debutNext'] = 'Y'
                else:
                    player_year['debutNext'] = 'N'
            else:
                player_year['debutNext'] = 'N'
            player_debut.append(player_year)
        player_debuts.extend(player_debut)
    player_debuts = pl.DataFrame(player_debuts)
    return player_debuts

def train_encoder(data, metric_cols,
                  hidden_dim=15, latent_dim=5, lr=1e-3, weight_decay=1e-3,
                  epochs=2500, batch_size=128):
  encoder = AutoEncoder(input_dim=len(metric_cols), hidden_dim=hidden_dim, latent_dim=latent_dim)
  loss_function = nn.MSELoss()
  optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  encoder.to(device)
  dataset = data.to_torch(
      return_type='dataset',
      features=metric_cols,
      dtype=pl.Float32,
  )

  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  final_loss = 0
  for epoch in range(epochs):
      losses = []
      for stats in dataloader:
          stats = stats[0]
          stats = stats.to(device)
          reconstructed_stats = encoder(stats)
          loss = loss_function(reconstructed_stats, stats)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          losses.append(loss.item())
      final_loss = np.array(losses).mean()
  print(f"Loss: {final_loss:.6f}")

  return encoder

def create_embeddings(data, encoder, metric_cols):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  encoder.to(device)
  dataset = data.to_torch(
      return_type='dataset',
      features=metric_cols,
      dtype=pl.Float32,
  )

  dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
  embeddings = []
  for stats in dataloader:
      stats = stats[0]
      stats = stats.to(device)
      embedding = encoder.encode(stats)
      embeddings.append(embedding.detach().cpu().numpy())
  embeddings = np.concatenate(embeddings, axis=0)
  return embeddings


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config = load_app_config('./config.yaml')
    session = LimiterSession(per_second=1024)
    #session = LimiterSession()
    seed_everything()

    if config.get_data == 0:
        games = load_game_data(config, session)
        pbp_pitching, pbp_batting, boxscores = get_box_scores(games, config.fields, config, session)
        save_pbp(pbp_batting, pbp_pitching, boxscores, config)
        batting_stats, pitching_stats = agg_stats(session, pbp_pitching, pbp_batting, boxscores, config)
        save_stats(batting_stats, pitching_stats, config)
    elif config.get_data == 1:
        pbp_batting, pbp_pitching, boxscores = load_pbp(config)
        batting_stats, pitching_stats = agg_stats(session, pbp_pitching, pbp_batting, boxscores, config)
        save_stats(batting_stats, pitching_stats, config)
    else:
        batting_stats, pitching_stats = load_stats(config)



    """
    metric_cols = ['ab', 'strikeOutRate', 'bbRate',
              'hitRate','groundOutRate', 'airOutRate',
              'hrRate', 'rbiRate', 'runRate','groundOutRatio',
              'airOutRatio', 'avgBases', 'SLG', 'OBP', 'OPS',
              'hardHitRate', 'hardOutRate', 'avgHitSpeed',
              'avgHitAngle', 'avgHRSpeed', 'avgHRAngle',
              'avgHRDistance', 'fieldingEfficiency']
    """

    batting_stats = batting_stats.fill_null(0).fill_nan(0)
    batting_stats = batting_stats.with_columns(
        pl.when(pl.col("runRate").is_infinite())
        .then(0.0)
        .otherwise(pl.col("runRate"))
        .alias("runRate")
    )

    pitching_stats = pitching_stats.fill_null(0).fill_nan(0)
    pitching_stats = pitching_stats.with_columns(
        pl.when(cs.float().is_infinite())
        .then(0)
        .otherwise(cs.float())
        .name.keep() 
    )

    print("All Batting Stats")
    batting_stats_test1 = batting_stats.with_columns(
        [
            (pl.col(col) - pl.col(col).mean())
            / (pl.col(col).std())
            for col in config.metrics['batting']
        ]
    )
    batting_encoder_one = train_encoder(batting_stats_test1, config.metrics['batting'])

  
    print("Basic Batting Stats")
    batting_stats_test2 = batting_stats.with_columns(
        [
            (pl.col(col) - pl.col(col).mean())
            / (pl.col(col).std())
            for col in config.metrics['batting_basic']
        ]
    )
    batting_encoder_two = train_encoder(batting_stats_test2, config.metrics['batting_basic'], hidden_dim=10, latent_dim=3)


    print("All Pitching Stats")
    pitching_stats_test1 = pitching_stats.with_columns(
        [
            (pl.col(col) - pl.col(col).mean())
            / (pl.col(col).std())
            for col in config.metrics['pitching']
        ]
    )
    pitching_encoder_one = train_encoder(pitching_stats_test1, config.metrics['pitching'], hidden_dim=22, latent_dim=8)


    print("Basic Pitching Stats")
    pitching_stats_test2 = pitching_stats.with_columns(
        [
            (pl.col(col) - pl.col(col).mean())
            / (pl.col(col).std())
            for col in config.metrics['pitching_basic']
        ]
    )
    pitching_encoder_two = train_encoder(pitching_stats_test2, config.metrics['pitching_basic'], hidden_dim=8, latent_dim=4)

    """
    ---
    Embedding DB
    ---
    """

    with torch.no_grad():
        batting_embedings = create_embeddings(batting_stats_test1, batting_encoder_one, config.metrics['batting'])
    
    batting_db = batting_stats.with_columns(
        pl.Series(batting_embedings).alias('embedding')
    )
    batting_db = batting_db.select(['season', 'level', 'player_id', 'debutNext', 'embedding'])

    
    with torch.no_grad():
        pitching_embedings = create_embeddings(pitching_stats_test1, pitching_encoder_one, config.metrics['pitching'])
    
    pitching_db = pitching_stats.with_columns(
        pl.Series(pitching_embedings).alias('embedding')
    )
    pitching_db = pitching_db.select(['season', 'level', 'player_id', 'debutNext', 'embedding'])
    
    """
    ---
    K-Opimization
    ---
    """

    # Batting
    reference_space = batting_db.filter((pl.col('debutNext') != 'X') & (pl.col('season') == 2023))
    reference_space = reference_space.with_columns(
        pl.col("debutNext").replace_strict({"Y": 1, "N": 0}, default=None).alias("labels")
    )
    reference_space_inputs = np.vstack(reference_space['embedding'].to_numpy())
    reference_space_labels = reference_space['labels'].to_numpy()

    lookup_space = batting_db.filter((pl.col('debutNext') != 'X') & (pl.col('season') == 2024))
    lookup_space = lookup_space.with_columns(
        pl.col("debutNext").replace_strict({"Y": 1, "N": 0}, default=None).alias("labels")
    )
    lookup_space_inputs = np.vstack(lookup_space['embedding'].to_numpy())
    lookup_space_labels = lookup_space['labels'].to_numpy()
    print("Batting Optimal K")
    print("Reference Stats")
    print(reference_space.get_column("debutNext").value_counts())
    print("Lookup Stats")
    print(lookup_space.get_column("debutNext").value_counts())
    for i in range(1, 20, 1):
        knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
        knn.fit(reference_space_inputs, reference_space_labels)
        pred_i = knn.predict(lookup_space_inputs)

        error_rate = np.mean(pred_i != lookup_space_labels)
        f_score = fbeta_score(lookup_space_labels, pred_i, beta=1.0)
        print(f"Neighbors: {i}, F1 Score: {f_score:.4f}, Error Rate: {error_rate:.4f}")


    # Pitching
    reference_space = pitching_db.filter((pl.col('debutNext') != 'X') & (pl.col('season') == 2023))
    reference_space = reference_space.with_columns(
        pl.col("debutNext").replace_strict({"Y": 1, "N": 0}, default=None).alias("labels")
    )
    reference_space_inputs = np.vstack(reference_space['embedding'].to_numpy())
    reference_space_labels = reference_space['labels'].to_numpy()

    lookup_space = pitching_db.filter((pl.col('debutNext') != 'X') & (pl.col('season') == 2024))
    lookup_space = lookup_space.with_columns(
        pl.col("debutNext").replace_strict({"Y": 1, "N": 0}, default=None).alias("labels")
    )
    lookup_space_inputs = np.vstack(lookup_space['embedding'].to_numpy())
    lookup_space_labels = lookup_space['labels'].to_numpy()
    print("Pitching Optimal K")
    print("Reference Stats")
    print(reference_space.get_column("debutNext").value_counts())
    print("Lookup Stats")
    print(lookup_space.get_column("debutNext").value_counts())
    for i in range(1, 20, 1):
        knn = KNeighborsClassifier(n_neighbors=i, metric='cosine', weights='distance')
        knn.fit(reference_space_inputs, reference_space_labels)
        pred_i = knn.predict(lookup_space_inputs)

        error_rate = np.mean(pred_i != lookup_space_labels)
        f_score = fbeta_score(lookup_space_labels, pred_i, beta=1.0)
        print(f"Neighbors: {i}, F1 Score: {f_score:.4f}, Error Rate: {error_rate:.4f}")

    """
    ---
    Scoring
    ---
    """

    # Batting
    reference_space = batting_db.filter((pl.col('debutNext') != 'X') & (pl.col('season') == 2024))
    reference_space = reference_space.with_columns(
        pl.col("debutNext").replace_strict({"Y": 1, "N": 0}, default=None).alias("labels")
    )
    reference_space_inputs = np.vstack(reference_space['embedding'].to_numpy())
    reference_space_labels = reference_space['labels'].to_numpy()

    lookup_space = batting_db.filter((pl.col('debutNext') != 'X') & (pl.col('season') == 2025))
    lookup_space = lookup_space.with_columns(
        pl.col("debutNext").replace_strict({"Y": 1, "N": 0}, default=None).alias("labels")
    )
    lookup_space_inputs = np.vstack(lookup_space['embedding'].to_numpy())
    lookup_space_labels = lookup_space['labels'].to_numpy()

    knn = KNeighborsClassifier(n_neighbors=8, weights='distance')
    knn.fit(reference_space_inputs, reference_space_labels)
    pred_i = knn.predict(lookup_space_inputs)

    f0_5 = fbeta_score(lookup_space_labels, pred_i, beta=0.5)
    f1_0 = fbeta_score(lookup_space_labels, pred_i, beta=1.0)
    f2_0 = fbeta_score(lookup_space_labels, pred_i, beta=2.0)
    print("Batting Scoring")
    print("Reference Stats")
    print(reference_space.get_column("debutNext").value_counts())
    print("Lookup Stats")
    print(lookup_space.get_column("debutNext").value_counts())
    print(f"F0.5 (Precision focus): {f0_5:.4f}")
    print(f"F1.0 (Balanced): {f1_0:.4f}")
    print(f"F2.0 (Recall focus): {f2_0:.4f}")

    # Pitching
    reference_space = pitching_db.filter((pl.col('debutNext') != 'X') & (pl.col('season') == 2024))
    reference_space = reference_space.with_columns(
        pl.col("debutNext").replace_strict({"Y": 1, "N": 0}, default=None).alias("labels")
    )
    reference_space_inputs = np.vstack(reference_space['embedding'].to_numpy())
    reference_space_labels = reference_space['labels'].to_numpy()

    lookup_space = pitching_db.filter((pl.col('debutNext') != 'X') & (pl.col('season') == 2025))
    lookup_space = lookup_space.with_columns(
        pl.col("debutNext").replace_strict({"Y": 1, "N": 0}, default=None).alias("labels")
    )
    lookup_space_inputs = np.vstack(lookup_space['embedding'].to_numpy())
    lookup_space_labels = lookup_space['labels'].to_numpy()

    knn = KNeighborsClassifier(n_neighbors=6, metric='cosine', weights='distance')
    knn.fit(reference_space_inputs, reference_space_labels)
    pred_i = knn.predict(lookup_space_inputs)

    f0_5 = fbeta_score(lookup_space_labels, pred_i, beta=0.5)
    f1_0 = fbeta_score(lookup_space_labels, pred_i, beta=1.0)
    f2_0 = fbeta_score(lookup_space_labels, pred_i, beta=2.0)
    print("Pitching Scoring")
    print("Reference Stats")
    print(reference_space.get_column("debutNext").value_counts())
    print("Lookup Stats")
    print(lookup_space.get_column("debutNext").value_counts())
    print(f"F0.5 (Precision focus): {f0_5:.4f}")
    print(f"F1.0 (Balanced): {f1_0:.4f}")
    print(f"F2.0 (Recall focus): {f2_0:.4f}")

if __name__ == "__main__":
    main()