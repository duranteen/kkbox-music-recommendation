import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import scipy.sparse as sp

# Read data
path = 'data/'
train_data = pd.read_csv(path + 'train.csv')
test_data = pd.read_csv(path + 'test.csv')
members_data = pd.read_csv(path + 'members.csv')
songs_data = pd.read_csv(path + 'songs.csv')
song_extinfo_data = pd.read_csv(path + 'song_extra_info.csv', parse_dates=['expiration_date', 'registration_init_time'])

# Count the songs that appear in the training and test data
existing_songs = set(train_data['song_id'].append(test_data['song_id']))
# Filtrate songs_data file
songs_data['existing'] = songs_data['song_id'].apply(lambda x: 1 if x in existing_songs else 0)
songs_data = songs_data[songs_data.existing == 1]
songs_data.drop('existing', axis=1, inplace=True)
# Filtrate song_extinfo_data file
song_extinfo_data['existing'] = song_extinfo_data['song_id'].apply(lambda x: 1 if x in existing_songs else 0)
song_extinfo_data = song_extinfo_data[song_extinfo_data.existing == 1]
song_extinfo_data.drop('existing', axis=1, inplace=True)
# Filtrate members_data file
existing_mem = set(train_data['msno'].append(train_data['msno']))
members_data['existing'] = members_data['msno'].apply(lambda x: 1 if x in existing_mem else 0)
members_data = members_data[members_data.existing == 1]
members_data.drop('existing', axis=1, inplace=True)

# labelEncoding msno and song id
# msno
msno_encoder = LabelEncoder()
msno_encoder.fit(members_data['msno'].values)
members_data['msno'] = msno_encoder.transform(members_data['msno'])
train_data['msno'] = msno_encoder.transform(train_data['msno'])
test_data['msno'] = msno_encoder.transform(test_data['msno'])
# song id
sid_encoder = LabelEncoder()
sid_encoder.fit(songs_data['song_id'])
songs_data['song_id'] = sid_encoder.transform(songs_data['song_id'])
train_data['song_id'] = sid_encoder.transform(train_data['song_id'])
test_data['song_id'] = sid_encoder.transform(test_data['song_id'])

# LabelEncoding remaining category features
# source_system_tab,source_screen_name,source_type
columns = ['source_system_tab', 'source_screen_name', 'source_type']
for col in columns:
    encoder = LabelEncoder()
    if train_data[col].dtypes == '0':
        encoder.fit(train_data[col].fillna('nan').append(test_data[col].fillna('nan')))
        train_data[col] = encoder.transform(train_data[col].fillna('nan'))
        test_data[col] = encoder.transform(test_data[col].fillna('nan'))
    else:
        encoder.fit(train_data[col].fillna(-1).append(test_data[col].fillna(-1)))
        train_data[col] = encoder.transform(train_data[col].fillna(-1))
        test_data[col] = encoder.transform(test_data[col].fillna(-1))
# city,gender,registered_via
columns = ['city', 'gender', 'registered_via']
for col in columns:
    encoder = LabelEncoder()
    if members_data[col].dtypes == '0':
        encoder.fit(members_data[col].fillna('nan'))
        members_data[col] = encoder.transform(members_data[col].fillna('nan'))
    else:
        encoder.fit(members_data[col].fillna(-1))
        members_data[col] = encoder.transform(members_data[col].fillna(-1))


# genre_idx: | delimiter -> 4 col
def get_genreids_split(df):
    genreids_split = np.zeros((len(df), 4))
    for i in range(len(df)):
        if df[i] == 'nan':
            continue
        else:
            num_genre = str(df[i]).count('|')
            splits = str(df[i]).split('|')
            if num_genre + 1 > 2:
                genreids_split[i, 0] = int(splits[0])
                genreids_split[i, 1] = int(splits[1])
                genreids_split[i, 2] = int(splits[2])
            elif num_genre + 1 > 1:
                genreids_split[i, 0] = int(splits[0])
                genreids_split[i, 1] = int(splits[1])
            elif num_genre + 1 == 1:
                genreids_split[i, 0] = int(splits[0])
            genreids_split[i, 3] = num_genre + 1
    return genreids_split


genreids_split = get_genreids_split(songs_data['genre_ids'].fillna('nan').values)
songs_data['first_genre_id'] = genreids_split[:, 0]
songs_data['second_genre_id'] = genreids_split[:, 1]
songs_data['third_genre_id'] = genreids_split[:, 2]
songs_data['fourth_genre_id'] = genreids_split[:, 3]
# label encoding
genre_encoder = LabelEncoder()
genre_encoder.fit(songs_data['first_genre_id'].append(songs_data['second_genre_id']).append(songs_data['third_genre_id']))
songs_data['first_genre_id'] = genre_encoder.transform(songs_data['first_genre_id'])
songs_data['second_genre_id'] = genre_encoder.transform(songs_data['second_genre_id'])
songs_data['third_genre_id'] = genre_encoder.transform(songs_data['third_genre_id'])


# artist_name -> get first
def artist_count(x):
    return x.count('and') + x.count(',') + x.count(' feat') + x.count('&') + 1


def get_1st_artist(x):
    if x.count('and') > 0:
        x = x.split('and')[0]
    if x.count(',') > 0:
        x = x.split(',')[0]
    if x.count(' feat') > 0:
        x = x.split(' feat')
    if x.count('&') > 0:
        x = x.split('&')
    return x


songs_data['artist_cnt'] = songs_data['artist_name'].apply(artist_count).astype(np.int8)
songs_data['first_artist_name'] = songs_data['artist_name'].apply(get_1st_artist)


# lyricist and composer
def lyricist_or_composer_count(x):
    try:
        return x.count('and') + x.count('/') + x.count('|') + x.count('\\') + x.count(';') + x.count('&') + 1
    except:
        return 0


def get_first_lyricist_or_composer(x):
    try:
        if x.count('and') > 0:
            x = x.split('and')[0]
        if x.count(',') > 0:
            x = x.split(',')[0]
        if x.count(' feat') > 0:
            x = x.split(' feat')[0]
        if x.count('&') > 0:
            x = x.split('&')[0]
        if x.count('|') > 0:
            x = x.split('|')[0]
        if x.count('/') > 0:
            x = x.split('/')[0]
        if x.count('\\') > 0:
            x = x.split('\\')[0]
        if x.count(';') > 0:
            x = x.split(';')[0]
        return x.strip()
    except:
        return x


songs_data['lyricist_cnt'] = songs_data['lyricist'].apply(lyricist_or_composer_count).astype(np.int8)
songs_data['composer_cnt'] = songs_data['composer'].apply(lyricist_or_composer_count).astype(np.int8)
songs_data['first_lyricist_name'] = songs_data['lyricist'].apply(get_first_lyricist_or_composer)
songs_data['first_composer_name'] = songs_data['composer'].apply(get_first_lyricist_or_composer)
# label encoding
columns = ['first_artist_name','first_lyricist_name','first_composer_name']
for col in columns:
    print(col)
    encoder = LabelEncoder()
    encoder.fit(songs_data[col].fillna('nan'))
    songs_data[col] = encoder.transform(songs_data[col].fillna('nan'))

