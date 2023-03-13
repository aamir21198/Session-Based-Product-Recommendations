import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# data config (all methods)
DATA_PATH = '../data/retailrocket/raw/'
DATA_PATH_PROCESSED = '../data/retailrocket/prepared/'
DATA_FILE = 'events'
SESSION_LENGTH = 30 * 60  # 30 minutes

# filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

# min date config
MIN_DATE = '2014-04-01'

# days test default config
DAYS_TEST = 1

# slicing default config
NUM_SLICES = 10
DAYS_OFFSET = 0
DAYS_SHIFT = 5
DAYS_TRAIN = 25
DAYS_TEST = 2


def load_data(file):
    # load csv
    data = pd.read_csv(file+'.csv', sep=',', header=0, usecols=[0, 1, 2, 3], dtype={
                       0: np.int64, 1: np.int32, 2: str, 3: np.int32})
    # specify header names
    data.columns = ['Time', 'UserId', 'Type', 'ItemId']
    data['Time'] = (data.Time / 1000).astype(int)

    # sessionize
    data.sort_values(by=['UserId', 'Time'], ascending=True, inplace=True)

    # compute the time difference between queries
    tdiff = np.diff(data['Time'].values)

    # check which of them are bigger then session_th
    split_session = tdiff > SESSION_LENGTH
    split_session = np.r_[True, split_session]

    # check when the user chenges is data
    new_user = data['UserId'].values[1:] != data['UserId'].values[:-1]
    new_user = np.r_[True, new_user]

    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)

    # compute the session ids
    session_ids = np.cumsum(new_session)
    data['SessionId'] = session_ids

    data.sort_values(['SessionId', 'Time'], ascending=True, inplace=True)

    cart = data[data.Type == 'addtocart']
    data = data[data.Type == 'view']
    del data['Type']

    print(data)
    print(data.Time.min())
    print(data.Time.max())

    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat()))

    return data, cart


def filter_data(data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH):
    # filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(
        data.ItemId, item_supports[item_supports >= min_item_support].index)]

    # filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(
        data.SessionId, session_lengths[session_lengths >= min_session_length].index)]

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat()))

    return data


def split_data(data, output_file, days_test):
    """
    Split the entire dataset into training and testing data
    """
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days_test)

    # Generate training and testing sessions
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times <
                                      test_from.timestamp()].index
    session_test = session_max_times[session_max_times >=
                                     test_from.timestamp()].index

    # Generate training and testing sets
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]

    # Filter out all single-event sessions
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    # Output csv files
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(
        len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)

    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(
        len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)


def slice_data(data, output_file, num_slices, days_offset, days_shift, days_train, days_test):
    # For each slice of data, split to training and testing sets
    for slice_id in range(0, num_slices):
        split_data_slice(data, output_file, slice_id, days_offset +
                         (slice_id*days_shift), days_train, days_test)


def split_data_slice(data, output_file, slice_id, days_offset, days_train, days_test):
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(), data_end.isoformat()))

    start = datetime.fromtimestamp(
        data.Time.min(), timezone.utc) + timedelta(days_offset)
    middle = start + timedelta(days_train)
    end = middle + timedelta(days_test)

    # prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >=
                                      start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(
        data.SessionId, greater_start.intersection(lower_end))]

    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format(slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat(), end.date().isoformat()))

    # split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times <
                                       middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >=
                                      middle.timestamp()].index

    # Training set
    train = data[np.in1d(data.SessionId, sessions_train)]

    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat()))

    train.to_csv(output_file + '_train_full.' +
                 str(slice_id)+'.txt', sep='\t', index=False)

    # Testing set
    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]

    # Filter out single event sessions
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format(slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(), end.date().isoformat()))

    test.to_csv(output_file + '_test.'+str(slice_id) +
                '.txt', sep='\t', index=False)


if __name__ == '__main__':
    pass
