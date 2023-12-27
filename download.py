import vitaldb
import pickle
import pandas as pd
import numpy as np

caseid = []

for id in caseid:
    track_names = ['SNUADC/ART', 'SNUADC/ECG_II', 'SNUADC/PLETH', 'Primus/CO2']
    interval = 1/100
    vf = vitaldb.VitalFile(id, track_names)

    data_dict = {}
    for track_name in track_names:
        data_dict[track_name] = np.array(vf.get_samples(track_name, interval)[0]).astype(float)[0]
    df = pd.DataFrame(data_dict)
    df.fillna(method='ffill', inplace=True)

    pkl_file = f"./converted/{id}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(df, f)
