# import vitaldb
# import pickle
# import pandas as pd

# caseid = [137
#           ]

# for id in caseid:
#     track_names = ['SNUADC/ART', 'SNUADC/ECG_II', 'SNUADC/PLETH', 'Primus/CO2']
#     interval = 1/100
#     vf = vitaldb.VitalFile(id, track_names)
#     csv_filename = f'./137.csv'
#     vf.to_csv(csv_filename, track_names, interval)
#     print(type(vf))
import vitaldb
import pickle
import pandas as pd
import numpy as np

caseid = [112, 114, 116, 117, 118, 119, 124, 125, 126, 128, 130, 132, 135, 136, 138, 139, 140, 142, 143, 145, 148, 149, 150, 152, 156, 157, 158, 161, 163, 164, 166, 167, 172, 175, 177, 178, 180, 183, 184, 185, 186, 189, 190, 191, 195, 197, 198, 199, 200
          ]

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
