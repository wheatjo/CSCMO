import numpy as np
import pickle
import json
import csv
import yaml

pickle_file = 'a.pickle'
# a = np.random.random((600, 15))
# p = {}
# # print(a)
# p['a'] = a
# c = np.random.random((600, 15))
# p['c'] = c
# np.save('a.npy', p)
# f = open('a.pickle', 'rb')
# res = pickle.load(f)
# print(res['a'])
problem_name = 'mw1'
igd = 10.03
hv = 20.04
indicator = [{'indicator': 'igd', 'value': igd}, {'indicator': 'hv', 'value': hv}]
# indicator_str = json.dumps(indicator, indent=4)
# j_file = open('a.json', 'w')
# json.dump(indicator_str, j_file)

# with open('a.csv', 'w', encoding='utf-8', newline='') as csvfile:
#     name = {'indicator', 'value'}
#     writer = csv.DictWriter(csvfile, fieldnames=name)
#     for i in range(len(indicator)):
#         item_dict = indicator[i]
#         writer.writerow(item_dict)

message = {'problem_name': problem_name, 'max_FE': 600, 'indicator': {'igd': igd, 'hv': hv}}
yml = yaml.dump(message, sort_keys=False)
print(yml)
with open('a.yml', 'w') as f:
    yaml.dump(message, f, sort_keys=False)

