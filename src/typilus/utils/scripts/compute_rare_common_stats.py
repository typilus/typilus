import json
import sys
import numpy as np

if len(sys.argv) != 2:
    print('Usage <file_to_analyze>')
    sys.exit()

with open(sys.argv[1]) as f:
    data = json.load(f)

common_threshold = 100
sum_common_accuracy = 0
sum_common_accuracy_upto_parametric = 0
num_common_samples = 0

sum_rare_accuracy = 0
sum_rare_accuracy_upto_parametric = 0
num_rare_samples = 0

for type_name, type_data in data['per_type_stats'].items():
    if type_data['count'] >= common_threshold:
        num_common_samples += type_data['count']
        sum_common_accuracy += type_data['count'] * type_data['accuracy']
        sum_common_accuracy_upto_parametric += type_data['count'] * type_data['accuracy_up_to_parametric_type']
    else:
        num_rare_samples += type_data['count']
        sum_rare_accuracy += type_data['count'] * type_data['accuracy']
        sum_rare_accuracy_upto_parametric += type_data['count'] * type_data['accuracy_up_to_parametric_type']

print(f'Accuracy of Common {sum_common_accuracy/num_common_samples:%}')
print(f'Accuracy Upto Parametric of Common {sum_common_accuracy_upto_parametric/num_common_samples:%}')

print(f'Accuracy of Rare {sum_rare_accuracy/num_rare_samples:%}')
print(f'Accuracy Upto Parametric of Rare {sum_rare_accuracy_upto_parametric/num_rare_samples:%}')

import matplotlib
matplotlib.use('agg')

font = dict(family='normal', weight='bold', size=23)

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['ps.useafm'] = True

import matplotlib.pyplot as plt

num_samples = np.array([2,5,10,20,50,100,200,500,1000,10000])
acc = np.zeros_like(num_samples, dtype=np.float)
acc_up_to_parametric = np.zeros_like(num_samples, dtype=np.float)
counts = np.zeros_like(num_samples)


for type_name, type_data in data['per_type_stats'].items():
    count_bin = np.digitize(type_data['count'], num_samples)
    acc[count_bin] += type_data['accuracy']
    acc_up_to_parametric[count_bin] += type_data['accuracy_up_to_parametric_type']
    counts[count_bin] += 1


fig, axs = plt.subplots(1, 2, figsize=(5, 3), sharey=True)
axs[0].bar(np.arange(len(num_samples)), 100*acc/counts, tick_label=num_samples)
axs[0].set_xticklabels(num_samples, rotation=90)
axs[0].set_ylabel('Exact Match (\%)')
axs[0].set_xlabel('Annotation Count')
axs[0].grid(axis='y')

axs[1].bar(np.arange(len(num_samples)), 100*acc_up_to_parametric/counts, tick_label=num_samples)
axs[1].set_xticklabels(num_samples, rotation=90)
axs[1].set_ylabel('Match up to Parametric (\%)')
axs[1].set_xlabel('Annotation Count')
plt.grid(axis='y')

plt.tight_layout()
plt.savefig('acc_hist.pdf', dpi=300)
