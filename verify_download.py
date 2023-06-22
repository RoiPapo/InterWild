import os
import os.path as osp
from tqdm import tqdm

checksum_filename = 'InterHand2.6M.images.5.fps.v1.0.tar.CHECKSUM'
if not osp.isfile(checksum_filename):
    print(checksum_filename + ' is missing. Please download it again.')
    exit()
with open(checksum_filename) as f:
    checksums = f.readlines()

good = True
results = []
for line in tqdm(checksums):
    md5sum, filename = line.split()

    if not osp.isfile(filename):
        good = False
        results.append(filename + ' is missing. Please download it again.')
        continue
    
    os.system('md5sum ' + filename + ' > YOUR_CHECKSUM')
    with open('YOUR_CHECKSUM') as f:
        md5sum_yours, _ = f.readline().split()
    os.system('rm YOUR_CHECKSUM')
    
    if md5sum == md5sum_yours:
        results.append('md5sum of ' + filename + ' is correct.')
    else:
        good = False
        results.append('md5sum of ' + filename + ' is wrong. Please download it again')

if good:
    print('All of downloaded files are verified.')
else:
    print('Some of downloaded files are not verified.')

with open('results.txt', 'w') as f:
    for result in results:
        f.write(result + '\n')
print('The results are saved in results.txt')

