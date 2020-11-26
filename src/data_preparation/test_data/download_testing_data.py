import os

from tqdm import tqdm

if not os.path.exists('test_repositories'):
    os.mkdir('test_repositories')

os.chdir('test_repositories')

for repo_name in tqdm(open('../repositories_list.txt', 'r').readlines()):
    repo_name = repo_name.strip()
    if repo_name != "":
        os.system(f"git clone --quiet --depth 1 {repo_name}")
