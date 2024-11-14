import os
import random
import shutil

train_ratio = 0.7  # 70% of the data
test_ratio = 0.15  # 15% for testing
eval_ratio = 0.15  # 15% for evaluation

label_folder = "./aur-684_multimodal_document_parsing_v2/label"

filenames = os.listdir(label_folder)
random.shuffle(filenames)

total = len(filenames)
train_end = int(total * train_ratio)
test_end = train_end + int(total * test_ratio)

train_files = filenames[:train_end]
test_files = filenames[train_end:test_end]
eval_files = filenames[test_end:]

for name in train_files:
    src = os.path.join(label_folder, name)
    dest = os.path.join("./train", name)
    shutil.copy(src, dest)

for name in test_files:
    src = os.path.join(label_folder, name)
    dest = os.path.join("./test", name)
    shutil.copy(src, dest)

for name in eval_files:
    src = os.path.join(label_folder, name)
    dest = os.path.join("./dev", name)
    shutil.copy(src, dest)
    