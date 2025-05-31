from datasets import load_from_disk

dset = load_from_disk("./data/chess_puzzles")
print(dset[0])
print(dset.column_names)
print(dset.shape)
