from utils.eval_utils import get_filepaths

if __name__ == "__main__":
    # json file to check
    json_file = pd.read_json(
        "/home/SharedFolder/CurrentDatasets/bdd100k_sorted_quickaug/train_A_ganaug_050/bdd100k_sorted_train_A_ganaug_050_augonlyasbase.json")
    # root directory to search through
    data_root = "/home/SharedFolder/CurrentDatasets/bdd100k_sorted/train_A"
    # verify that all file names listed in json are somewhere within the root directory
    get_filepaths(data_root, json_file.name)
    print("Found all")
