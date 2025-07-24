from new_train_nfm import train_nfm

if __name__ == "__main__":
    user_path =  "C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/new_Test_FM/data_files/user_features.csv"
    note_path = "C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/new_Test_FM/data_files/user_to_notes.csv"

    train_nfm(
        dataset_path_user=user_path,
        dataset_path_note=note_path,
        embedding_dim=16,
        epochs=100,
        batch_size=64
    )
