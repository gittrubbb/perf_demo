
from train_deepfm import train_deepfm

if __name__ == "__main__":
    user_path = "C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/new_Test_DeepFM/data_files/pickply_user_features.csv"
    note_path = "C:/Users/ADMIN/Desktop/Data/AI perfume/Model comparison/AE/new_Test_DeepFM/data_files/pickply_user_to_notes.csv"

    train_deepfm(
        user_path=user_path,
        note_path=note_path,
        embedding_dim=10,
        hidden_dims=[128, 64],
        dropout=0.3,
        epochs=100,
        batch_size=64
    )
