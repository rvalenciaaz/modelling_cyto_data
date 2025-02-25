import glob
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import json
import datetime
import os

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend as K

# Optuna for hyperparameter optimization
import optuna

# Sklearn imports for splitting, scaling, and PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(['science', 'nature'])

# ---------------------------------------------------------
# 0. Logging utility for timestamps
# ---------------------------------------------------------
log_steps = []
def log_message(message):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_steps.append(f"[{now_str}] {message}")
    print(message)

# ---------------------------------------------------------
# 1. Reading & subsampling CSV files
# ---------------------------------------------------------
log_message("Reading CSV files and subsampling...")
csv_files = glob.glob("species*.csv")  # e.g. species1.csv, species2.csv, etc.
df_list = []
for file_path in csv_files:
    temp_df = pd.read_csv(file_path)
    # (Optionally subsample; e.g., uncomment the following line)
    # temp_df = temp_df.sample(n=min(len(temp_df), 10_000), random_state=42)
    label = file_path.split('.')[0]  # e.g. "species1"
    temp_df['Label'] = label
    df_list.append(temp_df)
combined_df = pd.concat(df_list, ignore_index=True)
# Keep the label for later visualization of the latent space.
combined_df["Label"] = combined_df["Label"].str.replace("species", "", regex=False)
log_message(f"Combined dataset shape: {combined_df.shape}")

# ---------------------------------------------------------
# 2. Filtering features based on MAD
# ---------------------------------------------------------
log_message("Filtering numeric features based on MAD...")
numerical_data = combined_df.select_dtypes(include=[np.number])
cv_results = {}
for col in numerical_data.columns:
    mean_val = numerical_data[col].mean()
    std_val = numerical_data[col].std()
    cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    mad = median_abs_deviation(numerical_data[col].values, scale='normal')
    cv_results[col] = [col, cv, mad]
cv_df = pd.DataFrame(cv_results.values(), columns=['Feature', 'CV', 'MAD'])
MAD_THRESHOLD = 5
features_to_keep = cv_df.loc[cv_df["MAD"] >= MAD_THRESHOLD, "Feature"].tolist()  # filter by MAD
cols_to_keep = features_to_keep + ["Label"]
final_df = combined_df[cols_to_keep].copy()
log_message(f"Number of features kept after MAD filtering: {len(features_to_keep)}")

# ---------------------------------------------------------
# 3. Train/Test split (unsupervised; labels are kept for latent space plotting)
# ---------------------------------------------------------
log_message("Splitting data into training and test sets...")
X = final_df.drop(columns=["Label"]).values
y = final_df["Label"].values  # kept for later visualization
# (Encode labels only for plotting purposes)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_encoded
)
log_message(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Pre-scale data (the scaler will be saved for replication if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Define helper function and VAE model in Keras
# ---------------------------------------------------------
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
       Arguments:
         args (tensor): mean and log of variance of Q(z|X)
       Returns:
         z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_vae_model(input_dim, hidden_size, latent_dim, num_layers):
    """
    Builds a VAE using the Keras functional API.
    Returns the compiled VAE model, the encoder, and the decoder.
    """
    # Encoder
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = inputs
    for i in range(num_layers):
        x = layers.Dense(hidden_size, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = latent_inputs
    for i in range(num_layers):
        x = layers.Dense(hidden_size, activation='relu')(x)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    
    # VAE = encoder + decoder
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    
    # Define VAE loss
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    
    return vae, encoder, decoder

# ---------------------------------------------------------
# 5. Optimize hyperparameters with Optuna
# ---------------------------------------------------------
def objective(trial):
    # Hyperparameters to optimize
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32])
    latent_dim = trial.suggest_categorical("latent_dim", [2, 5])
    num_layers = trial.suggest_categorical("num_layers", [1, 2])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    epochs = trial.suggest_int("epochs", 30, 30)  # Fixed 30 epochs for optuna optimization

    input_dim = X_train_scaled.shape[1]
    vae, _, _ = create_vae_model(input_dim, hidden_size, latent_dim, num_layers)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    
    # Use 20% of training data as validation split
    history = vae.fit(X_train_scaled, X_train_scaled,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_split=0.2,
                      verbose=0)
    # We use the final validation loss as the objective
    final_val_loss = history.history['val_loss'][-1]
    return final_val_loss

log_message("Starting Optuna hyperparameter optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best_params = study.best_trial.params
log_message(f"Best hyperparameters: {best_params}")

# Save best hyperparameters to file
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=2)

# ---------------------------------------------------------
# 6. Train final VAE model using best hyperparameters
# ---------------------------------------------------------
# Increase epochs for the final training run
best_params['epochs'] = 100

input_dim = X_train_scaled.shape[1]
vae_final, encoder_final, decoder_final = create_vae_model(
    input_dim,
    hidden_size=best_params["hidden_size"],
    latent_dim=best_params["latent_dim"],
    num_layers=best_params["num_layers"]
)
vae_final.compile(optimizer=tf.keras.optimizers.Adam(best_params["learning_rate"]))

log_message("Training final VAE model with best hyperparameters...")
history_final = vae_final.fit(X_train_scaled, X_train_scaled,
                              epochs=best_params['epochs'],
                              batch_size=best_params["batch_size"],
                              validation_split=0.2,
                              verbose=1)

# Save training history (loss and val_loss) for replication
np.savez("vae_training_history.npz", 
         loss=np.array(history_final.history['loss']),
         val_loss=np.array(history_final.history['val_loss']))

# ---------------------------------------------------------
# 7. Evaluate on the test set and extract latent space encoding
# ---------------------------------------------------------
log_message("Evaluating final VAE model on test set...")
test_loss = vae_final.evaluate(X_test_scaled, X_test_scaled, 
                               batch_size=best_params["batch_size"], verbose=1)
log_message(f"Test loss: {test_loss:.4f}")

log_message("Extracting latent space encoding for test set...")
# Get latent representations (using the encoder's mean output)
z_mean, _, _ = encoder_final.predict(X_test_scaled)
# Save latent encodings along with original labels
np.savez("latent_space_data.npz", latent_encodings=z_mean, labels=y_test)

# If the latent dimension is not 2, reduce via PCA for plotting
if z_mean.shape[1] == 2:
    latent_2d = z_mean
else:
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(z_mean)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                      c=label_encoder.transform(y_test), cmap='viridis', alpha=0.7)
plt.title("Latent Space Encoding (Test Set)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
cbar = plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)))
cbar.ax.set_ylabel("Encoded Label")
plt.tight_layout()
plt.savefig("latent_space_plot.png", bbox_inches='tight')
plt.close()
log_message("Saved latent space plot to 'latent_space_plot.png'.")

# ---------------------------------------------------------
# 8. Plot and save training history
# ---------------------------------------------------------
epochs_range = np.arange(1, best_params['epochs'] + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history_final.history['loss'], label="Train Loss", color='blue')
plt.plot(epochs_range, history_final.history['val_loss'], label="Validation Loss", color='orange')
plt.title("Training and Validation Loss (VAE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("vae_training_loss.png", bbox_inches='tight')
plt.close()
log_message("Saved training loss plot to 'vae_training_loss.png'.")

# ---------------------------------------------------------
# 9. Save final model, scaler, and logs
# ---------------------------------------------------------
# Save the final model weights
vae_final.save_weights("best_vae_weights.h5")
log_message("Saved final VAE model weights to 'best_vae_weights.h5'.")

# Save the scaler for later replication
import pickle
with open("data_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
log_message("Saved data scaler to 'data_scaler.pkl'.")

# Save the detailed log
with open("vae_log_steps.json", "w") as f:
    json.dump(log_steps, f, indent=2)
log_message("Saved detailed log with timestamps to 'vae_log_steps.json'.")

log_message("All done!")
