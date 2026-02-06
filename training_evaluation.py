from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, X_train, y_train, X_val, y_val, save_path='best_model.h5'):
    """
    Train the model with early stopping and checkpointing
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    return history
