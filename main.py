import argparse
import os
import numpy as np
import pandas as pd

import helpers as hp

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


def build_model(time_steps, num_features, num_classes):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(time_steps, num_features)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def save_training_plots(history, out_dir):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(out_dir, 'training_loss.png'))
    plt.close()

    # accuracy (may not exist if metrics changed)
    if 'accuracy' in history.history:
        plt.figure()
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.legend()
        plt.title('Accuracy')
        plt.savefig(os.path.join(out_dir, 'training_accuracy.png'))
        plt.close()


def evaluate_and_save(model, X_test, y_test, y_test_cat, label_map, out_dir):
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(out_dir, exist_ok=True)

    # predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test

    # classification report
    target_names = [str(k) for k in sorted(label_map.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()


def main(args):
    print('Loading data...')
    cdata, clabels = hp.get_data()

    print('Total chunks:', len(cdata))

    # Preprocess
    time_steps = cdata.shape[1]
    X, y = hp.split_data(cdata, clabels) #TODO this has to preprocess cnlabels too
    print('After preprocessing, X shape:', X.shape, 'y shape:', y.shape)

    # Filter classes: keep only labels that appear at least k times
    min_count = 20
    counts = Counter(y)
    keep_labels = {lab for lab,cnt in counts.items() if cnt >= min_count}
    mask = np.array([lab in keep_labels for lab in y])
    X = X[mask]
    y = y[mask]
    print('After filtering rare classes, samples:', X.shape[0], 'classes kept:', len(keep_labels))

    # Remap labels to contiguous range
    label_map = {old:i for i,old in enumerate(sorted(list(keep_labels)))}
    y = np.array([label_map[lab] for lab in y])
    num_classes = len(label_map)
    print('Num classes:', num_classes)

    # Train/val/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval)

    print('Train/Val/Test shapes:', X_train.shape, X_val.shape, X_test.shape)

    # One-hot encode labels using numpy (avoids importing TF for dry-run)
    y_train_cat = np.eye(num_classes)[y_train]
    y_val_cat = np.eye(num_classes)[y_val]
    y_test_cat = np.eye(num_classes)[y_test]

    if args.dry_run:
        print('Dry run - exiting after data preprocessing')
        return

    # Build model
    model = build_model(time_steps, 1, num_classes)
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ckpt = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    # Ensure output dir exists before creating TB log directories
    os.makedirs(args.output_dir, exist_ok=True)
    # TensorBoard: create both a timestamped archive dir and a stable 'current' dir
    import datetime
    log_dir_ts = os.path.join(args.output_dir, 'tb_logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    log_dir_current = os.path.join(args.output_dir, 'tb_logs', 'current')
    os.makedirs(log_dir_ts, exist_ok=True)
    os.makedirs(log_dir_current, exist_ok=True)
    # Two callbacks: one for archival (timestamped) and one stable path for VS Code extension to watch
    tb_cb_ts = TensorBoard(log_dir=log_dir_ts, histogram_freq=1, update_freq='epoch', profile_batch=0)
    tb_cb_current = TensorBoard(log_dir=log_dir_current, histogram_freq=1, update_freq='epoch', profile_batch=0)

    # Train
    epochs = 5 if args.quick else args.epochs
    history = model.fit(X_train, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        epochs=epochs,
                        batch_size=args.batch_size,
                        callbacks=[early_stop, ckpt, tb_cb_ts, tb_cb_current])

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f'Test loss: {loss:.4f}, Test accuracy: {acc:.4f}')

    # Save metrics and plots
    out_dir = args.output_dir
    save_training_plots(history, out_dir)
    evaluate_and_save(model, X_test, y_test, y_test_cat, label_map, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Only preprocess and show shapes')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, dest='batch_size', help='Training batch size')
    parser.add_argument('--output-dir', type=str, default='results', dest='output_dir', help='Directory to save metrics and plots')
    parser.add_argument('--quick', action='store_true', help='Quick train (small number of epochs)')
    args = parser.parse_args()
    main(args)

