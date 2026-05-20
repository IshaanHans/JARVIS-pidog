import os, sys, pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
sys.path.append(os.path.expanduser('~/pidog'))
from model.signs import INDEX_TO_SIGN

DATA_PATH  = os.path.expanduser('~/pidog/data_collection/training_data.npz')
MODEL_PATH = os.path.expanduser('~/pidog/model/model.pkl')

def main():
    if not os.path.exists(DATA_PATH):
        print(f'ERROR: No training data at {DATA_PATH}')
        print('Run data_collection/collect.py first.')
        return
    data = np.load(DATA_PATH)
    X, y = data['X'], data['y']
    print(f'Loaded {len(X)} samples, {X.shape[1]} features, {len(set(y))} classes')
    for idx in sorted(set(y)):
        print(f'  {INDEX_TO_SIGN.get(idx)}: {np.sum(y == idx)} samples')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print('\nTraining RandomForest...')
    clf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                  random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f'Test accuracy: {accuracy:.1%}')
    labels = [INDEX_TO_SIGN.get(i) for i in sorted(set(y))]
    print(classification_report(y_test, y_pred, target_names=labels))
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f'Model saved to {MODEL_PATH}')

if __name__ == '__main__':
    main()
