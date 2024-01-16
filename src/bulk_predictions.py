import os
import multiprocessing as mp
import pandas as pd
from functools import partial
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

#make function that aligns train data and target data
def merge_target_and_data(data_df, target_df, target_col):
    targets = target_df[[target_col]]
    data_df = data_df.merge(targets, 
                            how='left',
                            left_index=True,
                            right_index=True)
    data_df = data_df.dropna(subset=target_col)
    data_df = data_df.drop('date', axis=1)
    data_df = data_df.fillna(0)
    return data_df

def make_test_train_holdout(df, train_pct, holdout_size, target_col):
    #make train, test, and val
    total_length = df.shape[0]
    train_end = round(train_pct*total_length)
    train = df.iloc[0:train_end,:]
    test = df.iloc[train_end:-holdout_size,:]
    val = df.iloc[-holdout_size:,:]

    #split train, test, val into x,y
    train_x = train.drop(target_col, axis=1)
    train_y = train[[target_col]]

    test_x = test.drop(target_col, axis=1)
    test_y = test[[target_col]]


    return train_x, train_y, test_x, test_y

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, recall, precision, f1

def data_end_to_end(target_file, feature_file, target_col, holdout, test_size):
    #read in data
    targets = pd.read_csv(target_file)
    data = pd.read_csv(feature_file)

    #convert date column to datetime and set as index
    data['date'] = pd.to_datetime(data.date)
    targets['date'] = pd.to_datetime(targets.date)
    data = data.set_index(data.date)
    targets = targets.set_index(targets.date)

    #merge targets and features
    combined = merge_target_and_data(data, targets, target_col)

    #make X and y arrays
    x = combined.drop(target_col, axis=1)
    y = combined[[target_col]]

    #make validation arrays
    val_x = x.iloc[-holdout:,:]
    val_y = y.iloc[-holdout:,:]

    #make train test
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, stratify=y)

    return train_x, test_x, train_y, test_y, val_x, val_y

def predict_stock(package,
                  model_grid, 
                  holdout,
                  target_columns, 
                  output_dir,
                  interim_dir):
    #iterate over all stocks and categorical targets and model, capture outputs
    results = []
    failures = []
    target_files = package[0]
    data_files = package[1]
    for tgt_col in target_columns:
        for targets, data in zip(target_files, data_files):
            try:
                stock = targets.split('_')[0]
                target_file = os.path.join(interim_dir, targets)
                feature_file = os.path.join(interim_dir, data)
                train_x, test_x, train_y, test_y, val_x, val_y = data_end_to_end(target_file,
                                                                                feature_file,
                                                                                tgt_col,
                                                                                holdout,
                                                                                test_size=.33)
                #iterate over models train and capture results
                for model in model_grid:
                    pipe = Pipeline([('scaler',StandardScaler()),
                                    ('dim_reduction', PCA(n_components=.95)),
                                    ('classifier', model)])
                    
                    pipe = pipe.fit(train_x, train_y)
                    train_predictions = pipe.predict(train_x)
                    test_predictions = pipe.predict(test_x)
                    val_predictions = pipe.predict(val_x)
                    model_name = pipe.steps[-1][1].__repr__()
                    train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(train_y, train_predictions)
                    test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(test_y, test_predictions)
                    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(val_y, val_predictions)
                    train_results = (model_name, stock, tgt_col, "train", train_accuracy, train_precision, train_recall, train_f1)
                    test_results = (model_name, stock, tgt_col, "test", test_accuracy, test_precision, test_recall, test_f1)
                    val_results = (model_name, stock, tgt_col, "val", val_accuracy, val_precision, val_recall, val_f1)
                    results.append(train_results)
                    results.append(test_results)
                    results.append(val_results)
                    print('\n'*3)
                    print('*'*80)
                    print(f"{model_name}:{stock}:{tgt_col}")
                    print("Train Classification Report\n")
                    print(classification_report(train_y, train_predictions))
                    print('\n'*2)
                    print("Test Classification Report")
                    print(classification_report(test_y, test_predictions))
                    print('\n'*2)
                    print("Val Classification Report")
                    print(classification_report(val_y, val_predictions))
                    print('*'*80)
                    print('\n'*3)
            except:
                failures.append((stock,tgt_col))
    outputs = pd.DataFrame(data=results, columns=['model','stock','tgt','split','accuracy','precision','recall','f1'])
    outputs.to_csv(os.path.join(output_dir,f'{os.getpid()}-model_target_search.csv', mode='a'))
    return failures

if __name__ == "__main__":
    #set global params
    #get all feature and target files
    INTERIM_DIR = r"C:\Users\aburtnerabt\Documents\Continuing Education\Algo Trading\algo_trading\data\interim"
    OUTPUT_DIR = r"C:\Users\aburtnerabt\Documents\Continuing Education\Algo Trading\algo_trading\data\processed"
    feature_files = [x for x in os.listdir(INTERIM_DIR) if "indicators" in x]
    target_files = [x for x in os.listdir(INTERIM_DIR) if "targets" in x]
    HOLDOUT = 100
    CHUNK_SIZE = 5
    WORKERS = 20
    
    #read in a target file to get target columns
    target_file = os.path.join(INTERIM_DIR,target_files[0])
    targets = pd.read_csv(target_file)

    #get all binary target columns
    target_columns = [x for x in targets.columns if "drawdown" in x]

    #make models list
    models = [
    RandomForestClassifier(),
    LogisticRegression(),
    MLPClassifier(),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    SVC(),
    LinearSVC()
    ]

    #make lists of file names
    target_chunks = [target_files[x:x+CHUNK_SIZE] for x in range(0,len(target_files), CHUNK_SIZE)]
    feature_chunks = [feature_files[x:x+CHUNK_SIZE] for x in range(0,len(feature_files), CHUNK_SIZE)]
    packages = [(x,y) for x,y in zip(target_chunks, feature_chunks)]

    #make sure files are aligned
    for target_file, feature_file in zip(target_files,feature_files):
        target_stock = target_file.split("_")[0]
        feature_stock = feature_file.split("_")[0]
        if target_stock != feature_stock:
            print(f"{target_stock} and {feature_stock} mis-aligned, check the files")
            exit()

    #make partial function
    run_models = partial(predict_stock, 
                         model_grid=models, 
                         holdout=HOLDOUT,
                         target_columns=target_columns,
                         output_dir=OUTPUT_DIR,
                         interim_dir=INTERIM_DIR)

    #execute multiprocessing 
    with mp.Pool(processes=WORKERS) as pool:
        all_failures = list(tqdm(pool.imap(run_models, packages, chunksize=1), total=len(packages)))

    if len(all_failures) > 0:
        with open("first_fun_failures.pkl", 'w') as f:
            pickle.dump(all_failures)