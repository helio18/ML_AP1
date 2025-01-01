import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def load_and_preprocess_data():
    """
    Loads and preprocesses the training and testing data from multiple Excel files.

    - Concatenates multiple training files into a single DataFrame
    - Loads the test file into a separate DataFrame
    - Converts columns to numeric where needed
    - Converts 'Data' column to datetime
    - Cleans and converts 'Medie Consum[MW]' to numeric (if present)
    - Drops rows with NaN
    """

    train_files = [
        r"../data/2014-2016.xlsx",
        r"../data/2017-2019.xlsx",
        r"../data/2020-2022.xlsx",
        r"../data/2023-2024-no-dec.xlsx"
    ]
    test_file = r"../data/2024-dec.xlsx"

    train_dfs = []
    for fpath in train_files:
        df = pd.read_excel(fpath, sheet_name="Grafic SEN")
        train_dfs.append(df)

    train_data = pd.concat(train_dfs, ignore_index=True)

    test_data = pd.read_excel(test_file, sheet_name="Grafic SEN")

    print("TEST(passed): First rows (TRAIN):")
    print(train_data.head())
    print("\nTRAIN Info:")
    print(train_data.info())

    print("\nTEST(passed): First rows (TEST):")
    print(test_data.head())
    print("\nTEST Info:")
    print(test_data.info())

    numeric_columns = [
        'Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
        'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]',
        'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]', 'Sold[MW]'
    ]
    for col in numeric_columns:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

    train_data['Data'] = pd.to_datetime(train_data['Data'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    test_data['Data'] = pd.to_datetime(test_data['Data'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

    if 'Medie Consum[MW]' in train_data.columns:
        train_data['Medie Consum[MW]'] = (
            train_data['Medie Consum[MW]'].astype(str).str.replace(',', '.')
        )
        train_data['Medie Consum[MW]'] = pd.to_numeric(train_data['Medie Consum[MW]'], errors='coerce')

    if 'Medie Consum[MW]' in test_data.columns:
        test_data['Medie Consum[MW]'] = (
            test_data['Medie Consum[MW]'].astype(str).str.replace(',', '.')
        )
        test_data['Medie Consum[MW]'] = pd.to_numeric(test_data['Medie Consum[MW]'], errors='coerce')

    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    print("\nPost preprocessing TRAIN:")
    print(train_data.info())
    print(train_data.head())

    print("\nPost preprocessing TEST:")
    print(test_data.info())
    print(test_data.head())

    return train_data, test_data


def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler. This is optional.
    Returns the scaled X_train, X_test, and the scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def optimize_id3(X_train, y_train):
    """
    Hyperparameter tuning for DecisionTreeClassifier (ID3) via GridSearchCV.
    We use 'neg_mean_squared_error' as the scoring to align with a regression-like approach.
    """
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [3, 5, 8, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    id3 = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=id3,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("Best ID3 parameters:", grid_search.best_params_)
    return grid_search.best_estimator_


def evaluate_with_cross_validation(model, X_train, y_train, cv_folds=5):
    """
    Evaluates a model using cross-validation and prints the average RMSE.
    """
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    rmse_scores = (-scores) ** 0.5
    print(f"Cross-validated RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")


def method_1(train_data, test_data, use_scaling=False, use_hparam_tuning=False, iteration_tag=""):
    """
    Method 1:
    - Predicting 'Sold[MW]' values using only this column
    - Using ID3 (DecisionTreeClassifier) and Bayesian (GaussianNB) adapted for regression via bucketing

    Args:
        use_scaling: Whether to scale features using StandardScaler.
        use_hparam_tuning: Whether to run GridSearchCV for ID3 hyperparameter optimization.
        iteration_tag: A string to uniquely identify the iteration for saving plots.
    """

    print(f"\n[Method 1] Predicting Sold[MW] - (ID3 + Bayesian) [Tag: {iteration_tag}]")

    train_df = train_data[['Data', 'Sold[MW]']].copy()
    test_df = test_data[['Data', 'Sold[MW]']].copy()

    for df in [train_df, test_df]:
        df['hour'] = df['Data'].dt.hour
        df['day'] = df['Data'].dt.day
        df['month'] = df['Data'].dt.month
        df['dayofweek'] = df['Data'].dt.dayofweek

    X_train = train_df[['hour', 'day', 'month', 'dayofweek']]
    y_train = train_df['Sold[MW]']
    X_test = test_df[['hour', 'day', 'month', 'dayofweek']]
    y_test_actual = test_df['Sold[MW]'].values

    num_bins = 5
    train_df['Sold_Bucket'] = pd.qcut(train_df['Sold[MW]'], q=num_bins, labels=False, duplicates='drop')
    bin_edges = pd.qcut(train_df['Sold[MW]'], q=num_bins, retbins=True, duplicates='drop')[1]
    test_df['Sold_Bucket'] = pd.cut(test_df['Sold[MW]'], bins=bin_edges, labels=False, include_lowest=True)
    test_df['Sold_Bucket'] = test_df['Sold_Bucket'].fillna(0).astype(int)
    y_train_buckets = train_df['Sold_Bucket']

    if use_scaling:
        X_train, X_test, _ = scale_features(X_train, X_test)

    if use_hparam_tuning:
        print("\n[Method 1] Hyperparameter tuning ID3...")
        id3_model = optimize_id3(X_train, y_train_buckets)
    else:
        print("\n[Method 1] Using default ID3 parameters...")
        id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

    id3_model.fit(X_train, y_train_buckets)
    pred_buckets_id3 = id3_model.predict(X_test)

    bucket_means = train_df.groupby('Sold_Bucket')['Sold[MW]'].mean().to_dict()
    pred_values_id3 = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_id3])

    bayes_model = GaussianNB()
    bayes_model.fit(X_train, y_train_buckets)
    pred_buckets_bayes = bayes_model.predict(X_test)

    pred_values_bayes = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_bayes])

    print("\nCross-validation for ID3 (Method 1) [Optional]:")
    evaluate_with_cross_validation(id3_model, X_train, y_train_buckets)

    print("\nCross-validation for Bayesian (Method 1) [Optional]:")
    evaluate_with_cross_validation(bayes_model, X_train, y_train_buckets)

    mse_id3 = mean_squared_error(y_test_actual, pred_values_id3)
    rmse_id3 = mse_id3 ** 0.5
    mae_id3 = mean_absolute_error(y_test_actual, pred_values_id3)
    r2_id3 = r2_score(y_test_actual, pred_values_id3)

    mse_bayes = mean_squared_error(y_test_actual, pred_values_bayes)
    rmse_bayes = mse_bayes ** 0.5
    mae_bayes = mean_absolute_error(y_test_actual, pred_values_bayes)
    r2_bayes = r2_score(y_test_actual, pred_values_bayes)

    print(f"\n[Method 1] ID3 Performance:")
    print(f"RMSE: {rmse_id3:.2f}, MAE: {mae_id3:.2f}, R²: {r2_id3:.2f}")

    print(f"\n[Method 1] Bayesian Performance:")
    print(f"RMSE: {rmse_bayes:.2f}, MAE: {mae_bayes:.2f}, R²: {r2_bayes:.2f}")

    plt.figure(figsize=(14, 7))
    plt.plot(test_df['Data'], y_test_actual, label='Real Values', color='blue')
    plt.plot(test_df['Data'], pred_values_id3, label='ID3 Predictions', alpha=0.7, color='orange')
    plt.plot(test_df['Data'], pred_values_bayes, label='Bayesian Predictions', alpha=0.7, color='green')
    plt.xlabel('Date')
    plt.ylabel('Sold[MW]')
    plt.title(f'[Method 1] Real vs. ID3 & Bayesian Predictions - {iteration_tag}')
    plt.legend()

    plt.savefig(f"Method1_{iteration_tag}.png", bbox_inches='tight')
    plt.close()

    return rmse_id3, rmse_bayes, mae_id3, mae_bayes


def method_2(train_data, test_data, use_scaling=False, use_hparam_tuning=False, iteration_tag=""):
    """
    Method 2:
    - Predict each component (Consum[MW], Productie[MW], etc.) with ID3 & Bayesian
    - Then compute Sold[MW] = Productie[MW] - Consum[MW]
    """
    print(f"\n[Method 2] Predicting columns, computing Sold[MW] - [Tag: {iteration_tag}]")

    for df in [train_data, test_data]:
        df['hour'] = df['Data'].dt.hour
        df['day'] = df['Data'].dt.day
        df['month'] = df['Data'].dt.month
        df['dayofweek'] = df['Data'].dt.dayofweek

    target_columns = [
        'Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
        'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]',
        'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]'
    ]

    predicted_columns_id3 = {}
    predicted_columns_bayes = {}
    rmse_id3 = rmse_bayes = mae_id3 = mae_bayes = 9999

    actual_sold = test_data['Sold[MW]'].values

    for target in target_columns:
        print(f"\nTraining ID3 for column: {target}")

        X_train = train_data[['hour', 'day', 'month', 'dayofweek']]
        y_train = train_data[target]
        X_test = test_data[['hour', 'day', 'month', 'dayofweek']]

        if use_scaling:
            X_train, X_test, _ = scale_features(X_train, X_test)

        num_bins = 5
        try:
            train_data[f'{target}_Bucket'] = pd.qcut(y_train, q=num_bins, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"Error in bucketing for {target}: {e}")
            continue

        bin_edges = pd.qcut(train_data[target], q=num_bins, retbins=True, duplicates='drop')[1]
        test_data[f'{target}_Bucket'] = pd.cut(
            test_data[target], bins=bin_edges, labels=False, include_lowest=True
        )
        test_data[f'{target}_Bucket'] = test_data[f'{target}_Bucket'].fillna(0).astype(int)

        y_train_buckets = train_data[f'{target}_Bucket']

        if use_hparam_tuning:
            print(f"[Method 2] Hyperparameter tuning ID3 for {target} ...")
            id3_model = optimize_id3(X_train, y_train_buckets)
        else:
            id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)

        id3_model.fit(X_train, y_train_buckets)
        pred_buckets_id3 = id3_model.predict(X_test)

        bucket_means = train_data.groupby(f'{target}_Bucket')[target].mean().to_dict()
        pred_values_id3 = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_id3])

        predicted_columns_id3[target] = pred_values_id3

    for target in target_columns:
        print(f"\nTraining Bayesian for column: {target}")

        X_train = train_data[['hour', 'day', 'month', 'dayofweek']]
        y_train = train_data[target]
        X_test = test_data[['hour', 'day', 'month', 'dayofweek']]

        if use_scaling:
            X_train, X_test, _ = scale_features(X_train, X_test)

        num_bins = 5
        try:
            train_data[f'{target}_Bucket'] = pd.qcut(y_train, q=num_bins, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"Error in bucketing for {target}: {e}")
            continue

        bin_edges = pd.qcut(train_data[target], q=num_bins, retbins=True, duplicates='drop')[1]
        test_data[f'{target}_Bucket'] = pd.cut(
            test_data[target], bins=bin_edges, labels=False, include_lowest=True
        )
        test_data[f'{target}_Bucket'] = test_data[f'{target}_Bucket'].fillna(0).astype(int)

        y_train_buckets = train_data[f'{target}_Bucket']

        bayes_model = GaussianNB()
        bayes_model.fit(X_train, y_train_buckets)
        pred_buckets_bayes = bayes_model.predict(X_test)

        bucket_means = train_data.groupby(f'{target}_Bucket')[target].mean().to_dict()
        pred_values_bayes = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_bayes])

        predicted_columns_bayes[target] = pred_values_bayes

    if 'Productie[MW]' in predicted_columns_id3 and 'Consum[MW]' in predicted_columns_id3:
        predicted_sold_id3 = predicted_columns_id3['Productie[MW]'] - predicted_columns_id3['Consum[MW]']
        mse_id3 = mean_squared_error(actual_sold, predicted_sold_id3)
        rmse_id3 = mse_id3 ** 0.5
        mae_id3 = mean_absolute_error(actual_sold, predicted_sold_id3)
        r2_id3 = r2_score(actual_sold, predicted_sold_id3)
        print(f"\n[Method 2] ID3 => Sold[MW]: RMSE={rmse_id3:.2f}, MAE={mae_id3:.2f}, R²={r2_id3:.2f}")
    else:
        print("Error: Missing 'Productie[MW]' or 'Consum[MW]' in ID3 predictions.")

    if 'Productie[MW]' in predicted_columns_bayes and 'Consum[MW]' in predicted_columns_bayes:
        predicted_sold_bayes = predicted_columns_bayes['Productie[MW]'] - predicted_columns_bayes['Consum[MW]']
        mse_bayes = mean_squared_error(actual_sold, predicted_sold_bayes)
        rmse_bayes = mse_bayes ** 0.5
        mae_bayes = mean_absolute_error(actual_sold, predicted_sold_bayes)
        r2_bayes = r2_score(actual_sold, predicted_sold_bayes)
        print(f"[Method 2] Bayesian => Sold[MW]: RMSE={rmse_bayes:.2f}, MAE={mae_bayes:.2f}, R²={r2_bayes:.2f}")
    else:
        print("Error: Missing 'Productie[MW]' or 'Consum[MW]' in Bayesian predictions.")

    plt.figure(figsize=(14, 7))
    plt.plot(test_data['Data'], actual_sold, label='Real Sold[MW]', color='blue')
    if 'predicted_sold_id3' in locals():
        plt.plot(test_data['Data'], predicted_sold_id3, label='ID3 Sold[MW]', alpha=0.7, color='orange')
    if 'predicted_sold_bayes' in locals():
        plt.plot(test_data['Data'], predicted_sold_bayes, label='Bayesian Sold[MW]', alpha=0.7, color='green')
    plt.xlabel('Date')
    plt.ylabel('Sold[MW]')
    plt.title(f'[Method 2] Real vs. ID3 & Bayesian Predictions - {iteration_tag}')
    plt.legend()

    plt.savefig(f"Method2_{iteration_tag}.png", bbox_inches='tight')
    plt.close()

    return rmse_id3, rmse_bayes, mae_id3, mae_bayes


def method_3(train_data, test_data, use_scaling=False, use_hparam_tuning=False, iteration_tag=""):
    """
    Method 3:
    - Aggregating production into categories: "Intermittent" (solar/wind) + "Constant" (nuclear, etc.)
    - Using ID3 & Bayesian with bucketing for regression
    - Sold[MW] = Production_Intermittent + Production_Constant - Consum[MW]
    """
    print(f"\n[Method 3] Aggregating production, computing Sold[MW] - [Tag: {iteration_tag}]")

    train_data['Production_Intermittent'] = train_data['Eolian[MW]'] + train_data['Foto[MW]']
    train_data['Production_Constant'] = (
        train_data['Nuclear[MW]'] + train_data['Hidrocarburi[MW]'] +
        train_data['Ape[MW]'] + train_data['Carbune[MW]'] + train_data['Biomasa[MW]']
    )

    test_data['Production_Intermittent'] = test_data['Eolian[MW]'] + test_data['Foto[MW]']
    test_data['Production_Constant'] = (
        test_data['Nuclear[MW]'] + test_data['Hidrocarburi[MW]'] +
        test_data['Ape[MW]'] + test_data['Carbune[MW]'] + test_data['Biomasa[MW]']
    )

    for df in [train_data, test_data]:
        df['hour'] = df['Data'].dt.hour
        df['day'] = df['Data'].dt.day
        df['month'] = df['Data'].dt.month
        df['dayofweek'] = df['Data'].dt.dayofweek

    target_columns = ['Production_Intermittent', 'Production_Constant', 'Consum[MW]']

    predicted_columns_id3 = {}
    predicted_columns_bayes = {}
    rmse_id3 = rmse_bayes = mae_id3 = mae_bayes = 9999

    actual_sold = test_data['Sold[MW]'].values

    for target in target_columns:
        print(f"\n[Method 3] Training ID3 for target: {target}")

        X_train = train_data[['hour', 'day', 'month', 'dayofweek']]
        y_train = train_data[target]
        X_test = test_data[['hour', 'day', 'month', 'dayofweek']]

        if use_scaling:
            X_train, X_test, _ = scale_features(X_train, X_test)

        num_bins = 5
        try:
            train_data[f'{target}_Bucket'] = pd.qcut(y_train, q=num_bins, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"Error in bucketing for {target}: {e}")
            continue

        try:
            bin_edges = pd.qcut(train_data[target], q=num_bins, retbins=True, duplicates='drop')[1]
        except ValueError as e:
            print(f"Error extracting bin edges for {target}: {e}")
            continue

        test_data[f'{target}_Bucket'] = pd.cut(
            test_data[target], bins=bin_edges, labels=False, include_lowest=True
        )
        test_data[f'{target}_Bucket'] = test_data[f'{target}_Bucket'].fillna(0).astype(int)

        y_train_buckets = train_data[f'{target}_Bucket']

        if use_hparam_tuning:
            print(f"[Method 3] Hyperparameter tuning ID3 for {target} ...")
            id3_model = optimize_id3(X_train, y_train_buckets)
        else:
            id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)

        id3_model.fit(X_train, y_train_buckets)
        pred_buckets_id3 = id3_model.predict(X_test)

        bucket_means = train_data.groupby(f'{target}_Bucket')[target].mean().to_dict()
        pred_values_id3 = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_id3])

        predicted_columns_id3[target] = pred_values_id3

    for target in target_columns:
        print(f"\n[Method 3] Training Bayesian for target: {target}")

        X_train = train_data[['hour', 'day', 'month', 'dayofweek']]
        y_train = train_data[target]
        X_test = test_data[['hour', 'day', 'month', 'dayofweek']]

        if use_scaling:
            X_train, X_test, _ = scale_features(X_train, X_test)

        num_bins = 5
        try:
            train_data[f'{target}_Bucket'] = pd.qcut(y_train, q=num_bins, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"Error in bucketing for {target}: {e}")
            continue

        try:
            bin_edges = pd.qcut(train_data[target], q=num_bins, retbins=True, duplicates='drop')[1]
        except ValueError as e:
            print(f"Error extracting bin edges for {target}: {e}")
            continue

        test_data[f'{target}_Bucket'] = pd.cut(
            test_data[target], bins=bin_edges, labels=False, include_lowest=True
        )
        test_data[f'{target}_Bucket'] = test_data[f'{target}_Bucket'].fillna(0).astype(int)

        y_train_buckets = train_data[f'{target}_Bucket']

        bayes_model = GaussianNB()
        bayes_model.fit(X_train, y_train_buckets)
        pred_buckets_bayes = bayes_model.predict(X_test)

        bucket_means = train_data.groupby(f'{target}_Bucket')[target].mean().to_dict()
        pred_values_bayes = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_bayes])

        predicted_columns_bayes[target] = pred_values_bayes

    if 'Production_Intermittent' in predicted_columns_id3 and 'Consum[MW]' in predicted_columns_id3:
        predicted_sold_id3 = (
            predicted_columns_id3['Production_Intermittent'] +
            predicted_columns_id3['Production_Constant'] -
            predicted_columns_id3['Consum[MW]']
        )
        mse_id3 = mean_squared_error(actual_sold, predicted_sold_id3)
        rmse_id3 = mse_id3 ** 0.5
        mae_id3 = mean_absolute_error(actual_sold, predicted_sold_id3)
        r2_id3 = r2_score(actual_sold, predicted_sold_id3)
        print(f"\n[Method 3] ID3 => Sold[MW]: RMSE={rmse_id3:.2f}, MAE={mae_id3:.2f}, R²={r2_id3:.2f}")
    else:
        print("[Method 3] Error: 'Production_Intermittent' or 'Consum[MW]' missing from ID3 predictions.")

    if 'Production_Intermittent' in predicted_columns_bayes and 'Consum[MW]' in predicted_columns_bayes:
        predicted_sold_bayes = (
            predicted_columns_bayes['Production_Intermittent'] +
            predicted_columns_bayes['Production_Constant'] -
            predicted_columns_bayes['Consum[MW]']
        )
        mse_bayes = mean_squared_error(actual_sold, predicted_sold_bayes)
        rmse_bayes = mse_bayes ** 0.5
        mae_bayes = mean_absolute_error(actual_sold, predicted_sold_bayes)
        r2_bayes = r2_score(actual_sold, predicted_sold_bayes)
        print(f"[Method 3] Bayesian => Sold[MW]: RMSE={rmse_bayes:.2f}, MAE={mae_bayes:.2f}, R²={r2_bayes:.2f}")
    else:
        print("[Method 3] Error: 'Production_Intermittent' or 'Consum[MW]' missing from Bayesian predictions.")

    plt.figure(figsize=(14, 7))
    plt.plot(test_data['Data'], actual_sold, label='Real Values', color='blue')
    if 'predicted_sold_id3' in locals():
        plt.plot(test_data['Data'], predicted_sold_id3, label='ID3 Sold[MW]', alpha=0.7, color='orange')
    if 'predicted_sold_bayes' in locals():
        plt.plot(test_data['Data'], predicted_sold_bayes, label='Bayesian Sold[MW]', alpha=0.7, color='green')
    plt.xlabel('Date')
    plt.ylabel('Sold[MW]')
    plt.title(f'[Method 3] Real vs. ID3 & Bayesian Predictions - {iteration_tag}')
    plt.legend()

    plt.savefig(f"Method3_{iteration_tag}.png", bbox_inches='tight')
    plt.close()

    return rmse_id3, rmse_bayes, mae_id3, mae_bayes


if __name__ == "__main__":
    
    train_data, test_data = load_and_preprocess_data()

    def run_method(method_index, use_scaling, use_hparam_tuning, iteration_tag):
        """
        Runs the specified method (1,2,3) with the given flags
        and returns (rmse_id3, rmse_bayes, mae_id3, mae_bayes).
        """
        if method_index == 1:
            rmse_id3, rmse_bayes, mae_id3, mae_bayes = method_1(
                train_data, test_data,
                use_scaling=use_scaling,
                use_hparam_tuning=use_hparam_tuning,
                iteration_tag=iteration_tag
            )
        elif method_index == 2:
            rmse_id3, rmse_bayes, mae_id3, mae_bayes = method_2(
                train_data, test_data,
                use_scaling=use_scaling,
                use_hparam_tuning=use_hparam_tuning,
                iteration_tag=iteration_tag
            )
        else:
            rmse_id3, rmse_bayes, mae_id3, mae_bayes = method_3(
                train_data, test_data,
                use_scaling=use_scaling,
                use_hparam_tuning=use_hparam_tuning,
                iteration_tag=iteration_tag
            )
        return rmse_id3, rmse_bayes, mae_id3, mae_bayes

    combos = []
    for m in [1, 2, 3]:
        for scale in [False, True]:
            for tune in [False, True]:
                combos.append((m, scale, tune))

    with open("rmse-mae.txt", "w", encoding="utf-8") as f:
        f.write("Method, use_scaling, use_hparam_tuning, RMSE_ID3, RMSE_Bayes, MAE_ID3, MAE_Bayes\n")

        for (m_idx, scaling, tuning) in combos:
            iteration_tag = f"M{m_idx}_S{scaling}_T{tuning}"
            f.write(f"Running: Method={m_idx}, scale={scaling}, tune={tuning}\n")
            print(f"\n=== Running: Method={m_idx}, scaling={scaling}, tuning={tuning} ===")

            rmse_id3, rmse_bayes, mae_id3, mae_bayes = run_method(m_idx, scaling, tuning, iteration_tag)

            f.write(
                f"{m_idx}, {scaling}, {tuning}, "
                f"{rmse_id3:.2f}, {rmse_bayes:.2f}, {mae_id3:.2f}, {mae_bayes:.2f}\n"
            )
            f.flush()

    print("\nAll iterations completed. Results (and plots) saved. Check rmse-mae.txt for metrics.")