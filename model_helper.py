import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import arange
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans




def standardize_data(df):
    """
    Given

    df (Dataframe) :      input data table

    Returns:
    Data table standadized so that each column has a mean of 0 and a standard deviation of 1
    """
    std_scaler = StandardScaler()

    df_scaled = std_scaler.fit_transform(df.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns.values.tolist())

    return df_scaled


def split_train_test_data(input_data, test_ratio, standardize = False):
    """
    Given input data table, splits it into training and test
    data sets, with the size of the test data set being set
    by the input 'test_ratio', if the boolean input of
    standardize is set to True, also standardizes the X
    data before returning it as test and train sets

    input_data (Dataframe) :        data table of patient record data
    test_ratio (float):             decimal input that specifies what ratio of the data should be allocated for the test data set
    standardize (Boolean):          boolean input that specifies whether X data should be standardized using standard scaler before being returned

    Returns:
    Histogram plot and statistical breakdown of patient ages at first or second stroke-incident for a specific stroke type
    """
    y = input_data.outcome
    input_data.drop(['outcome'], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=test_ratio, random_state=42)

    if standardize:
        X_train = standardize_data(X_train)
        X_test = standardize_data(X_test)

    return X_train, X_test, y_train, y_test


def confusion_matrix_plot(y, y_pred, data_type):
    """
    Given

    d (Dataframe) :      d
    n (String):          s

    Returns:

    """
    cf_matrix = confusion_matrix(y, y_pred)
    ax = plt.axes()
    sns.heatmap(cf_matrix, annot=True, cmap='Oranges')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    ax.set_title(data_type + ' Confusion Matrix', fontsize=16)
    plt.show()


def model_log_reg(X_train, X_test, y_train, y_test, acc_print, matrix):
    """
    Given

    X_train (Dataframe) :        data table input training data for the model
    X_test (Dataframe):          data table input testing data for the model

    Returns:

    """
    reg = LogisticRegression(class_weight={0: 1, 1: 3}).fit(X_train, y_train)
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)
    if acc_print:
        print("Training logistic regression accuracy is: ", reg.score(X_train, y_train))
        print("Training mean absolute error is: ", mean_absolute_error(y_train, y_pred_train))
        print("Training mean squared error is: ", mean_squared_error(y_train, y_pred_train))
        print()
        print("Test logistic regression accuracy is: ", reg.score(X_test, y_test))
        print("Test mean absolute error is: ", mean_absolute_error(y_test, y_pred_test))
        print("Test mean squared error is: ", mean_squared_error(y_test, y_pred_test))

    if matrix:
        confusion_matrix_plot(y_train, y_pred_train, "Training Data")
        confusion_matrix_plot(y_test, y_pred_test, 'Test Data')
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    return train_acc, test_acc


def model_pca(X, num_components):
    """
    Given

    d (Dataframe) :      d
    n (String):          s

    Returns:

    """
    pca = PCA(n_components = num_components)
    X_pca = pca.fit_transform(X)
    return X_pca


def optimize_pca(X_train, X_test, y_train, y_test):
    best_num_c = 0
    best_test_acc = 0
    best_train_acc = 0
    
    for num_components in range(8,49,2):
        X_pca_train = model_pca(X_train, num_components)
        X_pca_test = model_pca(X_test, num_components)
        
        curr_train, curr_test = model_log_reg(X_pca_train, X_pca_test, y_train, y_test, False, False)

        if curr_test > best_test_acc:
            best_test_acc = curr_test
            best_train_acc = curr_train
            best_num_c = num_components
        
        
    print("Best Number of Principal Components for PCA :", best_num_c)
    print("Best Train Accuracy : ", best_train_acc)
    print("Best Test Accuracy : ", best_test_acc)
    
    X_pca_train = model_pca(X_train, best_num_c)
    X_pca_test = model_pca(X_test, best_num_c)

    curr_train, curr_test = model_log_reg(X_pca_train, X_pca_test, y_train, y_test, False, True)    


def model_random_forest(X_train, X_test, y_train, y_test, num_estimators, matrix):
    """
    Given

    X_train (Dataframe) :        data table input training data for the model
    X_test (Dataframe):          data table input testing data for the model

    Returns:

    """
    forest = RandomForestClassifier(n_estimators=num_estimators, oob_score=True)

    forest.fit(X_train, y_train.ravel())

    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    if matrix:
            confusion_matrix_plot(y_train, y_train_pred, "Training Data")
            confusion_matrix_plot(y_test, y_test_pred, 'Test Data')
            
    return num_estimators, train_acc, test_acc


def optimize_random_forest(X_train, X_test, y_train, y_test):
    best_est = 0
    best_test_acc = 0
    best_train_acc = 0
    
    
    for num_estimators in range(20,49,2):
        curr_est, curr_train, curr_test = model_random_forest(X_train, X_test, y_train, y_test, num_estimators, False) 
        if curr_test > best_test_acc:
            best_test_acc = curr_test
            best_train_acc = curr_train
            best_est = curr_est

    print("Best Number of Estimators :", best_est)
    print("Best Train Accuracy : ", best_train_acc)
    print("Best Test Accuracy : ", best_test_acc)
    
    model_random_forest(X_train, X_test, y_train, y_test, best_est, True) 
    return 

   

def model_gradient_booster(X_train, X_test, y_train, y_test, lr_rate, matrix):
    
    gradient_booster = GradientBoostingClassifier(learning_rate=lr_rate)
    gradient_booster.fit(X_train,y_train)
    
    y_train_pred = gradient_booster.predict(X_train)
    y_test_pred = gradient_booster.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    if matrix:
            confusion_matrix_plot(y_train, y_train_pred, "Training Data")
            confusion_matrix_plot(y_test, y_test_pred, 'Test Data')
            
    return lr_rate, train_acc, test_acc


def optimize_gradient_booster(X_train, X_test, y_train, y_test):
    best_lr = 0
    best_test_acc = 0
    best_train_acc = 0
    
    
    for lr in arange(0.1,0.21,0.01):
        curr_lr, curr_train, curr_test = model_gradient_booster(X_train, X_test, y_train, y_test, lr, False) 
        if curr_test > best_test_acc:
            best_test_acc = curr_test
            best_train_acc = curr_train
            best_lr = lr
            
    print("Best Learning Rate :", best_lr)
    print("Best Train Accuracy : ", best_train_acc)
    print("Best Test Accuracy : ", best_test_acc)
    
    model_gradient_booster(X_train, X_test, y_train, y_test, best_lr, True) 

    
def model_cluster(df, num_clusters):
    """
    Given

    X_train (Dataframe) :        data table input training data for the model
    X_test (Dataframe):          data table input testing data for the model

    Returns:

    """
    # Convert DataFrame to matrix
    mat = df.values
    # Using sklearn
    km = KMeans(n_clusters = num_clusters)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    df['cluster'] = labels
    
    return df

def optimize_cluster(df):
    """
    Given

    X_train (Dataframe) :        data table input training data for the model
    X_test (Dataframe):          data table input testing data for the model

    Returns:

    """
    X_train, X_test, y_train, y_test = split_train_test_data(df, 0.3, True)
    optimize_random_forest(X_train, X_test, y_train, y_test)
    optimize_gradient_booster(X_train, X_test, y_train, y_test)
    
    return 
    
     
def model_svm(X_train, X_test, y_train, y_test):
    """
    Given

    X_train (Dataframe) :        data table input training data for the model
    X_test (Dataframe):          data table input testing data for the model

    Returns:

    """
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print(" Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print(" Test Accuracy:", accuracy_score(y_test, y_test_pred))