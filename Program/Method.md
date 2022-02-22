def getCryptos(cryptos=['BTC'], columns=['volumefrom', 'open', 'low', 'high', 'close']):
    cryptoDic = {}
    index = []
    for cryptoName in cryptos:
        df = prepareData(cryptoName, 'EUR')
        if (df.empty == False) and (df[df.index.duplicated()].empty == True):           
            cryptoDic = df[columns]
            index = df.index
        
    df2 = pd.DataFrame(cryptoDic, index=index)
    df2['volume'] = df2['volumefrom']
    df2 = df2.drop(['volumefrom'], axis=1)
    return df2
    

#bereitet die geladenen Daten zur Analyse
def prepareData(cryptoName='BTC', FIAT='EUR'):
    btcHist = cryptocompare.get_historical_price_day(cryptoName, curr=FIAT, limit=2000)
    df_btc = pd.DataFrame(btcHist) 

    df_btc['time'] = [pd.to_datetime(dt, unit='s').date() for dt in df_btc['time']]
    df_btc.set_index('time', inplace=True)
    print('columns', df_btc.columns)
    return df_btc


    
#The values of the dataframe are standardized for graphical display.
def standard_scale_and_plot_line(df):
    stdc = StandardScaler()

    X_train_std = stdc.fit_transform(df)

    columns = df.columns
    index = df.index

    X_train_std_df = pd.DataFrame(X_train_std) 
    X_train_std_df.columns = columns
    X_train_std_df.index = index

    X_train_std_df.plot.line()
    
    
def select_features_from_model(model, tryingFeatArrayNumber, columns, X, y):
    featies = []
    model.fit(X, y)
    for idmf,mf in enumerate(tryingFeatArrayNumber):
        sfm = SelectFromModel(model, threshold=None, max_features=mf, prefit=True)#
        x_selected = sfm.transform(X)
        feature_idx = sfm.get_support()
        feature_names = columns[feature_idx]
        featies.append(feature_names)
    #
    print(f'features selected from a model', featies)
    return featies[0].tolist()


# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_regression, k=5)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    #
    feat_names = X_cols[fs.get_support()]
    feat_indexes = fs.get_support()
    print(f'SELECTED NAMES: {feat_names}')
    #
    return X_train_fs, X_test_fs, fs, feat_names.tolist(), feat_indexes


def get_importance_regression_features_by_names(X,y,X_columns):
    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print('Ranking of feature importance:')
    
    feats = []
    for f in range(X.shape[1]):#column length from shape[1]
        if f > 0: #importance value
            print("%2d) %-*s %f" % (f, 30, X_columns[indices[f]], importances[indices[f]]))
            feats.append(X_columns[indices[f]])
    return feats
    
    
    #Generiere mir Time series daten in der richtigen Form
def create_dataset(X, y, time_steps=1):#current time prediction = 1
    Xs, ys = [], [] # history sets
    for i in range(len(X) - time_steps): #history row size
        v = X.iloc[i:(i + time_steps)].values # get the history value each row
        Xs.append(v) # append it into the list
        ys.append(y.iloc[i + time_steps]) # do the same for the target
    return np.array(Xs), np.array(ys) # return the history data
