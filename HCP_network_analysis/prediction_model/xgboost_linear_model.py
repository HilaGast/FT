import numpy as np
import xgboost

from HCP_network_analysis.prediction_model.predict_traits_by_networks import create_X_y



def zscore_X_y(X, y):
    from scipy.stats import zscore
    ynorm = zscore(y, nan_policy='omit')
    Xnorm = zscore(X, axis = 0, nan_policy='omit')

    return Xnorm, ynorm

def norm_X_y(X,y):
    ''' Normalize values between 0 to 1'''
    ynorm = (y-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y))
    Xnorm  = (X-np.nanmin(X,axis=0))/(np.nanmax(X, axis=0)-np.nanmin(X, axis=0))

    return Xnorm, ynorm



def train_gradient_boost_model_cv_logloss(networks_components, trait_vector, network_list, n_components, model_params, etas, n_estimators, weight_by):
    from sklearn.model_selection import GridSearchCV
    import xgboost as xg
    import statsmodels.api as sm
    import numpy as np
    import matplotlib.pyplot as plt

    X, y = create_X_y(networks_components, trait_vector, network_list, n_components)
    #X, y = zscore_X_y(X, y)
    X = sm.add_constant(X)

    param_grid = dict(learning_rate=etas, n_estimators=n_estimators)
    xgb_r = xg.XGBRegressor(objective='reg:squarederror', booster = 'gblinear',
                            reg_alpha=model_params['alpha'], random_state=42)

    grid_search = GridSearchCV(xgb_r, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=10, verbose=1)
    grid_result = grid_search.fit(X, y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    # plot results
    scores = np.array(means).reshape(len(etas), len(n_estimators))
    for i, value in enumerate(etas):
        plt.plot(n_estimators, scores[i], label='eta: ' + str(value))
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('MAE')
    plt.title(f"n_estimators vs learning rate \n for {weight_by} \n Alpha = {model_params['alpha']}")
    plt.show()

def gradient_boost_model(networks_components, trait_vector, network_list, n_components, model_params, model_results_dict, trait_name, weight_by, figs_folder, return_r=False):
    from sklearn.model_selection import train_test_split
    import xgboost as xg
    from statsmodels.tools.eval_measures import rmse
    import statsmodels.api as sm
    import numpy as np
    from calc_corr_statistics.spearman_r_calc import calc_corr

    X, y = create_X_y(networks_components, trait_vector, network_list, n_components)
    #X, y = zscore_X_y(X, y)
    X = sm.add_constant(X)


    # Splitting:
    train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                        test_size=model_params['test_size'], random_state=1)#full model:17 wb model: 2

    # Instantiation:

    xgb_r = xg.XGBRegressor(objective='reg:squarederror', booster = 'gblinear',
                            n_estimators=model_params['n_estimators'], learning_rate = model_params['eta'], reg_alpha=model_params['alpha'], random_state=42)
    # Fitting the model
    xgb_r.fit(train_X, train_y)

    r2 = xgb_r.score(train_X, train_y)
    adjusted_r2 = 1-((1-r2)*(len(train_y)-1)/(len(train_y)-np.sum(np.abs(xgb_r.coef_)>0)-1))
    # Predict the model
    pred_y = xgb_r.predict(test_X)

    #rmse = rmse(test_y, pred_y)
    #model_results_dict['R2'][trait_name][weight_by] = r2
    #model_results_dict['RMSE'][trait_name][weight_by] = rmse

    r,p = calc_corr(test_y, pred_y ,trait_name, weight_by, figs_folder)
    #print(f'Pearson r: {r},   p: {p}')
    if return_r:
        return xgb_r, model_results_dict, r, p
    else:
        return xgb_r, model_results_dict


def gradient_boost_permutation_test(networks_components, trait_vector, network_list, n_components, model_params, trait_name, weight_by, figs_folder, n=1000):
    from sklearn.model_selection import train_test_split
    import xgboost as xg
    import statsmodels.api as sm
    from calc_corr_statistics.spearman_r_calc import calc_corr
    from HCP_network_analysis.prediction_model.choose_random_state import random_state
    X, y = create_X_y(networks_components, trait_vector, network_list, n_components)
    X = sm.add_constant(X)
    all_r = []
    for i in range(n):
        # Splitting:
        train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                            test_size=model_params['test_size'], random_state=random_state())

        # Instantiation:

        xgb_r = xg.XGBRegressor(objective='reg:squarederror', booster='gblinear',
                                n_estimators=model_params['n_estimators'], learning_rate=model_params['eta'],
                                reg_alpha=model_params['alpha'], random_state=42)
        # Fitting the model
        xgb_r.fit(train_X, train_y)
        pred_y = xgb_r.predict(test_X)
        r = calc_corr(test_y, pred_y, trait_name, weight_by, figs_folder, show=False)[0]
        all_r.append(r)

    return all_r



