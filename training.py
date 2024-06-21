"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""
def train_save_model(cleaned_df, outcome_df):
 ## This script contains a bare minimum working example
    
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    
    # Logistic regression model
    model = LogisticRegression()

    # Fit the model
    model.fit(model_df[["cf20m024","cf20m128","cf20m180","ci20m006","ci20m261"]], model_df['new_child'])

    # Save the model
    joblib.dump(model, "model.joblib")

predict_outcomes(mydata)
