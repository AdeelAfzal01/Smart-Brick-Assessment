from .utils import CorrelationAnalysis, FeatureImportance, RecursiveFeatureElimination,\
                    UnivariateFeatureSelection, CombineSelectedFeatures

def FeatureSelection(df_, target_column, mode="train"):

    df = df_.copy()
    corr_feats = CorrelationAnalysis(df)
    randomforest_feats = FeatureImportance(df,  target_column )
    rfe_feats = RecursiveFeatureElimination(df, target_column)
    univariate_feats = UnivariateFeatureSelection(df, target_column)
    final_selected_features = CombineSelectedFeatures(
        corr_feats,
        randomforest_feats,
        rfe_feats,
        univariate_feats)
    
    if target_column in final_selected_features:
        final_selected_features.remove(target_column)
    return final_selected_features