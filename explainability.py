import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import os


class ExplainableAI:
    def __init__(self, model, X_background: np.ndarray = None):
        self.model = model
        self.explainer = None
        
        if X_background is not None:
            self.initialize_explainer(X_background)
    
    def initialize_explainer(self, X_background: np.ndarray):
        if len(X_background) > 100:
            X_background = shap.sample(X_background, 100)
        
        self.explainer = shap.TreeExplainer(self.model.model)
    
    def explain_prediction(self, X: np.ndarray, feature_names: List[str] = None) -> Dict:
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        shap_values = self.explainer.shap_values(X)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            shap_values = shap_values.reshape(1, -1)
        
        explanations = []
        for i in range(len(X)):
            if feature_names:
                feature_contributions = {
                    feature_names[j]: float(shap_values[i, j])
                    for j in range(len(feature_names))
                }
            else:
                feature_contributions = {
                    f"feature_{j}": float(shap_values[i, j])
                    for j in range(shap_values.shape[1])
                }
            
            sorted_contributions = dict(
                sorted(feature_contributions.items(), 
                       key=lambda x: abs(x[1]), 
                       reverse=True)[:10]
            )
            
            explanations.append({
                "top_features": sorted_contributions,
                "base_value": float(self.explainer.expected_value),
                "prediction_value": float(self.explainer.expected_value + shap_values[i].sum())
            })
        
        return explanations[0] if len(explanations) == 1 else explanations
    
    def get_global_importance(self, X: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        shap_values = self.explainer.shap_values(X)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)
        
        return importance_df
    
    def plot_waterfall(self, X: np.ndarray, feature_names: List[str] = None, 
                       save_path: str = None, index: int = 0):
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        shap_values = self.explainer.shap_values(X)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            shap_values = shap_values.reshape(1, -1)
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[index],
                base_values=self.explainer.expected_value,
                data=X[index],
                feature_names=feature_names
            )
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
    
    def plot_summary(self, X: np.ndarray, feature_names: List[str] = None, 
                     save_path: str = None):
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        shap_values = self.explainer.shap_values(X)
        
        shap.summary_plot(
            shap_values, X, 
            feature_names=feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()


def explain_customer_churn(model, customer_data: np.ndarray, 
                          feature_names: List[str], 
                          X_background: np.ndarray = None) -> Dict:
    explainer = ExplainableAI(model, X_background)
    explanation = explainer.explain_prediction(customer_data, feature_names)
    return explanation
