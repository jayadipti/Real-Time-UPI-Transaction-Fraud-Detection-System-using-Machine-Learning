import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class UPIFraudDetector:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the UPI transactions data"""
        print("Loading and preprocessing data...")
        
        df = pd.read_csv(file_path)
        
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week_num'] = df['timestamp'].dt.dayofweek
        
        
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount (INR)'])
        
        # Transaction features
        df['same_sender_receiver'] = (df['sender_bank'] == df['receiver_bank']).astype(int)
        
        # Select features for modeling
        feature_columns = [
            'transaction type', 'merchant_category', 'amount (INR)', 'amount_log',
            'transaction_status', 'sender_age_group', 'receiver_age_group', 
            'sender_state', 'sender_bank', 'receiver_bank', 'device_type', 
            'network_type', 'hour_of_day', 'day_of_week', 'is_weekend',
            'month', 'day', 'hour', 'day_of_week_num', 'is_morning', 'is_afternoon',
            'is_evening', 'is_night', 'same_sender_receiver'
        ]
        
        X = df[feature_columns].copy()
        y = df['fraud_flag']
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        print(f"Data shape: {X.shape}")
        print(f"Fraud rate: {y.mean():.4f} ({y.sum()} fraud cases out of {len(y)} total)")
        
        return X, y
    
    def handle_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        print("Handling class imbalance using SMOTE...")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X, y)
            
        print(f"Balanced data shape: {X_balanced.shape}")
        print(f"Balanced fraud rate: {y_balanced.mean():.4f}")
        
        return X_balanced, y_balanced
    
    def train_models(self, X, y):
        """Train Random Forest and Logistic Regression models"""
        print("Training Random Forest and Logistic Regression models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=1000
            )
        }
        
        # Train and evaluate models
        results = {}
        best_score = 0
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # For fraud detection, prioritize precision and F1-score
            score = (precision * 0.4 + f1 * 0.4 + auc * 0.2)
            
            if score > best_score:
                best_score = score
                self.best_model = model
                self.best_model_name = name
                
            print(f"  {name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Store feature importance for the best model
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return results, X_test, y_test
    
    def evaluate_model(self, results, X_test, y_test):
        """Comprehensive model evaluation"""
        print(f"\n=== BEST MODEL: {self.best_model_name} ===")
        
        best_result = results[self.best_model_name]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, best_result['y_pred']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, best_result['y_pred'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, best_result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'{self.best_model_name} (AUC = {best_result["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curve.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, best_result['y_pred_proba'])
        plt.plot(recall, precision, label=f'{self.best_model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('precision_recall_curve.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Feature importance plot (only for Random Forest)
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importance - {self.best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
            plt.close()
    
    def save_model_results(self, results):
        """Save model results to a file"""
        with open('model_results.txt', 'w') as f:
            f.write("UPI FRAUD DETECTION MODEL RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for name, result in results.items():
                f.write(f"{name}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall: {result['recall']:.4f}\n")
                f.write(f"  F1-Score: {result['f1']:.4f}\n")
                f.write(f"  AUC: {result['auc']:.4f}\n")
                f.write("\n")
            
            f.write(f"\nBEST MODEL: {self.best_model_name}\n")
            if self.feature_importance is not None:
                f.write("\nTop 10 Most Important Features:\n")
                for idx, row in self.feature_importance.head(10).iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

def main():
    """Main function to run the complete fraud detection pipeline"""
    print("=== UPI FRAUD DETECTION MODEL ===")
    print("=" * 40)
    
    # Initialize detector
    detector = UPIFraudDetector()
    
    # Load and preprocess data
    X, y = detector.load_and_preprocess_data('upi_transactions_2024.csv')
    
    # Handle class imbalance
    X_balanced, y_balanced = detector.handle_imbalance(X, y)
    
    # Train models
    results, X_test, y_test = detector.train_models(X_balanced, y_balanced)
    
    # Evaluate best model
    detector.evaluate_model(results, X_test, y_test)
    
    # Save results
    detector.save_model_results(results)
    
    print("\n=== MODEL TRAINING COMPLETED ===")
    print("Generated files:")
    print("- confusion_matrix.png")
    print("- roc_curve.png") 
    print("- precision_recall_curve.png")
    print("- feature_importance.png")
    print("- model_results.txt")
    print(f"\nBest model: {detector.best_model_name}")

if __name__ == "__main__":
    main()
