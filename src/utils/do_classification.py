from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, auc

def feature_classification_pipeline(
        svc_kernel,
        svc_c_value,
        svc_gamma_value,
        path_to_train_features,
        path_to_train_labels,
        path_to_test_features,
        path_to_test_labels,
        path_to_save_classifier: str = None,
):
    """
    JUST NEED TO PROVIDE PATHS 
    TO FOLDER(S) CONTAINING TRAIN AND TEST FEATURES AND LABELS

    NEED TO HANDLE TRAIN AND TEST LOADER
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_features = np.load(path_to_train_features) # "train_features_sn.npy"
    train_labels = np.load(path_to_train_labels) # "train_labels_sn.npy"

    test_features = np.load(path_to_test_features) # "test_features_sn.npy"
    test_labels = np.load(path_to_test_labels) # "test_labels_sn.npy"

    # Train an SVM on the extracted features
    print("Training SVM...")
    svm_classifier = make_pipeline(
        StandardScaler(), 
        SVC(
            kernel=svc_kernel,  # 'rbf'
            probability=True, 
            C = svc_c_value, # 100
            gamma = svc_gamma_value, # 0.0001
            random_state=42
            )
            )
    svm_classifier.fit(train_features, train_labels)

    # Save the trained SVM model
    joblib.dump(svm_classifier, path_to_save_classifier) # "svm_classifier_bp_new.pkl"

    # Evaluate on the test set
    print("Evaluating on test set...")
    test_predictions = svm_classifier.predict(test_features)

    print("Test Set Performance:")
    print(classification_report(test_labels, test_predictions))
    print("Test Accuracy:", accuracy_score(test_labels, test_predictions))

    
    '''# Create confusion matrix for test output
    cm = confusion_matrix(test_labels, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Schizophrenic', 'Schizophrenic'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix- Test Set")
    plt.show()'''

    # Get predictions on the training set
    test_true = []
    test_pred = []
    test_scores= []

    # Generate predictions
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.numpy()
        features = feature_extractor(inputs).detach().cpu().numpy()
        preds = svm_classifier.predict(features)
        scores = svm_classifier.decision_function(features)  # Get scores for the current batch
        test_true.extend(labels)
        test_pred.extend(preds)
        test_scores.extend(scores)
        #

    fpr_test, tpr_test, thresholds_test = roc_curve(test_true, test_scores)

    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot the ROC curve
    '''plt.figure()
    plt.plot(fpr_test, tpr_test, color='orange', lw=2, label=f'Test set (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve- Gaussian Noise')
    plt.legend(loc="lower right")
    plt.show()'''

    # Get predictions on the training set
    train_true = []
    train_pred = []
    train_scores= []

    # Generate predictions
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.numpy()
        features = feature_extractor(inputs).detach().cpu().numpy()
        preds = svm_classifier.predict(features)
        scores = svm_classifier.decision_function(features)  # Get scores for the current batch
        train_true.extend(labels)
        train_pred.extend(preds)
        train_scores.extend(scores)
        #

    fpr_test, tpr_test, thresholds_test = roc_curve(test_true, test_scores)
    fpr_train, tpr_train, thresholds_train = roc_curve(train_true, train_scores)
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot the ROC curve
    '''plt.figure()
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train set (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='orange', lw=2, label=f'Test set (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve- Gaussian Noise')
    plt.legend(loc="lower right")
    plt.show()'''     

    pass
