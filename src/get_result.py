#encoding=utf-8
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import read_dataset
from sklearn.metrics import roc_curve, auc

def draw_roc_curve(fpr, tpr, auc, path):
    # draw the roc curve
    plt.figure()
    lw = 2  # line width
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('The ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The parameters of the program.")
    
    parser.add_argument('--input_path', type=str, help='The input data of the program.')
    parser.add_argument('--start', type=int, help='The start for reading.')
    parser.add_argument('--end', type=int, help='The end for reading.')
    parser.add_argument('--task', type=str, help='The task target, for example, \'multi\' or \'roc\'.')
    parser.add_argument('--output_path', type=str, help='The output data for the program.(used only in roc test)')
    
    args = parser.parse_args()
    if args.task == 'multi':
        data = read_dataset(args.input_path, start=args.start, end=args.end)
        cnt = 0
        for item in data:
            if item['target_key'] == item['detection_key']:
                cnt += 1
        res = cnt / (args.end - args.start) * 100
        print(f'The result for this multi test is {res}')
    elif args.task == 'roc':
        data = read_dataset(args.input_path, start=args.start, end=args.end)
        y_test = list()
        y_score = list()
        for item in data:
            y_test.append(item['label'])
            y_score.append(item['score'])
        y_test = np.array(y_test)
        y_score = np.array(y_score)
        y_score = -np.log(y_score)
        y_score_min = np.min(y_score)
        y_score_max = np.max(y_score)
        y_score = (y_score - y_score_min) / (y_score_max - y_score_min)
        
        # calculate the roc and auc
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        # calculate the tpr while fpr close to 0.01 and 0.05
        # Find points where FPR is closest to 0.01 and 0.05
        index = np.argmin(np.abs(fpr - 0.01))
        
        # Print TPR values at these points
        print(f'TPR at FPR ≈ 0.01: {tpr[index]:.3f}')
        
        # Calculate metrics at threshold corresponding to FPR ≈ 0.01
        threshold = thresholds[index]
        predictions = (y_score >= threshold).astype(int)
        
        # Calculate confusion matrix elements
        TP = np.sum((predictions == 1) & (y_test == 1))
        TN = np.sum((predictions == 0) & (y_test == 0))
        FP = np.sum((predictions == 1) & (y_test == 0))
        FN = np.sum((predictions == 0) & (y_test == 1))
        
        # Calculate metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        print(f'At threshold {threshold:.3f}:')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'F1-score: {f1:.3f}')
        print(f'Accuracy: {accuracy:.3f}')
        auc = auc(fpr, tpr)
        # draw the picture
        draw_roc_curve(fpr, tpr, auc, args.output_path)
        