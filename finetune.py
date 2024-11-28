import os
import argparse
import random

import torch
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel, ClassificationArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    args.model_path = str(args.model_name_or_path).split('/')[-1]
    args.output_dir = os.path.join('outputs', args.model_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    return args

def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def evaluation_score(result):
    tp = result['tp']
    fp = result['fp']
    tn = result['tn']
    fn = result['fn']
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    specificity = tn / (tn + fp)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
        "specificity": round(specificity, 3)
    }


def load_dataset(path):
    train_df = pd.read_csv('{}/train_new.csv'.format(path))
    test_df = pd.read_csv('{}/test_new.csv'.format(path))
    return train_df, test_df

def main():
    # initialization
    args = init_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read data
    train_df, test_df = load_dataset('dataset')
    train_df.columns = ["text", "labels"]
    test_df.columns = ["text", "labels"]

    # Define model arguments
    model_args = {
        'num_train_epochs': 5,
        'learning_rate': 2e-5,
        'overwrite_output_dir': True,
        'train_batch_size': 16,
        'eval_batch_size': 16,
        'output_dir': args.output_dir,
        'save_steps': -1,
        'save_model_every_epoch': False,
        'use_multiprocessing': False,
        'use_multiprocessing_for_evaluation': False,
        'multiprocessing_chunksize': 1,
        'dataloader_num_workers': 1
    }

    # Create a ClassificationModel
    model = ClassificationModel(
        args.model_type,        # Model type (can be roberta, xlnet, etc.)
        args.model_name_or_path,    # Model name
        args=model_args,
        use_cuda=device
    )

    # Train the model
    model.train_model(train_df, show_running_loss=True)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_df)
    eval_score = evaluation_score(result)
    new_eval = {"Model": args.model_path, 'TP': eval_score['tp'], 'FP': eval_score['fp'], 'FN': eval_score['fn'], 'TN': eval_score['tn'], 'Accuracy': eval_score['accuracy'],
                'Precision': eval_score['precision'], 'Recall': eval_score['recall'], 'F1': eval_score['f1_score']}

    # Update the overall models' performance evaluation score
    if os.path.exists(os.path.join('outputs', 'overall_evaluation.xlsx')):
        eval_df = pd.read_excel(os.path.join('outputs', 'overall_evaluation.xlsx'))
        eval_df = pd.concat(
            [eval_df, pd.DataFrame([new_eval])], ignore_index=True)
        eval_df.to_excel(os.path.join('outputs', 'overall_evaluation.xlsx'), index=False)
    else:
        eval_df = pd.DataFrame(new_eval, index=[0])
        eval_df.to_excel(os.path.join('outputs', 'overall_evaluation.xlsx'), index=False)

    # Save the predicted labels to perform error analysis
    predictions, raw_outputs = model.predict(test_df['text'].to_list())
    test_df['preds'] = predictions
    test_df.to_excel(os.path.join(args.output_dir, f"prediction_{args.model_path}.xlsx"))


if __name__ == "__main__":
    main()