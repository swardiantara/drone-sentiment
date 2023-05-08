# Fine-tuning code
import os
import sys
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
import logging
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def load_dataset(path):
    train_df = pd.read_csv('{}/train.csv'.format(path))
    test_df = pd.read_csv('{}/test.csv'.format(path))
    return train_df, test_df


def get_model_name(model_type):
    if model_type == "bert":
        model_name = "bert-base-cased"
    elif model_type == "roberta":
        model_name = "roberta-base"
    elif model_type == "distilbert":
        model_name = "distilbert-base-cased"
    elif model_type == "distilroberta":
        model_type = "roberta"
        model_name = "distilroberta-base"
    elif model_type == "electra-base":
        model_type = "electra"
        model_name = "google/electra-base-discriminator"
    elif model_type == "xlnet":
        model_name = "xlnet-base-cased"

    return model_type, model_name


def get_model_args(model_type):
    args = ClassificationArgs()
    model_type, model_name = get_model_name(model_type)
    args.reprocess_input_data = True
    args.overwrite_output_dir = True,
    args.use_cached_eval_features = True,
    args.output_dir = f"outputs/{model_type}"
    args.best_model_dir = f"outputs/{model_type}/best_model"
    args.evaluate_during_training = False
    args.max_seq_length = 64
    args.num_train_epochs = 4
    # args.evaluate_during_training_steps = 100
    # args.wandb_project = "drone-sentiment"
    # args.wandb_kwargs = {"name": model_name}
    args.save_model_every_epoch = False
    args.save_eval_checkpoints = False
    args.train_batch_size = 8
    args.eval_batch_size = 8

    return args


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


def main():
    model_type = sys.argv[1]
    device = True if torch.cuda.is_available() else False
    train_df, test_df = load_dataset('dataset')

    train_args = get_model_args(model_type)
    model_type, model_name = get_model_name(model_type)

    # Create a ClassificationModel
    model = ClassificationModel(
        model_type, model_name, args=train_args, use_cuda=device)
    output_dir = getattr(train_args, "output_dir")

    # Fine-tune the model using our own dataset
    model.train_model(train_df, eval_data=test_df, acc=accuracy_score)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_df)
    eval_score = evaluation_score(result)
    new_eval = {"Model": model_name, 'TP': eval_score['tp'], 'FP': eval_score['fp'], 'FN': eval_score['fn'], 'TN': eval_score['tn'], 'Accuracy': eval_score['accuracy'],
                'Precision': eval_score['precision'], 'Recall': eval_score['recall'], 'F1': eval_score['f1_score'], 'Specificity': eval_score['specificity']}

    # Update the overall models' performance evaluation score
    if os.path.exists('overall_evaluation.csv'):
        eval_df = pd.read_csv('overall_evaluation.csv')
        eval_df = pd.concat([eval_df, pd.DataFrame([new_eval])], ignore_index=True)
        eval_df.to_csv('overall_evaluation.csv', index=False)
    else:
        eval_df = pd.DataFrame(new_eval, index=[0])
        eval_df.to_csv('overall_evaluation.csv', index=False)

    # Save the predicted labels to perform error analysis
    predictions, raw_outputs = model.predict(test_df['text'])
    test_df['preds'] = predictions
    test_df.to_csv("{}/prediction_{}.csv".format(output_dir, model_name))


if __name__ == "__main__":
    main()
