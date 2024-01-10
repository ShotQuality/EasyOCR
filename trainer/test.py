import time
import os
import yaml
from pathlib import Path
import fire
import pandas as pd

# import string
# import argparse
import random
import torch

# import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager, AttrDict
from dataset import hierarchical_dataset, AlignCollate
from model import Model
from tqdm import tqdm

THIS_FILEPATH = Path(os.path.dirname(os.path.abspath(__file__)))


def get_config(file_path):
    with open(file_path, "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == "None":
        characters = ""
        for data in opt["select_data"].split("-"):
            csv_path = os.path.join(opt["train_data"], data, "labels.csv")
            df = pd.read_csv(
                csv_path,
                sep="^([^,]+),",
                engine="python",
                usecols=["filename", "words"],
                keep_default_na=False,
            )
            all_char = "".join(df["words"])
            characters += "".join(set(all_char))
        characters = sorted(set(characters))
        opt.character = "".join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f"./saved_models/{opt.experiment_name}", exist_ok=True)
    return opt


def validation(model, criterion, evaluation_loader, converter, opt, device):
    """validation or evaluation"""
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in tqdm(enumerate(evaluation_loader)):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(
            device
        )
        text_for_pred = (
            torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        )

        text_for_loss, length_for_loss = converter.encode(
            labels, batch_max_length=opt.batch_max_length
        )

        start_time = time.time()
        if "CTC" in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC decoder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = criterion(
                preds.log_softmax(2).permute(1, 0, 2),
                text_for_loss,
                preds_size,
                length_for_loss,
            )

            if opt.decode == "greedy":
                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode_greedy(preds_index.data, preds_size.data)
            elif opt.decode == "beamsearch":
                preds_str = converter.decode_beamsearch(preds, beamWidth=2)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, : text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1),
            )

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []

        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if "Attn" in opt.Prediction:
                gt = gt[: gt.find("[s]")]
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1

            """
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription." 
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            """

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return (
        valid_loss_avg.val(),
        accuracy,
        norm_ED,
        preds_str,
        confidence_score_list,
        labels,
        infer_time,
        length_of_data,
    )


def main(
    config_file=str(
        THIS_FILEPATH
        / "config_files"
        / "shotquality_sb_config_TPS_ResNet_BiLSTM_CTC.yaml"
    ),
):
    max_show_number = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = get_config(file_path=config_file)
    converter = (
        CTCLabelConverter(opt.character)
        if "CTC" in opt.Prediction
        else AttnLabelConverter(opt.character)
    )

    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)

    print(
        "model input parameters",
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction,
    )

    if opt.saved_model != "":
        pretrained_dict = torch.load(opt.saved_model)
        if opt.new_prediction:
            model.Prediction = nn.Linear(
                model.SequenceModeling_output,
                len(pretrained_dict["module.Prediction.weight"]),
            )

        model = torch.nn.DataParallel(model).to(device)
        print(f"loading pretrained model from {opt.saved_model}")
        if opt.FT:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
        if opt.new_prediction:
            model.module.Prediction = nn.Linear(
                model.module.SequenceModeling_output, opt.num_class
            )
            for name, param in model.module.Prediction.named_parameters():
                if "bias" in name:
                    init.constant_(param, 0.0)
                elif "weight" in name:
                    init.kaiming_normal_(param)
            model = model.to(device)

    criterion = (
        torch.nn.CTCLoss(zero_infinity=True).to(device)
        if "CTC" in opt.Prediction
        else torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    )

    AlignCollate_valid = AlignCollate(
        imgH=opt.imgH,
        imgW=opt.imgW,
        keep_ratio_with_pad=opt.PAD,
        contrast_adjust=opt.contrast_adjust,
    )
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=min(32, opt.batch_size),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        prefetch_factor=512,
        collate_fn=AlignCollate_valid,
        pin_memory=True,
    )

    log_dir = THIS_FILEPATH / "saved_models" / opt.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(
        str(log_dir / "log_eval.txt"),
        "a",
        encoding="utf8",
    ) as log:
        model.eval()
        with torch.no_grad():
            (
                valid_loss,
                current_accuracy,
                current_norm_ED,
                preds,
                confidence_score,
                labels,
                infer_time,
                length_of_data,
            ) = validation(model, criterion, valid_loader, converter, opt, device)

        # training loss and validation loss
        loss_log = f"Valid loss: {valid_loss:0.5f}"
        current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'
        loss_model_log = f"{loss_log}\n{current_model_log}"
        print(loss_model_log)
        log.write(loss_model_log + "\n")

        # show some predicted results
        show_number = min(len(labels), max_show_number)
        dashed_line = "-" * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"

        # show_number = min(show_number, len(labels))

        start = random.randint(0, len(labels) - show_number)
        for gt, pred, confidence in zip(
            labels[start : start + show_number],
            preds[start : start + show_number],
            confidence_score[start : start + show_number],
        ):
            if "Attn" in opt.Prediction:
                gt = gt[: gt.find("[s]")]
                pred = pred[: pred.find("[s]")]

            predicted_result_log += (
                f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
            )
        predicted_result_log += f"{dashed_line}"
        print(predicted_result_log)
        log.write(predicted_result_log + "\n")

    print("end of evaluation!")


if __name__ == "__main__":
    fire.Fire(main)
