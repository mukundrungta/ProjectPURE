import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from prepro import RETACREDProcessor
from evaluation import get_f1
from model import REModel
from torch.cuda.amp import GradScaler
import wandb
import higher


def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'ss': batch[3].to(args.device),
                      'os': batch[4].to(args.device),
                      }
            outputs = model(**inputs)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                wandb.log({'loss': loss.item()}, step=num_steps)

            if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                for tag, features in benchmarks:
                    f1, output = evaluate(args, model, features, tag=tag)
                    wandb.log(output, step=num_steps)

    for tag, features in benchmarks:
        f1, output = evaluate(args, model, features, tag=tag)
        wandb.log(output, step=num_steps)

def train_meta_learning(args, model, train_features_meta_train, train_features_meta_test, benchmarks):
    train_dataloader_meta_train = DataLoader(train_features_meta_train, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    train_dataloader_meta_test = DataLoader(train_features_meta_test, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    total_steps = int(len(train_dataloader_meta_train) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    inner_opt = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, (batch_meta_train, batch_meta_test) in enumerate(tqdm(zip(train_dataloader_meta_train, train_dataloader_meta_test))):
            model.train()
            inputs_meta_train = {'input_ids': batch_meta_train[0].to(args.device),
                      'attention_mask': batch_meta_train[1].to(args.device),
                      'labels': batch_meta_train[2].to(args.device),
                      'ss': batch_meta_train[3].to(args.device),
                      'os': batch_meta_train[4].to(args.device),
                      }
            
            inputs_meta_test = {'input_ids': batch_meta_test[0].to(args.device),
                      'attention_mask': batch_meta_test[1].to(args.device),
                      'labels': batch_meta_test[2].to(args.device),
                      'ss': batch_meta_test[3].to(args.device),
                      'os': batch_meta_test[4].to(args.device),
                      }

            # trains the model.
            with torch.backends.cudnn.flags(enabled=False), higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fast_model, diffopt):
                fast_model.train()
                meta_train_outputs = fast_model(**inputs_meta_train)
                meta_train_loss = meta_train_outputs[0] / args.gradient_accumulation_steps


                diffopt.step(meta_train_loss) # computing temporary params on meta-train set
                meta_test_outputs = fast_model(**inputs_meta_test)
                meta_test_loss = meta_test_outputs[0] / args.gradient_accumulation_steps

                total_loss = meta_train_loss + meta_test_loss
                scaler.scale(total_loss).backward()

            # outputs = model(**inputs_meta_train)
            # loss = outputs[0] / args.gradient_accumulation_steps
            # scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                wandb.log({'loss': total_loss.item()}, step=num_steps)

            if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                for tag, features in benchmarks:
                    f1, output = evaluate(args, model, features, tag=tag, steps=step)
                    wandb.log(output, step=num_steps)

    for tag, features in benchmarks:
        f1, output = evaluate(args, model, features, tag=tag)
        wandb.log(output, step=num_steps)


def evaluate(args, model, features, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    _, _, max_f1 = get_f1(keys, preds)

    output = {
        tag + "_f1": max_f1 * 100,
    }
    print(output)
    return max_f1, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/retacred", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=40)
    parser.add_argument("--evaluation_steps", type=int, default=500,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--run_name", type=str, default="re-tacred")

    args = parser.parse_args()
    wandb.init(project=args.project_name, name=args.run_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    model = REModel(args, config)
    model.to(0)

    train_file = os.path.join(args.data_dir, "train.json")
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_dir, "test.json")

    processor = RETACREDProcessor(args, tokenizer)
    train_features = processor.read(train_file)
    train_features_meta_train, train_features_meta_test = processor.read_meta_learning(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)

    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
    )

    # train(args, model, train_features, benchmarks)
    train_meta_learning(args, model, train_features_meta_train, train_features_meta_test, benchmarks)

if __name__ == "__main__":
    main()
