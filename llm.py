
pip install "ray[air]" transformers datasets accelerate evaluate torch



import kagglehub

# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Path to dataset files:", path)


import ray
from ray.train.torch import TorchTrainer
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train import ScalingConfig, RunConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import evaluate

# Step 1: Define Tokenization
def tokenize_data(example, tokenizer, max_length=128):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)

# Step 2: Prepare Dataset
def prepare_dataset(tokenizer):
    dataset = load_dataset("imdb", split="train[:2000]")  # small subset for demo
    tokenized_dataset = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    return tokenized_dataset

# Step 3: Define Trainer Initialization Function
def trainer_init_per_worker(train_dataset, eval_dataset, tokenizer_name):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_dir="./logs",
        report_to="none"
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

# Step 4: Launch Ray and Fine-Tune
if __name__ == "__main__":
  ray.shutdown()
  ray.init(ignore_reinit_error=True)

  model_name = "distilbert-base-uncased"  # Small, lightweight model
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  tokenized = prepare_dataset(tokenizer)

  trainer = TransformersTrainer(
      trainer_init_per_worker=lambda: trainer_init_per_worker(
          tokenized["train"], tokenized["test"], model_name
        ),
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        run_config=RunConfig(name="lightweight-llm-finetune")
    )

  result = trainer.fit()
  print("Training completed:", result)
  ray.shutdown()



