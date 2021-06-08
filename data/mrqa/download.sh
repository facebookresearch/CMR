OUTPUT=mrqa_train
mkdir -p $OUTPUT
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz -O $OUTPUT/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz -O $OUTPUT/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz -O $OUTPUT/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz -O $OUTPUT/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz -O $OUTPUT/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz -O $OUTPUT/NaturalQuestions.jsonl.gz

OUTPUT=mrqa_dev # in-domain
mkdir -p $OUTPUT
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz -O $OUTPUT/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz -O $OUTPUT/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz -O $OUTPUT/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz -O $OUTPUT/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz -O $OUTPUT/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz -O $OUTPUT/NaturalQuestions.jsonl.gz