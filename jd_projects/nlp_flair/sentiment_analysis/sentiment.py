{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b80e8a21",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'transformers'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m/tmp/ipykernel_34433/2842442717.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mtransformers\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mAutoModelForSequenceClassification\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mtransformers\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mTFAutoModelForSequenceClassification\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mtransformers\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mAutoTokenizer\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mscipy\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mspecial\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0msoftmax\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'transformers'"
     ]
    }
   ],
   "source": [
    "from transformers import AutoModelForSequenceClassification\n",
    "from transformers import TFAutoModelForSequenceClassification\n",
    "from transformers import AutoTokenizer\n",
    "import numpy as np\n",
    "from scipy.special import softmax\n",
    "\n",
    "# Preprocess text (username and link placeholders)\n",
    "def preprocess(text):\n",
    "    new_text = [] \n",
    "    for t in text.split(\" \"):\n",
    "        t = '@user' if t.startswith('@') and len(t) > 1 else t\n",
    "        t = 'http' if t.startswith('http') else t\n",
    "        new_text.append(t)\n",
    "    return \" \".join(new_text)\n",
    "\n",
    "# Tasks:\n",
    "# emoji, emotion, hate, irony, offensive, sentiment\n",
    "# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary\n",
    "\n",
    "task='sentiment'\n",
    "MODEL = f\"cardiffnlp/twitter-roberta-base-{task}\"\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL)\n",
    "\n",
    "# download label mapping\n",
    "labels = [\"negative\",\"neutral\",\"positive\"]\n",
    "\n",
    "# PT\n",
    "model = AutoModelForSequenceClassification.from_pretrained(MODEL)\n",
    "# model.save_pretrained(MODEL)\n",
    "\n",
    "def SentimentAnalysis(text):\n",
    "    text = preprocess(text)\n",
    "    encoded_input = tokenizer(text, return_tensors='pt')\n",
    "    output = model(**encoded_input)\n",
    "    scores = output[0][0].detach().numpy()\n",
    "    scores = softmax(scores)\n",
    "    ranking = np.argsort(scores)\n",
    "    ranking = ranking[::-1]\n",
    "    idx = scores.argmax()\n",
    "    if idx == 0 or idx == 2:\n",
    "        return { \"sentiment\" : labels[idx] , \"score\" : float(format(scores[idx],\".4f\")), \"error\" : 0}\n",
    "    # print(labels[idx],scores[idx])\n",
    "    elif idx == 1 and scores[idx] < 0.52 and len(text.split())<3:\n",
    "        idx = np.argsort(scores)[::-1][1]\n",
    "        return { \"sentiment\" : labels[idx] , \"score\" : float(format(scores[idx],\".4f\")), \"error\" : 0}\n",
    "        # print(labels[idx],scores[idx])\n",
    "    else:\n",
    "        return { \"sentiment\" : labels[idx] , \"score\" : float(format(scores[idx],\".4f\")), \"error\" : 0}\n",
    "        # print(labels[idx],scores[idx])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3cec2959",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
