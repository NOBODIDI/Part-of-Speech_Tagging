# Part-of-Speech Tagging

This project implements a Part-of-Speech (POS) Tagger using a Hidden Markov Model (HMM). The tagger is trained on a set of annotated training files and can be used to tag new sentences with their corresponding POS tags. This is a useful technique for Natural Language Processing (NLP).

## Requirements

- Python 3.x
- NumPy

## Usage

1. **Training the Model:**

   Prepare a list of training files where each line contains a word and its corresponding POS tag separated by " : ".

2. **Testing the Model:**

   Prepare a test file where each line contains a word. Sentences should be separated by punctuation marks like ".", "?", "!", or "-".

3. **Running the Tagger:**

   Use the following command to run the tagger:

   ```bash
   python3 tagger.py --trainingfiles <training_file1> <training_file2> ... --testfile <test_file> --outputfile <output_file>
   ```

   Example:

   ```bash
   python3 tagger.py --trainingfiles training/training1.txt --testfile tests/test1.txt --outputfile tests/output1.txt
   ```

## Files

- `tagger.py`: The main script that implements the POS tagging using HMM.
- `training/`: Directory containing training files.
- `tests/`: Directory containing test files and output files.

## Functions

- `read_training_files(training_list)`: Reads the training files and returns a list of words with their tags.
- `getM_fTag(words)`: Creates the M and fTag dictionaries from the list of words.
- `getI(words)`: Creates the initial probability matrix I.
- `getT(words)`: Creates the transition probability matrix T.
- `getDistTag(fTag, nbWords)`: Calculates the distribution of tags.
- `read_testing_file(file)`: Reads the testing file and returns a list of sentences.
- `doViterbi(distTag, sent, I, T, M, knownWds)`: Performs the Viterbi algorithm to find the most probable sequence of tags for a sentence.

## Example

An example of a training file (`training1.txt`):

```
Detective : NP0
Chief : NP0
Inspector : NP0
John : NP0
McLeish : NP0
gazed : VVD
doubtfully : AV0
at : PRP
the : AT0
plate : NN1
before : CJS
him : PNP
. : PUN
```

An example of a test file (`test1.txt`):

```
Detective
Chief
Inspector
John
McLeish
gazed
doubtfully
at
the
plate
before
him
.
```

The output file (`output1.txt`) will contain the tagged words:

```
Detective : NP0
Chief : NP0
Inspector : NP0
John : NP0
McLeish : NP0
gazed : VVD
doubtfully : AV0
at : PRP
the : AT0
plate : NN1
before : CJS
him : PNP
. : PUN
```
