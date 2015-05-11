### Introduction

Latent Semantic Analysis (LSA) is a widely used method for measuring semantic similarity between sentences and, in particular, for paraphrase detection. We use the LSA with text-to-text representation (i.e. for each utterance the word embeddings are averaged and then the dot product is taken between the two resulting utterance vectors), because it robust by design has been found to yield excellent performance in Rus et al. The embeddings were trained on the Wikipedia corpus (Model W4) from early January 2013 and the TASA corpus. Recently, it has been shown that the Latent Dirichlet Allocation (LDA) model is competetive with LSA in terms of paraphrase detection by Rus et al. We use LDA with greedy word matching because this was found to work best in Rus et al.

We also use evaluation metrics based on structured knowledge bases, because such methods define semantic similarity explicitly and are less biased than those based solely on text corpora. We use the measure proposed by Corley et al. A variant of this measure, by the same authors, performed excellent on paraphrase detection in the survey by Achananuparp et al. To compliment this we use the a WordNet-based greedy word matching procedure first proposed by Lin, and described in details in \url{http://aclweb.org/anthology//W/W12/W12-2018.pdf}. We use WordNet 3.0 and Stanford's Suite of NLP Tools 1.3.0 together with Porter's Stemmer.

We also use BLEU and Meteor, because these have been applied widely in machine translation and image caption generation.


### Toolkit

Luckily for us, most of these semantic similarity measures have been implemented in "SEMILAR: A Semantic Similarity Toolkit":

    http://deeptutor2.memphis.edu/Semilar-Web/public/semilar-api.html


### Local Installation

Download the following files from the website:
- SEMILAR-API-1.0.zip,
- SemilarExampleCodes.zip,
- LSA-MODELS.zip,
- LDA-MODELS.zip
- Wiki 4.zip (http://deeptutor2.memphis.edu/Semilar-Web/public/lsa-models-lrec2014.html)

Then extract SEMILAR-API-1.0.zip and create a directory "Data" inside "SEMILAR-API-1.0". Then extract LDA-MODELS.zip and LSA-MODELS.zip into it. Then extract "Wiki 4.zip" inside the "LSA-Models" directory. Then created directory "bin" inside "SEMILAR-API-1.0", and extract SemilarExampleCodes.zip in it. Then copy over Sentence2SentencSimilarityTool.java (found in the same directory as this readme) into the "bin" directory.

Finally, in directory "SEMILAR-API-1.0/Data/LSA-MODELS/Wiki 4", rename lsaModel to lsaModel.txt and voc to voc.txt.

Open a terminal, cd into "SEMILAR-API-1.0" directory and run:

   javac -d bin -sourcepath bin -cp Semilar-1.0.jar:bin bin/Sentence2SentenceSimilarityTool.java
   java -Xmx4096m -cp Semilar-1.0.jar:bin semilardemo.Sentence2SentenceSimilarityTool <SEMILAR-API_1.0> <TARGETS_FILE> <SAMPLES_FILE> <LINE_START> <LINE_END> <OUTPUT_FILE>

Here <SEMILAR-API_1.0> is the directory SEMILAR-API_1.0.zip was extracted to, <TARGETS_FILE> is the file with target utterances (one utterance per line, optionally tokenized with NLTK), <SAMPLES_FILE> is the file with the model samples (e.g. produced with stochastic sampling or beam search), <LINE_START> and <LINE_END> denote the utterances to evaluate  (e.g. 1 10 will evaluate the semantic semilarity of utterances 1 to 10) and <OUTPUT_FILE> is the file wher the results are written to. The metrics in <OUTPUT_FILE> are separated by spaces corresponding to:
- greedyComparerWNLin
- greedyComparerLSATasa
- greedyComparerLSAWiki
- greedyComparerLDATasa
- optimumComparerLSATasa
- optimumComparerLSAWiki
- lsaComparer
- cmComparer
- meteorComparer
- bleuComparer

The flag -Xmx4096m is to allow the program to use 4 GB memory (required for LSA). It can be increased if necessary


### Cluster Installation

It seems that only Guillimin has the newest Java version. We therefore need to transfer all the software to Guillimin:

First, SSH into Guillimin and create appropriate directories:

    ssh <user>@guillimin.clumeq.ca
    cd /home/<user>
    mkdir Software
    cd Software
    mkdir SEMILAR

Then copy over software and unzip as appropriate

    scp LDA-Models.zip LSA-Models.zip SEMILAR-API-1.0.zip Wiki\ 4.zip <user>@guillimin.clumeq.ca:/home/<user>/Software/SEMILAR
    unzip SEMILAR-API-1.0.zip
    cd SEMILAR-API-1.0
    mkdir Data
    mkdir bin
    cd ..
    mv LDA-Models.zip LSA-Models.zip Wiki\ 4 SEMILAR-API-1.0/Data
    cd SEMILAR-API-1.0/Data
    unzip LDA-Models.zip
    unzip LSA-Models.zip
    mv Wiki\ 4 Data/
    unzip Wiki\ 4

Finally (from laptop) copy over Sentence2SentenceSimilarityTool:

    cd bin
    scp Sentence2SentenceSimilarityTool.java <user>@guillimin.clumeq.ca:/home/<user>/Software/SEMILAR/SEMILAR-API-1.0/bin
 
In "SEMILAR-API-1.0/Data/LSA-MODELS/Wiki 4", rename lsaModel to lsaModel.txt and voc to voc.txt.

Launch interactive job to test:

    qsub -I -l walltime=1:59:59

Wait for it to start then run Sentence2SentenceSimilarityTool as before locally.

It should now be straightforward to evaluate an entire model in less than an hour by running hundres of jobs, each evaluating utterances at different line intervals.


### References

"SEMILAR: The Semantic Similarity Toolkit", Rus et al.
“Latent Semantic Analysis Models on Wikipedia and TASA”, Stefanescu et al. 
"Similarity Measures Based on Latent Dirichlet Allocation", Rus et al.
"A semantic similarity approach to paraphrase detection", Fernando et al. 
"Measuring the semantic similarity of texts", Corley et al.
"The Evaluation of Sentence Similarity Measures" by Achananuparp et al
"Automatic retrieval and clustering of similar words", D. Lin
