package semilardemo;

import semilar.config.ConfigManager;
import semilar.data.Sentence;
import semilar.sentencemetrics.BLEUComparer;
import semilar.sentencemetrics.CorleyMihalceaComparer;
import semilar.sentencemetrics.DependencyComparer;
import semilar.sentencemetrics.GreedyComparer;
import semilar.sentencemetrics.LSAComparer;
import semilar.sentencemetrics.LexicalOverlapComparer;
import semilar.sentencemetrics.MeteorComparer;
import semilar.sentencemetrics.OptimumComparer;
import semilar.sentencemetrics.PairwiseComparer.NormalizeType;
import semilar.sentencemetrics.PairwiseComparer.WordWeightType;
import semilar.tools.preprocessing.SentencePreprocessor;
import semilar.tools.semantic.WordNetSimilarity;
import semilar.wordmetrics.LDAWordMetric;
import semilar.wordmetrics.LSAWordMetric;
import semilar.wordmetrics.WNWordMetric;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Based on the example found in the Sentence2SentenceSimilarityTest
 *
 * @author Iulian Vlad Serban
 */
public class Sentence2SentenceSimilarityTool {

    /* NOTE:
     * The greedy matching and Optimal matching methods rely on word to word similarity method.
     *(please see http://aclweb.org/anthology//W/W12/W12-2018.pdf for more details). So, based on the unique word to
     * word similarity measure, they have varying output (literally, many sentence to sentence similarity methods from
     * the combinations).
     */
    //greedy matching (see the available word 2 word similarity in the separate example file).

    GreedyComparer greedyComparerWNLin; //greedy matching, use wordnet LIN method for Word 2 Word similarity
    GreedyComparer greedyComparerLSATasa; // use LSA based word 2 word similarity (using TASA corpus LSA model).
    GreedyComparer greedyComparerLDATasa; // use LDA based word 2 word similarity (using TASA corpus LDA model).
    GreedyComparer greedyComparerLSAWiki; // use LDA based word 2 word similarity (using TASA corpus LDA model).


    //Overall optimum matching method.. you may try all possible word to word similarity measures. Here I show some.
    OptimumComparer optimumComparerLSATasa;
    OptimumComparer optimumComparerLSAWiki;

    //Please see paper Corley, C. and Mihalcea, R. (2005). Measuring the semantic similarity of texts.
    CorleyMihalceaComparer cmComparer;

    //METEOR method (introduced for machine translation evaluation): http://www.cs.cmu.edu/~alavie/METEOR/
    MeteorComparer meteorComparer;

    //BLEU (introduced for machine translation evaluation):http://acl.ldc.upenn.edu/P/P02/P02-1040.pdf 
    BLEUComparer bleuComparer;
    LSAComparer lsaComparer;
    LexicalOverlapComparer lexicalOverlapComparer; // Just see the lexical overlap.

    public Sentence2SentenceSimilarityTool(String semilarRootFolder) {

        /* Word to word similarity expanded to sentence to sentence .. so we need word metrics */
        boolean wnFirstSenseOnly = false; //applies for WN based methods only.
        WNWordMetric wnMetricLin = new WNWordMetric(WordNetSimilarity.WNSimMeasure.LIN, wnFirstSenseOnly);
        WNWordMetric wnMetricLeskTanim = new WNWordMetric(WordNetSimilarity.WNSimMeasure.LESK_TANIM, wnFirstSenseOnly);
        //provide the LSA model name you want to use.
        LSAWordMetric lsaMetricTasa = new LSAWordMetric("LSA-MODEL-TASA-LEMMATIZED-DIM300");

        //provide the LSA model name you want to use.
        LSAWordMetric lsaMetricWiki = new LSAWordMetric("Wiki 4");

        //provide the LDA model name you want to use.
        LDAWordMetric ldaMetricTasa = new LDAWordMetric("LDA-MODEL-TASA-LEMMATIZED-TOPIC300");

        greedyComparerWNLin = new GreedyComparer(wnMetricLin, 0.3f, false);
        greedyComparerLSATasa = new GreedyComparer(lsaMetricTasa, 0.3f, false);
        greedyComparerLSAWiki = new GreedyComparer(lsaMetricWiki, 0.3f, false);
        greedyComparerLDATasa = new GreedyComparer(ldaMetricTasa, 0.3f, false);

        optimumComparerLSATasa = new OptimumComparer(lsaMetricTasa, 0.3f, false, WordWeightType.NONE, NormalizeType.AVERAGE);

        optimumComparerLSAWiki = new OptimumComparer(lsaMetricWiki, 0.3f, false, WordWeightType.NONE, NormalizeType.AVERAGE);

        /* methods without using word metrics */
        cmComparer = new CorleyMihalceaComparer(0.3f, false, "NONE", "par");
        //for METEOR, please provide the **Absolute** path to your project home folder (without / at the end), And the
        // semilar library jar file should be in your project home folder.
        meteorComparer = new MeteorComparer(semilarRootFolder);
        bleuComparer = new BLEUComparer();

        //lsaComparer: This is different from lsaMetricTasa, as this method will
        // directly calculate sentence level similarity whereas  lsaMetricTasa
        // is a word 2 word similarity metric used with Optimum and Greedy methods.
        lsaComparer = new LSAComparer("LSA-MODEL-TASA-LEMMATIZED-DIM300");
    }

    public void printSimilarities(Sentence sentenceA, Sentence sentenceB) {
        System.out.println("Sentence 1:" + sentenceA.getRawForm());
        System.out.println("Sentence 2:" + sentenceB.getRawForm());
        System.out.println("------------------------------");
        System.out.println("greedyComparerWNLin : " + greedyComparerWNLin.computeSimilarity(sentenceA, sentenceB));
        System.out.println("greedyComparerLSATasa : " + greedyComparerLSATasa.computeSimilarity(sentenceA, sentenceB));
        System.out.println("greedyComparerLSAWiki : " + greedyComparerLSAWiki.computeSimilarity(sentenceA, sentenceB));

        System.out.println("greedyComparerLDATasa : " + greedyComparerLDATasa.computeSimilarity(sentenceA, sentenceB));
        System.out.println("optimumComparerLSATasa : " + optimumComparerLSATasa.computeSimilarity(sentenceA, sentenceB));

        System.out.println("optimumComparerLSAWiki : " + optimumComparerLSAWiki.computeSimilarity(sentenceA, sentenceB));
        System.out.println("lsaComparer : " + lsaComparer.computeSimilarity(sentenceA, sentenceB));
        System.out.println("cmComparer : " + cmComparer.computeSimilarity(sentenceA, sentenceB));
        System.out.println("meteorComparer : " + meteorComparer.computeSimilarity(sentenceA, sentenceB));
        System.out.println("bleuComparer : " + bleuComparer.computeSimilarity(sentenceA, sentenceB));
        System.out.println("                              ");
    }

    public String getSimilarities(Sentence sentenceA, Sentence sentenceB) {
        return greedyComparerWNLin.computeSimilarity(sentenceA, sentenceB) + " " + greedyComparerLSATasa.computeSimilarity(sentenceA, sentenceB) + " " + greedyComparerLSAWiki.computeSimilarity(sentenceA, sentenceB) + " " + greedyComparerLDATasa.computeSimilarity(sentenceA, sentenceB) + " " + optimumComparerLSATasa.computeSimilarity(sentenceA, sentenceB) + " " + optimumComparerLSAWiki.computeSimilarity(sentenceA, sentenceB) + " " + lsaComparer.computeSimilarity(sentenceA, sentenceB) + " " + cmComparer.computeSimilarity(sentenceA, sentenceB) + " " + meteorComparer.computeSimilarity(sentenceA, sentenceB) + " " + bleuComparer.computeSimilarity(sentenceA, sentenceB);
    }

    public String getMetricNames() {
        return "greedyComparerWNLin greedyComparerLSATasa greedyComparerLSAWiki greedyComparerLDATasa optimumComparerLSATasa optimumComparerLSAWiki lsaComparer cmComparer meteorComparer bleuComparer";
    }


    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // Read commandline arguments
        if (args.length != 6) {
            System.out.println("Sentence2SentenceSimilarityTool expects exactly 6 arguments : ");            
            System.out.println("- SEMILAR Package Root Directory");            
            System.out.println("- References / Targets filename");            
            System.out.println("- Samples filename");            
            System.out.println("- Start line number");            
            System.out.println("- End line number");            
            System.out.println("- Output filename");
        }

        String semilarRootFolder = args[0];
        String semilarDataFolder = semilarRootFolder + "Data/";
        String refFilePath = args[1];
        String sampleFilePath = args[2];
        int startLine = Integer.parseInt(args[3]);
        int endLine = Integer.parseInt(args[4]);
        String outputFilePath = args[5];

        // Set the semilar data folder path (ending with /).
        ConfigManager.setSemilarDataRootFolder(semilarDataFolder);


        // Prepare to open reference and samples file
        BufferedReader refBuf = null;
        BufferedReader sampleBuf = null;

        try {
              // Open reference and samples file
	       refBuf = new BufferedReader(new FileReader(refFilePath));
	       sampleBuf = new BufferedReader(new FileReader(sampleFilePath));

               // Open output file for writing
	       File outputFile = new File(outputFilePath);
	       BufferedWriter output = new BufferedWriter(new FileWriter(outputFile));

	       String currentRef;
	       String currentSample;

	       // Initialize preprocessor
	       System.out.println("Initializing semantic similarity models... \n");
	       SentencePreprocessor preprocessor = new SentencePreprocessor(SentencePreprocessor.TokenizerType.STANFORD, SentencePreprocessor.TaggerType.STANFORD, SentencePreprocessor.StemmerType.PORTER, SentencePreprocessor.ParserType.STANFORD);
	       Sentence2SentenceSimilarityTool s2sSimilarityMeasurer = new Sentence2SentenceSimilarityTool(semilarRootFolder);

               // Loop over each line while keeping track of the line number
	       int i = 0;
	       while (((currentRef = refBuf.readLine()) != null) && ((currentSample = sampleBuf.readLine()) != null)) {
		i = i + 1;

                // Only process lines in the interval specified by the commandline arguments
                if ((startLine <= i) && (i <= endLine)) {
			System.out.println("Processing sample " + i + "\n");

			Sentence sentence1;
			Sentence sentence2;

		        // Replace puntucations and odd symbols. This could be done in a cleaner way...
		        currentRef = currentRef.replace(" ' ", "'");
		        currentRef = currentRef.replace("-", "");
		        currentRef = currentRef.replace(".", "");
		        currentRef = currentRef.replace(",", "");
		        currentRef = currentRef.replace("!", "");
		        currentRef = currentRef.replace(";", "");
		        currentRef = currentRef.replace(":", "");
		        currentRef = currentRef.replace("?", "");
		        currentRef = currentRef.replace("'", "");
		        currentRef = currentRef.replace("\"", "");
		        currentRef = currentRef.replace("<person>", "person");
		        currentRef = currentRef.replace("<number>", "number");
		        currentRef = currentRef.replace("<continued_utterance>", "");
		        currentRef = currentRef.replace("<unk>", "");
		        currentRef = currentRef.replace("`", "");
		        currentRef = currentRef.replace("[", "");
		        currentRef = currentRef.replace("]", "");
		        currentRef = currentRef.replace("  ", " ");
		        currentRef = currentRef.replace("  ", " ");

		        currentSample = currentSample.replace(" ' ", "'");
		        currentSample = currentSample.replace("-", "");
		        currentSample = currentSample.replace(".", "");
		        currentSample = currentSample.replace(",", "");
		        currentSample = currentSample.replace("!", "");
		        currentSample = currentSample.replace(";", "");
		        currentSample = currentSample.replace(":", "");
		        currentSample = currentSample.replace("'", "");
		        currentSample = currentSample.replace("\"", "");
		        currentSample = currentSample.replace("<person>", "person");
		        currentSample = currentSample.replace("<number>", "number");
		        currentSample = currentSample.replace("<continued_utterance>", "");
		        currentSample = currentSample.replace("<unk>", "");
		        currentSample = currentSample.replace("`", "");
		        currentSample = currentSample.replace("[", "");
		        currentSample = currentSample.replace("]", "");
		        currentSample = currentSample.replace("  ", " ");
		        currentSample = currentSample.replace("  ", " ");

		        // Preprocess sentences, compute similarities and write them to disc
			sentence1 = preprocessor.preprocessSentence(currentRef);
			sentence2 = preprocessor.preprocessSentence(currentSample);
			output.write(s2sSimilarityMeasurer.getSimilarities(sentence1, sentence2) + "\n");

			//s2sSimilarityMeasurer.printSimilarities(sentence1, sentence2);
		}
	       }
	       output.close();
       } catch (IOException e) {
		e.printStackTrace();
       } finally {
		try {
			if (refBuf != null) refBuf.close();
			if (sampleBuf != null) sampleBuf.close();
		} catch (IOException ex) {
			ex.printStackTrace();
		}
       }

        System.out.println("\nDone!");
    }
}
