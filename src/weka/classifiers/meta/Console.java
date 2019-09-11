package weka.classifiers.meta;

import java.io.BufferedReader;
import java.io.FileReader;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Range;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.CommandLine;

/**
 * @author hqf 2019-7-25 命令行 java -jar ***.jar -f trainfile -p testfile -c cv
 */
public class Console {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		System.out.println("start...............................................");
		String trainfile = null;
		String testfile = null;
		int cv = 5;
		CommandLineParser parser = new DefaultParser();
		Options options = new Options();
		options.addOption("f", "fit", true, "train set");
		options.addOption("p", "predict", true, "testing set");
		options.addOption("c", "cv", true, "cv");
		// Parse the program arguments
		CommandLine commandLine = parser.parse(options, args);

		if (commandLine.hasOption('f')) {
			trainfile = commandLine.getOptionValue('f');

		}
		if (commandLine.hasOption('p')) {
			testfile = commandLine.getOptionValue('p');
		}
		if (commandLine.hasOption('c')) {
			cv = Integer.parseInt(commandLine.getOptionValue('c'));
		}

		try {

			BufferedReader br = new BufferedReader(new FileReader(trainfile));

			Instances ins = new Instances(br);

			ins.setClassIndex(ins.numAttributes() - 1);
			HQFSVM_SA c1 = new HQFSVM_SA();
//			String[] options = weka.core.Utils.splitOptions("-S 0 ");// -K 4
//			c1.setOptions(options);

			if (testfile != null) { /*----------------------- supply the test set------------------------------ */

				c1.buildClassifier(ins);
//				SerializationHelper.write("HQFSVM.model", c1);
				BufferedReader br1 = new BufferedReader(new FileReader(testfile));
				Instances ins1 = new Instances(br1);
				ins1.setClassIndex(ins1.numAttributes() - 1);
				Evaluation eval = new Evaluation(ins);
				eval.evaluateModel(c1, ins1);
				System.out.println("--------------------------");
				System.out.println(eval.toClassDetailsString());
				System.out.println("--------------------------");
				System.out.println(eval.toSummaryString());
				System.out.println("--------------------------");
				System.out.println(eval.toMatrixString());

			} else {/*-------------------- 这里是交叉验证------------------------------------ */

				Evaluation eval = new Evaluation(ins);
				Random r = new Random(1);
				Object[] o = new Object[3];
				o[0] = new StringBuffer();
				o[1] = new Range();
				o[2] = new Boolean(true);
				eval.crossValidateModel(c1, ins, cv, r, o);
				System.out.println(eval.toClassDetailsString());
				System.out.println("-----------------------");
				System.out.println(eval.toSummaryString());
				System.out.println("----------------------");
				System.out.println(eval.toMatrixString());
			}

		}

		catch (Exception ex) {
			System.out.println(ex.getMessage());
		}
		System.out.println("end...............................................");

	}

}
