package weka.classifiers.meta;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class Model {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		System.out.println("start...............................................");
		String testfile = null;
		String mm = null;
		if (args[0].equals("-o")) {
			testfile = args[1];

		}
		if (args[2].equals("-m")) {
			mm = args[3];

		}
		Classifier c1 = (Classifier) weka.core.SerializationHelper.read(mm);
		BufferedReader br1 = new BufferedReader(new FileReader(testfile));

		Instances ins1 = new Instances(br1);
		ins1.setClassIndex(ins1.numAttributes() - 1);

		int sum = ins1.numInstances();

		System.out.println("label path :" + System.getProperty("user.dir") + "/labeled.txt");
		BufferedWriter bw1 = new BufferedWriter(new FileWriter(System.getProperty("user.dir") + "/labeled.txt"));

		for (int i = 0; i < sum; i++) {

			double clsLabel = 1.0;

			if (c1.classifyInstance(ins1.instance(i)) == 0) {
				clsLabel = 1.0;
			} else {
				clsLabel = -1.0;
			}

			bw1.write(String.valueOf(clsLabel));
			bw1.newLine();

		}
		bw1.flush();
		bw1.close();
		System.out.println("end...............................................");

	}

}
