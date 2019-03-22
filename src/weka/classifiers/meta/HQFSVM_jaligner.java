package weka.classifiers.meta;

import java.io.IOException;

import java.util.ArrayList;
import java.util.StringTokenizer;

import jaligner.Alignment;
import jaligner.Sequence;
import jaligner.SmithWatermanGotoh;
import jaligner.matrix.Matrix;
import jaligner.matrix.MatrixLoaderException;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Type;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;

/**
 * @author hqf 传递核矩阵给svm
 */

public class HQFSVM_jaligner extends LibSVM {

	private static final int SIZE = 127;
	float[][] scores = new float[SIZE][SIZE];
	public Instances ii;// 存储训练集
	public float[] index = new float[2];
	public int flag = 0;// 1表示dna/其余protein
	int flag1 = 1;// 序号

	int length;
	String[] m_ClassNames;

	@Override
	public void buildClassifier(Instances insts) throws Exception {
		// TODO Auto-generated method stub
		length = insts.numClasses();

		m_ClassNames = new String[length];

		for (int i = 0; i < length; i++) {

			m_ClassNames[i] = insts.classAttribute().value(i);
		}
		if (insts.numAttributes() <= 2) {

			ii = insts;

			String[][] kk = kernel_function(insts);

			ArrayList<Attribute> l = attributes(kk[0].length - 1);

			Instances instances = new Instances("kernel", l, 0);

			instances.setClassIndex(instances.numAttributes() - 1);

			int sum = insts.numInstances();

			for (int i = 0; i < sum; i++) {

				double num[] = new double[instances.numAttributes()];

				Instance instance1 = new DenseInstance(1, num);
				int g;
				num[0] = i + 1;

				for (g = 1; g < num.length - 1; g++) {

					num[g] = Double.parseDouble(kk[i][g + 1]);
				}

				num[g] = instances.attribute(g).indexOfValue(kk[i][0]);

				instances.add(instance1);

			}
			// System.out.println(instances);
			super.buildClassifier(instances);
		} else {
			super.buildClassifier(insts);
		}
	}

	public ArrayList<Attribute> attributes(int i) {

		ArrayList<Attribute> attributes = new ArrayList<>();

		for (int l = 0; l < i; l++) {

			attributes.add(new Attribute("num" + l));

		}

		ArrayList<String> labels = new ArrayList<String>();

		for (int n = 0; n < length; n++) {

			labels.add(m_ClassNames[n]);

		}

		attributes.add(new Attribute("class", labels));

		return attributes;

	}

	public ArrayList<Attribute> attributes(String[] a) {

		ArrayList<Attribute> attributes = new ArrayList<>();

		for (int l = 0; l < a.length; l++) {

			attributes.add(new Attribute(a[l]));

		}

		ArrayList<String> labels = new ArrayList<String>();

		for (int n = 0; n < length; n++) {

			labels.add(m_ClassNames[n]);

		}

		attributes.add(new Attribute("class", labels));

		return attributes;

	}

	/*
	 * @author hqf 这个方法中就是对单个实例就行预测，怎么预测呢，这就需要判断置信度了， 对单个实例返回置信度较大的标签，
	 */
	@Override
	public double classifyInstance(Instance instance) throws Exception {

		double dist[] = new double[2];

		if (instance.numAttributes() <= 2) {

			String[] kk = k_function(ii, instance);// 计算核矩阵 ii为训练集

			ArrayList<Attribute> l = attributes(ii.numInstances() + 1);

			Instances instances = new Instances("kernel", l, 0);

			instances.setClassIndex(instances.numAttributes() - 1);

			double num[] = new double[instances.numAttributes()];

			Instance instance1 = new DenseInstance(1, num);

			int k = 0;
			num[0] = 1;// 待检查
			for (k = 1; k < instance1.numAttributes() - 1; k++) {

				num[k] = Double.parseDouble(kk[k + 1]);

			}
			num[k] = instances.attribute(k).indexOfValue(kk[0]);

			instance1.setDataset(instances);

			instance1.setClassMissing();

			instances.add(instance1);

			dist = super.distributionForInstance(instances.instance(0));

		} else {

			dist = super.distributionForInstance(instance);

		}

		if (dist == null) {

			throw new Exception("Null distribution predicted");

		}
		switch (instance.classAttribute().type()) {

		case Attribute.NOMINAL:

			double max = 0;

			int maxIndex = 0;

			for (int i = 0; i < dist.length; i++) {

				if (dist[i] > max) {

					maxIndex = i;

					max = dist[i];

				}
			}
			if (max > 0) {

				return maxIndex;

			} else {

				return Utils.missingValue();

			}
		case Attribute.NUMERIC:

		case Attribute.DATE:

			return dist[0];

		default:

			return Utils.missingValue();

		}

	}

	/**
	 * @author sponsor-fly 交叉验证 / 模型评估走这里
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		double dist[] = new double[2];

		if (instance.numAttributes() <= 2) {

			String[] kk = k_function(ii, instance);// 计算核矩阵 ii为训练集

			ArrayList<Attribute> l = attributes(ii.numInstances() + 1);

			Instances instances = new Instances("DNA", l, 0);

			instances.setClassIndex(instances.numAttributes() - 1);

			double num[] = new double[instances.numAttributes()];

			Instance instance1 = new DenseInstance(1, num);

			int k = 0;
			num[0] = flag1;// 待检查
			flag1++;
			for (k = 1; k < instance1.numAttributes() - 1; k++) {

				num[k] = Double.parseDouble(kk[k + 1]);
			}
			num[k] = instances.attribute(k).indexOfValue(kk[0]);

			instance1.setDataset(instances);

			instance1.setClassMissing();

			instances.add(instance1);

			dist = super.distributionForInstance(instances.instance(0));

		} else {

			dist = super.distributionForInstance(instance);

		}
		return dist;
	}

	/**
	 * @author hqf 这个方法设置在weka的classify的choose下是否为灰色；也就是定义该分类器可以处理什么类型数据
	 */

	@Override
	public Capabilities getCapabilities() {

		Capabilities result = super.getCapabilities();

		result.disableAll();

		// attributes
		result.enable(Capability.NUMERIC_ATTRIBUTES);

		result.enable(Capability.STRING_ATTRIBUTES);

		result.enable(Capability.MISSING_VALUES);
		// class
		result.enable(Capability.NOMINAL_CLASS);

		result.enable(Capability.NUMERIC_CLASS);

		result.enable(Capability.MISSING_CLASS_VALUES);

		return result;
	}

	public int test() {

		int flag = 0;

		double rate = 0;// 统计ATCGU的含量

		rate += stringCount(ii.instance(0).stringValue(0), "a");

		rate += stringCount(ii.instance(0).stringValue(0), "t");

		rate += stringCount(ii.instance(0).stringValue(0), "g");

		rate += stringCount(ii.instance(0).stringValue(0), "c");

		rate += stringCount(ii.instance(0).stringValue(0), "u");

		rate += stringCount(ii.instance(0).stringValue(0), "x");

		if (rate > 0.9) {

			flag = 1;
		}

		return flag;
	}

	public String[] k_function(Instances train, Instance test) throws MatrixLoaderException {

		int trainnum = train.numInstances();

		String kk[] = new String[trainnum + 2];

		double[] s;

		// 导入打分矩阵
		Matrix matrix = null;

		if (flag == 1) {// dna

			matrix = new Matrix("matrix", scores);

		} else {// protein

			matrix = new Matrix("matrix", scores);
		}
		String teststring;
		String trainstring;
		String teststring1;
		String trainstring1;
		String testins = null;
		String trainins = null;

		teststring1 = String.valueOf(test);

		teststring = teststring1.toUpperCase();

		if (teststring.trim().length() != 0) {

			String a[] = teststring.split(",");

			testins = a[0];

			kk[0] = a[1];

			kk[1] = String.valueOf((0 + 1));

		}
		for (int j = 0; j < trainnum; j++) {

			trainstring1 = String.valueOf(train.instance(j));

			trainstring = trainstring1.toUpperCase();

			if (trainstring.trim().length() != 0) {

				String c[] = trainstring.split(",");

				trainins = c[0];

			}
			Sequence s1 = new Sequence(testins);

			Sequence s2 = new Sequence(trainins);

			Alignment alignment = SmithWatermanGotoh.align(s1, s2, matrix, index[0], index[1]);

			float similarity = alignment.getSimilarity();

			float len = alignment.getSequence1().length;

			float sim = similarity / len;

			String dd = String.valueOf(sim);

			kk[j + 2] = dd;

		}

		return kk;

	}

	public String[][] kernel_function(Instances data) throws IOException, MatrixLoaderException {

		Matrix matrix = null;

		flag = test();

		if (flag == 1) {// DNA

			index[0] = 5.0f;
			index[1] = 2.0f;

			matrix = new Matrix("matrix", DnaMatrix());

		} else {// Protein

			index[0] = 10.0f;
			index[1] = 0.5f;

			matrix = new Matrix("matrix", ProteinMatrix());
		}

		int sum = data.numInstances();

		String tag;
		String b[] = new String[sum];// 存放字符串
		String kk[][] = new String[sum][sum + 2];

		for (int n = 0; n < sum; n++) {

			tag = String.valueOf(data.instance(n));

			if (tag.trim().length() != 0) {

				String a[] = tag.split(",");

				b[n] = a[0].toUpperCase();

				kk[n][0] = a[1];

				kk[n][1] = String.valueOf((n + 1));

			}
		}

		// 计算上三角相似度
		for (int j = 0; j < b.length; j++) {

			for (int k = j + 1; k < b.length; k++) {

				Sequence s1 = new Sequence(b[j]);

				Sequence s2 = new Sequence(b[k]);

				Alignment alignment = SmithWatermanGotoh.align(s1, s2, matrix, index[0], index[1]);

				float similarity = alignment.getSimilarity();

				float len = alignment.getSequence1().length;

				float sim = similarity / len;

				String dd = String.valueOf(sim);

				kk[j][k + 2] = dd;
			}

		}
		// 对角线相似度都是1，减少计算量
		for (int i = 0; i < kk.length; i++) {
			kk[i][i + 2] = "1";
		}
		// 上三角相似度赋值给下三角
		for (int i = 0; i < kk.length; i++) {
			for (int j = i; j < kk.length; j++) {
				kk[j][i + 2] = kk[i][j + 2];

			}
		}

		return kk;

	}

	public static float stringCount(String str, String key) {

		String str1 = str.toUpperCase();

		String key1 = key.toUpperCase();

		int index = 0;

		float count = 0;

		float sum = str1.length() - 1;

		float rate = 0.0f;

		while ((index = str1.indexOf(key1)) != -1) {

			str1 = str1.substring(index + key1.length());

			count++;
		}

		rate = count / sum;

		return rate;
	}

	public float[][] ProteinMatrix() {
		char[] acid = { 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
				'V', 'B', 'Z', 'X', '*' };
		String[] ll = { "4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4 ",
				"-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4",
				"-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4",
				"-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4",
				"0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4",
				"-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4",
				"-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4",
				"0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4",
				"-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4",
				"-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4",
				"-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4",
				"-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4",
				"-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4",
				"-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4",
				"-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4",
				"1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4",
				"0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4",
				"-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4",
				"-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4",
				"0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4",
				"-2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4",
				"-1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4",
				"0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4",
				"-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1" };

		for (int n = 0; n < ll.length; n++) {
			StringTokenizer tokenizer = new StringTokenizer(ll[n]);

			for (int j = 0; tokenizer.hasMoreTokens(); j++) {

				scores[acid[n]][acid[j]] = Float.parseFloat(tokenizer.nextToken());
			}
		}
		return scores;

	}

	public float[][] DnaMatrix() {
		char[] acid = { 'A', 'T', 'C', 'G' };
		String[] ll = { "5 -4 -4 -4", "-4  5 -4  -4", "-4 -4  5 -4", "-4 -4 -4  5" };

		for (int n = 0; n < ll.length; n++) {
			StringTokenizer tokenizer = new StringTokenizer(ll[n]);

			for (int j = 0; tokenizer.hasMoreTokens(); j++) {

				scores[acid[n]][acid[j]] = Float.parseFloat(tokenizer.nextToken());
			}
		}

		return scores;

	}

	@Override
	public String globalInfo() {
		return "A wrapper class for the libsvm library. This wrapper supports the classifiers implemented in the libsvm "
				+ "library.\n"
				+ "Note: To be consistent with other SVMs in WEKA, the target attribute is now normalized before "
				+ "\n" + getTechnicalInformation().toString();
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;
		TechnicalInformation additional;
		TechnicalInformation additional1;

		result = new TechnicalInformation(Type.MISC);
		result.setValue(TechnicalInformation.Field.AUTHOR, "hqf");
		result.setValue(TechnicalInformation.Field.YEAR, "2019");
		result.setValue(TechnicalInformation.Field.TITLE, "HQFSVM-jaligner");
		result.setValue(TechnicalInformation.Field.NOTE, "LibSVM was originally developed as 'HQFSVM'");
		result.setValue(TechnicalInformation.Field.URL, "#");
		result.setValue(TechnicalInformation.Field.NOTE,
				"The Weka classifier works with version 1.0 of HQFSVM-kmer-jaligner");

		additional1 = result.add(Type.MISC);
		additional1.setValue(TechnicalInformation.Field.AUTHOR, "Yasser EL-Manzalawy");
		additional1.setValue(TechnicalInformation.Field.YEAR, "2005");
		additional1.setValue(TechnicalInformation.Field.TITLE, "WLSVM");
		additional1.setValue(TechnicalInformation.Field.NOTE, "LibSVM was originally developed as 'WLSVM'");
		additional1.setValue(TechnicalInformation.Field.URL, "http://www.cs.iastate.edu/~yasser/wlsvm/");
		additional1.setValue(TechnicalInformation.Field.NOTE,
				"You don't need to include the WLSVM package in the CLASSPATH");

		additional = result.add(Type.MISC);
		additional.setValue(TechnicalInformation.Field.AUTHOR, "Chih-Chung Chang and Chih-Jen Lin");
		additional.setValue(TechnicalInformation.Field.TITLE, "LIBSVM - A Library for Support Vector Machines");
		additional.setValue(TechnicalInformation.Field.YEAR, "2001");
		additional.setValue(TechnicalInformation.Field.URL, "http://www.csie.ntu.edu.tw/~cjlin/libsvm/");
		additional.setValue(TechnicalInformation.Field.NOTE, "The Weka classifier works with version 2.82 of LIBSVM");

		return result;
	}

	@Override
	public String modelFileTipText() {
		return "The file to save the HQFSVM-jaligner-internal model to; no model is saved if pointing to a directory.";
	}

	@Override
	public String toString() {
		return "HQFSVM-jaligner wrapper, original code by hqf";
	}

	public static void main(String[] args) {
		runClassifier(new HQFSVM_jaligner(), args);
	}

}
