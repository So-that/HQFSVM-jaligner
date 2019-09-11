package weka.classifiers.meta;

import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;
import jaligner.Sequence;
import jaligner.matrix.Matrix;
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

public class HQFSVM_SA extends LibSVM {

	private static final int SIZE = 127;
	float[][] scores = new float[SIZE][SIZE];// 存储罚分矩阵
	public Instances ii;// 存储训练集
	public float[] index = new float[2];// 仿射罚分的 罚分设置
	public static int flag = 0;// 1表示dna/其余protein
	static int length;
	static String[] m_ClassNames;

	public void classname(Instances insts) {// 记录标签名字

		length = insts.numClasses();

		m_ClassNames = new String[length];

		for (int i = 0; i < length; i++) {

			m_ClassNames[i] = insts.classAttribute().value(i);
		}

	}

	@Override
	public void buildClassifier(Instances insts) throws Exception {
		// TODO Auto-generated method stub

		classname(insts);

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
			num[0] = 1;
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
			num[0] = 1;

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

	public int test(Instances data) {

		int flag = 0;

		double rate = 0;// 统计ATCGU的含量

		rate += stringCount(data.instance(0).stringValue(0), "a");

		rate += stringCount(data.instance(0).stringValue(0), "t");

		rate += stringCount(data.instance(0).stringValue(0), "g");

		rate += stringCount(data.instance(0).stringValue(0), "c");

		rate += stringCount(data.instance(0).stringValue(0), "u");

		rate += stringCount(data.instance(0).stringValue(0), "x");

		if (rate > 0.9) {

			flag = 1;
		}

		return flag;
	}

	public String[] k_function(Instances train, Instance test) {

		double max1 = 1;
		double min1 = 0;

		int trainnum = train.numInstances();

		String kk[] = new String[trainnum + 2];

		// 导入打分矩阵
		Matrix matrix = null;

		if (flag == 1) {// dna

			matrix = new Matrix("matrix", scores);

		} else {// protein

			matrix = new Matrix("matrix", scores);
		}
		String teststring;
		String trainstring;
		String testins = null;
		String trainins = null;

		teststring = String.valueOf(test).toUpperCase();

		if (teststring.trim().length() != 0) {

			String a[] = teststring.split(",");

			testins = a[0];

//			kk[0] = a[1];
			kk[0] = "?";

			kk[1] = String.valueOf(1);

		}
		for (int j = 0; j < trainnum; j++) {

			trainstring = String.valueOf(train.instance(j)).toUpperCase();

			if (trainstring.trim().length() != 0) {

				String c[] = trainstring.split(",");

				trainins = c[0];

			}
			Sequence s1 = new Sequence(testins);

			Sequence s2 = new Sequence(trainins);

			double sim = align(s1, s2, matrix, index[0], index[1]);

			String dd = String.valueOf(sim);

			kk[j + 2] = dd;

		}

		if (flag == 0) {// protein
			for (int j = 0; j < trainnum; j++) {
				max1 = Double.parseDouble(kk[2]);
				min1 = max1;

				double cc = Double.parseDouble(kk[j + 2]);

				if (cc > max1) {
					max1 = cc;
				}
				if (cc < min1) {
					min1 = cc;
				}

			}

			for (int j = 0; j < trainnum; j++) {
				double tt = Double.parseDouble(kk[j + 2]);
				double ss = (tt - min1) / (max1 - min1);

				String dd = String.valueOf(ss);

				kk[j + 2] = dd;
			}

		}

		return kk;

	}

	public String[][] kernel_function(Instances data) throws IOException {

		Matrix matrix = null;

		flag = test(data);

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

			for (int k = j; k < b.length; k++) {

				Sequence s1 = new Sequence(b[j]);

				Sequence s2 = new Sequence(b[k]);

				double sim = align(s1, s2, matrix, index[0], index[1]);

				String dd = String.valueOf(sim);

				kk[j][k + 2] = dd;

			}

		}
		if (flag == 0) {// protein
			double max = Double.parseDouble(kk[0][2]);
			double min = max;
			for (int j = 0; j < b.length; j++) {

				for (int k = j; k < b.length; k++) {
					double dd = Double.parseDouble(kk[j][k + 2]);
					if (dd > max) {
						max = dd;

					}
					if (dd < min) {
						min = dd;

					}

				}
			}

			for (int j = 0; j < b.length; j++) {

				for (int k = j + 1; k < b.length; k++) {

					double tt = Double.parseDouble(kk[j][k + 2]);

					double ss = (tt - min) / (max - min);

					String dd = String.valueOf(ss);

					kk[j][k + 2] = dd;

				}
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

	public static double align(Sequence s1, Sequence s2, Matrix matrix, float o, float e) {

		float[][] scores = matrix.getScores();

		int m = s1.length() + 1;
		int n = s2.length() + 1;

		byte[] pointers = new byte[m * n];

		// Initializes the boundaries of the traceback matrix to STOP.

		short[] sizesOfVerticalGaps = new short[m * n];
		short[] sizesOfHorizontalGaps = new short[m * n];
		for (int i = 0, k = 0; i < m; i++, k += n) {
			for (int j = 0; j < n; j++) {
				sizesOfVerticalGaps[k + j] = sizesOfHorizontalGaps[k + j] = 1;
			}
		}

		double sim = construct(s1, s2, scores, o, e, sizesOfVerticalGaps, sizesOfHorizontalGaps);

		return sim;
	}

	private static double construct(Sequence s1, Sequence s2, float[][] matrix, float o, float e,
			short[] sizesOfVerticalGaps, short[] sizesOfHorizontalGaps) {

		char[] a1 = s1.toArray();
		char[] a2 = s2.toArray();

		int m = s1.length() + 1;
		int n = s2.length() + 1;

		float f; // score of alignment x1...xi to y1...yi if xi aligns to yi
		float[] g = new float[n]; // score if xi aligns to a gap after yi
		float h; // score if yi aligns to a gap after xi
		float[] v = new float[n]; // best score of alignment x1...xi to y1...yi
		float vDiagonal;

		g[0] = Float.NEGATIVE_INFINITY;
		h = Float.NEGATIVE_INFINITY;
		v[0] = 0;

		for (int j = 1; j < n; j++) {
			g[j] = Float.NEGATIVE_INFINITY;
			v[j] = 0;
		}
		float[][] hh = new float[m + 1][n + 1];
		hh[0][0] = 0;
		for (int i = 1; i < n + 1; i++) {
			hh[0][i] = (-o) + i * (-e);

		}
		for (int j = 1; j < m + 1; j++) {
			hh[j][0] = (-o) + j * (-e);

		}

		float similarityScore, g1, g2, h1, h2;

		for (int i = 1, k = n; i < m; i++, k += n) {
			h = Float.NEGATIVE_INFINITY;
			vDiagonal = v[0];

			for (int j = 1, l = k + 1; j < n; j++, l++) {
				similarityScore = matrix[a1[i - 1]][a2[j - 1]];

				// Fill the matrices
				// f = vDiagonal + similarityScore;
				f = hh[i - 1][j - 1] + similarityScore;

				g1 = g[j] - e;
				// g2 = v[j] - o;// 垂直

				g2 = hh[i - 1][j] - o - e;

				if (g1 > g2) {
					g[j] = g1;
					sizesOfVerticalGaps[l] = (short) (sizesOfVerticalGaps[l - n] + 1);
				} else {
					g[j] = g2;
				}

				h1 = h - e;
				// h2 = v[j - 1] - o;//水平

				h2 = hh[i][j - 1] - o - e;

				if (h1 > h2) {
					h = h1;
					sizesOfHorizontalGaps[l] = (short) (sizesOfHorizontalGaps[l - 1] + 1);
				} else {
					h = h2;
				}

				vDiagonal = v[j];
				v[j] = maximum(f, g[j], h);
				hh[i][j] = v[j];

			}
		}

		double sim = 0;
		if (flag == 1) {
			m--;
			n--;

			int fc = 0;
			double x = 0;
			double y = 0;

			if (m < n) {
				fc = m;
				m = n;
				n = fc;
			}
			/* 这里的-3代表替换得分，（o-e）代表（gap open - gap extend） */
			x = n - e * (m - n);

			y = -3 * n - e * (m - n);

			if (m != n) {
				x = x - (o - e);
				y = y - (o - e);
			}

			sim = (hh[a1.length][a2.length] - y) / (x - y);

		} else {// protein
			sim = hh[a1.length][a2.length];
		}

		return sim;
	}

	private static float maximum(float a, float b, float c) {
		if (a > b) {
			if (a > c) {
				return a;
			} else {
				return c;
			}
		} else if (b > c) {
			return b;
		} else {
			return c;
		}
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
		String[] ll = {

				"  4 -1  0  0 -3  1  0  0 -2  0 -1  0  1 -2 -1  1  1 -5 -4  1  0  0  0 -7 ",
				" -1  8 -2 -1 -2  3 -1 -2 -1 -3 -2  1  0 -1 -1 -1 -3  0  0 -1 -2  0 -1 -7 ",
				"  0 -2  8  1 -1 -1 -1  0 -1  0 -2  0  0 -1 -3  0  1 -7 -4 -2  4 -1  0 -7 ",
				"  0 -1  1  9 -3 -1  1 -1 -2 -4 -1  0 -3 -5 -1  0 -1 -4 -1 -2  5  0 -1 -7 ",
				" -3 -2 -1 -3 17 -2  1 -4 -5 -2  0 -3 -2 -3 -3 -2 -2 -2 -6 -2 -2  0 -2 -7 ",
				"  1  3 -1 -1 -2  8  2 -2  0 -2 -2  0 -1 -3  0 -1  0 -1 -1 -3 -1  4  0 -7 ",
				"  0 -1 -1  1  1  2  6 -2  0 -3 -1  2 -1 -4  1  0 -2 -1 -2 -3  0  5 -1 -7 ",
				"  0 -2  0 -1 -4 -2 -2  8 -3 -1 -2 -1 -2 -3 -1  0 -2  1 -3 -3  0 -2 -1 -7 ",
				" -2 -1 -1 -2 -5  0  0 -3 14 -2 -1 -2  2 -3  1 -1 -2 -5  0 -3 -2  0 -1 -7 ",
				"  0 -3  0 -4 -2 -2 -3 -1 -2  6  2 -2  1  0 -3 -1  0 -3 -1  4 -2 -3  0 -7 ",
				" -1 -2 -2 -1  0 -2 -1 -2 -1  2  4 -2  2  2 -3 -2  0 -2  3  1 -1 -1  0 -7 ",
				"  0  1  0  0 -3  0  2 -1 -2 -2 -2  4  2 -1  1  0 -1 -2 -1 -2  0  1  0 -7 ",
				"  1  0  0 -3 -2 -1 -1 -2  2  1  2  2  6 -2 -4 -2  0 -3 -1  0 -2 -1  0 -7 ",
				" -2 -1 -1 -5 -3 -3 -4 -3 -3  0  2 -1 -2 10 -4 -1 -2  1  3  1 -3 -4 -1 -7 ",
				" -1 -1 -3 -1 -3  0  1 -1  1 -3 -3  1 -4 -4 11 -1  0 -3 -2 -4 -2  0 -1 -7 ",
				"  1 -1  0  0 -2 -1  0  0 -1 -1 -2  0 -2 -1 -1  4  2 -3 -2 -1  0 -1  0 -7 ",
				"  1 -3  1 -1 -2  0 -2 -2 -2  0  0 -1  0 -2  0  2  5 -5 -1  1  0 -1  0 -7 ",
				" -5  0 -7 -4 -2 -1 -1  1 -5 -3 -2 -2 -3  1 -3 -3 -5 20  5 -3 -5 -1 -2 -7 ",
				" -4  0 -4 -1 -6 -1 -2 -3  0 -1  3 -1 -1  3 -2 -2 -1  5  9  1 -3 -2 -1 -7 ",
				"  1 -1 -2 -2 -2 -3 -3 -3 -3  4  1 -2  0  1 -4 -1  1 -3  1  5 -2 -3  0 -7 ",
				"  0 -2  4  5 -2 -1  0  0 -2 -2 -1  0 -2 -3 -2  0  0 -5 -3 -2  5  0 -1 -7 ",
				"  0  0 -1  0  0  4  5 -2  0 -3 -1  1 -1 -4  0 -1 -1 -1 -2 -3  0  4  0 -7 ",
				"  0 -1  0 -1 -2  0 -1 -1 -1  0  0  0  0 -1 -1  0  0 -2 -1  0 -1  0 -1 -7 ",
				" -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7 -7  1 "

		};

		for (int n = 0; n < ll.length; n++) {
			StringTokenizer tokenizer = new StringTokenizer(ll[n]);

			for (int j = 0; tokenizer.hasMoreTokens(); j++) {

				scores[acid[n]][acid[j]] = Float.parseFloat(tokenizer.nextToken());
			}
		}
		return scores;

	}

	public float[][] DnaMatrix() {
		char[] acid = { 'A', 'T', 'C', 'G', 'U' };
		String[] ll = { "1 -3 -3 -3 -3", "-3  1 -3  -3 -3", "-3 -3  1 -3 -3", "-3 -3 -3  1 -3", "-3 -3 -3  -3 1" };

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

		result = new TechnicalInformation(Type.MISC);
		result.setValue(TechnicalInformation.Field.AUTHOR, "hqf");
		result.setValue(TechnicalInformation.Field.YEAR, "2019");
		result.setValue(TechnicalInformation.Field.TITLE, "HQFSVM-SA");
		result.setValue(TechnicalInformation.Field.NOTE, "LibSVM was originally developed as 'HQFSVM'");
		result.setValue(TechnicalInformation.Field.URL, "#");
		result.setValue(TechnicalInformation.Field.NOTE,
				"The Weka classifier works with version 1.0 of HQFSVM-SA");

		return result;
	}

	@Override
	public String toString() {
		return "HQFSVM-SA wrapper, original code by hqf";
	}

	public static void main(String[] args) {

	}

}
