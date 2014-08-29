package com.weka.learning;

import java.io.FileReader;

import weka.classifiers.trees.J48;
import weka.core.Instances;

public class L02Classifier {

	public static void main(String[] args) throws Exception {
		FileReader fr = new FileReader(L01Instances.class.getClassLoader()
				.getResource("com/weka/learning/contact-lenses.arff").getFile());
		Instances instances = new Instances(fr);
		instances.setClassIndex(instances.numAttributes() - 1);
		J48 classifier = new J48();
		classifier.buildClassifier(instances);
		double classificationVal = classifier.classifyInstance(instances.instance(1));
		System.out.println(getContactLensesValue((int) classificationVal));
	}

	private static String getContactLensesValue(int intVal) {
		switch (intVal) {
		case 0:
			return "soft";
		case 1:
			return "hard";
		case 2:
			return "none";

		default:
			return "null";
		}
	}

}
