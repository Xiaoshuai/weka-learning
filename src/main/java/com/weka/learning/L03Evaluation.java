package com.weka.learning;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class L03Evaluation {

	public static void main(String[] args) throws Exception {
		FileReader fr = new FileReader(L01Instances.class.getClassLoader()
				.getResource("com/weka/learning/contact-lenses.arff").getFile());
		Instances instances = new Instances(fr);
		instances.setClassIndex(instances.numAttributes() - 1);
		System.out.println(">--------------------crossValidateModel--------------------<");
		J48 classifier = new J48();
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random(1));
		System.out.println(eval.toClassDetailsString());
		System.out.println(">----------------------------------------<");
		System.out.println(eval.toSummaryString());
		System.out.println(">----------------------------------------<");
		System.out.println(eval.toMatrixString());
		fr.close();

		System.out.println(">--------------------evaluateModel--------------------<");
		classifier = new J48();
		classifier.buildClassifier(instances);
		eval = new Evaluation(instances);
		eval.evaluateModel(classifier, instances);
		System.out.println(eval.toClassDetailsString());
		System.out.println(">----------------------------------------<");
		System.out.println(eval.toSummaryString());
		System.out.println(">----------------------------------------<");
		System.out.println(eval.toMatrixString());
	}

}
